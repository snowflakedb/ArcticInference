//! Turn execution mechanics: running a completion's tool calls and classifying
//! a tool-less completion. Policy-agnostic — the orchestrator above decides what
//! the outcomes *mean* (commit accounting, which prompt to nudge with); this
//! layer just drives the model's requested work and reports what happened.

use crate::ai::StopReason;
use crate::ai::protocol::{ToolCall, ToolContent, ToolResult};
use crate::harness::stream;
use crate::tool::clamp::{Limits, clamp_reply};
use crate::tool::{ToolBox, ToolReply};

/// Run the turn's tool calls in order, printing each call + result, and return
/// the results in call order. Pure w.r.t. run state and free of any policy
/// vocabulary (commit/eval/…): the caller derives those from the tool-call names
/// it issued plus these results.
pub async fn dispatch_tools(toolbox: &ToolBox, tool_calls: Vec<ToolCall>) -> Vec<ToolResult> {
    let limits = Limits::default();
    let mut results = Vec::with_capacity(tool_calls.len());
    for call in tool_calls {
        stream::print_tool_call(&call.name, &call.input);
        // Central backstop: every result — Ok reply AND Err string — passes one
        // mandatory clamp so no tool (including an under-capping/new one, or a
        // tool that returns a huge *error*) can inject an oversized `tool_result`
        // that replays on every retry. A correctly self-capping tool is a no-op here.
        //
        // The unparseable-arguments case is handled inside `ToolBox::dispatch` and
        // arrives here as an ordinary error, so it clamps on the same path as any
        // other tool failure. `ToolError::never_ran()` distinguishes the two if a
        // caller ever wants to; the model sees an error `tool_result` either way.
        let (content, is_error) = match toolbox.dispatch(&call).await {
            Ok(reply) => (Vec::<ToolContent>::from(clamp_reply(reply, &limits).reply), false),
            Err(e) => {
                let clamped = clamp_reply(ToolReply(vec![ToolContent::Text(e.to_string())]), &limits);
                (Vec::<ToolContent>::from(clamped.reply), true)
            }
        };
        stream::print_tool_result(&content, is_error);
        results.push(ToolResult {
            tool_call_id: call.id,
            content,
            is_error,
        });
    }
    results
}

/// Synthetic error results for the tool calls of a completion that was cut off at
/// the output-token cap: the JSON arguments may be incomplete or garbled, so we do
/// NOT execute them. Each `tool_use` is answered with an error `tool_result` (which
/// keeps the `tool_use/tool_result` pairing valid for the next provider call) telling
/// the model to re-issue with complete arguments. Mirrors the tool-call/result
/// printing of [`dispatch_tools`] so transcripts look consistent.
pub fn truncated_tool_results(tool_calls: &[ToolCall], message: &str) -> Vec<ToolResult> {
    tool_calls
        .iter()
        .map(|call| {
            stream::print_tool_call(&call.name, &call.input);
            let content = vec![ToolContent::Text(message.to_string())];
            stream::print_tool_result(&content, true);
            ToolResult {
                tool_call_id: call.id.clone(),
                content,
                is_error: true,
            }
        })
        .collect()
}

/// A completion that issued no tool calls: either the model hit the output cap
/// mid-thought (resume it) or it stopped without acting (nudge it to continue).
/// The orchestrator maps each variant to the prompt it nudges with.
pub enum Nonproductive {
    Truncated,
    Idle,
}

/// Classify a completed turn from its stop reason. `None` means the agent acted
/// (has tool calls to dispatch); `Some(_)` means an unproductive turn that needs
/// a nudge. Pure over provider semantics — carries no policy prompts.
pub const fn classify_nonproductive(reason: &StopReason, has_tools: bool) -> Option<Nonproductive> {
    match reason {
        // `Done` is handled by the run-complete break before this is reached.
        StopReason::ToolUse | StopReason::Done => None,
        // Truncated but with tool calls: the caller must NOT execute them (args may
        // be garbled) — it answers each with a re-issue error via
        // [`truncated_tool_results`]. Not a no-tool nudge, so `None` here.
        StopReason::MaxTokens if has_tools => None,
        StopReason::MaxTokens => Some(Nonproductive::Truncated),
        StopReason::EndTurn | StopReason::Other(_) => Some(Nonproductive::Idle),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_tool_results_are_reissue_errors_matching_calls() {
        let calls = vec![
            ToolCall {
                id: "c1".into(),
                name: "evaluate".into(),
                input: serde_json::json!({"stage": "full"}),
                parse_error: None,
            },
            ToolCall {
                id: "c2".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
                parse_error: None,
            },
        ];
        let results = truncated_tool_results(&calls, "re-issue with complete arguments");
        assert_eq!(results.len(), 2, "one result per tool_use keeps the pairing valid");
        assert_eq!(results[0].tool_call_id, "c1");
        assert_eq!(results[1].tool_call_id, "c2");
        assert!(
            results.iter().all(|r| r.is_error),
            "truncated calls are error results, not runs"
        );
        assert!(matches!(&results[0].content[0], ToolContent::Text(t) if t.contains("re-issue")));
    }

    #[test]
    fn truncated_with_tools_is_not_a_no_tool_nudge() {
        // has_tools truncation is handled by the dispatch-side guard, not a nudge.
        assert!(classify_nonproductive(&StopReason::MaxTokens, true).is_none());
        // No tools ⇒ genuine truncation nudge.
        assert!(matches!(
            classify_nonproductive(&StopReason::MaxTokens, false),
            Some(Nonproductive::Truncated)
        ));
    }
}
