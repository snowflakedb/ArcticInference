//! Headless self-test: drive the real AVO loop with a scripted client.
//!
//! [`ScriptedClient`] implements [`ProtocolClient`] with a fixed sequence of
//! turns instead of calling a model. Running `run_avo` with it exercises the
//! *entire* orchestration end-to-end with no network and no API budget: sandbox
//! setup, the read-only trusted evaluator on the real GPU/MPS backend, every
//! tool (`read` / `gpu_job` / `evaluate` / `write` / `search_view`),
//! candidate archiving, git-backed promotion, budget accounting, the manifest,
//! and crash-safe journaling.
//!
//! The script rewrites v0 (the naive `Reference` baseline) with a small,
//! self-contained torch solution that swaps its materialized-score softmax for
//! torch's fused scaled-dot-product attention. It then runs a full evaluation
//! and attempts a submit. Whether the candidate is promoted depends on the
//! backend — on CUDA fused SDPA easily beats the naive baseline, while on MPS
//! the baseline's explicit matmuls are already competitive, so the submit may be
//! accepted or rejected purely on measured speed. Either way the self-test
//! drives the entire pipeline
//! (sandbox → run → evaluate → write → submit → lineage → export) end-to-end
//! with no network, asserting the *mechanics* ran, not a particular verdict.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ai::protocol::{ToolCall, ToolDescriptor, ToolResult};
use crate::ai::{BlockPhase, CompletionError, Event, ProtocolClient, StopReason, ThinkingEffort, Usage};
use crate::domain::types::Manifest;
use crate::orchestrator::strategy;

/// Verify the manifest `avo --selftest` just wrote is fully instrumented.
///
/// Post-run assertion for `avo --selftest`: re-read the manifest the loop just
/// wrote and verify the instrumentation is populated — the policy id, tree
/// history file, a
/// non-empty `params` snapshot, and at least the v0 best-vs-budget point. The
/// loop itself can't `assert!` the verdict (promotion is backend-dependent), so
/// this checks the *instrumentation* that must be present either way. Returns
/// `Err` naming the first missing field.
///
/// # Errors
///
/// Returns an `Err` naming the first problem found: `manifest.json` missing or
/// unreadable, malformed JSON, a `policy_id`/`strategy_id` that is not one of the
/// known policies or that disagrees with `params.policy`, an empty `params.model`,
/// a zero `params.max_output_tokens`, an unset `params.min_improvement_frac`, a
/// missing `history/nodes.jsonl`, or an empty best-vs-budget curve.
pub fn verify_run(run_dir: &Path) -> Result<(), String> {
    let path = run_dir.join("manifest.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("self-test: cannot read {}: {e}", path.display()))?;
    let m: Manifest = serde_json::from_slice(&bytes).map_err(|e| format!("self-test: manifest did not parse: {e}"))?;

    if !matches!(m.policy_id.as_str(), strategy::POLICY_AVO | strategy::POLICY_BEAM) || m.strategy_id != m.policy_id {
        return Err(format!(
            "self-test: unexpected policy_id/strategy_id {:?}/{:?}",
            m.policy_id, m.strategy_id
        ));
    }
    if !run_dir.join("history/nodes.jsonl").exists() {
        return Err("self-test: history/nodes.jsonl was not written".to_string());
    }
    if m.params.policy != m.policy_id {
        return Err(format!("self-test: unexpected params.policy {:?}", m.params.policy));
    }
    if m.params.model.is_empty() {
        return Err("self-test: params.model is empty".to_string());
    }
    if m.params.max_output_tokens == 0 {
        return Err("self-test: params.max_output_tokens is zero".to_string());
    }
    if m.params.min_improvement_frac <= 0.0 {
        return Err("self-test: params.min_improvement_frac is not set".to_string());
    }
    if m.curve.is_empty() {
        return Err(
            "self-test: curve has no points (expected >= 1 once the first correct candidate is scored)".to_string(),
        );
    }
    Ok(())
}

/// The candidate the script writes: a small, dependency-free torch solution
/// that replaces the naive baseline's materialized-score softmax with torch's
/// fused scaled-dot-product attention. It is correct everywhere (standard
/// attention math, matching the naive `Reference` within tolerance) and depends
/// on no external solution file, so the scripted write→evaluate→submit path runs
/// end-to-end with no network — whether the submission is accepted depends on the
/// backend's measured speed.
const SDPA_SOLUTION: &str = r"from __future__ import annotations

import torch
import torch.nn as nn


class Solution(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(self, Q, K, V):
        # Correct fused attention matching the naive Reference: GQA expand, the
        # reference's bottom-right-aligned causal mask, fp32 compute -> bf16 out.
        Lq, Lk = Q.shape[-2], K.shape[-2]
        rep = Q.shape[1] // K.shape[1]
        Kf = K.repeat_interleave(rep, dim=1) if rep > 1 else K
        Vf = V.repeat_interleave(rep, dim=1) if rep > 1 else V
        attn_mask = None
        if self.causal:
            q_pos = torch.arange(Lq, device=Q.device).unsqueeze(1) + (Lk - Lq)
            k_pos = torch.arange(Lk, device=Q.device).unsqueeze(0)
            attn_mask = k_pos <= q_pos  # bool: True = allowed (SDPA convention)
        out = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), Kf.float(), Vf.float(), attn_mask=attn_mask
        )
        return out.to(Q.dtype)
";

/// Minimal provider-shaped message for the scripted client.
///
/// The mock ignores transcript content (it is driven by a turn counter), but the
/// type must round-trip through serde so the run journal / resume path is
/// exercised like any real provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScriptedMessage {
    User(String),
    Assistant(String),
    ToolResults(usize),
}

/// A scripted completion client: the Nth `complete` call emits the Nth turn.
pub struct ScriptedClient {
    turn: AtomicUsize,
}

impl ScriptedClient {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            turn: AtomicUsize::new(0),
        }
    }
}

impl Default for ScriptedClient {
    fn default() -> Self {
        Self::new()
    }
}

fn tool_call(id: &str, name: &str, input: Value) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        input,
        parse_error: None,
    }
}

/// The fixed script. Returns `(assistant_text, tool_calls, stop_reason)` for
/// turn `index`; once exhausted it emits [`StopReason::Done`] so the loop stops.
fn step(index: usize) -> (String, Vec<ToolCall>, StopReason) {
    let (text, calls): (&str, Vec<ToolCall>) = match index {
        0 => (
            "Reading the current candidate.",
            vec![tool_call("s0", "read", json!({"path": "solution/solution.py"}))],
        ),
        1 => (
            "Checking the search view so far.",
            vec![tool_call("s1", "search_view", json!({}))],
        ),
        2 => (
            "Profiling the seed to understand the workload.",
            vec![tool_call(
                "s2",
                "gpu_job",
                json!({"command": "python3 _trusted/run_kernel.py 0 5"}),
            )],
        ),
        3 => (
            "Sanity-checking the seed's correctness.",
            vec![tool_call("s3", "evaluate", json!({"stage": "correctness"}))],
        ),
        4 => (
            "Replacing the materialized-score softmax with torch's fused SDPA.",
            vec![tool_call(
                "s4",
                "write",
                json!({"path": "solution/solution.py", "file_text": SDPA_SOLUTION}),
            )],
        ),
        5 => (
            "Full evaluation of the fused-SDPA rewrite — a scored candidate for the search.",
            vec![tool_call("s5", "evaluate", json!({"stage": "full"}))],
        ),
        6 => (
            "Checking the search view again.",
            vec![tool_call("s6", "search_view", json!({}))],
        ),
        _ => {
            return ("Self-test script complete.".to_string(), Vec::new(), StopReason::Done);
        }
    };
    (text.to_string(), calls, StopReason::ToolUse)
}

impl ProtocolClient for ScriptedClient {
    type Message = ScriptedMessage;
    type Stream = futures::stream::Iter<std::vec::IntoIter<Result<Event<ScriptedMessage>, CompletionError>>>;

    async fn complete(
        &self,
        _ctx: &[Self::Message],
        _tools: &[ToolDescriptor],
        _max_output_tokens: usize,
        _thinking: ThinkingEffort,
    ) -> Result<Self::Stream, CompletionError> {
        let index = self.turn.fetch_add(1, Ordering::SeqCst);
        let (text, tool_calls, reason) = step(index);
        let events = vec![
            Ok(Event::Text(BlockPhase::Start)),
            Ok(Event::Text(BlockPhase::Delta(text.clone()))),
            Ok(Event::Text(BlockPhase::End)),
            Ok(Event::Stop {
                reason,
                usage: Usage::default(),
                message: ScriptedMessage::Assistant(text),
                tool_calls,
            }),
        ];
        Ok(futures::stream::iter(events))
    }

    fn user_message(&self, text: String) -> Self::Message {
        ScriptedMessage::User(text)
    }

    fn tool_result_messages(&self, results: Vec<ToolResult>) -> Vec<Self::Message> {
        vec![ScriptedMessage::ToolResults(results.len())]
    }

    /// Mirror the real providers: only an assistant message is a safe tail start
    /// (a `ToolResults` turn would orphan; a `User` turn would double up).
    fn is_compaction_tail_start(&self, msg: &Self::Message) -> bool {
        matches!(msg, ScriptedMessage::Assistant(_))
    }
}
