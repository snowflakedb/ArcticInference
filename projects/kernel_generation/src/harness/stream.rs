//! Shared turn-streaming, retry, and terminal rendering.
//!
//! The AVO orchestrator drives turns through [`stream_turn_with_retry`]: open
//! the stream, render
//! every event as it arrives, and capture the terminal [`Event::Stop`]. The
//! retry wrapper keeps a long-running agent alive across rate limits and
//! transient provider failures, only surfacing [`CompletionError::Fatal`] (or
//! exhausted attempts) to the caller.

use std::io::{self, Write};
use std::time::Duration;

use futures::StreamExt;
use serde_json::Value;

use crate::ai::protocol::{ToolCall, ToolContent, ToolDescriptor};
use crate::ai::{BlockPhase, CompletionError, Event, ProtocolClient, StopReason, ThinkingEffort, Usage};
use crate::ui::{BOLD, DIM, GREEN, MAGENTA, RED, RESET, WHITE, YELLOW};

#[derive(Default)]
struct InlineRenderer {
    bold: bool,
    pending_star: bool,
}

impl InlineRenderer {
    fn render_delta(&mut self, text: &str, base_style: &str) -> String {
        let mut out = String::with_capacity(text.len());

        for ch in text.chars() {
            if ch == '*' {
                if self.pending_star {
                    self.pending_star = false;
                    self.bold = !self.bold;
                    if self.bold {
                        out.push_str(BOLD);
                    } else {
                        out.push_str(RESET);
                        out.push_str(base_style);
                    }
                } else {
                    self.pending_star = true;
                }
            } else {
                if self.pending_star {
                    out.push('*');
                    self.pending_star = false;
                }
                out.push(ch);
            }
        }

        out
    }

    fn finish(&mut self) -> String {
        self.bold = false;
        if std::mem::take(&mut self.pending_star) {
            "*".to_string()
        } else {
            String::new()
        }
    }
}

/// Aggregated result of one streamed model completion — the assistant *half* of
/// a turn (thinking + text + `tool_use` blocks + stop reason), captured before its
/// tool calls run. A full turn = this completion plus its tool results; the
/// orchestrator assembles that into a node.
pub struct Completion<M> {
    pub message: M,
    pub tool_calls: Vec<ToolCall>,
    pub reason: StopReason,
    pub usage: Usage,
    /// The agent's *visible* assistant text this turn (its short summaries like
    /// "X regressed — revert"), concatenated. Excludes hidden thinking. Captured
    /// so the orchestrator can feed the supervisor what the agent has tried and
    /// concluded, not just the committed lineage.
    pub text: String,
}

/// Backoff policy for [`stream_turn_with_retry`].
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 12,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_mins(2),
        }
    }
}

/// Exponential backoff `base * 2^(attempt-1)`, capped at `max_delay`, plus up
/// to 25% jitter to avoid thundering-herd retries across parallel workers.
fn backoff_delay(attempt: u32, cfg: &RetryConfig) -> Duration {
    let exp = cfg.base_delay.saturating_mul(1u32 << attempt.saturating_sub(1).min(16));
    let capped = exp.min(cfg.max_delay);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let jitter_frac = f64::from(nanos % 1000) / 1000.0 * 0.25;
    capped.saturating_add(Duration::from_secs_f64(capped.as_secs_f64() * jitter_frac))
}

/// Stream one turn with retry-on-transient/rate-limit. Returns the aggregated
/// [`Completion`] on success, or the last [`CompletionError`] after exhausting
/// attempts (or immediately on [`CompletionError::Fatal`]).
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
pub async fn stream_turn_with_retry<C>(
    client: &C,
    messages: &[C::Message],
    tools: &[ToolDescriptor],
    max_output_tokens: usize,
    thinking: ThinkingEffort,
    cfg: &RetryConfig,
) -> Result<Completion<C::Message>, CompletionError>
where
    C: ProtocolClient,
{
    let mut attempt: u32 = 0;
    loop {
        match stream_turn(client, messages, tools, max_output_tokens, thinking).await {
            Ok(turn) => return Ok(turn),
            Err(err) => {
                attempt = attempt.saturating_add(1);
                let delay = match &err {
                    CompletionError::Fatal(_) => return Err(err),
                    CompletionError::RateLimited { retry_after, .. } => {
                        let backoff = backoff_delay(attempt, cfg);
                        retry_after.map_or(backoff, |provider_delay| provider_delay.max(backoff))
                    }
                    CompletionError::Transient(_) => backoff_delay(attempt, cfg),
                };
                if attempt >= cfg.max_attempts {
                    return Err(err);
                }
                eprintln!(
                    "\n{YELLOW}{err}; retrying in {:.1}s (attempt {attempt}/{}){RESET}",
                    delay.as_secs_f64(),
                    cfg.max_attempts,
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Drive one streaming turn end-to-end: open the stream, render every event to
/// the terminal as it arrives, capture the terminal [`Event::Stop`], and return
/// its payload. A single attempt — see [`stream_turn_with_retry`] for the
/// retrying wrapper.
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
pub async fn stream_turn<C>(
    client: &C,
    messages: &[C::Message],
    tools: &[ToolDescriptor],
    max_output_tokens: usize,
    thinking: ThinkingEffort,
) -> Result<Completion<C::Message>, CompletionError>
where
    C: ProtocolClient,
{
    let mut stream = client.complete(messages, tools, max_output_tokens, thinking).await?;

    let mut finalized: Option<Completion<C::Message>> = None;
    // Accumulate visible assistant text across deltas so the finalized
    // `Completion` carries it (see `Completion::text`).
    let mut text_acc = String::new();
    let mut text_renderer = InlineRenderer::default();

    while let Some(item) = stream.next().await {
        match item? {
            Event::Text(BlockPhase::Start) => {
                print!("\n{YELLOW}\u{23fa} {WHITE}");
                io::stdout().flush().ok();
            }
            Event::Text(BlockPhase::Delta(text)) => {
                print!("{}", text_renderer.render_delta(&text, WHITE));
                io::stdout().flush().ok();
                text_acc.push_str(&text);
            }
            Event::Text(BlockPhase::End) => {
                print!("{}{RESET}", text_renderer.finish());
                io::stdout().flush().ok();
            }

            Event::Thinking(BlockPhase::Start) => {
                print!("\n{YELLOW}\u{23fa} {DIM}");
                io::stdout().flush().ok();
            }
            Event::Thinking(BlockPhase::Delta(text)) => {
                print!("{text}");
                io::stdout().flush().ok();
            }
            Event::Thinking(BlockPhase::End) => {
                print!("{RESET}");
                io::stdout().flush().ok();
            }

            // Tool-use args stream as partial JSON; suppress here and render a
            // clean preview from the finalized call below.
            Event::Opaque(_) => {}

            Event::Stop {
                reason,
                usage,
                message,
                tool_calls,
            } => {
                print_stop(&reason, usage);
                finalized = Some(Completion {
                    message,
                    tool_calls,
                    reason,
                    usage,
                    text: std::mem::take(&mut text_acc),
                });
            }
        }
    }

    finalized.ok_or_else(|| CompletionError::Transient("stream ended without an Event::Stop".to_string()))
}

/// Stream one turn with no tools, rendering and returning the concatenated
/// assistant text. Used by context compaction to obtain a summary.
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
pub async fn collect_text_turn<C>(
    client: &C,
    messages: &[C::Message],
    max_output_tokens: usize,
    thinking: ThinkingEffort,
) -> Result<String, CompletionError>
where
    C: ProtocolClient,
{
    let mut stream = client.complete(messages, &[], max_output_tokens, thinking).await?;
    let mut text = String::new();
    let mut text_renderer = InlineRenderer::default();
    while let Some(item) = stream.next().await {
        match item? {
            Event::Text(BlockPhase::Start) => {
                print!("\n{DIM}");
                io::stdout().flush().ok();
            }
            Event::Text(BlockPhase::Delta(t)) => {
                print!("{}", text_renderer.render_delta(&t, DIM));
                io::stdout().flush().ok();
                text.push_str(&t);
            }
            Event::Text(BlockPhase::End) => {
                print!("{}{RESET}", text_renderer.finish());
                io::stdout().flush().ok();
            }
            _ => {}
        }
    }
    Ok(text)
}

pub fn print_stop(reason: &StopReason, usage: Usage) {
    let ctx = usage
        .input_tokens
        .saturating_add(usage.cache_write_tokens)
        .saturating_add(usage.cache_read_tokens)
        .saturating_add(usage.output_tokens);
    println!(
        "\n{MAGENTA}stop: {:?} | {}{BOLD}\u{2b61}\u{2b63} {RESET}{MAGENTA}{}, ctx: {}{RESET} | {DIM}cache w={} r={} reasoning={}{RESET}",
        reason,
        humanize(usage.input_tokens),
        humanize(usage.output_tokens),
        humanize(ctx),
        humanize(usage.cache_write_tokens),
        humanize(usage.cache_read_tokens),
        humanize(usage.reasoning_tokens),
    );
}

pub fn humanize(n: u32) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", f64::from(n) / 1_000.0)
    } else {
        format!("{:.2}M", f64::from(n) / 1_000_000.0)
    }
}

/// Render a tool invocation: `\u{23fa} Name(key=value, ...)`.
pub fn print_tool_call(name: &str, input: &Value) {
    let display_name = capitalize_first(name);
    let preview = arg_preview(input);
    println!("\n{GREEN}\u{23fa} {display_name}{RESET}({DIM}{preview}{RESET})");
}

/// Render the tool's response as a follow-up arrow under the call.
pub fn print_tool_result(content: &[ToolContent], is_error: bool) {
    let mut text_first_line: Option<&str> = None;
    let mut image_count = 0usize;
    for block in content {
        match block {
            ToolContent::Text(s) if text_first_line.is_none() => {
                text_first_line = s.lines().next();
            }
            ToolContent::Text(_) => {}
            ToolContent::Image { .. } => image_count = image_count.saturating_add(1),
        }
    }
    let preview = truncate_chars(text_first_line.unwrap_or(""), 100);
    let suffix = if image_count > 0 {
        format!(" [+{image_count} image{}]", if image_count == 1 { "" } else { "s" })
    } else {
        String::new()
    };
    let color = if is_error { RED } else { DIM };
    println!("{color}  \u{21aa} {preview}{suffix}{RESET}");
}

fn arg_preview(input: &Value) -> String {
    let raw = match input {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", "),
        _ => input.to_string(),
    };
    truncate_chars(&raw, 80)
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    chars
        .next()
        .map_or_else(String::new, |c| c.to_uppercase().chain(chars).collect())
}

fn truncate_chars(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}\u{2026}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_config_defaults_tolerate_long_rate_limit_windows() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_attempts, 12);
        assert_eq!(cfg.base_delay, Duration::from_secs(2));
        assert_eq!(cfg.max_delay, Duration::from_mins(2));
    }

    #[test]
    fn backoff_delay_is_exponential_and_capped() {
        let cfg = RetryConfig {
            max_attempts: 12,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(10),
        };

        let first = backoff_delay(1, &cfg);
        assert!(first >= Duration::from_secs(2));
        assert!(first <= Duration::from_millis(2500));

        let capped = backoff_delay(8, &cfg);
        assert!(capped >= Duration::from_secs(10));
        assert!(capped <= Duration::from_millis(12_500));
    }

    #[test]
    fn inline_renderer_bolds_markdown_pair() {
        let mut renderer = InlineRenderer::default();
        assert_eq!(
            renderer.render_delta("hello **aaa**", WHITE),
            format!("hello {BOLD}aaa{RESET}{WHITE}")
        );
        assert_eq!(renderer.finish(), "");
    }

    #[test]
    fn inline_renderer_bolds_across_split_deltas() {
        let mut renderer = InlineRenderer::default();
        assert_eq!(renderer.render_delta("hello *", WHITE), "hello ");
        assert_eq!(renderer.render_delta("*aa", WHITE), format!("{BOLD}aa"));
        assert_eq!(
            renderer.render_delta("a** done", WHITE),
            format!("a{RESET}{WHITE} done")
        );
        assert_eq!(renderer.finish(), "");
    }

    #[test]
    fn inline_renderer_flushes_unmatched_star() {
        let mut renderer = InlineRenderer::default();
        assert_eq!(renderer.render_delta("hello *", WHITE), "hello ");
        assert_eq!(renderer.finish(), "*");
    }

    #[test]
    fn inline_renderer_restores_requested_base_style() {
        let mut renderer = InlineRenderer::default();
        assert_eq!(renderer.render_delta("**aaa**", DIM), format!("{BOLD}aaa{RESET}{DIM}"));
    }
}
