//! Context compaction.
//!
//! A long optimization run will eventually fill the model's context window. Left
//! unchecked, the next request errors ("prompt too long") and the run dies. This
//! module keeps a run alive by summarizing the OLDER part of the transcript in
//! place while keeping the most recent messages VERBATIM, then continuing.
//!
//! Design (ported from pi + opencode; see `references/`):
//! - **Real-window trigger.** Compact when the last turn's context crosses
//!   `context_window - reserve`, where `reserve` leaves room for the model's next
//!   output plus one turn of growth (opencode `usable() = limit.input - min(20k,
//!   maxOut)`; pi `contextWindow - reserveTokens`). The window is the serving
//!   model's real input cap, plumbed from `--context-window-tokens`.
//! - **Verbatim recent tail.** Only the older prefix is summarized; the last
//!   ~[`KEEP_RECENT_TOKENS`] of messages are kept as-is, cut at a provider-safe
//!   boundary so a `tool_result` is never orphaned and the rebuilt transcript
//!   never doubles a user turn.
//! - **Rolling summary.** Any prior brief sits at the head of the summarized
//!   prefix and re-enters the summarizer, which is told to update-and-merge it
//!   rather than regenerate cold. A fixed section template ([`prompts::COMPACT_TEMPLATE`])
//!   keeps exact paths/symbols/hashes in stable slots.
//! - **Cheap prune pre-pass.** Before the (paid) summarizer call, stale tool
//!   OUTPUTS in the prefix are erased in place (calls kept) at zero LLM cost.
//! - **Overflow self-guard.** The summarizer request must itself fit the window;
//!   if the prefix is pathologically large the oldest prefix messages are dropped
//!   (logged) so the shrink-the-context call can't itself overflow (opencode).
//!
//! Lineage and the archive are unaffected (they live on disk), so the agent
//! loses chat scrollback but not its actual progress. A deterministic ground-truth
//! `anchor` (authoritative current best + on-disk-vs-best, from the git lineage) is
//! folded into the summary message AHEAD of the model brief and marked
//! authoritative, so a drifting brief can't make the agent disown its own commits
//! or mis-attribute which node produced the best score.

use serde::Serialize;

use crate::ai::{CompletionError, ProtocolClient, ThinkingEffort};
use crate::domain::convert::f64_to_u32_saturating;
use crate::harness::stream;
use crate::orchestrator::prompts;

/// Floor for the reserve carved out below the window.
///
/// Where opencode uses `min(20k, maxOut)`, we use `max(maxOut, 20k)` so a large
/// per-turn output cap always fits. Ensures we compact with room to spare even
/// if `max_output_tokens` is small.
pub const MIN_RESERVE_TOKENS: u32 = 20_000;
/// Extra headroom on top of the output reserve for one turn of input growth
/// (tool results appended before compaction can next run). Matches pi's default
/// `reserveTokens` (16384).
pub const TURN_HEADROOM_TOKENS: u32 = 16_384;
/// Tokens of the newest messages kept VERBATIM (not summarized).
///
/// Matches pi's `keepRecentTokens` default; opencode keeps ~2k-8k. 20k keeps
/// enough recent state that the agent doesn't lose its in-flight edit/eval
/// across a compaction.
pub const KEEP_RECENT_TOKENS: usize = 20_000;
/// Cap on the summary turn's output.
const SUMMARY_MAX_TOKENS: usize = 8_000;
/// Safety margin (tokens) between the summarizer's estimated input and the window,
/// on top of the reserved [`SUMMARY_MAX_TOKENS`] output: absorbs role/framing
/// overhead and the roughness of the bytes/4 estimate so the overflow self-guard
/// leaves real headroom.
const SUMMARY_INPUT_MARGIN_TOKENS: usize = 8_000;
/// Prune pre-pass: protect this many tokens of the newest messages from tool-output
/// erasure (roughly the last couple of turns).
const PRUNE_PROTECT_TOKENS: usize = 40_000;

/// The reserve (tokens) to keep free below the real window before compacting.
///
/// Room for the model's next output plus one turn of input growth.
/// `reserve = max(max_output_tokens, MIN_RESERVE) + TURN_HEADROOM`.
#[must_use]
pub fn reserve_tokens(max_output_tokens: u32) -> u32 {
    max_output_tokens
        .max(MIN_RESERVE_TOKENS)
        .saturating_add(TURN_HEADROOM_TOKENS)
}

/// Fraction of the usable budget (`window − reserve`) at which we proactively compact.
///
/// The old trigger fired only at ~100% of the budget, so a long plateau ran
/// entirely inside one un-compacted context and the agent never re-grounded
/// mid-run (it lost track of its own committed best). Firing at a fraction lets
/// a mid-run re-ground happen; kept high enough (0.65) that it does not thrash —
/// after each compaction the context resets and must grow back to this fraction
/// before compacting again.
pub const COMPACTION_BUDGET_FRACTION: f32 = 0.65;

/// Whether the last turn's context crossed the compaction threshold.
///
/// The threshold is [`COMPACTION_BUDGET_FRACTION`] of the usable budget (the
/// real window minus the reserve). Saturating so an over-large reserve (bigger
/// than the window) just yields a threshold of 0 (always compact) rather than
/// underflowing.
#[must_use]
pub fn should_compact(last_context_tokens: u32, context_window_tokens: u32, reserve_tokens: u32) -> bool {
    let usable = context_window_tokens.saturating_sub(reserve_tokens);
    let threshold = f64_to_u32_saturating(f64::from(usable) * f64::from(COMPACTION_BUDGET_FRACTION));
    last_context_tokens > threshold
}

/// Rough per-message token estimate: serialized bytes / 4 (the standard ~4
/// chars/token heuristic). Only used to size the recent-tail cut, so an
/// approximation is fine.
fn estimate_tokens<M: Serialize>(msg: &M) -> usize {
    serde_json::to_string(msg).map_or(0, |s| s.len() / 4)
}

/// Index at which the verbatim recent tail should begin.
///
/// Walk newest → oldest summing token estimates until the kept tail reaches
/// `keep_recent_tokens`, then move the boundary OLDER until it lands on a
/// provider-`safe_start` message — one that can legally follow the synthesized
/// summary user-turn (an assistant turn on Anthropic; a non-`tool_output` turn on
/// `OpenAI`). This guarantees the rebuilt `[summary, ...tail]` never orphans a
/// `tool_result` nor produces two consecutive user turns.
///
/// Returns a value in `0..=len`. `0` means "no safe older boundary / transcript
/// too short" — the caller keeps everything and skips summarization.
fn tail_start_index(safe_start: &[bool], token_est: &[usize], keep_recent_tokens: usize) -> usize {
    let len = safe_start.len();
    debug_assert_eq!(len, token_est.len());
    if len == 0 {
        return 0;
    }
    // 1) Newest index whose tail (messages[cut..]) holds >= keep_recent_tokens.
    let mut total = 0usize;
    let mut cut = 0usize;
    for (i, est) in token_est.iter().enumerate().rev() {
        total = total.saturating_add(*est);
        cut = i;
        if total >= keep_recent_tokens {
            break;
        }
    }
    // 2) Move the boundary older until it is a safe tail-start (keeps a bit more
    //    verbatim, never less). If none exists above 0, cut collapses to 0.
    safe_start
        .get(..=cut)
        .map_or(0, |head| head.iter().rposition(|&safe| safe).unwrap_or(0))
}

/// First prefix index to feed the summarizer, so the summarizer request itself
/// fits the window (opencode's self-guard). The prefix to summarize is
/// `messages[start..cut]`; we drop the OLDEST messages (advancing `start` from 0)
/// until `sum(token_est[start..cut]) + prompt_est <= summary_input_budget`. The
/// rolling brief + the ground-truth anchor carry the load-bearing state forward,
/// so the oldest raw turns are the safest to shed. Returns a value in `0..=cut`;
/// `cut` means the whole prefix had to be dropped (nothing left to summarize).
fn summary_prefix_start(token_est: &[usize], cut: usize, prompt_est: usize, summary_input_budget: usize) -> usize {
    let prefix = token_est.get(..cut).unwrap_or(token_est);
    let mut prefix_tokens: usize = prefix.iter().sum();
    let mut start = 0usize;
    for est in prefix {
        if prefix_tokens.saturating_add(prompt_est) <= summary_input_budget {
            break;
        }
        prefix_tokens = prefix_tokens.saturating_sub(*est);
        start = start.saturating_add(1);
    }
    start
}

/// Summarize the OLDER prefix of `messages` in place, keeping the most recent
/// ~`keep_recent_tokens` VERBATIM.
///
/// On success `messages` becomes `[summary user-msg (anchor folded in),
/// ...verbatim recent tail]`. On failure the transcript is left untouched (the
/// caller logs and continues uncompacted).
///
/// `anchor` is a ground-truth note (the authoritative current best from the git
/// lineage) folded into the *same* summary message — kept in one message rather
/// than a second user turn so providers requiring strict user/assistant
/// alternation (Anthropic) don't choke. A model-written summary can silently lag
/// on-disk reality, so the anchor is placed FIRST (authoritative-by-construction),
/// with the model brief after it: the future self reads ground truth before any
/// narrative and is told to trust the anchor where they disagree.
///
/// `context_window_tokens` is the serving model's real input cap; it bounds the
/// summarizer call itself (opencode's self-guard) so a pathologically large prefix
/// can't overflow the very request meant to shrink it.
///
/// # Errors
///
/// Returns the [`CompletionError`] from the summarizer turn (provider failure, or
/// retries exhausted), or [`CompletionError::Transient`] when the summarizer
/// returns a blank brief — replacing the prefix with nothing would silently erase
/// the transcript, so the caller keeps it uncompacted instead.
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
pub async fn compact<C: ProtocolClient>(
    client: &C,
    messages: &mut Vec<C::Message>,
    anchor: &str,
    gotchas: &str,
    keep_recent_tokens: usize,
    context_window_tokens: u32,
) -> Result<(), CompletionError> {
    // Durable, model-visible view (older heavy artifacts already elided to
    // placeholders by the provider).
    let mut visible = client.model_visible_history(messages);
    if visible.is_empty() {
        return Ok(());
    }

    // [NICE] Cheap prune pre-pass: erase OLD tool OUTPUTS in place (keep the
    // CALLS), protecting the newest PRUNE_PROTECT_TOKENS. Zero LLM cost; the
    // provider decides what/how (default no-op).
    let freed = client.prune_old_tool_outputs(&mut visible, PRUNE_PROTECT_TOKENS);
    if freed > 0 {
        println!("[compaction] prune pre-pass freed ~{freed} tokens of stale tool output");
    }

    // Split into a summarized prefix and a verbatim tail cut at a safe boundary.
    let safe_start: Vec<bool> = visible.iter().map(|m| client.is_compaction_tail_start(m)).collect();
    let token_est: Vec<usize> = visible.iter().map(estimate_tokens).collect();
    let cut = tail_start_index(&safe_start, &token_est, keep_recent_tokens);

    // Nothing older to summarize (short transcript, or no safe boundary): keep
    // the (possibly pruned) transcript verbatim.
    if cut == 0 {
        *messages = visible;
        return Ok(());
    }

    // Summarize the older prefix only. Any earlier brief sits at the head of the
    // prefix and re-enters the summarizer, which the prompt tells to roll forward.
    //
    // Overflow self-guard (opencode): the summarizer request itself must fit the
    // window. If the prefix is so large that prefix + prompt + reserved summary
    // output would overflow, drop the OLDEST prefix messages until it fits — the
    // rolling brief + the ground-truth anchor carry the load-bearing state forward,
    // so the oldest raw turns are the safest to shed. The prune pre-pass above runs
    // first, so under the proactive trigger this never fires; it only bites the
    // reactive overflow path (real window smaller than assumed). Never silent.
    let prompt_est = prompts::COMPACT.len().saturating_add(prompts::COMPACT_TEMPLATE.len()) / 4;
    // Unreachable (`context_window_tokens` is u32), but 0 is the direction that
    // fails safe: it sheds prefix, where `usize::MAX` would let the summarizer
    // request overflow the window.
    let summary_input_budget = usize::try_from(context_window_tokens)
        .unwrap_or(0)
        .saturating_sub(SUMMARY_MAX_TOKENS + SUMMARY_INPUT_MARGIN_TOKENS);
    let prefix_start = summary_prefix_start(&token_est, cut, prompt_est, summary_input_budget);
    if prefix_start > 0 {
        let dropped_tokens: usize = token_est.get(..prefix_start).unwrap_or_default().iter().sum();
        println!(
            "[compaction] summary-input guard dropped {prefix_start} oldest prefix message(s) (~{dropped_tokens} tok) so the summarizer call fits the window"
        );
    }

    let prefix = visible.get(prefix_start..cut).unwrap_or_default();
    // Extreme case: the guard had to drop the WHOLE prefix to fit the window
    // (window smaller than the summarizer prompt itself — essentially unreachable
    // for real windows). Skip the summarizer call and emit an anchor-only head so
    // the run still re-grounds and survives; the dropped older turns are gone but
    // the verbatim tail and the authoritative anchor remain.
    let summary = if prefix.is_empty() {
        String::new()
    } else {
        let mut probe = prefix.to_vec();
        probe.push(client.user_message(format!("{}\n\n{}", prompts::COMPACT, prompts::COMPACT_TEMPLATE)));
        let s = stream::collect_text_turn(client, &probe, SUMMARY_MAX_TOKENS, ThinkingEffort::Low).await?;
        if s.trim().is_empty() {
            return Err(CompletionError::Transient(
                "compaction produced an empty summary".to_string(),
            ));
        }
        s
    };

    // Assemble: [single user-msg = preamble + anchor (authoritative, read FIRST) +
    // solved-gotchas ledger (durable procedure) + model brief (narrative, read
    // after), ...verbatim recent tail]. The anchor precedes the summary so a
    // drifting brief can never be the first thing the future self reads; the
    // gotchas ledger is folded in so a score-perfect-but-lossy summary can't drop
    // a hard-won build flag / recipe (the item 3 loss). The preamble tells it to
    // trust the anchor over the brief.
    let body = assemble_brief(prompts::COMPACT_PREAMBLE, anchor, gotchas, &summary);
    let mut out = vec![client.user_message(body)];
    if let Some(tail) = visible.get(cut..) {
        out.extend_from_slice(tail);
    }
    *messages = out;
    Ok(())
}

/// Build the compaction head message: `preamble`, then the authoritative `anchor`
/// (placed FIRST so a drifting brief is never the first thing the future self
/// reads), then the durable `gotchas` ledger (hard-won fixes a lossy summary must
/// not drop), then the model-written `summary`. Empty sections are skipped. Pure
/// so the ordering is unit-testable without a client.
fn assemble_brief(preamble: &str, anchor: &str, gotchas: &str, summary: &str) -> String {
    let mut body = String::from(preamble);
    for section in [anchor, gotchas, summary] {
        if !section.trim().is_empty() {
            body.push_str("\n\n");
            body.push_str(section);
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_is_max_output_or_floor_plus_headroom() {
        // Large per-turn output cap dominates the floor.
        assert_eq!(reserve_tokens(64_000), 64_000 + TURN_HEADROOM_TOKENS);
        // Small output cap floors at MIN_RESERVE_TOKENS.
        assert_eq!(reserve_tokens(5_000), MIN_RESERVE_TOKENS + TURN_HEADROOM_TOKENS);
    }

    #[test]
    fn should_compact_fires_at_the_budget_fraction() {
        let window = 200_000u32;
        let reserve = reserve_tokens(64_000); // 80_384
        let usable = window - reserve; // 119_616
        let threshold = f64_to_u32_saturating(f64::from(usable) * f64::from(COMPACTION_BUDGET_FRACTION));
        assert!(!should_compact(threshold, window, reserve));
        assert!(!should_compact(threshold - 5_000, window, reserve));
        assert!(should_compact(threshold + 1, window, reserve));
        // Fires well before the hard window−reserve limit (mid-run re-ground).
        assert!(should_compact(usable - 1, window, reserve));
    }

    #[test]
    fn should_compact_saturates_when_reserve_exceeds_window() {
        // Reserve larger than the window ⇒ threshold pinned at 0, not underflow.
        assert!(!should_compact(0, 10_000, 80_000));
        assert!(should_compact(1, 10_000, 80_000));
    }

    // safe_start convention below: assistant turns = true (safe tail start);
    // user / tool_result turns = false (unsafe: would orphan a tool_result or
    // double a user turn after the summary).

    #[test]
    fn tail_cut_lands_on_a_safe_boundary() {
        //          idx: 0     1     2     3     4
        //          msg: U     A     T     A     T
        let safe = [false, true, false, true, false];
        let tok = [10usize, 10, 10, 10, 10];
        // keep 15: sum from end 10(i4),20(i3)>=15 ⇒ cut=3; safe[3]=true ⇒ tail [A,T].
        assert_eq!(tail_start_index(&safe, &tok, 15), 3);
        assert!(safe[tail_start_index(&safe, &tok, 15)]);
    }

    #[test]
    fn tail_cut_walks_back_off_an_orphaned_tool_result() {
        let safe = [false, true, false, true, false];
        let tok = [10usize, 10, 10, 10, 10];
        // keep 25: 10,20,30(i2)>=25 ⇒ cut=2; safe[2]=false ⇒ walk to i1 (assistant).
        let cut = tail_start_index(&safe, &tok, 25);
        assert_eq!(cut, 1);
        assert!(safe[cut], "tail must never start on a tool_result");
    }

    #[test]
    fn tail_keeps_everything_when_no_safe_older_boundary() {
        let safe = [false, false, false];
        let tok = [10usize, 10, 10];
        // keep 25: 10,20,30>=25 ⇒ cut=0 then no safe boundary ⇒ 0 (keep all).
        assert_eq!(tail_start_index(&safe, &tok, 25), 0);
    }

    #[test]
    fn tail_keeps_everything_when_transcript_smaller_than_target() {
        let safe = [true, false];
        let tok = [10usize, 10];
        // Never reaches keep target ⇒ cut collapses to 0.
        assert_eq!(tail_start_index(&safe, &tok, 100), 0);
    }

    #[test]
    fn summary_guard_keeps_whole_prefix_when_it_fits() {
        // prefix (idx 0..cut=3) sums to 60; budget 1000 ⇒ no drop.
        let tok = [20usize, 20, 20, 10, 10];
        assert_eq!(summary_prefix_start(&tok, 3, 5, 1000), 0);
    }

    #[test]
    fn summary_guard_drops_oldest_until_it_fits() {
        // prefix idx 0..cut=4 sums to 100; prompt_est 10; budget 55.
        // drop i0(40)->60+10=70>55; drop i1(30)->30+10=40<=55 ⇒ start=2.
        let tok = [40usize, 30, 20, 10, 5];
        assert_eq!(summary_prefix_start(&tok, 4, 10, 55), 2);
    }

    #[test]
    fn summary_guard_drops_whole_prefix_when_window_smaller_than_prompt() {
        // Even an empty prefix can't fit: prompt_est 100 > budget 50 ⇒ start == cut.
        let tok = [10usize, 10, 10, 5];
        assert_eq!(summary_prefix_start(&tok, 3, 100, 50), 3);
    }

    #[test]
    fn brief_orders_preamble_anchor_gotchas_summary() {
        // The authoritative anchor precedes the durable gotchas ledger, which
        // precedes the model brief — so neither a drifting summary nor a dropped
        // gotcha is the first thing the future self reads.
        let body = assemble_brief("PREAMBLE", "GROUND TRUTH anchor", "SOLVED gotchas", "model brief");
        let pre = body.find("PREAMBLE").unwrap();
        let anc = body.find("GROUND TRUTH anchor").unwrap();
        let got = body.find("SOLVED gotchas").unwrap();
        let sum = body.find("model brief").unwrap();
        assert!(
            pre < anc && anc < got && got < sum,
            "order must be preamble -> anchor -> gotchas -> summary: {body:?}"
        );
    }

    #[test]
    fn brief_skips_empty_sections() {
        assert_eq!(assemble_brief("PRE", "", "", ""), "PRE");
        assert_eq!(assemble_brief("PRE", "   ", "  ", "S"), "PRE\n\nS");
        assert_eq!(assemble_brief("PRE", "A", "", "   "), "PRE\n\nA");
        // Gotchas present, anchor/summary empty: still folded in.
        assert_eq!(assemble_brief("PRE", "", "G", ""), "PRE\n\nG");
    }
}
