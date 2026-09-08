//! Supervisor — workflow, stagnation, and failed-commit interventions.
//!
//! Detection is stateless by design: [`due`] observes the counters the
//! orchestrator already journals and names which intervention (if any) is due.
//! The no-eval and failed-commit rungs fire once on their exact threshold
//! crossing; the stagnation rung re-fires every `STAGNATION_THRESHOLD` turns a
//! plateau persists, periodically re-orienting the search (mirroring the paper's
//! repeated intervention) instead of nudging once and going silent.
//!
//! Intervention has two modes, matching the paper's self-supervision
//! ("reviews the overall evolutionary trajectory and steers the search toward
//! several candidate optimization directions"):
//!
//! - **Active** ([`review`]): when a (typically cheaper) supervisor model is
//!   configured, the orchestrator hands it the committed lineage + the current
//!   stall signal and it returns concrete, tailored directions to inject.
//! - **Static** ([`static_prompt`]): a fixed Markdown nudge, used as the
//!   fallback when no supervisor model is configured or the review call fails.

use crate::ai::{CompletionError, ProtocolClient, ThinkingEffort};
use crate::harness::stream;

/// Turns with no evaluation before the no-eval nudge fires.
///
/// Turns of docs-reading / edits / profiling with NO evaluation at all before
/// we remind the agent that only `evaluate` counts. This is intentionally
/// *not* too early: for hard CUDA/PTX work, reading the architecture docs and
/// toolchain notes is productive work, not a stall.
pub const NO_EVAL_THRESHOLD: u64 = 12;
/// Turns without the confirmed best improving before we firmly re-orient the
/// search toward fresh directions.
pub const STAGNATION_THRESHOLD: u64 = 16;

/// Output cap for the supervisor's review turn (concise, ranked directions).
const REVIEW_MAX_TOKENS: usize = 2_000;

/// Which intervention is due — the rungs in descending severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rung {
    /// A long stretch with no improvement to the best: re-orient toward fresh
    /// directions.
    Stagnation,
    /// Many turns with no trusted `evaluate`: likely mistaking own timings for
    /// the score.
    NoEval,
}

/// Progress signals the supervisor reasons over — all cheap counters the
/// control loop already maintains.
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    /// Turns since the confirmed best last improved (or since run start).
    pub turns_since_best_improved: u64,
    /// Total `evaluate` invocations so far this run.
    pub evaluations: u64,
}

/// Which intervention (if any) is due. Higher-severity rungs take priority; the
/// stagnation rung re-fires on every multiple of `STAGNATION_THRESHOLD` so long
/// plateaus keep getting re-oriented.
#[must_use]
pub const fn due(p: Progress) -> Option<Rung> {
    // A long stretch with no improvement: re-orient toward fresh directions. Re-
    // fires every STAGNATION_THRESHOLD turns so a persistent plateau keeps getting
    // steered (the paper re-intervenes rather than nudging once).
    if p.turns_since_best_improved >= STAGNATION_THRESHOLD
        && p.turns_since_best_improved.is_multiple_of(STAGNATION_THRESHOLD)
    {
        return Some(Rung::Stagnation);
    }
    // Never even evaluated: eventually remind the agent that trusted `evaluate`
    // is the score, but allow a real docs-reading/design runway first.
    if p.evaluations == 0 && p.turns_since_best_improved == NO_EVAL_THRESHOLD {
        return Some(Rung::NoEval);
    }
    None
}

/// The static fallback nudge for a rung — injected verbatim when no supervisor
/// model is configured, or when an active [`review`] call fails.
#[must_use]
pub const fn static_prompt(rung: Rung) -> &'static str {
    match rung {
        Rung::Stagnation => crate::orchestrator::prompts::SUPERVISOR_STAGNATION,
        Rung::NoEval => crate::orchestrator::prompts::SUPERVISOR_NO_EVAL,
    }
}

/// Build the prompt for an active supervisor review.
///
/// Bundles the review instructions, the target hardware, the current stall
/// signal, the rendered candidate trajectory (`P_t`), and `recent` — a bounded
/// digest of what the agent has actually tried and concluded, so the supervisor
/// builds on findings instead of re-proposing disproven ones.
#[must_use]
pub fn review_context(rung: Rung, trajectory: &str, recent: &str, p: Progress, hardware: &str) -> String {
    let signal = match rung {
        Rung::Stagnation => format!(
            "The main agent has gone {} turns with no improvement to the confirmed best — the current line has plateaued.",
            p.turns_since_best_improved
        ),
        Rung::NoEval => format!(
            "The main agent has run {} turns without a trusted `evaluate`; preserve useful docs-reading, but help it convert what it learned into the smallest scored experiment.",
            p.turns_since_best_improved
        ),
    };
    format!(
        "{}\n\n## Target hardware (probed this run — tailor every direction to it)\n{hardware}\n\n## Current signal\n{signal}\n\n## Committed lineage so far (the evolutionary trajectory)\n{trajectory}\n\n## What the agent has already tried & concluded since its last commit (build on this; do NOT re-propose what it already ruled out)\n{recent}",
        crate::orchestrator::prompts::SUPERVISOR_REVIEW
    )
}

/// Run the active supervisor review with the (typically cheaper) supervisor model.
///
/// A single no-tools turn that returns concrete directions to inject into the
/// main agent. Mirrors `compaction::compact`'s auxiliary-call pattern. On
/// failure (empty output / provider error) the caller falls back to the static
/// nudge.
///
/// # Errors
///
/// Returns the [`CompletionError`] from the underlying streamed turn (provider
/// error, or retries exhausted), or [`CompletionError::Transient`] when the
/// supervisor model replies with nothing but whitespace.
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
pub async fn review<C: ProtocolClient>(client: &C, context: String) -> Result<String, CompletionError> {
    let msgs = vec![client.user_message(context)];
    let text = stream::collect_text_turn(client, &msgs, REVIEW_MAX_TOKENS, ThinkingEffort::Medium).await?;
    if text.trim().is_empty() {
        return Err(CompletionError::Transient(
            "supervisor review produced empty output".to_string(),
        ));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(turns_since_best_improved: u64, evaluations: u64) -> Progress {
        Progress {
            turns_since_best_improved,
            evaluations,
        }
    }

    #[test]
    fn no_eval_rung_fires_once_when_never_evaluated() {
        assert_eq!(due(p(NO_EVAL_THRESHOLD - 1, 0)), None);
        assert_eq!(due(p(NO_EVAL_THRESHOLD, 0)), Some(Rung::NoEval));
        assert_eq!(due(p(NO_EVAL_THRESHOLD + 1, 0)), None);
    }

    #[test]
    fn no_eval_rung_suppressed_once_the_agent_has_evaluated() {
        // Same turn count, but it has used `evaluate` — different failure mode.
        assert_eq!(due(p(NO_EVAL_THRESHOLD, 1)), None);
    }

    #[test]
    fn stagnation_rung_re_fires_each_period() {
        assert_eq!(due(p(STAGNATION_THRESHOLD - 1, 2)), None);
        assert_eq!(due(p(STAGNATION_THRESHOLD, 2)), Some(Rung::Stagnation));
        assert_eq!(due(p(STAGNATION_THRESHOLD + 1, 2)), None);
        // Re-fires on each subsequent multiple so long plateaus keep being
        // re-oriented, while staying quiet on the turns between.
        assert_eq!(due(p(2 * STAGNATION_THRESHOLD - 1, 2)), None);
        assert_eq!(due(p(2 * STAGNATION_THRESHOLD, 2)), Some(Rung::Stagnation));
        assert_eq!(due(p(3 * STAGNATION_THRESHOLD, 2)), Some(Rung::Stagnation));
    }

    #[test]
    fn review_context_includes_trajectory_signal_and_recent() {
        let ctx = review_context(
            Rung::Stagnation,
            "turn 3: geomean 1.2000 (confirmed)",
            "- fp16 simdgroup_matrix gives no speedup on M4 (no matrix datapath) — reverted",
            p(16, 4),
            "Apple M4, 10-core GPU, Metal 3",
        );
        assert!(ctx.contains("plateaued"), "names the stall signal: {ctx}");
        assert!(ctx.contains("geomean 1.2000"), "embeds the trajectory: {ctx}");
        assert!(ctx.contains("Apple M4"), "embeds the target hardware: {ctx}");
        assert!(ctx.contains("no matrix datapath"), "embeds recent activity: {ctx}");
    }
}
