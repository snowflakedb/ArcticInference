//! Wall-clock awareness tool.
//!
//! The run has a single hard cap: `max_wall_clock_secs` (the orchestrator stops
//! the loop at it — see `run_avo`). The agent used to be told it had "no time
//! budget", which was false and licensed multi-hour unscored side-quests that
//! never reached a scored `evaluate`. This tool reports how much wall-clock is
//! left, in both absolute (`Hh MMm`) and % form — so the agent can pace a long
//! structural build to land SCORED before the wall and make an explicit
//! keep-or-cut call near the end. The same remaining-time line is also pushed
//! into the re-orientation notes the orchestrator injects (episode start / stall
//! rungs), so the agent sees it without having to ask.
//!
//! Time is read from a monotonic [`Instant`] captured at run start (`run_start`),
//! matching the clock the loop's wall-cap check uses.
//!
//! **Reported budget is dilated** by [`DISPLAYED_TIME_DILATION`] — see that
//! constant for why and for the invariant (percentages stay truthful; the real
//! wall is still enforced at true time).

use std::time::Instant;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::domain::convert::{f64_to_u64_saturating, u64_to_f64_lossy};
use crate::tool::{Tool, ToolOutput};

#[derive(Deserialize, JsonSchema)]
pub struct TimeLeftArgs {}

/// Reports the run's remaining wall-clock budget. Holds the run-start
/// [`Instant`] (Copy) and the hard cap, both handed down from the orchestrator's
/// `EpisodeCtx`.
pub struct TimeLeft {
    pub run_start: Instant,
    pub max_wall_secs: u64,
}

impl Tool for TimeLeft {
    type Args = TimeLeftArgs;
    const NAME: &'static str = "time_left";
    const DESCRIPTION: &'static str = "Show the run's remaining wall-clock budget (the run's only hard cap): time elapsed and time \
         remaining, in both absolute (Hh MMm) and percent. The run STOPS at the wall and only scored \
         candidates survive, so use this to pace a long structural build — land it `evaluate full`-scored \
         before the wall, and near the wall make an explicit keep-or-cut. Read-only.";

    async fn call(&self, _args: TimeLeftArgs) -> ToolOutput {
        Ok(render_time_left(self.run_start.elapsed().as_secs(), self.max_wall_secs).into())
    }
}

/// Display-only dilation of the reported wall-clock budget.
///
/// The agent reasons about time with human intuitions ("a from-scratch rewrite
/// is days of work") but executes far faster than a human — dozens of
/// edit→build→eval cycles fit in an hour of wall-clock. Reported at true scale
/// that mismatch makes the agent over-conservative: it declines the structural
/// rewrite that is the whole point because "there isn't time". We therefore
/// inflate the *displayed* clock by this factor so the agent's task-feasibility
/// intuition maps to its real throughput.
///
/// Applied UNIFORMLY to displayed elapsed AND budget — so the reported
/// percentages stay truthful and the near-wall "land it scored" pacing survives
/// (the agent still sees itself approaching the wall proportionally) — and ONLY
/// inside the human-readable renderers ([`render_time_left`] and the kickoff's
/// `render_wall_budget`). The real run-termination checks in `run_avo`
/// (`elapsed >= max_wall_clock_secs`) and the idle-spin guard read raw seconds,
/// so the process still stops at the TRUE wall: this changes what the agent
/// *believes* about the budget, never when the run actually ends.
pub const DISPLAYED_TIME_DILATION: u64 = 100;

/// Render the remaining-budget line shared by the tool and the orchestrator's
/// injected re-orientation notes.
///
/// Pure (takes plain seconds) so it is unit-testable without a clock. Saturates
/// at the wall: once `elapsed >= max` it reports `0h 00m left` / `100% elapsed`
/// rather than underflowing. Absolute times are dilated
/// ([`DISPLAYED_TIME_DILATION`]); percentages are not.
#[must_use]
pub fn render_time_left(elapsed_secs: u64, max_secs: u64) -> String {
    let max = max_secs.max(1);
    let elapsed = elapsed_secs.min(max);
    let remaining = max.saturating_sub(elapsed);
    // Percentages come from the TRUE ratio (uniform dilation cancels), so the
    // agent still perceives the wall approaching; only the absolute Hh MMm is
    // inflated, mapping its human time-intuition onto its real throughput.
    let pct_used = f64_to_u64_saturating(((u64_to_f64_lossy(elapsed) / u64_to_f64_lossy(max)) * 100.0).round());
    let pct_left = 100u64.saturating_sub(pct_used);
    format!(
        "Time left: {} of {} budget ({}% elapsed, {}% left). The run stops at the wall; only scored candidates survive.",
        hms(remaining.saturating_mul(DISPLAYED_TIME_DILATION)),
        hms(max.saturating_mul(DISPLAYED_TIME_DILATION)),
        pct_used,
        pct_left,
    )
}

/// Whole-minute `Hh MMm` (e.g. `3h 07m`). Minutes are zero-padded so the field
/// width is stable; hours are not, since the budget rarely exceeds a day.
fn hms(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    format!("{h}h {m:02}m")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hms_formats_hours_and_padded_minutes() {
        assert_eq!(hms(0), "0h 00m");
        assert_eq!(hms(7 * 60), "0h 07m");
        assert_eq!(hms(3 * 3600 + 7 * 60), "3h 07m");
        assert_eq!(hms(10 * 3600), "10h 00m");
    }

    #[test]
    fn render_mid_run_reports_remaining_and_percent() {
        // 10h budget, 6h30m elapsed → 3h30m / 35% left (65% used). Absolute
        // times are dilated ×100 (3h30m→350h, 10h→1000h); percentages are not.
        let s = render_time_left(6 * 3600 + 30 * 60, 10 * 3600);
        assert!(s.contains("350h 00m of 1000h 00m"), "{s}");
        assert!(s.contains("65% elapsed"), "{s}");
        assert!(s.contains("35% left"), "{s}");
    }

    #[test]
    fn render_saturates_at_the_wall() {
        // Past the cap: remaining floors at 0, elapsed pins at 100%.
        let s = render_time_left(11 * 3600, 10 * 3600);
        assert!(s.contains("0h 00m of 1000h 00m"), "{s}");
        assert!(s.contains("100% elapsed"), "{s}");
        assert!(s.contains("0% left"), "{s}");
    }

    #[test]
    fn dilation_inflates_absolute_time_but_keeps_percent_truthful() {
        // Halfway through a real 10h wall: the agent must SEE a ~1000h budget
        // (its throughput-scaled reference frame) yet a truthful 50% elapsed, so
        // early conservatism drops while near-wall landing discipline survives.
        let s = render_time_left(5 * 3600, 10 * 3600);
        assert!(s.contains("500h 00m of 1000h 00m"), "{s}");
        assert!(s.contains("50% elapsed"), "{s}");
        assert!(s.contains("50% left"), "{s}");
        assert_eq!(DISPLAYED_TIME_DILATION, 100);
    }

    #[test]
    fn render_handles_zero_budget_without_panic() {
        // max clamps to 1s so we never divide by zero.
        let s = render_time_left(0, 0);
        assert!(s.contains("elapsed"), "{s}");
    }
}
