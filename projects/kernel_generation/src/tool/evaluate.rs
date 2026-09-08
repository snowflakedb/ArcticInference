//! `evaluate` — the trusted, score-producing benchmark.
//!
//! Runs `scripts/evaluate.py` *inside the sandbox* (so the agent's `solution/`
//! tree is isolated) but against a *read-only* evaluator + `problem.py` (so the
//! score is unforgeable). Takes a [`GpuPool`] device lease so a benchmark always
//! runs on a quiescent machine. A full-stage result is memoized by fileset hash
//! so a later `submit` of the same bytes reuses the score.
//!
//! The same [`run_evaluator`] helper backs the `submit` tool's verification.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::domain::events::TurnEvents;
use crate::domain::types::{EvalMetrics, PerConfig, Stage};
use crate::domain::util::lock;
use crate::exec::queue::{GpuPool, warn_if_devices_still_held};
use crate::exec::sandbox::Sandbox;
use crate::tool::truncate::{MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES, clamp_with_spill, truncate_tail};
use crate::tool::{Tool, ToolOutput};

// Canonical workspace dir names live in `exec::snapshot_store` (the layer the
// snapshot-exclusion predicate is in). Re-exported here so existing
// `tool::evaluate::{SOLUTION_DIR, TRUSTED_DIR}` importers are unchanged, but there
// is exactly ONE definition. `TRUSTED_DIR` = the read-only evaluator/`problem.py`
// mount; `SOLUTION_DIR` = the agent's editable, snapshotted solution tree.
pub use crate::exec::snapshot_store::{SOLUTION_DIR, TRUSTED_DIR};

/// The required entrypoint inside [`SOLUTION_DIR`]; it exports `class Solution`
/// and may import its siblings (the dir is placed on `sys.path` by the
/// evaluator). This is the file the evaluator loads.
pub const SOLUTION_ENTRYPOINT: &str = "solution.py";

/// Workspace-relative path to the solution entrypoint (`solution/solution.py`).
#[must_use]
pub fn solution_entrypoint() -> String {
    format!("{SOLUTION_DIR}/{SOLUTION_ENTRYPOINT}")
}

/// Resolved sandbox-relative paths + interpreter the evaluator/commit tools run.
#[derive(Clone)]
pub struct EvalConfig {
    /// Python interpreter (must have torch). On `PATH` inside the sandbox.
    pub python: String,
    /// Sandbox-relative path to the read-only evaluator.
    pub evaluator: String,
    /// Sandbox-relative path to the read-only problem definition.
    pub problem: String,
    /// Sandbox-relative path to the agent's editable candidate.
    pub solution: String,
    /// Per-invocation execution timeout (seconds). Distinct from the queue
    /// acquire timeout.
    pub timeout_secs: u64,
    /// Per-run base seed for benchmark inputs. Each invocation derives a *fresh*
    /// seed `run_seed XOR eval_counter`, so inputs vary per evaluation (the
    /// agent can't predict a future eval's inputs) while the whole sequence is
    /// reproducible given `run_seed`.
    pub run_seed: u64,
    /// Monotonic invocation counter, shared across clones (so every
    /// `evaluate`/`submit` across both tool instances gets a distinct index).
    eval_counter: Arc<AtomicU64>,
    /// Per-run frozen performance baseline: per-config latency (ms) of the FIRST
    /// correct candidate the agent produces (the evaluator sets `ok` only after
    /// correctness passes). Frozen once, then used read-only to derive every
    /// later candidate's speedup (`speedup_vs_baseline`), so the score measures
    /// the agent's own optimization progress and machine drift cancels out of the
    /// solution-vs-solution comparison. The *reference* is never the baseline —
    /// it is the correctness oracle only. Shared across clones so the `evaluate`
    /// tool instances and `confirm_leader` see the same frozen baseline.
    baseline_latency: Arc<Mutex<BTreeMap<String, f64>>>,
}

impl EvalConfig {
    pub fn new(python: impl Into<String>, timeout_secs: u64, run_seed: u64) -> Self {
        Self {
            python: python.into(),
            evaluator: format!("{TRUSTED_DIR}/evaluate.py"),
            problem: format!("{TRUSTED_DIR}/problem.py"),
            solution: solution_entrypoint(),
            timeout_secs,
            run_seed,
            eval_counter: Arc::new(AtomicU64::new(0)),
            baseline_latency: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    #[must_use]
    pub fn run_kernel(&self) -> String {
        format!("{TRUSTED_DIR}/run_kernel.py")
    }

    /// Snapshot the frozen performance baseline: per-config latency (ms) of the
    /// first correct candidate. Empty until that candidate has been evaluated.
    /// Used by the orchestrator to score candidates and by `confirm_leader`.
    #[must_use]
    pub fn baseline(&self) -> BTreeMap<String, f64> {
        lock(&self.baseline_latency).clone()
    }

    /// Next per-evaluation input seed. Advances the shared counter, so two calls
    /// (even across the `evaluate` and `submit` tools) never share a seed within
    /// a run.
    fn next_eval_seed(&self) -> u64 {
        let idx = self.eval_counter.fetch_add(1, Ordering::Relaxed);
        self.run_seed ^ idx
    }
}

/// Run the read-only evaluator at `stage` inside the sandbox, holding the
/// execution lease for the duration. Returns the parsed trusted metrics, or an
/// `infrastructure error:`-prefixed string for queue / spawn / timeout / parse
/// failures (which are the orchestrator's fault, not the candidate's).
pub(crate) async fn run_evaluator(
    sandbox: &Sandbox,
    queue: &GpuPool,
    config: &EvalConfig,
    stage: Stage,
) -> Result<EvalMetrics, String> {
    let lease = queue
        .acquire_any()
        .await
        .map_err(|e| format!("infrastructure error: {e}"))?;

    let stage_str = stage.as_str();
    // Fresh-but-reproducible input seed for this invocation (see `EvalConfig`).
    let seed_str = config.next_eval_seed().to_string();
    let args: [&str; 7] = [
        &config.python,
        &config.evaluator,
        &config.problem,
        &config.solution,
        "--stage",
        stage_str,
        "--seed",
    ];
    let mut cmd: Vec<&str> = args.to_vec();
    cmd.push(&seed_str);
    let out = sandbox
        .run(
            &cmd,
            Some(Duration::from_secs(config.timeout_secs)),
            Some(lease.devices()),
        )
        .await
        .map_err(|e| format!("infrastructure error: evaluator spawn failed: {e}"))?;
    // Checked while the lease is still held, so the slot is not handed to the next
    // caller before we know whether its devices came back.
    if out.timed_out {
        warn_if_devices_still_held(lease.devices(), "evaluator").await;
    }
    drop(lease);

    if out.timed_out {
        return Err(format!(
            "infrastructure error: evaluator timed out after {}s at stage {stage_str}",
            config.timeout_secs
        ));
    }
    let metrics = extract_metrics(&out.stdout)?;
    // Freeze the performance baseline from the FIRST correct benchmarked eval —
    // the agent's first correct candidate (the evaluator sets `ok` only after
    // correctness passes). Frozen once, as a whole candidate's per-config
    // latencies, so every later candidate is scored against the same denominator
    // (`speedup_vs_baseline`) and machine drift cancels out of the comparison.
    if metrics.ok && !metrics.per_config.is_empty() {
        let mut baseline = lock(&config.baseline_latency);
        if baseline.is_empty() {
            for pc in &metrics.per_config {
                if pc.latency_ms > 0.0 {
                    baseline.insert(pc.name.clone(), pc.latency_ms);
                }
            }
        }
    }
    Ok(metrics)
}

/// Run the full evaluator `n` times back-to-back, collecting one [`EvalMetrics`]
/// per run. Used by the `submit` gate's Stage-2 paired confirmation: re-timing
/// the challenger and the champion in the same thermal window so drift cancels
/// and the winner's-curse is defeated by a fresh, repeated measurement.
///
/// Each run draws a fresh-but-reproducible input seed (via [`EvalConfig`]); when
/// the kernel's latency is value-independent (same shapes ⇒ same work),
/// different input draws add no bias to the timing comparison. Surfaces the
/// first `infrastructure error:` (queue/spawn/timeout/parse) it hits — the
/// caller treats that as *inconclusive* and falls back to the Stage-1 screen.
pub(crate) async fn run_evaluator_n(
    sandbox: &Sandbox,
    queue: &GpuPool,
    config: &EvalConfig,
    stage: Stage,
    n: usize,
) -> Result<Vec<EvalMetrics>, String> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(run_evaluator(sandbox, queue, config, stage).await?);
    }
    Ok(out)
}

/// The evaluator prints exactly one JSON object line, but torch can interleave
/// warnings on stderr (merged into `stdout` here), so scan from the end for the
/// JSON line.
fn extract_metrics(output: &str) -> Result<EvalMetrics, String> {
    for line in output.lines().rev() {
        let t = line.trim();
        if t.starts_with('{')
            && t.ends_with('}')
            && let Ok(m) = EvalMetrics::parse(t)
        {
            return Ok(m);
        }
    }
    // Evaluator output is the highest-reproduction-cost of the execution tools:
    // spill the full output and keep a generous tail (2000 lines / 50 KiB) in the
    // error, citing the spill path. Fall back to a bounded tail if the spill fails.
    let tail = clamp_with_spill(output, "kg-eval")
        .unwrap_or_else(|_| truncate_tail(output, MAX_OUTPUT_LINES, MAX_OUTPUT_BYTES).content);
    Err(format!(
        "infrastructure error: evaluator produced no parseable JSON. output tail:\n{tail}"
    ))
}

/// Roofline fraction above which an achieved figure is flagged as a likely
/// measurement artifact (you cannot do the real work faster than the hardware
/// ceiling). Left a touch above 1.0 so timing jitter right at a genuine ceiling
/// does not spuriously trip it.
const ROOFLINE_SUSPECT_FRAC: f64 = 1.02;

/// The *binding* roofline reading for one config: whichever of the FLOP-peak
/// (compute-bound) or bandwidth-peak (memory-bound) fraction the kernel is
/// closest to saturating. That is the trusted "how much is physically left"
/// signal. Printing only the binding fraction (with its achieved absolute value)
/// keeps the agent from mis-reading the tiny, irrelevant off-regime number — or
/// from hand-rolling its own byte count and getting the dtype wrong.
fn roofline_suffix(pc: &PerConfig) -> String {
    if pc.pct_bandwidth.is_none() && pc.pct_peak.is_none() {
        return String::new();
    }
    let bwf = pc.pct_bandwidth.unwrap_or(f64::NEG_INFINITY);
    let flf = pc.pct_peak.unwrap_or(f64::NEG_INFINITY);
    let (frac, body) = if bwf >= flf {
        (
            bwf,
            format!(
                "{:.1}% of bandwidth peak · {:.1} GB/s",
                bwf * 100.0,
                pc.achieved_gbps.unwrap_or(0.0)
            ),
        )
    } else {
        (
            flf,
            format!(
                "{:.1}% of FLOP peak · {:.2} TFLOP/s",
                flf * 100.0,
                pc.achieved_tflops.unwrap_or(0.0)
            ),
        )
    };
    // A fraction above the hardware roofline is physically impossible for the
    // real work, so it is a measurement artifact, not a result: small inputs
    // served from cache (SLC/L2) rather than DRAM, an unsynchronized/mis-timed
    // benchmark, or a kernel that skips work (should be caught by correctness).
    // Flag it inline so a too-good number gets scrutinized, not celebrated.
    if frac > ROOFLINE_SUSPECT_FRAC {
        format!(
            "  [{body}  (!) ABOVE ROOFLINE — impossible for the real work; likely cache reuse on a small config or unsynchronized timing, treat as suspect]"
        )
    } else {
        format!("  [{body}]")
    }
}

/// Concise, agent-facing rendering of a result. The evaluator is a pure
/// benchmark, so speedup is shown relative to the frozen `baseline` (the first
/// correct candidate); the reference is the correctness oracle and is never
/// timed. Before the baseline exists, the first correct candidate is announced
/// as the 1.0× anchor.
pub(crate) fn format_eval(m: &EvalMetrics, baseline: &BTreeMap<String, f64>) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "stage_reached: {}\nok: {}\ncorrect: {}",
        m.stage_reached, m.ok, m.correct
    );
    if !m.per_config.is_empty() {
        if baseline.is_empty() {
            s.push_str(
                "baseline: this is the FIRST correct candidate — it sets the 1.0x performance \
                 baseline; every later kernel is scored as speedup over it.\n",
            );
        } else {
            let g = crate::domain::types::speedup_vs_baseline(baseline, &m.per_config);
            let _ = writeln!(
                s,
                "geomean_speedup: {g:.4}x (vs your first-correct baseline; higher is better)"
            );
        }
        if let Some(nm) = m.noise_margin {
            let _ = writeln!(s, "noise_margin: {nm:.4}");
        }
        if m.peak_tflops.is_some() || m.peak_gbps.is_some() {
            let pt = m.peak_tflops.map_or_else(|| "?".into(), |v| format!("{v:.2} TFLOP/s"));
            let pg = m.peak_gbps.map_or_else(|| "?".into(), |v| format!("{v:.0} GB/s"));
            let _ = writeln!(
                s,
                "device roofline: {pt} (compute) · {pg} (bandwidth) — per-config [..] shows your BINDING fraction"
            );
        }
        for pc in &m.per_config {
            let speed = baseline
                .get(&pc.name)
                .filter(|b| **b > 0.0 && pc.latency_ms > 0.0)
                .map(|b| format!("{:.4}x  ", *b / pc.latency_ms))
                .unwrap_or_default();
            let _ = writeln!(
                s,
                "  {}: {}{:.3} ms{}",
                pc.name,
                speed,
                pc.latency_ms,
                roofline_suffix(pc)
            );
        }
    }
    if let Some(err) = &m.error {
        let _ = writeln!(s, "error: {err}");
    }
    if let Some(tb) = &m.traceback_tail {
        let _ = writeln!(s, "traceback:\n{tb}");
    }
    s
}

#[derive(Deserialize, JsonSchema)]
pub struct EvaluateArgs {
    /// How deep to evaluate. `compile` (imports + instantiates), `correctness`
    /// (asserts the output matches the reference on every config), or `full`
    /// (correctness + timing on all configs, repeated, with a noise margin). Use
    /// shallow stages while iterating; `full` is what `submit` requires.
    pub stage: Stage,
}

pub struct Evaluate {
    pub sandbox: Arc<Sandbox>,
    pub queue: GpuPool,
    pub config: EvalConfig,
    pub events: Arc<TurnEvents>,
}

impl Tool for Evaluate {
    type Args = EvaluateArgs;
    const NAME: &'static str = "evaluate";
    const DESCRIPTION: &'static str = "Benchmark the current solution (solution/solution.py + any helper files) \
         with the TRUSTED, read-only evaluator \
         (the only source of an official score). Stages: compile | correctness | full. Returns \
         correctness (output matched against the reference) and, for full, per-config latency plus \
         geomean speedup over your FIRST correct candidate (the 1.0x baseline) — the reference is the \
         correctness oracle only, never the speed target. This is the score commit checks against — you cannot fake it.";

    async fn call(&self, args: EvaluateArgs) -> ToolOutput {
        let files = self
            .sandbox
            .read_tree(SOLUTION_DIR)
            .map_err(|e| format!("infrastructure error: cannot read {SOLUTION_DIR}/ before evaluate: {e}"))?;
        let metrics = run_evaluator(&self.sandbox, &self.queue, &self.config, args.stage).await?;
        self.events.record_evaluation(metrics.clone(), files);
        Ok(format_eval(&metrics, &self.config.baseline()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field names must match `scripts/evaluate.py`, and `format_eval` must show
    /// the *binding* roofline per config: bandwidth for the memory-bound decode
    /// row, FLOP-peak for the compute-bound prefill row. Regression guard for
    /// the bug where these fields were dropped and the agent hand-rolled (and
    /// fumbled) its own bandwidth estimate.
    #[test]
    fn format_eval_shows_binding_roofline() {
        let json = r#"{
            "stage_reached": "full", "ok": true, "correct": true,
            "peak_tflops": 4.30, "peak_gbps": 120.0,
            "per_config": [
                {"name": "prefill_T4096", "latency_ms": 329.7,
                 "achieved_tflops": 0.44, "pct_peak": 0.103, "achieved_gbps": 0.2, "pct_bandwidth": 0.0006},
                {"name": "decode_kv32768", "latency_ms": 1.733,
                 "achieved_tflops": 0.001, "pct_peak": 0.0002, "achieved_gbps": 77.5, "pct_bandwidth": 0.645},
                {"name": "decode_kv1024", "latency_ms": 0.1,
                 "achieved_tflops": 0.01, "pct_peak": 0.002, "achieved_gbps": 170.4, "pct_bandwidth": 1.42}
            ]
        }"#;
        let m = EvalMetrics::parse(json).expect("parse");
        let out = format_eval(&m, &BTreeMap::new());
        assert!(
            out.contains("device roofline: 4.30 TFLOP/s (compute) · 120 GB/s (bandwidth)"),
            "{out}"
        );
        // decode row is memory-bound -> bandwidth fraction (NOT the tiny FLOP one)
        assert!(out.contains("decode_kv32768"));
        assert!(out.contains("[64.5% of bandwidth peak · 77.5 GB/s]"), "{out}");
        // prefill row is compute-bound -> FLOP fraction
        assert!(out.contains("[10.3% of FLOP peak · 0.44 TFLOP/s]"), "{out}");
        // a fraction above the roofline is flagged as suspect, not celebrated
        assert!(out.contains("142.0% of bandwidth peak"), "{out}");
        assert!(out.contains("ABOVE ROOFLINE"), "{out}");
        // ...while the legitimate sub-ceiling rows are NOT flagged
        assert_eq!(out.matches("ABOVE ROOFLINE").count(), 1, "{out}");
    }
}
