//! Shared, forward-compatible data types for the AVO orchestrator.
//!
//! These are deliberately plain serde structs: they are the on-disk schema for
//! the run manifest, the lineage/archive sidecars, and the resume journal, so
//! their shape is a compatibility surface. New optional fields are safe to add
//! (`#[serde(default)]`); renaming or removing fields breaks resume of older
//! runs.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::domain::convert::usize_to_f64_lossy;

/// A solution's source files, keyed by relative path.
///
/// The agent's editable solution as a set of files, keyed by path *relative to
/// the solution directory* (forward-slash separated, e.g. `"solution.py"`,
/// `"kernels/flash.metal"`). The entrypoint is always `solution.py`; it may
/// import sibling modules or ship kernel source alongside it. A `BTreeMap`
/// keeps the set canonically ordered so [`fileset_sha256`] is deterministic.
///
/// Files are treated as UTF-8 text (Python + kernel source); binary artifacts
/// are out of scope.
pub type SolutionFiles = BTreeMap<String, String>;

/// Identifier for one optimization run; also the directory name under `runs/`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunId(pub String);

impl RunId {
    /// Generate a fresh, lexicographically sortable id from the wall clock.
    #[must_use]
    pub fn generate() -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        Self(format!("run_{}_{:09}", now.as_secs(), now.subsec_nanos()))
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The three cumulative evaluation depths. Each stage runs everything the
/// shallower stages do, then more, and short-circuits on the first failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Compile,
    Correctness,
    Full,
}

impl Stage {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Compile => "compile",
            Self::Correctness => "correctness",
            Self::Full => "full",
        }
    }

    /// Ordinal depth used to rank evidence: a `Full` measurement outranks bare
    /// correctness, which outranks a compile. Never compare scores across stages
    /// without this.
    #[must_use]
    pub const fn rank(&self) -> u8 {
        match self {
            Self::Compile => 1,
            Self::Correctness => 2,
            Self::Full => 3,
        }
    }

    /// Map the evaluator's `stage_reached` string onto a stage. `"load"` is an
    /// alias for `compile`; anything unrecognized (including the empty string a
    /// parse failure yields) falls back to `Compile`, the shallowest depth.
    #[must_use]
    pub fn from_reached(s: &str) -> Self {
        match s {
            "full" => Self::Full,
            "correctness" => Self::Correctness,
            _ => Self::Compile,
        }
    }
}

/// The distilled, per-node outcome of one evaluation.
///
/// The domain type derived from [`EvalMetrics`] (the evaluator's untyped wire
/// DTO) at the tool boundary, shaped so illegal states can't be represented.
/// Each variant carries the [`Stage`] it reached; only [`Evaluation::Timed`]
/// carries a comparable score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Evaluation {
    /// Didn't compile, errored, or was incorrect. A real node (signals "this
    /// region doesn't work"), but never in the ranked frontier.
    Failed { stage: Stage, error: String },
    /// Passed correctness but was not timed (evaluated no deeper than
    /// `correctness`). Correct-but-unscored.
    Verified { stage: Stage },
    /// Correct and measured. The only variant with a comparable speedup.
    Timed {
        stage: Stage,
        geomean_speedup: f64,
        /// Fractional timing noise on the geomean; the frontier discounts the
        /// speedup by `1 + noise_margin` for a conservative score.
        noise_margin: f64,
        per_config: Vec<PerConfig>,
    },
}

impl Evaluation {
    /// Distill the evaluator's wire metrics into the domain outcome. A result
    /// that isn't `ok` or isn't `correct` is [`Failed`](Self::Failed); a correct
    /// result with per-config benchmark latencies is [`Timed`](Self::Timed),
    /// scored as geomean **speedup over the baseline**; a correct result with no
    /// benchmark (compile / correctness stage) is [`Verified`](Self::Verified).
    ///
    /// The evaluator is a *pure benchmark* — it emits absolute per-config
    /// latencies and no speedup. Speedup is derived here against `baseline` (the
    /// first correct candidate's frozen per-config latencies, held in
    /// `EvalConfig`). `baseline` is empty until that first candidate freezes it,
    /// at which point that candidate scores exactly 1.0×.
    #[must_use]
    pub fn from_metrics(m: &EvalMetrics, baseline: &std::collections::BTreeMap<String, f64>) -> Self {
        let stage = Stage::from_reached(&m.stage_reached);
        if !m.ok || !m.correct {
            let error = m.error.clone().or_else(|| m.traceback_tail.clone()).unwrap_or_else(|| {
                if m.ok {
                    "incorrect".into()
                } else {
                    "evaluation failed".into()
                }
            });
            return Self::Failed { stage, error };
        }
        if m.per_config.is_empty() {
            // Correct but not benchmarked (compile / correctness stage).
            return Self::Verified { stage };
        }
        Self::Timed {
            stage,
            geomean_speedup: speedup_vs_baseline(baseline, &m.per_config),
            noise_margin: m.noise_margin.unwrap_or(0.0),
            per_config: m.per_config.clone(),
        }
    }

    #[must_use]
    pub const fn stage(&self) -> Stage {
        match self {
            Self::Failed { stage, .. } | Self::Verified { stage } | Self::Timed { stage, .. } => *stage,
        }
    }
}

/// Geomean speedup of a candidate's per-config latencies over the frozen `baseline`.
///
/// The `baseline` is the first correct candidate's per-config latencies.
/// Computed over the **intersection** of config names, so it stays well-defined
/// even for a partial config set. Returns `1.0` when there is no overlap — i.e.
/// the baseline candidate scored against itself, or an empty baseline before any
/// correct candidate has frozen it.
#[must_use]
pub fn speedup_vs_baseline(baseline: &std::collections::BTreeMap<String, f64>, per_config: &[PerConfig]) -> f64 {
    let ratios: Vec<f64> = per_config
        .iter()
        .filter_map(|pc| {
            let base = baseline.get(&pc.name).copied()?;
            (pc.latency_ms > 0.0 && base > 0.0).then_some(base / pc.latency_ms)
        })
        .collect();
    if ratios.is_empty() {
        return 1.0;
    }
    (ratios.iter().map(|r| r.ln()).sum::<f64>() / usize_to_f64_lossy(ratios.len())).exp()
}

/// Per-config timing produced by `scripts/evaluate.py` at the `full` stage.
///
/// Mirrors the JSON object the evaluator emits per benchmark point. The
/// evaluator is a pure benchmark: it reports the candidate's absolute
/// `latency_ms` (+ roofline), never a speedup or reference latency — speedup is
/// derived in the search layer via [`speedup_vs_baseline`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerConfig {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub latency_ms: f64,
    #[serde(default)]
    pub latency_std_ms: f64,
    #[serde(default)]
    pub rel_noise: f64,
    /// Achieved compute throughput on this config (TFLOP/s) and as a fraction of
    /// the device FLOP peak (MFU) — the binding roofline for *compute-bound*
    /// regimes. `None` when the problem defines no
    /// `flops()` or the device peak is unknown.
    #[serde(default)]
    pub achieved_tflops: Option<f64>,
    #[serde(default)]
    pub pct_peak: Option<f64>,
    /// Achieved memory bandwidth on this config (GB/s) and as a fraction of the
    /// device bandwidth peak — the binding roofline for *memory-bound* regimes.
    /// `None` when the problem defines no
    /// `bytes_moved()` or the device peak is unknown.
    #[serde(default)]
    pub achieved_gbps: Option<f64>,
    #[serde(default)]
    pub pct_bandwidth: Option<f64>,
}

/// Deserialized result of one `scripts/evaluate.py` invocation — the *trusted*
/// score. Optional fields are absent for the shallow stages (compile /
/// correctness) and populated for full.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    #[serde(default)]
    pub stage_requested: String,
    #[serde(default)]
    pub stage_reached: String,
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub correct: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub per_config: Vec<PerConfig>,
    #[serde(default)]
    pub noise_margin: Option<f64>,
    /// Device roofline peaks, echoed for context on the per-config achieved
    /// figures. `None` when the active device's peak is unknown.
    #[serde(default)]
    pub peak_tflops: Option<f64>,
    #[serde(default)]
    pub peak_gbps: Option<f64>,
    #[serde(default)]
    pub traceback_tail: Option<String>,
}

impl EvalMetrics {
    /// Parse the evaluator's single JSON stdout line.
    ///
    /// # Errors
    ///
    /// Returns a message containing the `serde_json` error and the raw line if
    /// `json` is not a JSON object matching this shape — in practice when the
    /// evaluator crashed and its stdout is a traceback rather than a metrics
    /// line, or when it printed something extra alongside the metrics.
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("evaluator returned unparseable JSON: {e}\nraw: {json}"))
    }
}

/// Metadata recorded beside each promoted version's `solution/` fileset in git.
/// Self-contained so a promoted version's tree (the fileset + `sidecar.json`) is
/// an atomic, crash-safe snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sidecar {
    /// Schema version of this sidecar shape, for forward-compatible resume.
    pub schema_version: u32,
    /// Promoted spine version (`Some(n)` for `v_n`), or `None` for an
    /// archive-only candidate that was never accepted.
    #[serde(default)]
    pub version: Option<u32>,
    /// Version this candidate was derived from (its parent on the spine).
    #[serde(default)]
    pub parent_version: Option<u32>,
    /// Content hash of the exact solution *fileset* this metric set describes
    /// (see [`fileset_sha256`]). Two byte-identical solution trees collapse onto
    /// one archive entry and one spine identity.
    pub solution_sha256: String,
    /// Did it match the reference within tolerance on every config?
    pub correct: bool,
    /// Deepest evaluation stage these metrics came from.
    pub stage: String,
    /// The trusted metrics.
    pub metrics: EvalMetrics,
    /// Free-form labels the agent attached on submit (e.g. "tiling", "fused-softmax").
    #[serde(default)]
    pub tags: Vec<String>,
    /// The agent's stated reason for this change.
    #[serde(default)]
    pub rationale: String,
    /// Creation time (unix milliseconds).
    pub created_at_unix_ms: u128,
}

impl Sidecar {
    pub const SCHEMA_VERSION: u32 = 1;
}

/// Running tally of everything a run consumes — the spend side of the budget,
/// and the raw material for cross-strategy comparison later.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetLedger {
    pub started_at_unix_ms: u128,
    pub wall_clock_secs: u64,
    pub turns: u64,
    pub evaluations: u64,
    pub queue_busy_secs: f64,
    /// Cumulative non-cached input tokens (billed at the full input rate).
    pub input_tokens: u64,
    /// Cumulative output / completion tokens.
    pub output_tokens: u64,
    /// Cumulative cache-read input tokens — prompt-cache hits, billed at a
    /// fraction of the input rate. Tracked separately because the per-token
    /// price differs sharply from fresh input.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Cumulative cache-write input tokens — cache creation, billed at a
    /// premium over fresh input. Also tracked separately for the same reason.
    #[serde(default)]
    pub cache_write_tokens: u64,
    /// Cumulative reasoning / thinking tokens (`OpenAI` reports these; Anthropic
    /// reports 0). Tracked for visibility only — NOT folded into
    /// [`total_tokens`](Self::total_tokens), because providers already count
    /// reasoning within the output/completion total (avoids double-counting the
    /// token cap).
    #[serde(default)]
    pub reasoning_tokens: u64,
    pub commits: u64,
    pub candidates_archived: u64,
}

impl BudgetLedger {
    #[must_use]
    pub fn new() -> Self {
        Self {
            started_at_unix_ms: now_unix_ms(),
            ..Default::default()
        }
    }

    /// Total tokens exchanged with the provider across all four classes. The
    /// classes are billed at different rates, so this is a *volume* figure (what
    /// was consumed), not a cost. Use the per-class fields for any pricing math.
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.output_tokens)
            .saturating_add(self.cache_read_tokens)
            .saturating_add(self.cache_write_tokens)
    }
}

/// One sample on the best-vs-budget curve.
///
/// Appended at the v0 seed and on every promotion, so a finished run can be
/// plotted as "best geomean achieved vs budget spent" against whichever axis
/// (evals / tokens / wall-clock / queue-busy seconds) a later strategy
/// comparison cares about.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// Promotions so far (the point is sampled just after this many).
    pub commits: u64,
    /// `evaluate`/`submit` runs so far.
    pub evals: u64,
    /// Total tokens exchanged so far ([`BudgetLedger::total_tokens`]).
    pub tokens: u64,
    /// Seconds the execution queue had been held (machine-busy) so far.
    pub queue_busy_secs: f64,
    /// Wall-clock seconds elapsed so far.
    pub wall_clock_secs: u64,
    /// Best geomean speedup over the agent's own first-correct kernel (the `1.0x`
    /// baseline) at this point — NOT vs the `Reference`, which is the correctness
    /// oracle only and never the speed target.
    pub best_geomean: f64,
    pub at_unix_ms: u128,
}

/// Random seeds that make a run reproducible.
///
/// `evaluator_inputs` is the per-run base seed for benchmark inputs; a *derived*
/// seed (`base XOR eval_counter`) is threaded into each evaluator invocation so
/// inputs stay fresh per evaluation while the sequence is reproducible for a
/// given run (see `scripts/evaluate.py`). `agent` is the model sampling seed when
/// the provider supports one — `None` today (neither the Anthropic nor `OpenAI`
/// request path sets a `seed`, and Snowflake Cortex passthrough is unverified).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seeds {
    #[serde(default)]
    pub agent: Option<u64>,
    #[serde(default)]
    pub evaluator_inputs: u64,
}

impl Seeds {
    /// Fresh seeds derived from the wall clock (mixed so two runs started in the
    /// same millisecond still differ); `agent` left `None`.
    #[must_use]
    pub fn generate() -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        Self::from_base((now.as_secs() << 20) ^ u64::from(now.subsec_nanos()))
    }

    /// Deterministic seeds from a single base value, so a pinned `--seed` makes
    /// the evaluator-input stream reproducible across an A/B run.
    #[must_use]
    pub const fn from_base(base: u64) -> Self {
        Self {
            agent: None,
            evaluator_inputs: base ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Static knobs that define a run's configuration, snapshotted into the manifest
/// so two runs are comparable and the exact regime a result was obtained under
/// is recorded.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunParams {
    /// Search policy selected for this run.
    #[serde(default)]
    pub policy: String,
    pub model: String,
    pub thinking_effort: String,
    pub max_output_tokens: u64,
    /// Supervisor intervention thresholds (turns without the named event).
    pub no_eval_threshold: u64,
    pub stagnation_threshold: u64,
    /// Last-turn context size that triggers compaction.
    pub compaction_threshold_tokens: u64,
    /// Floor on the fractional improvement a confirmation requires — the
    /// ranking-and-selection indifference zone δ\* (noise-independent minimum).
    pub min_improvement_frac: f64,
    /// Paired-confirmation sample count for the policy's `Reevaluate` (`>= 2`
    /// enables the drift-cancelling re-timing). Defaults to `0` when absent so
    /// older manifests still load.
    #[serde(default)]
    pub confirm_samples: u64,
    /// One-sided z used for the confirmation CI lower bound (≈95% ⇒ 1.645).
    #[serde(default)]
    pub confirm_z: f64,
    /// Slug of the active supervisor model, or `None` for static-nudge fallback.
    #[serde(default)]
    pub supervisor_model: Option<String>,
    /// Number of active evidence-ranked frontier nodes retained by `beam`.
    #[serde(default)]
    pub beam_width: u32,
    /// Exploration constant `c` for the `ucb` policy's confidence bound
    /// (`norm_geomean + c*sqrt(ln T / n)`); `0` for non-UCB runs.
    #[serde(default)]
    pub ucb_c: f64,
    /// Beam-diversity (frontier-collapse fix): exact de-dup on/off.
    #[serde(default)]
    pub diversity_dedup: bool,
}

/// Experiment manifest written (atomically) for every run. Captures the
/// parameters that define the experiment plus the latest ledger snapshot, so
/// runs are reproducible and comparable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub run_id: RunId,
    pub problem: String,
    pub model: String,
    /// The run's wall-clock budget in seconds (the only hard cap).
    pub max_wall_clock_secs: u64,
    pub ledger: BudgetLedger,
    /// Search policy that drove the run (currently always `"avo"`).
    #[serde(default)]
    pub policy_id: String,
    /// Backward-compatible mirror of [`policy_id`](Self::policy_id).
    #[serde(default)]
    pub strategy_id: String,
    /// Static run configuration snapshot.
    #[serde(default)]
    pub params: RunParams,
    /// Reproducibility seeds.
    #[serde(default)]
    pub seeds: Seeds,
    /// Best-vs-budget samples (seed + one per confirmed best improvement).
    #[serde(default)]
    pub curve: Vec<CurvePoint>,
    /// The current best node id (a *view* over the search tree) and its geomean
    /// speedup. `best_version` is kept as a deprecated alias for older manifests.
    #[serde(default, alias = "best_version_node")]
    pub best_node_id: Option<String>,
    #[serde(default)]
    pub best_geomean_speedup: Option<f64>,
    /// Backend the run evaluated on (e.g. `"mps"`, `"cuda"`), copied from the
    /// current best's metrics so figures can label themselves. `None` until the
    /// first evaluated version lands.
    #[serde(default)]
    pub device: Option<String>,
    pub created_at_unix_ms: u128,
}

/// Current unix time in milliseconds.
#[must_use]
pub fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// Hex-encoded SHA-256 of `bytes` (used to dedup candidates and tie metrics to
/// exact content).
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(hasher.finalize())
}

/// Canonical content hash of a whole solution fileset — the multi-file
/// generalization of [`sha256_hex`].
///
/// Files are folded in `BTreeMap` (sorted) order, each contribution
/// length-prefixed (path then content) so that no concatenation of distinct
/// `(path, content)` pairs can collide with another. This is the dedup key for
/// the archive and the spine identity in [`Sidecar::solution_sha256`].
#[must_use]
pub fn fileset_sha256(files: &SolutionFiles) -> String {
    let mut hasher = Sha256::new();
    for (path, content) in files {
        // Saturating rather than `as`: byte-identical to the original cast for
        // every length a real fileset can have, and this feeds a hash input, so
        // the encoding must not change.
        hasher.update(u64::try_from(path.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update(u64::try_from(content.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(content.as_bytes());
    }
    hex(hasher.finalize())
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    use std::fmt::Write;
    let digest = digest.as_ref();
    let mut out = String::with_capacity(digest.len().saturating_mul(2));
    for b in digest {
        let _ = write!(out, "{b:02x}");
    }
    out
}
