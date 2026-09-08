//! Static run-config snapshot (manifest) and best-vs-budget curve sampling.

use crate::domain::run_state::RunMeta;
use crate::domain::types::{CurvePoint, Manifest, RunParams};
use crate::domain::util::atomic_write;

use super::{
    AvoConfig, CONFIRM_SAMPLES, MAX_OUTPUT_TOKENS, SearchTree, THINKING, compaction, now_unix_ms, strategy, supervisor,
};

/// Static run-config snapshot recorded in the manifest, so a result captures the
/// exact regime (thresholds, caps, model) it was obtained under.
pub(super) fn run_params(cfg: &AvoConfig) -> RunParams {
    RunParams {
        policy: cfg.policy_id.clone(),
        model: cfg.model_name.clone(),
        thinking_effort: format!("{THINKING:?}"),
        max_output_tokens: u64::try_from(MAX_OUTPUT_TOKENS).unwrap_or(u64::MAX),
        no_eval_threshold: supervisor::NO_EVAL_THRESHOLD,
        stagnation_threshold: supervisor::STAGNATION_THRESHOLD,
        compaction_threshold_tokens: u64::from(cfg.context_window_tokens.saturating_sub(compaction::reserve_tokens(
            u32::try_from(MAX_OUTPUT_TOKENS).unwrap_or(u32::MAX),
        ))),
        min_improvement_frac: strategy::MIN_IMPROVEMENT_FRAC,
        confirm_samples: CONFIRM_SAMPLES,
        confirm_z: strategy::CONFIRM_Z,
        supervisor_model: cfg.supervisor_model.clone(),
        beam_width: u32::try_from(cfg.beam_width).unwrap_or(u32::MAX),
        ucb_c: cfg.ucb_c,
        diversity_dedup: cfg.diversity_dedup,
    }
}

/// Snapshot the current ledger + best score as a best-vs-budget [`CurvePoint`].
/// The ledger's `wall_clock_secs` / `queue_busy_secs` are expected to be fresh
/// (the caller refreshes them right before sampling).
pub(super) fn curve_point(meta: &RunMeta, best_geomean: f64) -> CurvePoint {
    CurvePoint {
        commits: meta.ledger.commits,
        evals: meta.ledger.evaluations,
        tokens: meta.ledger.total_tokens(),
        queue_busy_secs: meta.ledger.queue_busy_secs,
        wall_clock_secs: meta.ledger.wall_clock_secs,
        best_geomean,
        at_unix_ms: now_unix_ms(),
    }
}

/// Write the experiment manifest (atomic).
pub(super) fn write_manifest(
    cfg: &AvoConfig,
    search_tree: &SearchTree,
    meta: &RunMeta,
    params: &RunParams,
) -> Result<(), String> {
    let best = search_tree.current_best();
    let manifest = Manifest {
        run_id: meta.run_id.clone(),
        problem: cfg.problem_name.clone(),
        model: cfg.model_name.clone(),
        max_wall_clock_secs: cfg.max_wall_clock_secs,
        ledger: meta.ledger.clone(),
        policy_id: cfg.policy_id.clone(),
        strategy_id: cfg.policy_id.clone(),
        params: params.clone(),
        seeds: meta.seeds.clone(),
        curve: meta.curve.clone(),
        best_node_id: best.as_ref().map(|b| b.node_id.0.clone()),
        best_geomean_speedup: best.as_ref().and_then(|b| b.geomean_speedup),
        // Device labelling now comes from the run's probed hardware, not a
        // per-version sidecar (retired with the accepted lineage).
        device: None,
        created_at_unix_ms: now_unix_ms(),
    };
    let json = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    atomic_write(&cfg.run_dir.join("manifest.json"), &json).map_err(|e| e.to_string())
}
