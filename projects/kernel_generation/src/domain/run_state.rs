//! Crash-safe run journal.
//!
//! `state.json` under `runs/<id>/` holds the [`RunMeta`] (ledger, counters, and
//! supervisor state) and is written atomically (tmp + fsync + rename) after
//! every turn and every tool batch.
//!
//! The provider transcript is *not* journaled here: it is derived on demand
//! from the session tree (`history/nodes.jsonl`), which is the source of truth.
//! On `--resume` the model-visible context is reconstructed by walking the tree
//! from the current leaf, so a killed node, OOM, or `Ctrl-C` resumes cleanly.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::domain::types::{BudgetLedger, CurvePoint, RunId, Seeds};
use crate::domain::util::atomic_write;

const STATE_FILE: &str = "state.json";

/// Run state: everything needed to resume the control loop other than the
/// provider transcript, which is derived from the session tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub run_id: RunId,
    pub ledger: BudgetLedger,
    /// Turns completed so far.
    pub turns: u64,
    /// Turns since the confirmed best last improved (drives stagnation detection).
    #[serde(alias = "turns_since_commit")]
    pub turns_since_best_improved: u64,
    /// Context size (tokens) reported by the last turn, for compaction.
    pub last_context_tokens: u32,
    /// Reproducibility seeds for this run (also surfaced in the manifest).
    #[serde(default)]
    pub seeds: Seeds,
    /// Best-vs-budget samples (seed + one per confirmed best improvement).
    /// Journaled here so the curve survives `--resume`; `write_manifest` copies it
    /// into the manifest.
    #[serde(default)]
    pub curve: Vec<CurvePoint>,
}

impl RunMeta {
    #[must_use]
    pub const fn new(run_id: RunId, ledger: BudgetLedger, seeds: Seeds) -> Self {
        Self {
            run_id,
            ledger,
            turns: 0,
            turns_since_best_improved: 0,
            last_context_tokens: 0,
            seeds,
            curve: Vec::new(),
        }
    }
}

/// Atomically persist the meta (`state.json`).
///
/// # Errors
///
/// Returns a stringified error if `meta` cannot be serialized to JSON, or if the
/// atomic write fails — the run directory not being creatable, the temp file not
/// being writable or fsyncable, or the rename over `state.json` failing.
pub fn save_meta(run_dir: &Path, meta: &RunMeta) -> Result<(), String> {
    let json = serde_json::to_vec_pretty(meta).map_err(|e| e.to_string())?;
    atomic_write(&run_dir.join(STATE_FILE), &json).map_err(|e| e.to_string())
}

/// Load the meta if a prior run journaled one.
#[must_use]
pub fn load_meta(run_dir: &Path) -> Option<RunMeta> {
    let bytes = std::fs::read(run_dir.join(STATE_FILE)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn meta_round_trips() {
        let dir = TempDir::new().unwrap();
        let mut meta = RunMeta::new(
            RunId("run_test".to_string()),
            BudgetLedger::new(),
            Seeds {
                agent: Some(7),
                evaluator_inputs: 42,
            },
        );
        meta.turns = 7;
        meta.turns_since_best_improved = 3;
        meta.last_context_tokens = 12345;
        meta.curve.push(CurvePoint {
            commits: 1,
            evals: 4,
            best_geomean: 1.5,
            ..Default::default()
        });
        save_meta(dir.path(), &meta).unwrap();

        let loaded = load_meta(dir.path()).expect("meta present");
        assert_eq!(loaded.turns, 7);
        assert_eq!(loaded.turns_since_best_improved, 3);
        assert_eq!(loaded.last_context_tokens, 12345);
        assert_eq!(loaded.run_id, RunId("run_test".to_string()));
        // Seeds + curve survive the journal round-trip.
        assert_eq!(loaded.seeds.agent, Some(7));
        assert_eq!(loaded.seeds.evaluator_inputs, 42);
        assert_eq!(loaded.curve.len(), 1);
        assert!(
            (loaded.curve[0].best_geomean - 1.5).abs() < f64::EPSILON,
            "{}",
            loaded.curve[0].best_geomean
        );
        assert_eq!(loaded.curve[0].evals, 4);
    }
}
