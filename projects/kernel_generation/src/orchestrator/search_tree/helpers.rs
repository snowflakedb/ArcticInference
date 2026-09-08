//! Free-function helpers for the search tree: node/state (de)serialization and
//! beam-frontier scoring. Kept out of the [`SearchTree`](super::SearchTree) impl
//! because they hold no lock and touch no `Inner` state. (Git plumbing +
//! workspace restore live in [`crate::exec::sandbox_manager`].)

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::domain::types::{EvalMetrics, Evaluation};

use super::{FrontierEntry, HistoryNode, HistoryState, NODES_FILE, STATE_FILE, TurnNodeId};

pub(super) fn load_history_state(history_dir: &Path) -> Option<HistoryState> {
    let bytes = fs::read(history_dir.join(STATE_FILE)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Rewrite `nodes.jsonl` in place (atomic) — used by `record_confirmation` to set
/// a node's `confirmed_score`. Nodes are append-only in the common case; this is
/// the rare exception (a promotion), serialized with appends by the write lock.
pub(super) fn rewrite_nodes(history_dir: &Path, nodes: &[HistoryNode]) -> Result<(), String> {
    let mut buf = Vec::new();
    for node in nodes {
        serde_json::to_writer(&mut buf, node).map_err(|e| e.to_string())?;
        buf.push(b'\n');
    }
    crate::domain::util::atomic_write(&history_dir.join(NODES_FILE), &buf).map_err(|e| e.to_string())
}

/// Node ids on the root→`node_id` path (inclusive): `node_id` and all its
/// ancestors, walked via `parent_id`. Scopes the agent-facing best/candidate
/// views to the current episode's own lineage under beam search (p>1), where the
/// global tree also holds sibling branches the agent never produced and cannot
/// act on. A `set.insert` that returns `false` breaks the walk (cycle guard —
/// impossible in a well-formed tree, but keeps a corrupt DAG from looping).
pub(super) fn lineage_id_set(nodes: &[HistoryNode], node_id: &TurnNodeId) -> std::collections::HashSet<String> {
    let by_id: std::collections::HashMap<&str, &HistoryNode> =
        nodes.iter().map(|n| (n.node_id.0.as_str(), n)).collect();
    let mut set = std::collections::HashSet::new();
    let mut cur = Some(node_id.0.clone());
    while let Some(id) = cur {
        if !set.insert(id.clone()) {
            break;
        }
        cur = by_id
            .get(id.as_str())
            .and_then(|n| n.parent_id.as_ref().map(|p| p.0.clone()));
    }
    set
}

pub(super) fn load_nodes(history_dir: &Path) -> Result<Vec<HistoryNode>, String> {
    let path = history_dir.join(NODES_FILE);
    let Ok(text) = fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut out = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        match serde_json::from_str(line) {
            Ok(node) => out.push(node),
            // Tolerate a corrupt/incomplete trailing line (crash mid-append); a bad
            // non-last line is real corruption, so still error. Mirrors
            // `TurnTree::from_jsonl`.
            Err(e) if i.saturating_add(1) == lines.len() => {
                eprintln!(
                    "[history] ignoring corrupt trailing history-node line {} (likely a crash mid-append): {e}",
                    i.saturating_add(1)
                );
            }
            Err(e) => return Err(format!("history node line {} parse error: {e}", i.saturating_add(1))),
        }
    }
    Ok(out)
}

/// Distill an [`Evaluation`] from a message delta that carries an `evaluations`
/// array (the fallback path when no typed evaluation was threaded in — chiefly
/// the tests). Takes the last object in the array as the effective metrics.
pub(super) fn latest_evaluation(message_delta_json: &Value) -> Option<Evaluation> {
    let obj = message_delta_json
        .get("evaluations")?
        .as_array()?
        .iter()
        .rev()
        .find(|v| v.is_object())?;
    let metrics: EvalMetrics = serde_json::from_value(obj.clone()).ok()?;
    // Fallback/test path only: no baseline is threaded here, so a benchmarked
    // eval scores against an empty baseline (1.0x). Production always threads the
    // typed, baseline-aware evaluation from `run_episode`.
    Some(Evaluation::from_metrics(&metrics, &std::collections::BTreeMap::new()))
}

pub(super) fn frontier_entry_for_node(
    node: &HistoryNode,
    evaluation: Option<&Evaluation>,
    workspace_sha256: Option<&str>,
) -> Option<FrontierEntry> {
    // Only scorable outcomes join the frontier. `Failed` is a real node but never
    // ranked; `Verified` (correct, untimed) ranks by stage depth alone; `Timed`
    // adds a noise-discounted speedup so deeper, faster evidence sorts first.
    let evaluation = evaluation?;
    let (stage, score) = match evaluation {
        Evaluation::Failed { .. } => return None,
        Evaluation::Verified { stage } => (*stage, f64::from(stage.rank()) * 1_000_000.0),
        Evaluation::Timed {
            stage,
            geomean_speedup,
            noise_margin,
            ..
        } => (
            *stage,
            // `rank()` is a `u8`, so the product is exact and `mul_add` rounds
            // identically to the `a * b + c` this replaces.
            f64::from(stage.rank()).mul_add(1_000_000.0, geomean_speedup / (1.0 + noise_margin)),
        ),
    };
    Some(FrontierEntry {
        node_id: node.node_id.0.clone(),
        workspace_snapshot: node.new_workspace.clone(),
        workspace_sha256: workspace_sha256.map(str::to_string),
        score,
        stage_rank: stage.rank(),
        turn: node.turn,
        reason: frontier_reason(evaluation),
    })
}

pub(super) fn frontier_reason(evaluation: &Evaluation) -> String {
    match evaluation {
        Evaluation::Timed {
            stage, geomean_speedup, ..
        } => format!("{} correct=true geomean={geomean_speedup:.4}", stage.as_str()),
        Evaluation::Verified { stage } => format!("{} correct=true", stage.as_str()),
        Evaluation::Failed { stage, .. } => format!("{} failed", stage.as_str()),
    }
}

pub(super) fn upsert_frontier(frontier: &mut Vec<FrontierEntry>, entry: FrontierEntry, beam_width: usize) {
    frontier.retain(|e| {
        e.node_id != entry.node_id
            && match (&e.workspace_sha256, &entry.workspace_sha256) {
                (Some(a), Some(b)) => a != b,
                _ => true,
            }
    });
    frontier.push(entry);
    frontier.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.turn.cmp(&a.turn))
    });
    frontier.truncate(beam_width);
}
