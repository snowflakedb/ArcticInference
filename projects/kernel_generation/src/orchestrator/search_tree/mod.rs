//! Session tree, workspace snapshots, and policy state façade.
//!
//! New history is centered on a pure [`TurnTree`](crate::exec::turn_tree::TurnTree):
//! nodes store only provider-message deltas plus the workspace snapshot created
//! at that node. The search tree is the virtual view over `Timed` nodes; `best`
//! is a computed view over that tree (top *confirmed* score), and confirmation is
//! a per-node annotation the policy writes — there is no stored accepted lineage
//! and no `submit`/`promote` (see `avo_search_tree_design.md` §10).

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::domain::types::{Evaluation, SolutionFiles, fileset_sha256, now_unix_ms};
use crate::domain::util::lock;
use crate::exec::sandbox_manager::SandboxManager;
use crate::exec::snapshot_store::WorkspaceSnapshot;
use crate::exec::turn_tree::{ConfirmedScore, TurnNode, TurnNodeId, TurnTree, WorkspaceSnapshotId};
#[cfg(test)]
use crate::orchestrator::strategy;

// One canonical definition of the solution dir name lives in `exec::snapshot_store`.
const SOLUTION_SUBDIR: &str = crate::exec::snapshot_store::SOLUTION_DIR;
const HISTORY_DIR: &str = "history";
const NODES_FILE: &str = "nodes.jsonl";
const STATE_FILE: &str = "state.json";

/// The current best candidate — a *view* over the tree, not a stored version.
///
/// It is the `Timed` node with the highest confirmed score (falling back to the
/// raw geomean for a node the policy has not re-timed yet).
#[derive(Clone, Debug)]
pub struct BestInfo {
    pub node_id: TurnNodeId,
    pub geomean_speedup: Option<f64>,
    /// Whether the winning score has been confirmed (paired re-timing) or is still
    /// a single-sample geomean.
    pub confirmed: bool,
    pub solution_sha256: String,
}

pub type HistoryNode = TurnNode<Value>;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistoryState {
    #[serde(default)]
    pub current_leaf_node_id: Option<String>,
    #[serde(default)]
    pub frontier: Vec<FrontierEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrontierEntry {
    pub node_id: String,
    #[serde(default)]
    pub workspace_snapshot: Option<WorkspaceSnapshotId>,
    #[serde(default)]
    pub workspace_sha256: Option<String>,
    pub score: f64,
    pub stage_rank: u8,
    pub turn: u64,
    pub reason: String,
}

struct Inner {
    manager: SandboxManager,
    history_dir: PathBuf,
    state: Mutex<HistoryState>,
    write: Mutex<()>,
}

#[derive(Clone)]
pub struct SearchTree {
    inner: Arc<Inner>,
}

/// Everything needed to append one turn node to the session tree: the message
/// delta, the optional workspace/solution snapshot to capture, and the frontier
/// bookkeeping inputs.
struct TurnNodeInput<'a, M> {
    turn: u64,
    beam_width: usize,
    new_messages: Vec<M>,
    message_delta_json: Value,
    workspace_root: &'a Path,
    workspace_changed: bool,
    evaluated_files: Option<&'a SolutionFiles>,
    evaluation: Option<Evaluation>,
    resets_context: bool,
    /// When `Some`, use this node as the explicit parent instead of reading the
    /// shared `current_leaf_node_id` from state. Required for concurrent episodes
    /// (`p>1`) so each episode parents its nodes independently of the shared
    /// playhead. `None` → fall back to the state (serial / loop-side appends).
    explicit_parent: Option<TurnNodeId>,
}

impl SearchTree {
    /// Open with a bare snapshot manager (no mount shell). Used by tests and any
    /// caller that only reads/writes the tree + snapshots and never spawns a real
    /// sandbox. Production uses [`with_manager`](Self::with_manager).
    ///
    /// # Errors
    ///
    /// Fails if the snapshot git repository under `run_dir/repo` cannot be created
    /// or opened, or if [`with_manager`](Self::with_manager) fails.
    pub fn open(run_dir: &Path) -> Result<Self, String> {
        let manager = SandboxManager::bare(run_dir.join("repo")).map_err(|e| e.to_string())?;
        Self::with_manager(run_dir, manager)
    }

    /// Open backed by a fully-configured [`SandboxManager`] (owns the snapshot repo
    /// + the mount shell used to spawn per-branch sandboxes).
    ///
    /// # Errors
    ///
    /// Fails if the `history/` directory cannot be created, or if reconciling with
    /// the persisted `state.json` cannot rewrite it.
    pub fn with_manager(run_dir: &Path, manager: SandboxManager) -> Result<Self, String> {
        let history_dir = run_dir.join(HISTORY_DIR);
        fs::create_dir_all(&history_dir).map_err(|e| e.to_string())?;
        let store = Self {
            inner: Arc::new(Inner {
                manager,
                history_dir,
                state: Mutex::new(HistoryState::default()),
                write: Mutex::new(()),
            }),
        };
        store.reconcile()?;
        Ok(store)
    }

    /// The snapshot manager backing this tree (git repo + sandbox minting).
    #[must_use]
    pub fn manager(&self) -> &SandboxManager {
        &self.inner.manager
    }

    #[must_use]
    pub fn has_tree_history(run_dir: &Path) -> bool {
        run_dir.join(HISTORY_DIR).join(NODES_FILE).exists()
    }

    #[must_use]
    pub fn history_nodes_path(&self) -> PathBuf {
        self.inner.history_dir.join(NODES_FILE)
    }

    pub fn current_node_id(&self) -> Option<TurnNodeId> {
        lock(&self.inner.state)
            .current_leaf_node_id
            .clone()
            .map(TurnNodeId::new)
    }

    /// Move the shared playhead (`current_leaf_node_id`) to `node_id`.
    ///
    /// # Errors
    ///
    /// Fails if `node_id` is not a node in the tree, if `nodes.jsonl` cannot be
    /// parsed, or if the updated `state.json` cannot be written.
    pub fn checkout_node(&self, node_id: &TurnNodeId) -> Result<(), String> {
        if self.turn_tree::<Value>()?.get(node_id).is_none() {
            return Err(format!("unknown session node {node_id}"));
        }
        let mut st = lock(&self.inner.state);
        st.current_leaf_node_id = Some(node_id.0.clone());
        drop(st);
        self.save_state_locked()
    }

    /// The whole session tree, parsed from `nodes.jsonl`.
    ///
    /// # Errors
    ///
    /// Fails if `nodes.jsonl` holds a malformed non-trailing line, a duplicate node
    /// id, or a node whose parent is missing. An absent file is not an error — it
    /// yields an empty tree.
    pub fn turn_tree<M>(&self) -> Result<TurnTree<M>, String>
    where
        M: Clone + Serialize + DeserializeOwned,
    {
        let text = fs::read_to_string(self.history_nodes_path()).unwrap_or_default();
        TurnTree::from_jsonl(&text)
    }

    /// The provider-message context accumulated along the root→`node_id` path.
    ///
    /// # Errors
    ///
    /// Fails if the tree cannot be parsed (see [`turn_tree`](Self::turn_tree)), or
    /// if `node_id` is unknown or its ancestry contains a cycle.
    pub fn context<M>(&self, node_id: &TurnNodeId) -> Result<Vec<M>, String>
    where
        M: Clone + Serialize + DeserializeOwned,
    {
        self.turn_tree::<M>()?.context(node_id)
    }

    /// Append a turn node from a full message list, deriving the delta JSON for
    /// callers.
    ///
    /// # Errors
    ///
    /// Fails if the workspace/solution snapshot cannot be committed, if the
    /// parent's effective workspace cannot be resolved, or if `nodes.jsonl` /
    /// `state.json` cannot be written.
    #[allow(clippy::too_many_arguments)]
    pub fn append_typed_turn_node<M>(
        &self,
        turn: u64,
        beam_width: usize,
        new_messages: Vec<M>,
        workspace_root: &Path,
        workspace_changed: bool,
        evaluated_files: Option<&SolutionFiles>,
        evaluation: Option<Evaluation>,
    ) -> Result<String, String>
    where
        M: Clone + Serialize + DeserializeOwned,
    {
        let message_delta_json = serde_json::to_value(&new_messages).unwrap_or(Value::Null);
        self.append_turn_node_inner(TurnNodeInput {
            turn,
            beam_width,
            new_messages,
            message_delta_json,
            workspace_root,
            workspace_changed,
            evaluated_files,
            evaluation,
            resets_context: false,
            explicit_parent: None,
        })
    }

    /// Like [`append_typed_turn_node`] but uses `parent` as the explicit parent
    /// node instead of the shared playhead. Required for concurrent expanders
    /// (`p>1`) so each episode parents its turns independently. The global
    /// `current_leaf_node_id` is still updated to the new node on each append,
    /// which is acceptable: it tracks "most recently appended" for resume
    /// purposes; the per-episode parent chain is carried locally in
    /// [`EpisodeInit::start_node_id`] and advanced by the caller.
    ///
    /// # Errors
    ///
    /// Same as [`append_typed_turn_node`](Self::append_typed_turn_node).
    #[allow(clippy::too_many_arguments)]
    pub fn append_typed_turn_node_from<M>(
        &self,
        parent: Option<&TurnNodeId>,
        turn: u64,
        beam_width: usize,
        new_messages: Vec<M>,
        workspace_root: &Path,
        workspace_changed: bool,
        evaluated_files: Option<&SolutionFiles>,
        evaluation: Option<Evaluation>,
    ) -> Result<String, String>
    where
        M: Clone + Serialize + DeserializeOwned,
    {
        let message_delta_json = serde_json::to_value(&new_messages).unwrap_or(Value::Null);
        self.append_turn_node_inner(TurnNodeInput {
            turn,
            beam_width,
            new_messages,
            message_delta_json,
            workspace_root,
            workspace_changed,
            evaluated_files,
            evaluation,
            resets_context: false,
            explicit_parent: parent.cloned(),
        })
    }

    /// Append a context-reset node: a turn that cuts the context chain, carrying no
    /// workspace snapshot and no evaluation.
    ///
    /// # Errors
    ///
    /// Fails if `nodes.jsonl` or `state.json` cannot be written.
    pub fn append_context_reset_node<M>(&self, turn: u64, new_messages: Vec<M>) -> Result<String, String>
    where
        M: Clone + Serialize + DeserializeOwned,
    {
        let message_delta_json = serde_json::to_value(&new_messages).unwrap_or(Value::Null);
        self.append_turn_node_inner(TurnNodeInput {
            turn,
            beam_width: 1,
            new_messages,
            message_delta_json,
            workspace_root: Path::new("."),
            workspace_changed: false,
            evaluated_files: None,
            evaluation: None,
            resets_context: true,
            explicit_parent: None,
        })
    }

    /// Seed the root node (node 0): the stub-solution workspace, **unscored**.
    /// The reference is the correctness oracle only, so the seed is not
    /// benchmarked and carries no `Timed` evaluation or `confirmed_score` — the
    /// performance baseline (the confirmed 1.0x incumbent) is established later by
    /// the agent's first correct candidate. The root is a plain workspace node.
    ///
    /// # Errors
    ///
    /// Fails if the seed snapshot cannot be committed to the snapshot repo, or if
    /// `nodes.jsonl` / `state.json` cannot be written.
    pub fn seed_root(&self, files: &SolutionFiles) -> Result<TurnNodeId, String> {
        let _write = lock(&self.inner.write);
        // Bind the leaf before branching so the state guard is released here,
        // rather than living until the end of the `if let`.
        let existing_leaf = lock(&self.inner.state).current_leaf_node_id.clone();
        if let Some(leaf) = existing_leaf {
            return Ok(TurnNodeId::new(leaf));
        }
        let node_id = Self::next_node_id_locked(0, &serde_json::json!({"seed": "v0"}), None);
        // Snapshot the seed solution into the node's workspace.
        let mut entries: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (rel, content) in files {
            entries.insert(format!("{SOLUTION_SUBDIR}/{rel}"), content.as_bytes().to_vec());
        }
        let snapshot_id = self.inner.manager.save_snapshot(&node_id, None, &entries)?;
        let node = HistoryNode {
            node_id: TurnNodeId::new(node_id.clone()),
            parent_id: None,
            resets_context: false,
            turn: 0,
            new_messages: Vec::new(),
            new_workspace: Some(snapshot_id),
            evaluation: None,
            confirmed_score: None,
            timestamp_unix_ms: now_unix_ms(),
        };
        self.append_node_locked(&node)?;
        let mut st = lock(&self.inner.state);
        st.current_leaf_node_id = Some(node_id.clone());
        drop(st);
        self.save_state_locked()?;
        Ok(TurnNodeId::new(node_id))
    }

    /// Append a turn node from an already-built message-delta JSON value.
    ///
    /// # Errors
    ///
    /// Same as [`append_typed_turn_node`](Self::append_typed_turn_node).
    pub fn append_turn_node(
        &self,
        turn: u64,
        beam_width: usize,
        message_delta_json: Value,
        workspace_root: &Path,
        workspace_changed: bool,
        evaluated_files: Option<&SolutionFiles>,
    ) -> Result<String, String> {
        self.append_turn_node_inner(TurnNodeInput {
            turn,
            beam_width,
            new_messages: vec![message_delta_json.clone()],
            message_delta_json,
            workspace_root,
            workspace_changed,
            evaluated_files,
            evaluation: None,
            resets_context: false,
            explicit_parent: None,
        })
    }

    fn append_turn_node_inner<M>(&self, input: TurnNodeInput<M>) -> Result<String, String>
    where
        M: Clone + Serialize + DeserializeOwned,
    {
        let TurnNodeInput {
            turn,
            beam_width,
            new_messages,
            message_delta_json,
            workspace_root,
            workspace_changed,
            evaluated_files,
            evaluation,
            resets_context,
            explicit_parent,
        } = input;
        let _write = lock(&self.inner.write);
        // Use the explicit parent when provided (concurrent episodes); otherwise
        // read from the shared state (serial / loop-side appends).
        let parent_id = explicit_parent
            .map(|n| n.0)
            .or_else(|| lock(&self.inner.state).current_leaf_node_id.clone());
        let node_id = Self::next_node_id_locked(turn, &message_delta_json, parent_id.as_deref());
        let (new_workspace, workspace_sha256) = if let Some(files) = evaluated_files {
            let sha = fileset_sha256(files);
            let mut snapshot = WorkspaceSnapshot::from_dir(workspace_root)?;
            let solution_prefix = format!("{SOLUTION_SUBDIR}/");
            snapshot
                .files
                .retain(|path, _| path != SOLUTION_SUBDIR && !path.starts_with(&solution_prefix));
            for (path, content) in files {
                snapshot
                    .files
                    .insert(format!("{SOLUTION_SUBDIR}/{path}"), content.as_bytes().to_vec());
            }
            let parent_snapshot = self.effective_workspace_locked(parent_id.as_deref())?;
            let snapshot_id = self
                .inner
                .manager
                .save_snapshot(&node_id, parent_snapshot.as_ref(), &snapshot.files)?;
            (Some(snapshot_id), Some(sha))
        } else if workspace_changed {
            let snapshot = WorkspaceSnapshot::from_dir(workspace_root)?;
            let parent_snapshot = self.effective_workspace_locked(parent_id.as_deref())?;
            let snapshot_id = self
                .inner
                .manager
                .save_snapshot(&node_id, parent_snapshot.as_ref(), &snapshot.files)?;
            (Some(snapshot_id), Some(snapshot.sha256))
        } else {
            (None, None)
        };
        // Resolve this node's evaluation: any typed evaluation the caller passed,
        // then a fallback distilled from the message delta (the path tests exercise).
        let evaluation = evaluation.or_else(|| latest_evaluation(&message_delta_json));
        let node = HistoryNode {
            node_id: TurnNodeId::new(node_id.clone()),
            parent_id: parent_id.map(TurnNodeId::new),
            resets_context,
            turn,
            new_messages: new_messages
                .into_iter()
                .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
                .collect(),
            new_workspace,
            evaluation: evaluation.clone(),
            // New nodes are unconfirmed; the policy sets this later via
            // `record_confirmation` (paired re-timing on promotion).
            confirmed_score: None,
            timestamp_unix_ms: now_unix_ms(),
        };
        self.append_node_locked(&node)?;
        let mut st = lock(&self.inner.state);
        if turn != 0
            && let Some(entry) = frontier_entry_for_node(&node, evaluation.as_ref(), workspace_sha256.as_deref())
        {
            upsert_frontier(&mut st.frontier, entry, beam_width.max(1));
        }
        st.current_leaf_node_id = Some(node_id.clone());
        drop(st);
        self.save_state_locked()?;
        Ok(node_id)
    }

    /// Recompute the beam frontier from scratch by replaying every stored node.
    ///
    /// # Errors
    ///
    /// Fails if `nodes.jsonl` cannot be read or parsed, or if `state.json` cannot be
    /// written.
    pub fn rebuild_frontier(&self, beam_width: usize) -> Result<(), String> {
        let _write = lock(&self.inner.write);
        let nodes = load_nodes(&self.inner.history_dir)?;
        let mut frontier = Vec::new();
        for node in nodes {
            if node.turn == 0 {
                continue;
            }
            let workspace_sha = node
                .new_workspace
                .as_ref()
                .and_then(|snap| self.inner.manager.load_solution_snapshot(snap).ok())
                .map(|files| fileset_sha256(&files));
            if let Some(entry) = frontier_entry_for_node(&node, node.evaluation.as_ref(), workspace_sha.as_deref()) {
                upsert_frontier(&mut frontier, entry, beam_width.max(1));
            }
        }
        let mut st = lock(&self.inner.state);
        st.frontier = frontier;
        drop(st);
        self.save_state_locked()
    }

    /// The current best candidate as a *view* over the tree: the highest-scored
    /// node whose score has been **confirmed** (paired re-timing). Unconfirmed
    /// high-raw-geomean nodes are only *candidates* — never crowned best until the
    /// policy re-times them (the winner's-curse/drift defense). `None` before the
    /// seed is confirmed.
    /// The best confirmed candidate across the WHOLE tree (all branches). Used by
    /// the orchestrator's global bookkeeping (manifest, run summary, policy).
    #[must_use]
    pub fn current_best(&self) -> Option<BestInfo> {
        self.best_confirmed(None)
    }

    /// The best confirmed candidate on `node_id`'s OWN lineage (the root→node path,
    /// inclusive) — not the global tree. This is what the agent-facing anchor uses
    /// under beam search (p>1): an expansion episode is a worker on one node and
    /// must see only its own line, never a sibling branch's result it neither
    /// produced nor can load (the "13.08x I never made" confusion). Falls out to
    /// [`current_best`] semantics on a single lineage, where the two coincide.
    #[must_use]
    pub fn lineage_best(&self, node_id: &TurnNodeId) -> Option<BestInfo> {
        self.best_confirmed(Some(node_id))
    }

    /// Shared core: best confirmed candidate, optionally restricted to the
    /// root→`lineage_of` ancestry. `None` filter ⇒ whole tree.
    fn best_confirmed(&self, lineage_of: Option<&TurnNodeId>) -> Option<BestInfo> {
        let nodes = load_nodes(&self.inner.history_dir).ok()?;
        let lineage = lineage_of.map(|n| lineage_id_set(&nodes, n));
        let keep = |id: &str| lineage.as_ref().is_none_or(|s| s.contains(id));
        let (node_id, score) = nodes
            .iter()
            .filter(|n| keep(n.node_id.0.as_str()))
            .filter_map(|n| n.confirmed_score.as_ref().map(|c| (n.node_id.0.clone(), c.geomean)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        let node = TurnNodeId::new(node_id);
        let sha = self
            .read_solution_at(&node)
            .ok()
            .map(|files| fileset_sha256(&files))
            .unwrap_or_default();
        Some(BestInfo {
            node_id: node,
            geomean_speedup: Some(score),
            confirmed: true,
            solution_sha256: sha,
        })
    }

    /// Record a confirmation annotation on a node (the policy's paired re-timing
    /// result), rewriting `nodes.jsonl` in place. Makes this node's score
    /// `confirmed` for the `best` view. Rare (only on promotion); serialized with
    /// appends by the write lock.
    ///
    /// # Errors
    ///
    /// Fails if `nodes.jsonl` cannot be read or parsed, if `node_id` is not in it,
    /// or if the rewritten file cannot be persisted.
    pub fn record_confirmation(&self, node_id: &TurnNodeId, geomean: f64, samples: usize) -> Result<(), String> {
        let _write = lock(&self.inner.write);
        let mut nodes = load_nodes(&self.inner.history_dir)?;
        let node = nodes
            .iter_mut()
            .find(|n| n.node_id.as_str() == node_id.as_str())
            .ok_or_else(|| format!("record_confirmation: unknown node {node_id}"))?;
        node.confirmed_score = Some(ConfirmedScore { geomean, samples });
        rewrite_nodes(&self.inner.history_dir, &nodes)
    }

    #[must_use]
    pub fn confirmed_score(&self, node_id: &TurnNodeId) -> Option<ConfirmedScore> {
        load_nodes(&self.inner.history_dir)
            .ok()?
            .into_iter()
            .find(|n| n.node_id.as_str() == node_id.as_str())?
            .confirmed_score
    }

    /// Read the solution tree materialized at a node (from its effective workspace
    /// snapshot). Replaces the old tag-based `read_solution(version)`.
    ///
    /// # Errors
    ///
    /// Fails if the tree cannot be parsed, if neither `node_id` nor any ancestor
    /// carries a workspace snapshot, or if the snapshot cannot be read back.
    pub fn read_solution_at(&self, node_id: &TurnNodeId) -> Result<SolutionFiles, String> {
        let snapshot = self
            .turn_tree::<Value>()?
            .effective_workspace(node_id)?
            .ok_or_else(|| format!("node {node_id} has no effective workspace snapshot"))?;
        self.inner.manager.load_solution_snapshot(&snapshot)
    }

    /// Materialize a node's FULL workspace — `solution/` plus `artifacts/` (verified
    /// standalone kernels, incl. binary profiler outputs), `notes/`, and any other
    /// source — directly into `dest_dir` via git's tree→worktree extraction
    /// ([`SandboxManager::extract_tree_into`]): byte-exact and a fixed 2 git
    /// processes, no per-file read. This is what the `checkout` tool uses to recover
    /// verified kernels the agent would otherwise re-derive. Returns the file count.
    ///
    /// # Errors
    ///
    /// Fails if the tree cannot be parsed, if neither `node_id` nor any ancestor
    /// carries a workspace snapshot, or if git cannot extract that tree into
    /// `dest_dir`.
    pub fn extract_workspace_into(&self, node_id: &TurnNodeId, dest_dir: &Path) -> Result<usize, String> {
        let snapshot = self
            .turn_tree::<Value>()?
            .effective_workspace(node_id)?
            .ok_or_else(|| format!("node {node_id} has no effective workspace snapshot"))?;
        self.inner.manager.extract_tree_into(snapshot.as_str(), dest_dir)
    }

    /// A compact, score-ranked view of the whole tree's `Timed` candidate nodes
    /// (all branches) for global bookkeeping. Highest score first.
    #[must_use]
    pub fn ranked_candidates(&self) -> Vec<CandidateView> {
        self.candidates_ranked(None)
    }

    /// Score-ranked candidates on `node_id`'s OWN lineage (root→node path). The
    /// agent-facing `search_view` uses this so the standings it sees are its own
    /// line, not sibling beam branches (see [`lineage_best`]).
    #[must_use]
    pub fn lineage_candidates(&self, node_id: &TurnNodeId) -> Vec<CandidateView> {
        self.candidates_ranked(Some(node_id))
    }

    /// Shared core: `Timed` candidates ranked best-first, optionally restricted to
    /// the root→`lineage_of` ancestry. `None` filter ⇒ whole tree.
    fn candidates_ranked(&self, lineage_of: Option<&TurnNodeId>) -> Vec<CandidateView> {
        let Ok(nodes) = load_nodes(&self.inner.history_dir) else {
            return Vec::new();
        };
        let lineage = lineage_of.map(|n| lineage_id_set(&nodes, n));
        let keep = |id: &str| lineage.as_ref().is_none_or(|s| s.contains(id));
        let mut out: Vec<CandidateView> = nodes
            .iter()
            .filter(|n| keep(n.node_id.0.as_str()))
            .filter_map(|node| match &node.evaluation {
                Some(Evaluation::Timed {
                    geomean_speedup,
                    noise_margin,
                    ..
                }) => {
                    let conf = node.confirmed_score.as_ref();
                    Some(CandidateView {
                        node_id: node.node_id.0.clone(),
                        turn: node.turn,
                        geomean: conf.map_or(*geomean_speedup, |c| c.geomean),
                        confirmed: conf.is_some(),
                        noise_margin: *noise_margin,
                        samples: conf.map(|c| c.samples),
                    })
                }
                _ => None,
            })
            .collect();
        out.sort_by(|a, b| b.geomean.partial_cmp(&a.geomean).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    #[must_use]
    pub fn frontier(&self) -> Vec<FrontierEntry> {
        lock(&self.inner.state).frontier.clone()
    }

    /// The solution files of an arbitrary workspace snapshot, by commit id.
    ///
    /// # Errors
    ///
    /// Fails if `commit` is not a snapshot the snapshot repo can read.
    pub fn read_workspace_snapshot(&self, commit: &str) -> Result<SolutionFiles, String> {
        self.inner
            .manager
            .load_solution_snapshot(&WorkspaceSnapshotId::new(commit))
    }

    /// The nearest workspace snapshot at or above `node_id` (`None` when no node is
    /// given, or when no node on the path carries one).
    ///
    /// # Errors
    ///
    /// Fails if `nodes.jsonl` cannot be read or parsed, if `node_id` is unknown, or
    /// if the ancestry contains a cycle.
    pub fn nearest_workspace_commit(&self, node_id: Option<&str>) -> Result<Option<String>, String> {
        let _write = lock(&self.inner.write);
        Ok(self.effective_workspace_locked(node_id)?.map(|id| id.0))
    }

    fn append_node_locked(&self, node: &HistoryNode) -> Result<(), String> {
        fs::create_dir_all(&self.inner.history_dir).map_err(|e| e.to_string())?;
        let path = self.inner.history_dir.join(NODES_FILE);
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut f, node).map_err(|e| e.to_string())?;
        f.write_all(b"\n").map_err(|e| e.to_string())?;
        Ok(())
    }

    fn save_state_locked(&self) -> Result<(), String> {
        let json = serde_json::to_vec_pretty(&*lock(&self.inner.state)).map_err(|e| e.to_string())?;
        crate::domain::util::atomic_write(&self.inner.history_dir.join(STATE_FILE), &json).map_err(|e| e.to_string())
    }

    /// Reconcile in-memory state with the persisted `state.json`: load the current
    /// leaf + frontier. Best/confirmed are views over the tree (recomputed from the
    /// nodes' `confirmed_score`), so nothing about them is stored to reconcile.
    fn reconcile(&self) -> Result<(), String> {
        let st = load_history_state(&self.inner.history_dir).unwrap_or_default();
        *lock(&self.inner.state) = st;
        self.save_state_locked()?;
        Ok(())
    }

    fn next_node_id_locked(turn: u64, message_delta_json: &Value, parent_id: Option<&str>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(turn.to_le_bytes());
        if let Some(parent) = parent_id {
            hasher.update(parent.as_bytes());
        }
        hasher.update(serde_json::to_vec(message_delta_json).unwrap_or_default());
        hasher.update(now_unix_ms().to_le_bytes());
        let hex = crate::domain::types::sha256_hex(&hasher.finalize());
        // `hex` is 64 ASCII chars, so `..16` is always a char boundary; `get` states
        // that safely instead of relying on the slice not panicking.
        format!("n{}", hex.get(..16).unwrap_or(&hex))
    }

    fn effective_workspace_locked(&self, node_id: Option<&str>) -> Result<Option<WorkspaceSnapshotId>, String> {
        let nodes = load_nodes(&self.inner.history_dir)?;
        let tree = TurnTree::from_nodes(nodes)?;
        let Some(node_id) = node_id else {
            return Ok(None);
        };
        tree.effective_workspace(&TurnNodeId::new(node_id))
    }
}

/// A score-ranked candidate node for agent/supervisor-facing views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateView {
    pub node_id: String,
    pub turn: u64,
    pub geomean: f64,
    pub confirmed: bool,
    /// Fractional single-eval timing noise on this candidate's geomean, surfaced so
    /// a display never presents a within-noise raw delta as a real improvement.
    pub noise_margin: f64,
    /// Paired re-timing sample count once confirmed (`None` while unconfirmed).
    /// `>=2` ⇒ a real paired confirmation; `Some(1)` ⇒ a single-sample provisional
    /// crown (baseline or inconclusive fallback), which the display marks distinctly.
    pub samples: Option<usize>,
}

mod helpers;
use helpers::{
    frontier_entry_for_node, latest_evaluation, lineage_id_set, load_history_state, load_nodes, rewrite_nodes,
    upsert_frontier,
};

#[cfg(test)]
mod tests;
