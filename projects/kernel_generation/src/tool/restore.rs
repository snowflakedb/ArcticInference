//! Agent-facing checkout tool.
//!
//! The search keeps every confirmed candidate safe on its own node, so the best
//! kernel and its verified standalone test kernels are never lost — but the
//! working tree routinely diverges (a later turn reverts `solution/`, a context
//! reset drops the code from view), and the agent was observed re-deriving a
//! verified standalone kernel ≥8×. This tool materializes a confirmed node's
//! FULL source — `solution/`, the verified standalone kernels under `artifacts/`,
//! and its `notes/` — into a fresh `checkout/` subdir of the episode's OWN
//! sandbox. It is **non-destructive**: the live `solution/` and the cumulative
//! `notes/` ledger are untouched, so the agent can compare both versions and copy
//! across what it needs. `checkout/` is excluded from every snapshot, so it never
//! affects the score or the search state (the same read-only-w.r.t.-search
//! contract as `write`/`edit`/`evaluate`; it deliberately does NOT touch
//! `checkout_node`, which moves the shared playhead and would race under beam).

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::sandbox::Sandbox;
use crate::exec::turn_tree::TurnNodeId;
use crate::orchestrator::search_tree::SearchTree;
use crate::tool::{Tool, ToolOutput};

// Scratch dir the recovered workspace is written into — the canonical name (and
// its snapshot exclusion) live in `exec::snapshot_store`.
use crate::exec::snapshot_store::CHECKOUT_DIR;

#[derive(Deserialize, JsonSchema)]
pub struct RestoreArgs {
    /// Which candidate to check out into `checkout/`. Omit (or pass `"best"`) for
    /// the best CONFIRMED candidate on your lineage. Otherwise pass a node id
    /// shown by `search_view` to recover that specific candidate's source.
    #[serde(default)]
    pub node: Option<String>,
}

pub struct Restore {
    pub search_tree: SearchTree,
    pub sandbox: Arc<Sandbox>,
    /// This episode's own node — scopes `"best"` to this lineage under beam search
    /// (p>1), mirroring `SearchView`. `None` ⇒ whole-tree best (cold start).
    pub lineage_node: Option<TurnNodeId>,
}

impl Tool for Restore {
    type Args = RestoreArgs;
    const NAME: &'static str = "checkout";
    const DESCRIPTION: &'static str = "Recover a CONFIRMED candidate's full verified source into a fresh `checkout/` dir \
         (NON-destructive — your live `solution/` and `notes/` are untouched, so you can diff and copy across what you need). \
         Includes `solution/` AND the verified standalone kernels under `artifacts/` — use it to recover a kernel you already \
         verified instead of re-deriving it after a revert or context reset. Omit `node` (or pass \"best\") for the best \
         confirmed kernel on your lineage, or pass a node id from `search_view`. Recovers SOURCE only (rebuild before you \
         `evaluate`); does not change your score or the search state.";

    async fn call(&self, args: RestoreArgs) -> ToolOutput {
        let want_best = args
            .node
            .as_deref()
            .is_none_or(|s| s.trim().is_empty() || s.trim().eq_ignore_ascii_case("best"));

        let (node, origin) = if !want_best && let Some(id) = args.node.as_deref() {
            // `want_best` already mapped a missing / blank / "best" argument to
            // `true`, so reaching here means the agent named a concrete node.
            (TurnNodeId::new(id.trim().to_string()), "requested node".to_string())
        } else {
            let best = self
                .lineage_node
                .as_ref()
                .map_or_else(|| self.search_tree.current_best(), |n| self.search_tree.lineage_best(n));
            match best {
                Some(b) => {
                    let g = b.geomean_speedup.unwrap_or(0.0);
                    (
                        b.node_id,
                        format!("best confirmed (geomean {g:.4}x vs first-correct baseline)"),
                    )
                }
                None => return Err("no confirmed candidate yet — evaluate one first".to_string()),
            }
        };

        // Materialize the node's full source (solution/ + artifacts/ + notes/) into
        // a fresh checkout/ via git's byte-exact tree extraction — the live tree is
        // untouched. Wipe any prior checkout/ first so it's an exact mirror.
        let dest = self.sandbox.workspace().join(CHECKOUT_DIR);
        if dest.exists() {
            std::fs::remove_dir_all(&dest).map_err(|e| format!("failed to clear {CHECKOUT_DIR}/: {e}"))?;
        }
        let n = self
            .search_tree
            .extract_workspace_into(&node, &dest)
            .map_err(|e| format!("cannot check out workspace at node {node}: {e}"))?;
        Ok(format!(
            "checked out {n} file(s) from {origin} (node {node}) into {CHECKOUT_DIR}/ \
             (e.g. {CHECKOUT_DIR}/solution/, {CHECKOUT_DIR}/artifacts/). Your live solution/ and notes/ are unchanged — \
             copy across what you need, then rebuild before evaluating."
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkout_tool_name() {
        assert_eq!(Restore::NAME, "checkout");
    }

    #[tokio::test]
    async fn checkout_writes_into_checkout_dir_non_destructively() {
        use crate::exec::snapshot_store::SOLUTION_DIR;
        use crate::orchestrator::search_tree::SearchTree;
        use std::collections::BTreeMap;

        // A node whose workspace has solution/ + artifacts/ source.
        let run_dir = tempfile::tempdir().unwrap();
        let tree = SearchTree::open(run_dir.path()).unwrap();
        let ws = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(ws.path().join("solution")).unwrap();
        std::fs::create_dir_all(ws.path().join("artifacts")).unwrap();
        std::fs::write(ws.path().join("solution/solution.py"), "RECOVERED").unwrap();
        std::fs::write(ws.path().join("artifacts/tc.cu"), "// verified gemm").unwrap();
        let node = tree
            .append_turn_node(1, 1, serde_json::json!({ "assistant": "edit" }), ws.path(), true, None)
            .unwrap();

        // A live sandbox whose solution/ has DIVERGED from that node.
        let sb = Arc::new(Sandbox::new().unwrap());
        let live: BTreeMap<String, String> = [("solution.py".to_string(), "LIVE-WIP".to_string())].into();
        sb.write_tree(SOLUTION_DIR, &live).unwrap();

        let tool = Restore {
            search_tree: tree,
            sandbox: sb.clone(),
            lineage_node: None,
        };
        tool.call(RestoreArgs { node: Some(node) }).await.unwrap();

        // Recovered source landed under checkout/ (solution AND artifacts)…
        assert_eq!(
            sb.read(&format!("{CHECKOUT_DIR}/solution/solution.py")).unwrap(),
            "RECOVERED"
        );
        assert!(
            sb.read(&format!("{CHECKOUT_DIR}/artifacts/tc.cu"))
                .unwrap()
                .contains("verified")
        );
        // …and the live solution/ was left untouched (non-destructive).
        assert_eq!(sb.read(&format!("{SOLUTION_DIR}/solution.py")).unwrap(), "LIVE-WIP");
    }
}
