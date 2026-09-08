//! Agent-facing search view tool.
//!
//! In the search-tree-first model there is no `submit`/promotion tool: the agent
//! only `evaluate`s, every scorable eval becomes a candidate node, and the search
//! keeps the best. This host-side, read-only tool lets the agent see where it
//! stands — the current best score and the top-ranked candidate nodes — via a
//! shared [`SearchTree`] handle (no GPU work, no queue).

use schemars::JsonSchema;
use serde::Deserialize;

use crate::exec::turn_tree::TurnNodeId;
use crate::orchestrator::search_tree::SearchTree;
use crate::tool::{Tool, ToolOutput};

// ─── search_view ───────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct SearchViewArgs {}

pub struct SearchView {
    pub search_tree: SearchTree,
    /// This episode's own node (its start node). The view is scoped to this
    /// node's lineage (root→node ancestry), so under beam search (p>1) the agent
    /// sees only its own line's standings — never a sibling branch's result it
    /// didn't produce and can't load. `None` ⇒ show the whole tree (cold start).
    pub lineage_node: Option<TurnNodeId>,
}

impl Tool for SearchView {
    type Args = SearchViewArgs;
    const NAME: &'static str = "search_view";
    const DESCRIPTION: &'static str = "Show YOUR lineage's search state: the best confirmed geomean speedup (over the first-correct baseline) on your own line and the top scored candidate \
         solutions on it (most recent evals become candidates automatically — there \
         is no separate submit step). Scoped to your lineage — other search branches are the orchestrator's to manage. Read-only.";

    async fn call(&self, _args: SearchViewArgs) -> ToolOutput {
        use std::fmt::Write as _;
        let mut s = String::new();
        let best = self.lineage_node.as_ref().map_or_else(
            || self.search_tree.current_best(),
            |node| self.search_tree.lineage_best(node),
        );
        match best {
            Some(b) => {
                let _ = writeln!(
                    s,
                    "best confirmed on your lineage: geomean {:.4}x vs first-correct baseline",
                    b.geomean_speedup.unwrap_or(0.0),
                );
            }
            None => s.push_str("best confirmed on your lineage: (none yet — evaluate a candidate)\n"),
        }
        let ranked = self.lineage_node.as_ref().map_or_else(
            || self.search_tree.ranked_candidates(),
            |node| self.search_tree.lineage_candidates(node),
        );
        if ranked.is_empty() {
            s.push_str("candidates: (none evaluated yet)\n");
        } else {
            s.push_str(
                "top candidates by raw geomean — ONLY the confirmed best above is a real improvement; \
                 unconfirmed candidates are pending paired re-timing and a higher raw number here may be within noise:\n",
            );
            for c in ranked.iter().take(10) {
                let tag = match c.samples {
                    Some(n) if n >= 2 => " (confirmed)".to_string(),
                    Some(n) => format!(" (provisional, {n} sample — not paired-confirmed)"),
                    None => " [unconfirmed — pending re-timing]".to_string(),
                };
                let _ = writeln!(
                    s,
                    "  turn {:>3}: geomean {:.4} ±{:.1}%{}",
                    c.turn,
                    c.geomean,
                    c.noise_margin * 100.0,
                    tag,
                );
            }
        }
        Ok(s.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_view_tool_name() {
        assert_eq!(SearchView::NAME, "search_view");
    }
}
