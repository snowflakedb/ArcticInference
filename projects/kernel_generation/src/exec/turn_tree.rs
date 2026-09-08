//! Pure replay tree for provider-message sessions.
//!
//! A [`TurnTree`] stores only node-local deltas: provider messages introduced
//! by a node and the workspace snapshot created at that node. Full context and
//! effective workspace are derived by walking ancestors.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::domain::types::Evaluation;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnNodeId(pub String);

impl TurnNodeId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TurnNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkspaceSnapshotId(pub String);

impl WorkspaceSnapshotId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WorkspaceSnapshotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A confirmation annotation over a `Timed` node.
///
/// The low-variance re-timed
/// geomean the policy produced (paired against the incumbent) + the paired sample
/// count. The winner's-curse/drift defense. Set on a node when the policy promotes
/// it; the search tree's `best` view is the highest confirmed score. Lives on the
/// node so `best`/`ranked_candidates` are computed purely from the tree.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConfirmedScore {
    pub geomean: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(bound(deserialize = "M: Deserialize<'de>"))]
pub struct TurnNode<M> {
    pub node_id: TurnNodeId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<TurnNodeId>,
    /// If true, replay starts at this node instead of walking ancestors.
    ///
    /// Context compaction intentionally replaces the provider transcript with a
    /// concise summary. Persisting that summary as an ordinary child node would
    /// not help because replay would still include every pre-compaction
    /// ancestor before it.
    #[serde(default)]
    pub resets_context: bool,
    pub turn: u64,
    #[serde(default)]
    pub new_messages: Vec<M>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_workspace: Option<WorkspaceSnapshotId>,
    /// The distilled outcome of this node's evaluation, if it produced one. The
    /// physical tree carries it directly; the search tree's "scorable" predicate
    /// reads it (a `Timed` node is a ranked candidate). `None` for turns that
    /// didn't evaluate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<Evaluation>,
    /// Confirmation annotation (paired re-timing result), set by the policy when
    /// this node is promoted. `None` until confirmed. The `best` view derives from
    /// the max confirmed score across the tree.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmed_score: Option<ConfirmedScore>,
    pub timestamp_unix_ms: u128,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(bound(deserialize = "M: Deserialize<'de>"))]
pub struct TurnTree<M> {
    nodes: BTreeMap<TurnNodeId, TurnNode<M>>,
}

impl<M: Clone> TurnTree<M> {
    #[must_use]
    pub const fn new() -> Self {
        Self { nodes: BTreeMap::new() }
    }

    /// # Errors
    ///
    /// Returns a message if two nodes share a `node_id`, or if a node names a
    /// `parent_id` that no earlier node in the iteration order introduced (see
    /// [`TurnTree::insert`]).
    pub fn from_nodes(nodes: impl IntoIterator<Item = TurnNode<M>>) -> Result<Self, String> {
        let mut tree = Self::new();
        for node in nodes {
            tree.insert(node)?;
        }
        Ok(tree)
    }

    /// # Errors
    ///
    /// Returns a message if `node.node_id` is already in the tree, or if
    /// `node.parent_id` is `Some` but names a node that is not present — the two
    /// invariants that keep ancestor walks total.
    pub fn insert(&mut self, node: TurnNode<M>) -> Result<(), String> {
        if self.nodes.contains_key(&node.node_id) {
            return Err(format!("duplicate session node {}", node.node_id));
        }
        if let Some(parent) = &node.parent_id
            && !self.nodes.contains_key(parent)
        {
            return Err(format!(
                "session node {} references missing parent {parent}",
                node.node_id
            ));
        }
        self.nodes.insert(node.node_id.clone(), node);
        Ok(())
    }

    #[must_use]
    pub fn get(&self, node_id: &TurnNodeId) -> Option<&TurnNode<M>> {
        self.nodes.get(node_id)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes(&self) -> impl Iterator<Item = &TurnNode<M>> {
        self.nodes.values()
    }

    /// # Errors
    ///
    /// Returns a message if `node_id` is not in the tree, if an ancestor link points
    /// at a missing node, or if the ancestor walk revisits a node (cycle).
    pub fn context(&self, node_id: &TurnNodeId) -> Result<Vec<M>, String> {
        let mut chain = self.ancestor_chain(node_id)?;
        chain.reverse();
        if let Some(reset_at) = chain.iter().rposition(|node| node.resets_context) {
            chain = chain.split_off(reset_at);
        }
        Ok(chain
            .into_iter()
            .flat_map(|node| node.new_messages.iter().cloned())
            .collect())
    }

    /// # Errors
    ///
    /// Returns a message if `node_id` or an ancestor is not in the tree, or if the
    /// walk toward the root revisits a node (cycle).
    pub fn effective_workspace(&self, node_id: &TurnNodeId) -> Result<Option<WorkspaceSnapshotId>, String> {
        let mut cur = Some(node_id.clone());
        let mut seen = BTreeSet::new();
        // `take()` rather than moving `cur` out: it leaves `cur` initialized (as
        // `None`), which is what lets the `clone_from` at the end of the body reuse
        // the existing allocation instead of re-assigning a fresh clone.
        while let Some(id) = cur.take() {
            if !seen.insert(id.clone()) {
                return Err(format!("cycle detected at session node {id}"));
            }
            let node = self
                .nodes
                .get(&id)
                .ok_or_else(|| format!("unknown session node {id}"))?;
            if let Some(snapshot) = &node.new_workspace {
                return Ok(Some(snapshot.clone()));
            }
            cur.clone_from(&node.parent_id);
        }
        Ok(None)
    }

    fn ancestor_chain(&self, node_id: &TurnNodeId) -> Result<Vec<&TurnNode<M>>, String> {
        let mut out = Vec::new();
        let mut cur = Some(node_id.clone());
        let mut seen = BTreeSet::new();
        // See `effective_workspace`: `take()` keeps `cur` initialized for `clone_from`.
        while let Some(id) = cur.take() {
            if !seen.insert(id.clone()) {
                return Err(format!("cycle detected at session node {id}"));
            }
            let node = self
                .nodes
                .get(&id)
                .ok_or_else(|| format!("unknown session node {id}"))?;
            out.push(node);
            cur.clone_from(&node.parent_id);
        }
        Ok(out)
    }
}

impl<M> TurnTree<M>
where
    M: Clone + Serialize + DeserializeOwned,
{
    /// # Errors
    ///
    /// Returns a message if `serde_json` cannot serialize a node — in practice a
    /// message type `M` whose `Serialize` impl fails (e.g. a non-string map key).
    pub fn to_jsonl(&self) -> Result<String, String> {
        let mut out = String::new();
        for node in self.nodes.values() {
            out.push_str(&serde_json::to_string(node).map_err(|e| e.to_string())?);
            out.push('\n');
        }
        Ok(out)
    }

    /// # Errors
    ///
    /// Returns a message if any line other than the last fails to parse as a
    /// [`TurnNode`] (genuine corruption — a corrupt *trailing* line is tolerated, see
    /// below), or if the parsed nodes violate the tree invariants checked by
    /// [`TurnTree::from_nodes`] (duplicate id, missing parent).
    pub fn from_jsonl(text: &str) -> Result<Self, String> {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let mut nodes = Vec::with_capacity(lines.len());
        for (i, line) in lines.iter().enumerate() {
            // 1-based line number for messages; bound once so the guard below and
            // the two messages cannot drift apart.
            let line_no = i.saturating_add(1);
            match serde_json::from_str(line) {
                Ok(node) => nodes.push(node),
                // Tolerate a corrupt/incomplete *trailing* line: a crash mid-append
                // can leave a partial last record, and failing the whole resume over
                // it would be worse than dropping that one turn. A bad *non-last*
                // line is genuine corruption (and would orphan later nodes' parent
                // refs anyway), so still error.
                Err(e) if line_no == lines.len() => {
                    eprintln!(
                        "[history] ignoring corrupt trailing session-node line {line_no} (likely a crash mid-append): {e}"
                    );
                }
                Err(e) => return Err(format!("session node line {line_no} parse error: {e}")),
            }
        }
        Self::from_nodes(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        AnthropicContentBlock, AnthropicMessage, AnthropicMessageRole, OpenaiInputContent, OpenaiItem, OpenaiMessage,
        OpenaiMessageRole,
    };

    fn node<M>(id: &str, parent: Option<&str>, turn: u64, messages: Vec<M>, workspace: Option<&str>) -> TurnNode<M> {
        TurnNode {
            node_id: TurnNodeId::new(id),
            parent_id: parent.map(TurnNodeId::new),
            resets_context: false,
            turn,
            new_messages: messages,
            new_workspace: workspace.map(WorkspaceSnapshotId::new),
            evaluation: None,
            confirmed_score: None,
            timestamp_unix_ms: u128::from(turn),
        }
    }

    fn reset_node<M>(id: &str, parent: Option<&str>, turn: u64, messages: Vec<M>) -> TurnNode<M> {
        TurnNode {
            resets_context: true,
            ..node(id, parent, turn, messages, None)
        }
    }

    #[test]
    fn context_returns_root_to_node_order() {
        let tree = TurnTree::from_nodes([
            node("root", None, 0, vec!["a"], None),
            node("left", Some("root"), 1, vec!["b", "c"], None),
            node("leaf", Some("left"), 2, vec!["d"], None),
        ])
        .unwrap();
        assert_eq!(
            tree.context(&TurnNodeId::new("leaf")).unwrap(),
            vec!["a", "b", "c", "d"]
        );
    }

    #[test]
    fn context_reset_discards_ancestors_before_replay() {
        let tree = TurnTree::from_nodes([
            node("root", None, 0, vec!["old prompt"], None),
            node("turn1", Some("root"), 1, vec!["old work"], None),
            reset_node("compact", Some("turn1"), 2, vec!["summary"]),
            node("turn3", Some("compact"), 3, vec!["new work"], None),
        ])
        .unwrap();
        assert_eq!(
            tree.context(&TurnNodeId::new("turn3")).unwrap(),
            vec!["summary", "new work"]
        );
    }

    #[test]
    fn effective_workspace_uses_own_or_nearest_ancestor() {
        let tree = TurnTree::from_nodes([
            node::<&str>("root", None, 0, vec![], Some("w0")),
            node("left", Some("root"), 1, vec![], None),
            node("leaf", Some("left"), 2, vec![], Some("w2")),
            node("sibling", Some("left"), 3, vec![], None),
        ])
        .unwrap();
        assert_eq!(
            tree.effective_workspace(&TurnNodeId::new("leaf"))
                .unwrap()
                .unwrap()
                .as_str(),
            "w2"
        );
        assert_eq!(
            tree.effective_workspace(&TurnNodeId::new("sibling"))
                .unwrap()
                .unwrap()
                .as_str(),
            "w0"
        );
    }

    #[test]
    fn siblings_have_independent_contexts_and_workspaces() {
        let tree = TurnTree::from_nodes([
            node("root", None, 0, vec!["root"], Some("w0")),
            node("a", Some("root"), 1, vec!["a"], Some("wa")),
            node("b", Some("root"), 1, vec!["b"], None),
        ])
        .unwrap();
        assert_eq!(tree.context(&TurnNodeId::new("a")).unwrap(), vec!["root", "a"]);
        assert_eq!(tree.context(&TurnNodeId::new("b")).unwrap(), vec!["root", "b"]);
        assert_eq!(
            tree.effective_workspace(&TurnNodeId::new("a"))
                .unwrap()
                .unwrap()
                .as_str(),
            "wa"
        );
        assert_eq!(
            tree.effective_workspace(&TurnNodeId::new("b"))
                .unwrap()
                .unwrap()
                .as_str(),
            "w0"
        );
    }

    #[test]
    fn anthropic_messages_round_trip() {
        let msg = AnthropicMessage {
            role: AnthropicMessageRole::User,
            content: vec![AnthropicContentBlock::Text {
                text: "hello".into(),
                cache_control: None,
            }],
        };
        let tree = TurnTree::from_nodes([node("root", None, 0, vec![msg], Some("w"))]).unwrap();
        let jsonl = tree.to_jsonl().unwrap();
        let decoded = TurnTree::<AnthropicMessage>::from_jsonl(&jsonl).unwrap();
        assert_eq!(decoded.context(&TurnNodeId::new("root")).unwrap().len(), 1);
    }

    #[test]
    fn openai_messages_round_trip() {
        let msg = OpenaiMessage {
            items: vec![OpenaiItem::Message {
                role: OpenaiMessageRole::User,
                content: vec![OpenaiInputContent::InputText { text: "hello".into() }],
            }],
        };
        let tree = TurnTree::from_nodes([node("root", None, 0, vec![msg], Some("w"))]).unwrap();
        let jsonl = tree.to_jsonl().unwrap();
        let decoded = TurnTree::<OpenaiMessage>::from_jsonl(&jsonl).unwrap();
        assert_eq!(decoded.context(&TurnNodeId::new("root")).unwrap().len(), 1);
    }

    #[test]
    fn from_jsonl_tolerates_corrupt_trailing_line() {
        // A crash mid-append can leave a partial final record. Resume must survive it.
        // Build the jsonl in append (topological) order — the file is always written
        // parents-before-children, unlike `to_jsonl`'s id-sorted output.
        let root = node::<String>("n0root", None, 0, vec!["a".into()], None);
        let leaf = node::<String>("n1leaf", Some("n0root"), 1, vec!["b".into()], None);
        let mut jsonl = format!(
            "{}\n{}\n",
            serde_json::to_string(&root).unwrap(),
            serde_json::to_string(&leaf).unwrap()
        );
        jsonl.push_str("{\"node_id\":\"leaf2\",\"turn\":2,\"new_mess"); // truncated, no newline
        let parsed = TurnTree::<String>::from_jsonl(&jsonl).unwrap();
        assert_eq!(parsed.len(), 2, "valid nodes survive; the corrupt tail is dropped");
        assert_eq!(
            parsed.context(&TurnNodeId::new("n1leaf")).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn from_jsonl_errors_on_corrupt_non_trailing_line() {
        // A bad line that is NOT the last one is genuine corruption, not a crash tail.
        let good = serde_json::to_string(&node::<String>("n0root", None, 0, vec!["a".into()], None)).unwrap();
        let leaf = serde_json::to_string(&node::<String>("n1leaf", Some("n0root"), 1, vec!["b".into()], None)).unwrap();
        let jsonl = format!("{good}\ngarbage-not-json\n{leaf}\n");
        assert!(
            TurnTree::<String>::from_jsonl(&jsonl).is_err(),
            "a corrupt non-trailing line must fail the parse"
        );
    }
}
