//! Per-branch working-plan (todo) tool.
//!
//! A long beam/AVO search runs for hundreds of turns across many context
//! resets. Without a durable plan the agent re-litigates ideas it already
//! tried — a real 10h run circled the same structural direction for ~230 turns.
//! This tool gives the agent a persistent, hand-editable working plan that
//! survives compaction and resume, and is re-shown after every context reset
//! (see `orchestrator::reinject_todos`).
//!
//! Storage: a single JSON file `notes/todo.json` in the BRANCH WORKSPACE. It is
//! per-branch for free — each expansion episode spawns its own sandbox from its
//! start node's workspace snapshot, and `notes/` is captured by the snapshot
//! walk (`WorkspaceSnapshot::from_dir`, not in `should_exclude_workspace_path`)
//! and restored on resume/expansion (`restore_full_workspace`). It deliberately
//! lives OUTSIDE `solution/` (which is the scored artifact), so a plan edit
//! never perturbs the solution fileset the evaluator scores.
//!
//! The file is the SOLE source of truth. Every call is a stateless
//! read-modify-write (no `Arc<Mutex>`, no in-process cache), so the plan is
//! tolerant of hand-edits and of concurrent branches: a missing OR corrupt file
//! is treated as an empty list and silently re-initialized on the next write.
//! This respects the harness's branch-state discipline — no cross-branch shared
//! mutable state, the same reason the search tree keeps per-node workspaces.
//!
//! Tools:
//! - `todo_write { items: [...] }` — a FULL-LIST write (mirrors pi/opencode's
//!   `todowrite`): items carrying an `id` update in place; items without one get
//!   a fresh monotonic id. Returns the rendered post-write list (with ids) so the
//!   model sees the assigned ids.
//! - `todo_read {}` — return the current rendered list.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::exec::sandbox::Sandbox;
use crate::tool::{Tool, ToolOutput};

/// Branch-workspace path of the plan file. Under `notes/` (snapshot-captured,
/// non-`solution/`) so it rides the per-branch workspace snapshot but never
/// touches the scored solution tree.
pub const TODO_PATH: &str = "notes/todo.json";

/// Char cap on the rendered plan (à la `orchestrator::ACTIVITY_NOTE_CHARS` /
/// `RECENT_ACTIVITY_CHARS`).
///
/// Open items render first, so if the plan overflows it is the settled
/// (completed/cancelled) tail that is elided, never the work still in flight.
pub const TODO_RENDER_CHARS: usize = 2_400;

// ─── data model ───────────────────────────────────────────────────────────────

/// Lifecycle of a plan item. `snake_case` on the wire (`in_progress`, …) so the
/// JSON file is readable and hand-editable.
///
/// `Cancelled` vs `Deferred` is a load-bearing distinction. `Cancelled` = an
/// evaluation proved the direction a DEAD END → do not re-explore it. `Deferred`
/// = parked for time/priority (e.g. a large structural build set aside), NOT
/// disproven → it stays a valid, re-openable lever while budget remains. Keeping
/// them separate stops "re-shown after every reset" from re-instructing the agent
/// to stay off a merely-deferred structural build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    #[default]
    Pending,
    InProgress,
    /// Parked for time/priority, not disproven — re-openable while budget remains.
    Deferred,
    Completed,
    Cancelled,
}

/// One plan item as persisted in `notes/todo.json`. `id` is assigned by the tool
/// (`t1`, `t2`, … monotonic); `active_form` is the optional present-continuous
/// label shown while the item is in progress.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_form: Option<String>,
}

/// One item as sent by the model to `todo_write`. `id` is optional (present ⇒
/// update in place, absent ⇒ create with a fresh id); `status` defaults to
/// `pending` so a bare new item is legal.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TodoItemInput {
    /// Stable id of an existing item to update in place (e.g. `t3`). Omit to
    /// create a new item — a fresh monotonic id is assigned and returned.
    #[serde(default)]
    pub id: Option<String>,
    /// The plan item: one optimization DIRECTION (e.g. "rewrite onto the newest
    /// tensor/compute instructions" or "restructure into an async pipeline"), not
    /// a micro-step.
    pub content: String,
    /// `pending` | `in_progress` | `deferred` | `completed` | `cancelled`.
    /// Defaults to `pending`. Mark `in_progress` when you start a direction;
    /// `completed` when it lands; `cancelled` ONLY when an evaluation proved it a
    /// dead end (you will not re-open it); `deferred` when you park a still-viable
    /// direction (e.g. a large structural build) for time/priority — it stays
    /// re-openable while budget remains. Put the outcome in `content`.
    #[serde(default)]
    pub status: Status,
    /// Optional present-continuous label shown while `in_progress`
    /// (e.g. "bringing up a new hardware feature standalone").
    #[serde(default)]
    pub active_form: Option<String>,
}

// ─── file I/O (stateless read-modify-write) ─────────────────────────────────

/// Load the plan from `notes/todo.json`, degrading to an empty list.
///
/// A missing file (fresh branch) OR a corrupt/hand-broken file both degrade to
/// an empty list — the plan is advisory memory, never load-bearing, so it must
/// never error a turn or a run.
pub fn load_todos(sandbox: &Sandbox) -> Vec<TodoItem> {
    sandbox
        .read(TODO_PATH)
        .map_or_else(|_| Vec::new(), |raw| serde_json::from_str(&raw).unwrap_or_default())
}

/// Persist the plan as pretty JSON (readable + hand-editable).
fn save_todos(sandbox: &Sandbox, items: &[TodoItem]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(items).map_err(|e| e.to_string())?;
    sandbox.write(TODO_PATH, &json).map_err(|e| e.to_string())?;
    Ok(())
}

/// Numeric suffix of a `t<N>` id, if it parses. Ids the model invents in some
/// other shape simply don't participate in the high-water mark (they're kept
/// verbatim on update); new ids always continue past the largest `t<N>` seen.
fn id_num(id: &str) -> Option<u64> {
    id.strip_prefix('t').and_then(|n| n.parse::<u64>().ok())
}

// ─── rendering ───────────────────────────────────────────────────────────────

const fn status_marker(s: Status) -> &'static str {
    match s {
        Status::InProgress => "[~]",
        Status::Pending => "[ ]",
        Status::Deferred => "[>]",
        Status::Completed => "[x]",
        Status::Cancelled => "[-]",
    }
}

/// Ordering key: live work first (`in_progress`, pending, then deferred — parked
/// but re-openable), then the settled items (completed, then cancelled). Stable
/// within a group, so the model's given order is preserved.
const fn status_rank(s: Status) -> u8 {
    match s {
        Status::InProgress => 0,
        Status::Pending => 1,
        Status::Deferred => 2,
        Status::Completed => 3,
        Status::Cancelled => 4,
    }
}

/// Render the plan compactly: open items first, capped at [`TODO_RENDER_CHARS`]
/// (settled tail elided first, with a count). Always shows at least one line.
#[must_use]
pub fn render_todos(items: &[TodoItem]) -> String {
    use std::fmt::Write as _;
    if items.is_empty() {
        return "(no plan items yet)".to_string();
    }
    let mut ordered: Vec<&TodoItem> = items.iter().collect();
    ordered.sort_by_key(|t| status_rank(t.status)); // stable sort
    let mut out = String::new();
    let mut rendered = 0usize;
    for t in &ordered {
        // While in progress, prefer the present-continuous label if given.
        let label = match (t.status, t.active_form.as_deref()) {
            (Status::InProgress, Some(af)) if !af.trim().is_empty() => af,
            _ => t.content.as_str(),
        };
        let line = format!("{} {}: {}\n", status_marker(t.status), t.id, label);
        // Cap, but always emit at least the first (highest-priority) line.
        if rendered > 0 && out.len().saturating_add(line.len()) > TODO_RENDER_CHARS {
            break;
        }
        out.push_str(&line);
        rendered = rendered.saturating_add(1);
    }
    if rendered < ordered.len() {
        let _ = writeln!(
            out,
            "… (+{} settled item(s) elided; full plan in {TODO_PATH})",
            ordered.len().saturating_sub(rendered)
        );
    }
    out
}

/// The rendered plan for re-injection, or `None` when the branch has no plan yet
/// (so the caller skips injecting an empty note — mirrors the ground-truth
/// anchor's empty-string skip).
pub fn render_for_injection(sandbox: &Sandbox) -> Option<String> {
    let items = load_todos(sandbox);
    if items.is_empty() {
        return None;
    }
    Some(render_todos(&items))
}

/// Apply a full-list write to `existing`, assigning fresh monotonic ids to items
/// that lack one. New ids continue past the largest `t<N>` in BOTH the on-disk
/// list and the incoming ids, so a freshly assigned id never collides with one
/// the model was just shown (even across a drop-and-re-add). Pure so it is unit
/// testable without a sandbox.
fn apply_write(existing: &[TodoItem], items: Vec<TodoItemInput>) -> Vec<TodoItem> {
    let mut high = existing.iter().filter_map(|t| id_num(&t.id)).max().unwrap_or(0);
    for it in &items {
        if let Some(n) = it.id.as_deref().and_then(id_num) {
            high = high.max(n);
        }
    }
    items
        .into_iter()
        .map(|it| {
            let id = match it.id {
                Some(id) if !id.trim().is_empty() => id,
                _ => {
                    high = high.saturating_add(1);
                    format!("t{high}")
                }
            };
            TodoItem {
                id,
                content: it.content,
                status: it.status,
                active_form: it.active_form,
            }
        })
        .collect()
}

// ─── todo_write ───────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct TodoWriteArgs {
    /// The FULL working plan, in display order. This REPLACES the stored list
    /// (it is not a patch): include every item you want to keep. Items that
    /// carry an `id` are updated in place; items without one are created with a
    /// fresh monotonic id.
    pub items: Vec<TodoItemInput>,
}

pub struct TodoWrite {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for TodoWrite {
    type Args = TodoWriteArgs;
    const NAME: &'static str = "todo_write";
    const DESCRIPTION: &'static str = "Write your persistent working plan (one item per optimization DIRECTION, not micro-steps). \
         This REPLACES the whole list: pass every item you want to keep. Items with an `id` update in place; \
         items without one get a fresh id (`t1`, `t2`, …). Mark an item `in_progress` when you start it, \
         `completed` when it lands, `cancelled` ONLY when an eval proved it a dead end, and `deferred` when you \
         park a still-viable direction (e.g. a large structural build) for time — deferred items stay re-openable \
         while budget remains (put the outcome in `content`). The plan is \
         stored per-branch and re-shown to you after every context reset — it is your memory of what you have \
         already tried, so keep it current. Returns the rendered plan with assigned ids.";

    async fn call(&self, args: TodoWriteArgs) -> ToolOutput {
        // Hold the per-path lock across the read-modify-write so two todo_write
        // calls in the same turn can't interleave and lose an update.
        let _guard = self.sandbox.lock_path(TODO_PATH).await.map_err(|e| e.to_string())?;
        let existing = load_todos(&self.sandbox);
        let updated = apply_write(&existing, args.items);
        save_todos(&self.sandbox, &updated)?;
        Ok(render_todos(&updated).into())
    }
}

// ─── todo_read ────────────────────────────────────────────────────────────────

#[derive(Deserialize, JsonSchema)]
pub struct TodoReadArgs {}

pub struct TodoRead {
    pub sandbox: Arc<Sandbox>,
}

impl Tool for TodoRead {
    type Args = TodoReadArgs;
    const NAME: &'static str = "todo_read";
    const DESCRIPTION: &'static str = "Show your persistent working plan (the per-branch todo list). Read-only.";

    async fn call(&self, _args: TodoReadArgs) -> ToolOutput {
        Ok(render_todos(&load_todos(&self.sandbox)).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;

    fn fresh_sandbox() -> Arc<Sandbox> {
        Arc::new(Sandbox::new().expect("sandbox"))
    }

    fn item(id: &str, content: &str, status: Status) -> TodoItem {
        TodoItem {
            id: id.to_string(),
            content: content.to_string(),
            status,
            active_form: None,
        }
    }

    // ─── id assignment ────────────────────────────────────────────────────────

    #[test]
    fn new_items_get_monotonic_ids() {
        let out = apply_write(
            &[],
            vec![
                TodoItemInput {
                    id: None,
                    content: "a".into(),
                    status: Status::Pending,
                    active_form: None,
                },
                TodoItemInput {
                    id: None,
                    content: "b".into(),
                    status: Status::Pending,
                    active_form: None,
                },
            ],
        );
        assert_eq!(out[0].id, "t1");
        assert_eq!(out[1].id, "t2");
    }

    #[test]
    fn new_ids_continue_past_existing_and_incoming_high_water_mark() {
        let existing = vec![item("t5", "old", Status::Completed)];
        // Incoming keeps t2 explicitly and adds one new item: the new id must be
        // t6 (past the on-disk t5), never t3 or a collision with t5.
        let out = apply_write(
            &existing,
            vec![
                TodoItemInput {
                    id: Some("t2".into()),
                    content: "keep".into(),
                    status: Status::Pending,
                    active_form: None,
                },
                TodoItemInput {
                    id: None,
                    content: "fresh".into(),
                    status: Status::Pending,
                    active_form: None,
                },
            ],
        );
        assert_eq!(out[0].id, "t2");
        assert_eq!(out[1].id, "t6");
    }

    // ─── id-preserving update ───────────────────────────────────────────────

    #[test]
    fn item_with_id_updates_in_place() {
        let existing = vec![item("t1", "draft kernel", Status::Pending)];
        let out = apply_write(
            &existing,
            vec![TodoItemInput {
                id: Some("t1".into()),
                content: "draft kernel — landed, 1.0x".into(),
                status: Status::Completed,
                active_form: None,
            }],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "t1");
        assert_eq!(out[0].status, Status::Completed);
        assert_eq!(out[0].content, "draft kernel — landed, 1.0x");
    }

    // ─── status transitions (through the tool + file) ────────────────────────

    #[tokio::test]
    async fn status_transitions_persist_across_writes() {
        let sb = fresh_sandbox();
        let tool = TodoWrite { sandbox: sb.clone() };

        // Create pending.
        tool.call_json(json!({ "items": [{ "content": "async pipeline rewrite" }] }))
            .await
            .unwrap();
        assert_eq!(load_todos(&sb)[0].status, Status::Pending);

        // → in_progress (update in place by id).
        tool.call_json(
            json!({ "items": [{ "id": "t1", "content": "async pipeline rewrite", "status": "in_progress" }] }),
        )
        .await
        .unwrap();
        assert_eq!(load_todos(&sb)[0].status, Status::InProgress);

        // → completed.
        tool.call_json(
            json!({ "items": [{ "id": "t1", "content": "async pipeline rewrite — 1.8x", "status": "completed" }] }),
        )
        .await
        .unwrap();
        let final_ = load_todos(&sb);
        assert_eq!(final_[0].status, Status::Completed);
        assert_eq!(final_[0].content, "async pipeline rewrite — 1.8x");
    }

    #[tokio::test]
    async fn write_returns_rendered_list_with_ids() {
        let sb = fresh_sandbox();
        let tool = TodoWrite { sandbox: sb };
        let out = tool
            .call_json(json!({ "items": [{ "content": "first", "status": "in_progress" }] }))
            .await
            .unwrap();
        let text = out.as_text();
        assert!(text.contains("[~] t1: first"), "{text}");
    }

    // ─── JSON round-trip ──────────────────────────────────────────────────────

    #[test]
    fn json_round_trips() {
        let items = vec![
            TodoItem {
                id: "t1".into(),
                content: "a".into(),
                status: Status::InProgress,
                active_form: Some("doing a".into()),
            },
            item("t2", "b", Status::Cancelled),
        ];
        let json = serde_json::to_string(&items).unwrap();
        let back: Vec<TodoItem> = serde_json::from_str(&json).unwrap();
        assert_eq!(items, back);
        // snake_case on the wire.
        assert!(json.contains("\"in_progress\""), "{json}");
    }

    #[tokio::test]
    async fn file_round_trips_through_sandbox() {
        let sb = fresh_sandbox();
        let tool = TodoWrite { sandbox: sb.clone() };
        tool.call_json(json!({ "items": [
            { "content": "a", "status": "in_progress", "active_form": "doing a" },
            { "content": "b", "status": "pending" }
        ] }))
        .await
        .unwrap();
        let loaded = load_todos(&sb);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].active_form.as_deref(), Some("doing a"));
    }

    // ─── missing → empty ; corrupt → re-init ─────────────────────────────────

    #[test]
    fn missing_file_loads_empty() {
        let sb = fresh_sandbox();
        assert!(load_todos(&sb).is_empty());
        assert!(render_for_injection(&sb).is_none());
    }

    #[tokio::test]
    async fn corrupt_file_treated_as_empty_and_reinitialized() {
        let sb = fresh_sandbox();
        sb.write(TODO_PATH, "{ this is not valid json ]").unwrap();
        // Load degrades to empty rather than erroring.
        assert!(load_todos(&sb).is_empty());
        // A subsequent write silently re-initializes the file cleanly.
        let tool = TodoWrite { sandbox: sb.clone() };
        tool.call_json(json!({ "items": [{ "content": "recovered" }] }))
            .await
            .unwrap();
        let loaded = load_todos(&sb);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "t1");
        assert_eq!(loaded[0].content, "recovered");
    }

    // ─── render cap + open-items-first ordering ──────────────────────────────

    #[test]
    fn render_orders_open_items_first() {
        let items = vec![
            item("t1", "done thing", Status::Completed),
            item("t2", "pending thing", Status::Pending),
            item("t3", "active thing", Status::InProgress),
            item("t4", "dropped thing", Status::Cancelled),
        ];
        let r = render_todos(&items);
        let pos = |needle: &str| r.find(needle).unwrap();
        // in_progress < pending < completed < cancelled in output order.
        assert!(pos("active thing") < pos("pending thing"));
        assert!(pos("pending thing") < pos("done thing"));
        assert!(pos("done thing") < pos("dropped thing"));
    }

    #[test]
    fn render_caps_and_elides_settled_tail() {
        // Two open items plus many completed ones; the completed tail is elided,
        // the open items always survive.
        let mut items = vec![
            item("t1", "OPEN-IN-PROGRESS", Status::InProgress),
            item("t2", "OPEN-PENDING", Status::Pending),
        ];
        for i in 0..400 {
            items.push(item(
                &format!("c{i}"),
                &format!("completed direction number {i} with a longish description"),
                Status::Completed,
            ));
        }
        let r = render_todos(&items);
        assert!(r.len() <= TODO_RENDER_CHARS + 120, "rendered {} chars", r.len());
        assert!(
            r.contains("OPEN-IN-PROGRESS"),
            "open in-progress item must always render"
        );
        assert!(r.contains("OPEN-PENDING"), "open pending item must always render");
        assert!(
            r.contains("elided"),
            "overflow note expected: {}",
            r.get(r.len().saturating_sub(80)..).unwrap_or(&r)
        );
    }

    #[test]
    fn empty_renders_placeholder() {
        assert_eq!(render_todos(&[]), "(no plan items yet)");
    }

    #[test]
    fn tool_names() {
        assert_eq!(TodoWrite::NAME, "todo_write");
        assert_eq!(TodoRead::NAME, "todo_read");
    }
}
