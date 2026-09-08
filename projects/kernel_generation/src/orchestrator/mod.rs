//! Agentic Variation Operator (AVO) orchestrator.
//!
//! Drives a single-lineage evolutionary search: seed a deliberately naive
//! `v0`, then let the agent repeatedly read the current best solution (the
//! `solution/` tree), profile it, edit it, and evaluate it against a read-only
//! trusted evaluator. A candidate that clears the screen is promoted by a loop-run
//! paired re-timing (`confirm_leader`), not by an agent `submit`. The
//! orchestrator owns the canonical git DAG, accounts the budget, and journals
//! enough state to resume after a crash.
//!
//! Layering (within this module):
//! - [`strategy`] — the single-lineage admission rule: a correct candidate that
//!   strictly beats the best geomean by more than the noise margin is promoted;
//!   everything else is rejected ("keep optimizing the same line").
//! - [`search_tree`] — `SearchTree`, the `Send + Sync` handle over the host-side
//!   git DAG that the search-tree tools share.
//! - [`supervisor`] — stagnation / failed-commit interventions.
//! - [`compaction`] — in-place context summarization.
//! - [`run_avo`] — the control loop tying it all together.
//!
//! Shared foundations it builds on live in [`crate::domain`] (schema, journal),
//! [`crate::exec`] (queue, snapshots, sandbox), and [`crate::env`] (hardware).

pub mod compaction;
pub mod prompts;
pub mod search_tree;
pub mod selftest;
pub mod strategy;
pub mod supervisor;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::ai::{CompletionError, ProtocolClient, StopReason, ThinkingEffort, Usage};
use crate::domain::events::TurnEvents;
use crate::domain::run_state::{self, RunMeta};
use crate::domain::types::{self, Evaluation, RunParams, Seeds, now_unix_ms};
use crate::domain::util;
use crate::exec::queue::GpuPool;
use crate::exec::sandbox::Sandbox;
use crate::exec::sandbox_manager::SandboxManager;
use crate::exec::turn_tree::{TurnNodeId, WorkspaceSnapshotId};
use crate::harness::skills::Skill;
use crate::harness::stream::{self, Completion, RetryConfig};
use crate::harness::turn::{self, Nonproductive};
use crate::tool::evaluate::{EvalConfig, Evaluate, SOLUTION_DIR, TRUSTED_DIR, run_evaluator_n, solution_entrypoint};
use crate::tool::restore::Restore;
use crate::tool::run::GpuJob;
use crate::tool::search::SearchView;
use crate::tool::time_left::{TimeLeft, render_time_left};
use crate::tool::{ToolBox, bash, edit, files, fs_tools, markdown, pdf, todo, view};

pub use search_tree::{BestInfo, SearchTree};

/// Per-turn output token cap (mirrors the REPL harness).
const MAX_OUTPUT_TOKENS: usize = 64_000;
/// Reasoning depth requested from the model during the loop.
const THINKING: ThinkingEffort = ThinkingEffort::High;
/// Paired-confirmation samples for a challenger the policy re-times against the
/// incumbent best (drift-cancelling winner's-curse defense; see
/// [`strategy::confirm_improvement`]). A fixed policy constant, not an operator
/// knob: `>= 2` enables the paired re-timing used on every real run.
const CONFIRM_SAMPLES: u64 = 3;
/// Resource slots handed to `SearchPolicy::schedule` at each boundary. Serial
/// expansion for now (one node per boundary); Area 2 S3 raises this to drive up
/// up to `beam_width` concurrent expander agents.
/// Beam-only episode backstop: after this many turns in one expansion episode
/// with no scorable (timed) evaluation, inject a firm nudge to yield a candidate
/// and reset. Set above the supervisor's `STAGNATION_THRESHOLD` (16) so, gated to
/// beam, it never double-fires with a supervisor rung.
const EPISODE_TURN_CAP: u64 = 20;
/// Hard ceiling on an unscored episode even when a structural build holds off the
/// `EPISODE_TURN_CAP` yield nudge (I7): a large structural rewrite gets room to run,
/// but a genuinely wedged episode still yields a candidate rather than spinning.
const EPISODE_TURN_HARD_CAP: u64 = 60;
/// Spin-loop guard (I9): end an idle episode after this many byte-identical idle
/// replies in a row (the agent is repeating itself with nothing left to do).
const MAX_IDENTICAL_IDLE: u64 = 3;
/// Spin-loop guard (I9): if this little wall-clock remains, an idle turn cannot
/// lead to another scored `evaluate`, so end rather than nudge into the wall.
const IDLE_AT_WALL_MARGIN_SECS: u64 = 120;
/// The v0 entrypoint (`solution/solution.py`) for a fresh run.
///
/// A problem is a single self-contained file whose `Reference` is the
/// **correctness oracle only** — never the performance baseline. So the seed is a
/// deliberately incomplete stub, not the reference: it exports a valid `Solution`
/// that fails correctness until the agent implements a real kernel. The agent's
/// FIRST correct kernel becomes the 1.0x performance baseline; every later kernel
/// is scored as speedup over it. `__init__(*args, **kwargs)` swallows any
/// problem-specific ctor args (e.g. a `causal=True` config flag); `forward` raises
/// until implemented, so the `compile` stage passes but `correctness` fails
/// cleanly (no astronomically-slow naive benchmark).
pub const SEED_STUB: &str = r#"""" v0 seed stub — implement a correct custom kernel here.

The Reference in problem.py is the correctness oracle ONLY. Your FIRST correct
Solution becomes the 1.0x performance baseline; every later kernel is scored as
speedup over it. Do NOT import or call the Reference / library ops as your
implementation — write a real custom kernel.
"""
import torch.nn as nn


class Solution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError("implement a correct custom kernel in solution/solution.py")
"#;

/// Everything the orchestrator needs that isn't a long-lived service handle.
pub struct AvoConfig {
    /// `runs/<id>/` — owns repo/, manifest.json, and the journal files.
    pub run_dir: PathBuf,
    /// Human-readable problem name (for the manifest / prompt).
    pub problem_name: String,
    /// Model slug (for the manifest).
    pub model_name: String,
    /// Search policy identifier (`avo` or `beam`).
    pub policy_id: String,
    /// Active beam width for `beam` policy. Ignored by `avo`.
    pub beam_width: usize,
    /// Per-node local expansion cap for `beam` policy (§14.5 widening-debt knob).
    /// `1` (default) = pure tree coverage / round-robin across the frontier;
    /// higher = local widening (a dominant node gets up to this many concurrent
    /// expansions per batch). Ignored by `avo`.
    pub beam_local_cap: usize,
    /// Exploration constant `c` for `--policy ucb` (`0` ⇒ policy default).
    /// Ignored by `avo`/`beam`.
    pub ucb_c: f64,
    /// Python interpreter on the sandbox `PATH` (must have torch).
    pub python: String,
    /// One-shot GPU/host identity probed at startup (`nvidia-smi` /
    /// `system_profiler`), injected into the kickoff and the supervisor context
    /// so both reason about the actual hardware (see [`crate::env::hardware::probe_hardware`]).
    pub hardware: String,
    /// The run's wall-clock budget in seconds — the only hard cap (it bounds
    /// token spend too, since tokens can't accrue without turns and turns burn
    /// the clock). A scripted/self-test run instead ends via [`StopReason::Done`].
    pub max_wall_clock_secs: u64,
    /// Resume an existing run rather than seeding a fresh one.
    pub resume: bool,
    /// Naive `v0` seed source (used only on a fresh run).
    pub seed_src: String,
    /// Per-evaluation execution timeout (seconds).
    pub eval_timeout_secs: u64,
    /// Optional fixed base seed (reproducible A/B runs). `None` generates one.
    pub seed: Option<u64>,
    /// Slug of the supervisor model, or `None` to disable the supervisor
    /// entirely (the default). When set, the supervisor reviews the committed
    /// lineage on a stall and injects concrete next directions; the supervisor
    /// *client* is passed separately to [`run_avo`] and must be the same
    /// provider/type as the main agent.
    pub supervisor_model: Option<String>,
    /// Beam-diversity: exact de-dup in UCB selection (default true) — never expand
    /// two byte-identical kernels in one round. The frontier-collapse fix. Free (no
    /// model). Ignored by `avo`/`beam`.
    pub diversity_dedup: bool,
    /// The serving model's real input-token window, used to decide WHEN to
    /// compact (`context_window - reserve`; see [`compaction::should_compact`]).
    /// Set from `--context-window-tokens`, defaulting to the `context_size`
    /// recorded for this model in its provider's catalog. Sized at or below the
    /// endpoint's actual limit so a request doesn't error before the proactive
    /// trigger fires (the reactive retry-on-overflow path is the backstop).
    pub context_window_tokens: u32,
    /// Repo-local startup skills discovered once by the harness before AVO starts.
    pub skills: Vec<Skill>,
}

/// Build the [`strategy::DiversityConfig`] for the run: exact content-hash dedup,
/// a free flag with no model behind it.
const fn build_diversity(cfg: &AvoConfig) -> strategy::DiversityConfig {
    strategy::DiversityConfig {
        dedup: cfg.diversity_dedup,
    }
}

/// Render the search tree's ranked candidates as a compact, human-readable
/// trajectory for the supervisor's active review: one line per scored candidate
/// with its geomean speedup and whether it is confirmed. This is the evolutionary
/// trajectory `P_t` the supervisor reasons over when proposing fresh directions.
fn render_trajectory(search_tree: &SearchTree) -> String {
    let candidates = search_tree.ranked_candidates();
    if candidates.is_empty() {
        return "(no scored candidates yet)".to_string();
    }
    let mut s = String::from(
        "(geomean = speedup vs the agent's first-correct baseline; only confirmed candidates are real improvements)\n",
    );
    for c in candidates.iter().take(20) {
        let tag = match c.samples {
            Some(n) if n >= 2 => " (confirmed)",
            Some(_) => " (provisional)",
            None => " [unconfirmed]",
        };
        let _ = writeln!(
            s,
            "turn {}: geomean {:.4} ±{:.1}%{tag}",
            c.turn,
            c.geomean,
            c.noise_margin * 100.0
        );
    }
    s
}

/// Per-entry char cap for a captured agent note in the activity log.
const ACTIVITY_NOTE_CHARS: usize = 320;
/// Total char budget for the recent-activity digest handed to the supervisor —
/// only the most recent entries survive if the stall window is longer, keeping
/// the (cheap) supervisor turn cheap.
const RECENT_ACTIVITY_CHARS: usize = 3_500;

/// Compact, first-seen-order `name×count` summary of a turn's tool calls, e.g.
/// `replace×2, evaluate`.
fn summarize_tool_calls<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let mut order: Vec<&str> = Vec::new();
    let mut counts: HashMap<&str, u32> = HashMap::new();
    for n in names {
        let count = counts.entry(n).or_insert(0);
        if *count == 0 {
            order.push(n);
        }
        *count = count.saturating_add(1);
    }
    order
        .iter()
        .filter_map(|n| counts.get(n).map(|c| (*n, *c)))
        .map(|(n, c)| if c > 1 { format!("{n}×{c}") } else { n.to_string() })
        .collect::<Vec<_>>()
        .join(", ")
}

/// One activity-log line: the agent's visible note plus a compact tag naming the
/// tools it used, so the supervisor can see what was tried since the last best.
fn format_activity_entry(note: &str, tools: &str) -> String {
    let tag = if tools.is_empty() {
        String::new()
    } else {
        format!(" [{tools}]")
    };
    if note.is_empty() {
        format!("- (no visible note){tag}")
    } else {
        format!("- {note}{tag}")
    }
}

/// Render the most recent activity entries for the supervisor, newest-biased and
/// capped at [`RECENT_ACTIVITY_CHARS`]. Empty window ⇒ an explicit note.
fn render_recent_activity(entries: &[String]) -> String {
    if entries.is_empty() {
        return "(no activity captured yet this stall window)".to_string();
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut total = 0usize;
    for e in entries.iter().rev() {
        total = total.saturating_add(e.len()).saturating_add(1);
        if total > RECENT_ACTIVITY_CHARS && !kept.is_empty() {
            kept.push("- …(earlier activity elided)");
            break;
        }
        kept.push(e.as_str());
    }
    kept.reverse();
    kept.join("\n")
}

/// Assemble the agent's tool set for a run. `turn_events` is shared with the
/// caller (via `Arc`) so the loop can drain per-turn evaluation events after
/// each turn. The order here is the order the tools are advertised to the model.
#[expect(
    clippy::too_many_arguments,
    reason = "8 independent handles; a param struct just relocates them"
)]
fn build_tools(
    sandbox: &Arc<Sandbox>,
    queue: &GpuPool,
    eval_config: &EvalConfig,
    search_tree: &SearchTree,
    turn_events: &Arc<TurnEvents>,
    run_start: Instant,
    max_wall_secs: u64,
    lineage_node: Option<TurnNodeId>,
) -> ToolBox {
    let mut toolbox = ToolBox::new();
    toolbox.register(Evaluate {
        sandbox: sandbox.clone(),
        queue: queue.clone(),
        config: eval_config.clone(),
        events: turn_events.clone(),
    });
    toolbox.register(GpuJob {
        sandbox: sandbox.clone(),
        queue: queue.clone(),
    });
    toolbox.register(bash::Bash {
        sandbox: sandbox.clone(),
    });
    toolbox.register(files::Read {
        sandbox: sandbox.clone(),
    });
    toolbox.register(files::Write {
        sandbox: sandbox.clone(),
    });
    toolbox.register(edit::Edit {
        sandbox: sandbox.clone(),
    });
    toolbox.register(markdown::MarkdownGetToc {
        sandbox: sandbox.clone(),
    });
    toolbox.register(markdown::MarkdownGrep {
        sandbox: sandbox.clone(),
    });
    toolbox.register(fs_tools::Grep {
        sandbox: sandbox.clone(),
    });
    toolbox.register(fs_tools::Find {
        sandbox: sandbox.clone(),
    });
    toolbox.register(fs_tools::Ls {
        sandbox: sandbox.clone(),
    });
    toolbox.register(pdf::PdfGetNumPages {
        sandbox: sandbox.clone(),
    });
    toolbox.register(view::View {
        sandbox: sandbox.clone(),
    });
    // Check out a confirmed candidate's full verified source (solution/ +
    // artifacts/) into checkout/ on demand — non-destructive, so the agent
    // recovers a verified kernel instead of re-deriving it.
    toolbox.register(Restore {
        search_tree: search_tree.clone(),
        sandbox: sandbox.clone(),
        lineage_node: lineage_node.clone(),
    });
    toolbox.register(SearchView {
        search_tree: search_tree.clone(),
        lineage_node,
    });
    // Wall-clock awareness: the run's only hard cap. Lets the agent pace a
    // long structural build to land SCORED before the wall (see `TimeLeft`);
    // the same remaining-time line is pushed into re-orientation notes.
    toolbox.register(TimeLeft {
        run_start,
        max_wall_secs,
    });
    // Per-branch working plan (todo): persistent memory of tried directions,
    // re-shown after every context reset (see `reinject_todos`). Branch-scoped
    // via `sandbox` — the file lives at `notes/todo.json` in this episode workspace.
    toolbox.register(todo::TodoWrite {
        sandbox: sandbox.clone(),
    });
    toolbox.register(todo::TodoRead {
        sandbox: sandbox.clone(),
    });
    toolbox
}

/// The handles one expansion episode needs — a bundle of **owned** (cheaply
/// clonable) service handles + run config, *not* a mutable state bag. Owned so an
/// episode can run as a `spawn_local` task with a `'static` future; the loop clones
/// this into each spawned expander. Each episode `spawn`s its **own** sandbox (from
/// the `manager`) + its own tools + `TurnEvents` (Area 2 S3), so concurrent
/// expanders never collide on the filesystem. The owned run state (transcript,
/// meta, activity) is threaded through [`run_episode`] by value.
struct EpisodeCtx<C: ProtocolClient> {
    client: Arc<C>,
    supervisor_client: Option<Arc<C>>,
    manager: SandboxManager,
    eval_config: EvalConfig,
    search_tree: SearchTree,
    queue: GpuPool,
    cfg: Arc<AvoConfig>,
    params: Arc<RunParams>,
    retry: RetryConfig,
    run_start: Instant,
}

// Manual `Clone` (a `#[derive]` would demand `C: Clone`, but every field is an
// `Arc`/handle that clones by refcount bump regardless of `C`).
impl<C: ProtocolClient> Clone for EpisodeCtx<C> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            supervisor_client: self.supervisor_client.clone(),
            manager: self.manager.clone(),
            eval_config: self.eval_config.clone(),
            search_tree: self.search_tree.clone(),
            queue: self.queue.clone(),
            cfg: self.cfg.clone(),
            params: self.params.clone(),
            retry: self.retry.clone(),
            run_start: self.run_start,
        }
    }
}

/// Why an expansion episode ended — the boundary transition [`run_avo`] acts on.
#[derive(Clone)]
enum EpisodeStop {
    /// The episode produced a `Timed` node (a scorable boundary). The loop
    /// `observe`s it (may flag a challenger to confirm) and re-schedules.
    Scorable { node: TurnNodeId, eval: Evaluation },
    /// The episode ended at a boundary with nothing to `observe`: either the beam
    /// episode backstop fired (a firm nudge was already injected — the normal,
    /// continuable case), or a scorable turn failed to persist its node. The
    /// latter drops the observation (the policy never sees that score); it is a
    /// rare infra failure, harmless at `p = 1`, but if the two cases ever diverge
    /// (e.g. S3 wants to retry a dropped observation) split this into two variants.
    /// Either way the loop just re-schedules.
    Boundary,
    /// The model declared the whole run complete (only the scripted self-test
    /// does this). The loop stops.
    Done,
    /// The wall-clock budget cap was reached mid-episode. The loop stops.
    Budget,
    /// Streaming failed after retries; the run gives up (gracefully — the loop
    /// records meta and stops, it does not propagate an error).
    CompletionError(String),
}

/// Per-turn accounting increment emitted by an expander episode over the event
/// channel. The loop owns [`RunMeta`] and folds these in; the episode no longer
/// calls `save_meta` or `write_manifest`. The loop persists after each delta,
/// giving the same per-turn crash-safety as before.
struct LedgerDelta {
    usage: Usage,
    evaluations: u64,
    candidates_archived: u64,
    /// Derived from `usage` (sum of input + `cache_r` + `cache_w` + output), stored
    /// ready to assign to `RunMeta::last_context_tokens`.
    last_context_tokens: u32,
}

/// Events an expander task sends over its unbounded channel.
enum EpisodeEvent {
    /// One model turn completed — fold into `RunMeta` and persist.
    Turn(LedgerDelta),
    /// Episode ended (terminal). Always sent exactly once, last.
    Done(EpisodeDone),
}

/// Terminal payload from an expander episode.
struct EpisodeDone {
    /// Activity entries accumulated during this episode (appended to the loop's
    /// global log on Done so subsequent episodes' supervisors see them).
    new_activity: Vec<String>,
    stop: EpisodeStop,
}

/// Everything an episode needs at spawn time (snapshot of loop state + the
/// starting transcript). The episode maintains its own local accounting
/// (`turns_since_best_improved`, `last_context_tokens`) from these seeds and
/// emits [`LedgerDelta`]s for the loop to fold; it never touches `RunMeta`
/// directly.
struct EpisodeInit<M> {
    transcript: Vec<M>,
    /// The tree node the episode expands from. Used for:
    /// (a) sandbox spawn — materialize its workspace snapshot;
    /// (b) explicit parent for node appends — so concurrent episodes parent
    ///     their turns correctly without racing on the shared playhead.
    start_node_id: Option<TurnNodeId>,
    /// `meta.turns` at spawn — the episode uses this as the starting turn index
    /// for node-append calls and increments it locally.
    turns: u64,
    turns_since_best_improved: u64,
    last_context_tokens: u32,
    evaluations: u64,
    /// `activity[activity_at_commit..]` snapshot at spawn time — what the episode's
    /// supervisor sees as "work since the last accepted improvement."
    recent_activity: Vec<String>,
}

/// Seconds past the wall-clock budget at which the watchdog gives up on the run.
///
/// Sized to exceed the longest a single turn can legitimately block, or a healthy run
/// finishing near the wall would be killed: acquiring a device lease waits up to 1800s
/// and a `gpu_job` up to 600s, so 2400s is the floor.
const WALL_CLOCK_WATCHDOG_GRACE_SECS: u64 = 3600;

/// Hard backstop for the wall-clock budget.
///
/// The episode loop tests the budget at the top of each turn, which cannot fire while a
/// turn is blocked — so any operation that outlives its own timeout takes the run's only
/// hard cap with it. This task shares no state and does nothing but exit, so it holds
/// regardless of where the run is stuck.
///
/// Exiting abandons the convenience export, not the work: the snapshot store and the
/// manifest are written as the run proceeds, so the best node is recoverable from
/// `runs/<id>/repo` via `refs/kg/nodes/<best_node_id>`.
#[expect(
    clippy::exit,
    reason = "a watchdog exists to end a run that cannot end itself; an error return would go \
              to the blocked caller that never resumes"
)]
fn spawn_wall_clock_watchdog(max_wall_clock_secs: u64) {
    let grace = WALL_CLOCK_WATCHDOG_GRACE_SECS;
    let deadline = max_wall_clock_secs.saturating_add(grace);
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(deadline)).await;
        eprintln!(
            "kernelguy: wall-clock watchdog firing — {grace}s past the {max_wall_clock_secs}s budget \
             with the run still not finished, so a turn is blocked and the budget check cannot run. \
             Exiting. Recover the best candidate from the run's snapshot store: \
             `git --git-dir=runs/<run_id>/repo/.git archive refs/kg/nodes/<best_node_id>` \
             (the id is in manifest.json)."
        );
        std::process::exit(124);
    });
}

/// Run the AVO loop to completion (budget exhausted, stagnation, or fatal
/// completion error). Returns `Err` only for setup failures that prevent
/// the loop from starting.
///
/// # Errors
///
/// Returns `Err` when the run cannot be set up or its state cannot be persisted:
/// seeding/restoring the baseline solution fails, the frontier cannot be rebuilt,
/// writing the journal (`save_meta`) or the manifest fails, or the expander tasks
/// close their event channel without reporting a terminal stop. A model/provider
/// failure inside the loop is NOT an error here — it ends the run gracefully.
#[expect(
    clippy::too_many_lines,
    reason = "one linear sequence; splitting trades length for state plumbing"
)]
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
pub async fn run_avo<C: ProtocolClient + 'static>(
    client: C,
    supervisor_client: Option<C>,
    sandbox: Arc<Sandbox>,
    search_tree: SearchTree,
    queue: GpuPool,
    cfg: AvoConfig,
) -> Result<(), String> {
    // Shared behind an `Arc` so it can be cloned into each spawned expander's
    // `EpisodeCtx` (Area 2 S2). Deref makes `cfg.field` / `&cfg` read unchanged.
    let cfg = Arc::new(cfg);
    // Reproducibility seeds: on resume, restore from the journal so the input
    // sequence continues; otherwise generate a fresh one. (Loaded once here and
    // reused by `load_or_init_session` so the eval config and the journaled
    // `RunMeta` agree on the seed.)
    let restored_meta = if cfg.resume {
        run_state::load_meta(&cfg.run_dir)
    } else {
        None
    };
    let seeds = restored_meta.as_ref().map_or_else(
        || cfg.seed.map_or_else(Seeds::generate, Seeds::from_base),
        |m| m.seeds.clone(),
    );

    let eval_config = EvalConfig::new(cfg.python.clone(), cfg.eval_timeout_secs, seeds.evaluator_inputs);
    // Wall clock covers the whole run, including the v0 seed evaluation below,
    // so it is always >= queue_busy_secs (which also counts that seed eval).
    let run_start = Instant::now();
    spawn_wall_clock_watchdog(cfg.max_wall_clock_secs);

    // Static run config snapshot for the manifest (so a result records the exact
    // regime it was obtained under). V1 supports the AVO policy only: a tree
    // store with accepted-lineage-only context/reporting.
    let params = Arc::new(run_params(&cfg));

    // ── init: seed the stub (or restore) into a loop-side scratch sandbox ────
    let best = init_baseline(&sandbox, &search_tree, &cfg)?;
    match &best {
        Some(b) => println!("baseline ready: geomean {:?} (node {})", b.geomean_speedup, b.node_id),
        None => println!("no baseline yet — the agent's first correct candidate will set the 1.0x baseline"),
    }
    search_tree.rebuild_frontier(cfg.beam_width.max(1))?;

    // ── transcript + meta (fresh or resumed) ─────────────────────────────────
    let (model, mut meta) = load_or_init_session(&client, &cfg, &search_tree, best.as_ref(), seeds, restored_meta);
    // The model-visible message list sent to the provider each turn. Compaction
    // and branch-switching replace it wholesale, so it is not append-only.
    let mut transcript = model;
    if !cfg.resume
        && let Err(e) = search_tree.append_typed_turn_node(
            0,
            cfg.beam_width.max(1),
            transcript.clone(),
            sandbox.workspace(),
            false,
            None,
            None,
        )
    {
        eprintln!("[history] failed to seed provider context node: {e}");
    }

    // Seed the initial best-vs-budget point (v0). On resume the curve is
    // restored from the journal, so only sample it when empty.
    if meta.curve.is_empty()
        && let Some(b) = search_tree.current_best()
    {
        meta.ledger.wall_clock_secs = run_start.elapsed().as_secs();
        meta.ledger.queue_busy_secs = queue.busy_secs();
        meta.curve.push(curve_point(&meta, b.geomean_speedup.unwrap_or(0.0)));
    }

    // ── control loop ──────────────────────────────────────────────────────────
    // One continuous, unbounded run: the agent reads the current best, profiles,
    // edits, and evaluates; strict improvements are confirmed onto the spine by
    // the loop's paired re-timing, until a budget cap, a stagnation stall, or a
    // fatal completion error.
    //
    // The search policy drives node selection at scorable boundaries (§6/§7). For
    // AVO (width-1) it always expands the current leaf — a no-op fork that keeps
    // the transcript + workspace linear. Every outer loop iteration is a boundary
    // now (schedule → run one episode → return at a boundary), so there is no
    // longer an `at_boundary` flag to track.
    let mut policy = strategy::make_policy(
        &cfg.policy_id,
        search_tree.clone(),
        cfg.beam_width,
        cfg.beam_local_cap,
        cfg.ucb_c,
        build_diversity(&cfg),
    );
    // In-memory activity log: one compact entry per turn (the agent's visible note
    // + tools). `activity_at_commit` marks where the current stall window begins,
    // so the supervisor is shown only what the agent has tried since its last
    // accepted improvement. Not journaled — on resume it starts empty.
    let mut activity: Vec<String> = Vec::new();
    let mut activity_at_commit: usize = 0;

    // Owned handles every episode shares (cloned into each spawned expander so its
    // future is `'static`). Each episode `spawn`s its own sandbox + tools +
    // `TurnEvents` from `manager`; the loop keeps its own scratch `sandbox` for
    // loop-side serial work (init / confirm).
    let ctx = EpisodeCtx {
        client: Arc::new(client),
        supervisor_client: supervisor_client.map(Arc::new),
        manager: search_tree.manager().clone(),
        eval_config: eval_config.clone(),
        search_tree: search_tree.clone(),
        queue: queue.clone(),
        cfg: cfg.clone(),
        params: params.clone(),
        retry: RetryConfig::default(),
        run_start,
    };

    // Single-threaded concurrency substrate: episodes run as `spawn_local` tasks on
    // a `LocalSet` (the `ProtocolClient` stream is deliberately non-`Send`, and the
    // workload is I/O-bound — overlapped awaits, not cores, are the win). The loop
    // stays the single owner of policy/ledger/manifest state; expanders rejoin over
    // an `mpsc`. For S2 exactly one expander runs at a time and the loop awaits its
    // completion — the spawn+channel path S3's `p>1` reuses (with a `select!` over
    // several expanders + async job events; here a blocking `recv` is the one-source
    // equivalent, deliberately not a `try_recv`).
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            loop {
                meta.ledger.wall_clock_secs = run_start.elapsed().as_secs();
                meta.ledger.queue_busy_secs = queue.busy_secs();
                // Wall-clock is the run's only hard cap (a scripted run ends on Done).
                if meta.ledger.wall_clock_secs >= cfg.max_wall_clock_secs {
                    println!("\nstopping: budget reached ({})", budget_summary(&meta));
                    break;
                }

                // Boundary-gated selection (§6/§7 + §10 redesign): every outer iteration
                // is a scorable boundary, so the policy decides what to run next each time.
                // It may ask to *confirm* a challenger (paired re-timing of the challenger
                // vs the incumbent best — the winner's-curse/drift defense, run here in the
                // loop, not in a tool) or to *expand* a node (run an agent episode). For AVO
                // the expand target is the latest node, so the fork is a no-op and the line
                // stays linear.
                let actions = policy.schedule(cfg.beam_width.max(1));
                let reevals: Vec<TurnNodeId> = actions
                    .iter()
                    .filter_map(|a| match a {
                        strategy::Action::Reevaluate(id) => Some(id.clone()),
                        strategy::Action::Expand(_) => None,
                    })
                    .collect();
                if let [challenger, incumbent, ..] = reevals.as_slice() {
                    // Confirm without an agent turn: re-time, decide, record. On a clear
                    // win this crowns a new best (a curve point + a best-improved reset).
                    match confirm_leader(
                        &sandbox,
                        &queue,
                        &eval_config,
                        &search_tree,
                        CONFIRM_SAMPLES,
                        challenger,
                        incumbent,
                    )
                    .await
                    {
                        Ok(Some(g)) => {
                            println!("[confirm] new best confirmed: geomean {g:.4}");
                            meta.ledger.commits = meta.ledger.commits.saturating_add(1);
                            meta.turns_since_best_improved = 0;
                            activity_at_commit = activity.len();
                            meta.ledger.wall_clock_secs = run_start.elapsed().as_secs();
                            meta.ledger.queue_busy_secs = queue.busy_secs();
                            meta.curve.push(curve_point(&meta, g));
                        }
                        Ok(None) => println!("[confirm] challenger did not clear the floor; best unchanged"),
                        Err(e) => eprintln!("[confirm] re-timing failed ({e}); best unchanged"),
                    }
                    run_state::save_meta(&cfg.run_dir, &meta)?;
                    write_manifest(&cfg, &search_tree, &meta, &params)?;
                    // The next schedule picks the expand target.
                    continue;
                }

                // Collect all Expand targets the policy wants to run concurrently.
                // Empty → one expander from the current leaf (AVO / empty frontier).
                let expand_targets: Vec<TurnNodeId> = actions
                    .iter()
                    .filter_map(|a| match a {
                        strategy::Action::Expand(id) => Some(id.clone()),
                        strategy::Action::Reevaluate(_) => None,
                    })
                    .collect();

                // Build per-expander (transcript, start_node_id) pairs.
                // For each target, derive the branch transcript from the tree.
                // Single-expander AVO and the empty-frontier case use the loop's
                // current transcript (taken by value so it isn't cloned needlessly).
                let expanders: Vec<(Vec<C::Message>, Option<TurnNodeId>)> = if expand_targets.is_empty() {
                    // No Expand action: run from current leaf (empty frontier /
                    // fresh seed-only run). Checkout the leaf so the playhead is set.
                    if let Some(leaf) = search_tree.current_node_id()
                        && let Err(e) = search_tree.checkout_node(&leaf)
                    {
                        eprintln!("[history] failed to checkout leaf {leaf}: {e}");
                    }
                    vec![(std::mem::take(&mut transcript), search_tree.current_node_id())]
                } else {
                    expand_targets
                        .iter()
                        .enumerate()
                        .map(|(i, node_id)| {
                            let t = match search_tree.context::<C::Message>(node_id) {
                                Ok(ctx) if !ctx.is_empty() => ctx,
                                _ => {
                                    if i == 0 {
                                        std::mem::take(&mut transcript)
                                    } else {
                                        transcript.clone()
                                    }
                                }
                            };
                            (t, Some(node_id.clone()))
                        })
                        .collect()
                };

                // Spawn p expanders on a shared event channel. The channel merges Turn
                // deltas and Done events from all expanders; the loop drains until it
                // has seen exactly p Done events.
                let p = expanders.len();
                let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<EpisodeEvent>();
                let recent = activity
                    .get(activity_at_commit.min(activity.len())..)
                    .unwrap_or_default()
                    .to_vec();
                for (transcript_i, start_node_id) in expanders {
                    let tx = event_tx.clone();
                    let task_ctx = ctx.clone();
                    let init = EpisodeInit {
                        transcript: transcript_i,
                        start_node_id,
                        turns: meta.turns,
                        turns_since_best_improved: meta.turns_since_best_improved,
                        last_context_tokens: meta.last_context_tokens,
                        evaluations: meta.ledger.evaluations,
                        recent_activity: recent.clone(),
                    };
                    tokio::task::spawn_local(async move {
                        run_episode(&task_ctx, init, tx).await;
                    });
                }
                // Drop the loop's sender clone so the channel closes when the last
                // expander task drops its clone.
                drop(event_tx);

                // Drain events until all p expanders have sent Done. Turn deltas are
                // folded into meta and persisted per-turn (crash-safety unchanged).
                // For each Scorable Done, observe immediately so the policy's state
                // is updated in arrival order. Terminal stops (Done/Budget/Error)
                // are collected; after the last expander finishes, the most-terminal
                // stop drives the outer loop decision.
                let mut done_count: usize = 0;
                let mut terminal: Option<EpisodeStop> = None;
                let stop = loop {
                    match event_rx.recv().await {
                        None => {
                            return Err("expander tasks closed channel without sending Done".to_string());
                        }
                        Some(EpisodeEvent::Turn(delta)) => {
                            meta.ledger.wall_clock_secs = run_start.elapsed().as_secs();
                            meta.ledger.queue_busy_secs = queue.busy_secs();
                            apply_delta(&mut meta, &delta);
                            run_state::save_meta(&cfg.run_dir, &meta)?;
                            write_manifest(&cfg, &search_tree, &meta, &params)?;
                        }
                        Some(EpisodeEvent::Done(done)) => {
                            activity.extend(done.new_activity);
                            match done.stop {
                                EpisodeStop::Scorable { ref node, ref eval } => {
                                    // The FIRST correct candidate establishes the 1.0x
                                    // performance baseline — crown it as the initial
                                    // confirmed best (this replaces the old
                                    // reference-seed crowning; `seed_root` no longer
                                    // scores) and seed the best-vs-budget curve. Do it
                                    // before `observe` so the policy doesn't also flag
                                    // it as a challenger to confirm.
                                    if search_tree.current_best().is_none()
                                        && let Evaluation::Timed { geomean_speedup, .. } = eval
                                    {
                                        match search_tree.record_confirmation(node, *geomean_speedup, 1) {
                                            Ok(()) => {
                                                meta.ledger.wall_clock_secs = run_start.elapsed().as_secs();
                                                meta.ledger.queue_busy_secs = queue.busy_secs();
                                                meta.curve.push(curve_point(&meta, *geomean_speedup));
                                            }
                                            Err(e) => {
                                                eprintln!("[history] failed to crown first candidate {node}: {e}");
                                            }
                                        }
                                    }
                                    policy.observe(node, eval);
                                }
                                ref s @ (EpisodeStop::Done | EpisodeStop::Budget | EpisodeStop::CompletionError(_)) => {
                                    if terminal.is_none() {
                                        terminal = Some(s.clone());
                                    }
                                }
                                EpisodeStop::Boundary => {}
                            }
                            done_count = done_count.saturating_add(1);
                            if done_count >= p {
                                break terminal.take().unwrap_or(EpisodeStop::Boundary);
                            }
                        }
                    }
                };
                // Re-derive transcript from the tree for the next episode.
                if let Some(node_id) = search_tree.current_node_id() {
                    match search_tree.context::<C::Message>(&node_id) {
                        Ok(ctx) if !ctx.is_empty() => transcript = ctx,
                        Ok(_) => {}
                        Err(e) => eprintln!("[history] failed to re-derive transcript: {e}"),
                    }
                }
                match stop {
                    // observe() was already called for each Scorable Done in the drain.
                    EpisodeStop::Scorable { .. } | EpisodeStop::Boundary => {}
                    EpisodeStop::Done => break,
                    EpisodeStop::Budget => {
                        println!("\nstopping: budget reached ({})", budget_summary(&meta));
                        break;
                    }
                    EpisodeStop::CompletionError(e) => {
                        eprintln!("\ncompletion failed (giving up this run): {e}");
                        run_state::save_meta(&cfg.run_dir, &meta)?;
                        break;
                    }
                }
            }

            // ── exit: export the best solution + final manifest ──────────────────────
            meta.ledger.wall_clock_secs = run_start.elapsed().as_secs();
            meta.ledger.queue_busy_secs = queue.busy_secs();
            if let Some(b) = search_tree.current_best() {
                if let Ok(files) = search_tree.read_solution_at(&b.node_id) {
                    let _ = util::write_file_tree(&cfg.run_dir.join("best_solution"), &files);
                }
                println!(
                    "done. best = geomean {:?}x vs first-correct baseline (node {}); exported {}/best_solution/",
                    b.geomean_speedup,
                    b.node_id,
                    cfg.run_dir.display()
                );
            }
            write_manifest(&cfg, &search_tree, &meta, &params)?;
            Ok::<(), String>(())
        })
        .await
}

/// Run one expansion episode: stream model turns until a scorable boundary, the
/// beam backstop, a model-declared stop, the wall-clock cap, or a stream error.
/// Emits [`EpisodeEvent::Turn`] deltas per turn (the loop folds them into
/// [`RunMeta`] and persists) and a terminal [`EpisodeEvent::Done`]. Never calls
/// `save_meta` or `write_manifest` itself. The loop owns all ledger state; the
/// episode owns only the per-episode context snapshot it received in [`EpisodeInit`].
#[expect(
    clippy::too_many_lines,
    reason = "one linear sequence; splitting trades length for state plumbing"
)]
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
async fn run_episode<C: ProtocolClient>(
    ctx: &EpisodeCtx<C>,
    init: EpisodeInit<C::Message>,
    event_tx: tokio::sync::mpsc::UnboundedSender<EpisodeEvent>,
) {
    // Unpack init into episode-local state.
    let mut transcript = init.transcript;
    let mut turns_since_best_improved = init.turns_since_best_improved;
    let mut last_context_tokens = init.last_context_tokens;
    let mut local_evaluations = init.evaluations;
    let mut activity: Vec<String> = Vec::new();
    let recent_activity = init.recent_activity; // pre-episode "since last commit" slice

    // Rebind ctx handles as borrows (the loop owns/clones the Arcs).
    let cfg: &AvoConfig = &ctx.cfg;
    let client: &C = &ctx.client;
    let supervisor_client: Option<&C> = ctx.supervisor_client.as_deref();
    let search_tree: &SearchTree = &ctx.search_tree;
    let queue: &GpuPool = &ctx.queue;
    let retry: &RetryConfig = &ctx.retry;
    let run_start = ctx.run_start;

    // Spawn this episode's OWN isolated sandbox — a fresh worktree materialized at
    // the start node's snapshot — with its own tools + `TurnEvents`. Concurrent
    // episodes (`p>1`) can't collide on the filesystem; the tempdir is cleaned on
    // `Drop` at episode end (per-turn appends snapshot state first).
    let base = init
        .start_node_id
        .as_ref()
        .and_then(|n| search_tree.nearest_workspace_commit(Some(n.as_str())).ok().flatten())
        .map(WorkspaceSnapshotId::new);
    let sandbox = match ctx.manager.spawn(base.as_ref()) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            let _ = event_tx.send(EpisodeEvent::Done(EpisodeDone {
                new_activity: vec![],
                stop: EpisodeStop::CompletionError(format!("episode sandbox spawn failed: {e}")),
            }));
            return;
        }
    };
    let turn_events = Arc::new(TurnEvents::default());
    // Descriptors are built once inside `register`, so advertising the toolset on
    // each turn is a borrow (`toolbox.descriptors()`), not a per-request projection.
    let toolbox = build_tools(
        &sandbox,
        queue,
        &ctx.eval_config,
        search_tree,
        &turn_events,
        run_start,
        cfg.max_wall_clock_secs,
        init.start_node_id.clone(),
    );
    let sandbox: &Sandbox = &sandbox;
    let turn_events: &TurnEvents = &turn_events;
    // Episode-local: resets at every boundary, so it never carries across episodes.
    let mut turns_since_scorable: u64 = 0;
    // Spin-loop guard state (I9): consecutive idle (no-tool) turns and the last idle
    // reply text, to detect a stuck "keep replying the same thing" loop.
    let mut consecutive_idle: u64 = 0;
    let mut last_idle_text: Option<String> = None;
    // Turn index for node-append calls: starts at the loop's `meta.turns` at spawn
    // and increments locally each turn (mirrors how the loop's `meta.turns` will
    // after folding each delta).
    let mut episode_turns = init.turns;
    // Explicit parent for append calls: each episode advances its own local chain,
    // so concurrent episodes (`p>1`) parent their nodes correctly without racing on
    // the shared `current_leaf_node_id` playhead.
    let mut parent_node: Option<TurnNodeId> = init.start_node_id.clone();
    // Ground-truth anchor (item 2a): re-orient the agent to the authoritative
    // current best from the git lineage at the START of every episode — decoupled
    // from compaction, so a promotion (which always precedes a fresh episode) and
    // a long run without any compaction still can't make the agent lose track of
    // its own committed progress. Folded in safely; empty until a first confirmed
    // best exists.
    {
        // Live wall-clock awareness (item 2): the same remaining-time line the
        // `time_left` tool renders, pushed in at episode start so the agent paces
        // its structural builds without having to ask. Always present (even before
        // a first confirmed best), then the anchor when one exists.
        let time_left = render_time_left(run_start.elapsed().as_secs(), cfg.max_wall_clock_secs);
        let anchor = ground_truth_anchor(
            search_tree,
            parent_node.as_ref(),
            on_disk_solution_sha(sandbox).as_deref(),
        );
        let note = if anchor.trim().is_empty() {
            time_left
        } else {
            format!("{time_left}\n\n{anchor}")
        };
        client.inject_user_note(&mut transcript, note);
    }
    // Working plan (todo): re-inject the branch's persistent plan (ephemeral,
    // fresh from notes/todo.json) so the agent opens each episode reminded of the
    // directions it has already tried. No-op until the branch has a plan.
    reinject_todos(client, &mut transcript, sandbox);
    // Doc-survey index: re-inject the branch's distilled design memory
    // (notes/DOC_INDEX.md — a one-line note per doc, which fit this problem and the
    // fast-path it is driving toward) so the flagged direction survives a reset
    // rather than being re-surveyed or lost to local tuning. No-op until it has content.
    reinject_doc_index(client, &mut transcript, sandbox);
    // Solved-gotchas ledger (item 3): re-inject the branch's durable hard-won
    // fixes (notes/SOLVED_GOTCHAS.md) so a build flag / error→fix recipe survives
    // a context reset and is never re-derived. No-op until the ledger has content.
    reinject_gotchas(client, &mut transcript, sandbox);
    // Build-invariants note (item C1): re-inject the SMALL, never-truncated set of
    // load-bearing invariants (exact build/arch flags, hard constraints) in FULL,
    // so a flag that scrolled off the truncated gotchas ledger is never re-derived.
    // Injected last, giving the never-re-derive build facts the highest recency salience.
    reinject_invariants(client, &mut transcript, sandbox);
    // Reactive retry-once guard (item 5): true only immediately after a
    // compact-and-retry, so we never recover from an overflow twice in a row.
    let mut recovered_last_turn = false;
    let stop = loop {
        // Wall-clock is the run's only hard cap; check it at the top of every turn.
        // The episode reads the clock directly (no meta access).
        if run_start.elapsed().as_secs() >= cfg.max_wall_clock_secs {
            break EpisodeStop::Budget;
        }

        // Supervisor: on a stall, review the trajectory (or fall back to a static
        // nudge) and inject its directions.
        let progress = supervisor::Progress {
            turns_since_best_improved,
            evaluations: local_evaluations,
        };
        if let Some(rung) = supervisor::due(progress) {
            // Build the supervisor slice: pre-episode history + this episode so far.
            let super_slice: Vec<String> = recent_activity.iter().chain(activity.iter()).cloned().collect();
            let directions = supervisor_directions(
                rung,
                supervisor_client,
                search_tree,
                progress,
                &super_slice,
                &cfg.hardware,
            )
            .await;
            println!("[supervisor] intervention injected ({rung:?})");
            // Re-anchor to ground truth at every stall rung (item 2b), prefixed
            // with the live remaining-time line so a late-run stall triggers an
            // explicit keep-or-cut (item 2). Inject SAFELY: fold into a trailing
            // user turn rather than appending a second consecutive one (Anthropic
            // rejects that with a fatal 400).
            let time_left = render_time_left(run_start.elapsed().as_secs(), cfg.max_wall_clock_secs);
            let anchor = ground_truth_anchor(
                search_tree,
                parent_node.as_ref(),
                on_disk_solution_sha(sandbox).as_deref(),
            );
            let note = if anchor.trim().is_empty() {
                format!("{time_left}\n\n{directions}")
            } else {
                format!("{time_left}\n\n{anchor}\n\n{directions}")
            };
            client.inject_user_note(&mut transcript, note);
        }

        // Compaction: summarize the OLDER prefix in place as context nears the
        // REAL window (item 1). Trigger = last_context > window - reserve, where
        // reserve leaves room for the next output + one turn of growth.
        if compaction::should_compact(
            last_context_tokens,
            cfg.context_window_tokens,
            compaction::reserve_tokens(u32::try_from(MAX_OUTPUT_TOKENS).unwrap_or(u32::MAX)),
        ) {
            println!(
                "\n[compaction] summarizing context (ctx {last_context_tokens} tok near window {})...",
                cfg.context_window_tokens
            );
            match compact_context(
                client,
                &transcript,
                search_tree,
                parent_node.as_ref(),
                sandbox,
                cfg.context_window_tokens,
                run_start.elapsed().as_secs(),
                cfg.max_wall_clock_secs,
            )
            .await
            {
                Ok(compacted) => {
                    transcript = compacted;
                    // Persist the reset AND continue the lineage from it, so a
                    // resume rebuilds the compacted context (not the full history).
                    match search_tree.append_context_reset_node(episode_turns, transcript.clone()) {
                        Ok(id) => {
                            parent_node = Some(TurnNodeId::new(id));
                            episode_turns = episode_turns.saturating_add(1);
                        }
                        Err(e) => eprintln!(
                            "[compaction] failed to persist context reset ({e}); continuing with in-memory compacted context"
                        ),
                    }
                    // Compaction may have summarized away the ephemeral plan note;
                    // re-add it fresh AFTER the reset node was persisted, so it stays
                    // out of stored node messages and never accumulates in the tree.
                    reinject_todos(client, &mut transcript, sandbox);
                    // last_context_tokens resets; the delta from the next real turn will persist it.
                }
                Err(e) => eprintln!("[compaction] failed ({e}); continuing uncompacted"),
            }
        }

        // Stream one turn. A hard kill (SIGINT/SIGTERM) drops the process here;
        // resume replays from the last persisted leaf, so we don't trap signals.
        let completion = match stream::stream_turn_with_retry(
            client,
            &transcript,
            toolbox.descriptors(),
            MAX_OUTPUT_TOKENS,
            THINKING,
            retry,
        )
        .await
        {
            Ok(t) => {
                recovered_last_turn = false;
                t
            }
            // Reactive recovery (item 5): a mid-episode context overflow (the
            // proactive trigger under-fired — e.g. the real window was smaller
            // than assumed) shouldn't kill the run. Compact once and retry the
            // SAME turn. Guarded so we never spin if compaction didn't free enough.
            Err(e) if e.is_context_overflow() && !recovered_last_turn => {
                println!("[compaction] context overflow ({e}); compacting and retrying the turn once");
                match compact_context(
                    client,
                    &transcript,
                    search_tree,
                    parent_node.as_ref(),
                    sandbox,
                    cfg.context_window_tokens,
                    run_start.elapsed().as_secs(),
                    cfg.max_wall_clock_secs,
                )
                .await
                {
                    Ok(compacted) => {
                        transcript = compacted;
                        match search_tree.append_context_reset_node(episode_turns, transcript.clone()) {
                            Ok(id) => {
                                parent_node = Some(TurnNodeId::new(id));
                                episode_turns = episode_turns.saturating_add(1);
                            }
                            Err(pe) => eprintln!(
                                "[compaction] failed to persist reactive context reset ({pe}); continuing in-memory"
                            ),
                        }
                        // Re-add the ephemeral plan note the summary may have dropped
                        // (after the reset node persisted, so it never accumulates).
                        reinject_todos(client, &mut transcript, sandbox);
                        recovered_last_turn = true;
                        continue;
                    }
                    Err(ce) => {
                        eprintln!("[compaction] reactive compaction failed ({ce}); ending episode");
                        break EpisodeStop::CompletionError(e.to_string());
                    }
                }
            }
            Err(e) => break EpisodeStop::CompletionError(e.to_string()),
        };

        let Completion {
            message,
            tool_calls,
            reason,
            usage,
            text,
        } = completion;
        transcript.push(message);

        // Update episode-local accounting (mirrors record_turn; the loop will fold
        // a LedgerDelta with the same numbers into RunMeta).
        let last_ctx_tokens = usage
            .input_tokens
            .saturating_add(usage.cache_read_tokens)
            .saturating_add(usage.cache_write_tokens)
            .saturating_add(usage.output_tokens);
        last_context_tokens = last_ctx_tokens;
        turns_since_best_improved = turns_since_best_improved.saturating_add(1);

        // The client signalled the whole run is complete — stop (only the
        // scripted self-test does this; a real run ends on a budget cap or
        // stagnation stall, never by the model declaring itself done).
        if matches!(reason, StopReason::Done) {
            // Emit a zero-delta turn so the loop increments turns and persists.
            let _ = event_tx.send(EpisodeEvent::Turn(LedgerDelta {
                usage,
                evaluations: 0,
                candidates_archived: 0,
                last_context_tokens: last_ctx_tokens,
            }));
            break EpisodeStop::Done;
        }

        // No tool calls: nudge the agent, journal the nudge, and run the next turn
        // in the same episode. Classification lives in the harness; choosing the
        // prompt is policy, so it stays here.
        if let Some(kind) = turn::classify_nonproductive(&reason, !tool_calls.is_empty()) {
            // Spin-loop guard (I9): an idle turn at/near the wall, or the same idle
            // reply repeated, means the episode is spinning with nothing left to
            // score — end it instead of nudging into the wall forever.
            if matches!(kind, Nonproductive::Idle) {
                let remaining = cfg.max_wall_clock_secs.saturating_sub(run_start.elapsed().as_secs());
                let idle_text = text.trim().to_string();
                if last_idle_text.as_deref() == Some(idle_text.as_str()) {
                    consecutive_idle = consecutive_idle.saturating_add(1);
                } else {
                    consecutive_idle = 1;
                    last_idle_text = Some(idle_text);
                }
                if remaining <= IDLE_AT_WALL_MARGIN_SECS || consecutive_idle >= MAX_IDENTICAL_IDLE {
                    println!(
                        "[episode] idle-spin guard: {} — ending episode (idle x{consecutive_idle}, {remaining}s left)",
                        if remaining <= IDLE_AT_WALL_MARGIN_SECS {
                            "at the wall"
                        } else {
                            "repeated identical idle replies"
                        }
                    );
                    break EpisodeStop::Budget;
                }
            }
            let prompt = match kind {
                Nonproductive::Truncated => prompts::TRUNCATED,
                Nonproductive::Idle => prompts::CONTINUE,
            };
            // Idle nudges carry the live budget so "keep optimizing" never reads as "no deadline".
            let nudge_text = match kind {
                Nonproductive::Idle => format!(
                    "{prompt}\n\n{}",
                    render_time_left(run_start.elapsed().as_secs(), cfg.max_wall_clock_secs)
                ),
                Nonproductive::Truncated => prompt.to_string(),
            };
            let nudge = client.user_message(nudge_text);
            if let Some(n) = append_message_only_node(
                search_tree,
                cfg.beam_width,
                episode_turns,
                sandbox.workspace(),
                transcript
                    .last()
                    .cloned()
                    .into_iter()
                    .chain(std::iter::once(nudge.clone()))
                    .collect(),
                parent_node.as_ref(),
            ) {
                parent_node = Some(n);
            }
            episode_turns = episode_turns.saturating_add(1);
            transcript.push(nudge);
            // Emit the delta (no evals on a nudge turn).
            let _ = event_tx.send(EpisodeEvent::Turn(LedgerDelta {
                usage,
                evaluations: 0,
                candidates_archived: 0,
                last_context_tokens: last_ctx_tokens,
            }));
            continue;
        }

        // A productive (tool-executing) turn breaks any idle streak.
        consecutive_idle = 0;
        last_idle_text = None;

        // Capture the turn's note/tools before dispatch consumes `tool_calls`.
        let agent_note = util::truncate(
            &text.split_whitespace().collect::<Vec<_>>().join(" "),
            ACTIVITY_NOTE_CHARS,
        );
        let tool_summary = summarize_tool_calls(tool_calls.iter().map(|c| c.name.as_str()));
        // A completion cut off at the output-token cap may carry incomplete/garbled
        // tool-call JSON. Do NOT execute it — answer with re-issue errors.
        let truncated = matches!(reason, StopReason::MaxTokens) && !tool_calls.is_empty();
        let eval_calls = if truncated {
            0
        } else {
            u64::try_from(tool_calls.iter().filter(|c| c.name == "evaluate").count()).unwrap_or(u64::MAX)
        };

        let workspace_before = workspace_digest(sandbox.workspace()).ok();
        let results = if truncated {
            turn::truncated_tool_results(&tool_calls, prompts::TRUNCATED_TOOL)
        } else {
            turn::dispatch_tools(&toolbox, tool_calls).await
        };

        let assistant_message = transcript.last().cloned();
        let mut tool_result_messages = client.tool_result_messages(results);

        let workspace_changed = workspace_before != workspace_digest(sandbox.workspace()).ok();
        let evaluations = turn_events.take_evaluations();
        let evaluated_files = evaluations.last().map(|e| &e.files);
        let evaluation = evaluations
            .last()
            .map(|e| Evaluation::from_metrics(&e.metrics, &ctx.eval_config.baseline()));
        let scorable = matches!(evaluation, Some(Evaluation::Timed { .. }));

        // Beam-only backstop: after EPISODE_TURN_CAP turns in one episode with no
        // scorable eval, fold a firm "yield a candidate" nudge INTO this turn's
        // tool-result user message and force a boundary. Deciding it *before* the
        // node is recorded is load-bearing: the nudge must ride the same
        // tool-result message, never a separate user turn. A standalone nudge
        // message (the old path) left the rebuilt context with two consecutive
        // user turns AND a duplicated tool_result — an invalid request the
        // provider rejects with a fatal HTTP 400, killing the run.
        // Suppress the yield nudge while a structural build is in progress (I7) — a
        // long unscored structural build is expected — but only up to a hard cap so
        // a genuinely wedged episode still yields a ranked candidate.
        let over_soft_cap = turns_since_scorable.saturating_add(1) >= EPISODE_TURN_CAP;
        let over_hard_cap = turns_since_scorable.saturating_add(1) >= EPISODE_TURN_HARD_CAP;
        let structural_hold = over_soft_cap
            && !over_hard_cap
            && todo::load_todos(sandbox)
                .iter()
                .any(|t| t.status == todo::Status::InProgress);
        let backstop = strategy::uses_beam_frontier(&cfg.policy_id) && !scorable && over_soft_cap && !structural_hold;
        if backstop {
            println!(
                "[episode] backstop: {} turns with no score — nudging to evaluate",
                turns_since_scorable.saturating_add(1)
            );
            client.append_instruction_to_results(&mut tool_result_messages, prompts::EPISODE_YIELD.to_string());
        }
        transcript.extend(tool_result_messages.clone());

        let observed_eval = evaluation.clone();
        let mut node_messages: Vec<C::Message> = assistant_message.into_iter().collect();
        node_messages.extend(tool_result_messages);
        let node_id = match search_tree.append_typed_turn_node_from(
            parent_node.as_ref(),
            episode_turns,
            cfg.beam_width.max(1),
            node_messages,
            sandbox.workspace(),
            workspace_changed,
            evaluated_files,
            evaluation,
        ) {
            Ok(id) => {
                parent_node = Some(TurnNodeId::new(id.clone()));
                Some(id)
            }
            Err(e) => {
                eprintln!("[history] failed to append trajectory node: {e}");
                None
            }
        };
        episode_turns = episode_turns.saturating_add(1);

        // Boundary transition (§6): a scorable node ends the episode; otherwise
        // stay in-episode unless the backstop above fired (nudge already folded in).
        let end = if scorable {
            match (node_id, observed_eval) {
                (Some(id), Some(eval)) => Some(EpisodeStop::Scorable {
                    node: TurnNodeId::new(id),
                    eval,
                }),
                _ => Some(EpisodeStop::Boundary),
            }
        } else {
            turns_since_scorable = turns_since_scorable.saturating_add(1);
            if backstop { Some(EpisodeStop::Boundary) } else { None }
        };

        // Emit the per-turn delta (loop folds into RunMeta and persists).
        local_evaluations = local_evaluations.saturating_add(eval_calls);
        let entry = format_activity_entry(&agent_note, &tool_summary);
        activity.push(entry);
        let _ = event_tx.send(EpisodeEvent::Turn(LedgerDelta {
            usage,
            evaluations: eval_calls,
            candidates_archived: eval_calls,
            last_context_tokens: last_ctx_tokens,
        }));

        if let Some(stop) = end {
            break stop;
        }
    };

    let _ = event_tx.send(EpisodeEvent::Done(EpisodeDone {
        new_activity: activity,
        stop,
    }));
}

/// The directions to inject when the supervisor is due: an active review of the
/// committed trajectory when a supervisor model is configured, else the static
/// nudge for the rung. A failed review falls back to the static nudge.
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
async fn supervisor_directions<C: ProtocolClient>(
    rung: supervisor::Rung,
    supervisor: Option<&C>,
    search_tree: &SearchTree,
    progress: supervisor::Progress,
    recent_activity: &[String],
    hardware: &str,
) -> String {
    let Some(sup) = supervisor else {
        return supervisor::static_prompt(rung).to_string();
    };
    println!("\n[supervisor] reviewing trajectory ({rung:?})...");
    let recent = render_recent_activity(recent_activity);
    let ctx = supervisor::review_context(rung, &render_trajectory(search_tree), &recent, progress, hardware);
    match supervisor::review(sup, ctx).await {
        Ok(directions) => format!("{}\n\n{directions}", prompts::SUPERVISOR_REVIEW_PREAMBLE),
        Err(e) => {
            eprintln!("[supervisor] review failed ({e}); using static nudge");
            supervisor::static_prompt(rung).to_string()
        }
    }
}

/// Summarize `model` into a fresh compacted context: the OLDER prefix is
/// replaced by a structured brief, the recent tail is kept verbatim, and the
/// deterministic ground-truth anchor is folded into the brief so the summary
/// can't drift from on-disk truth. Returns the new context; the caller adopts it.
#[expect(
    clippy::too_many_arguments,
    reason = "8 independent handles; a param struct just relocates them"
)]
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
async fn compact_context<C: ProtocolClient>(
    client: &C,
    model: &[C::Message],
    search_tree: &SearchTree,
    current_node: Option<&TurnNodeId>,
    sandbox: &Sandbox,
    context_window_tokens: u32,
    elapsed_secs: u64,
    max_wall_secs: u64,
) -> Result<Vec<C::Message>, CompletionError> {
    // Item B2: the post-compaction self used to lose the clock — the brief folded in
    // the anchor + gotchas but not the remaining-time line. Prepend it to the anchor
    // (which is placed FIRST in the brief) so the summarized-forward self still paces
    // against the wall.
    let anchor = format!(
        "{}\n\n{}",
        render_time_left(elapsed_secs, max_wall_secs),
        ground_truth_anchor(search_tree, current_node, on_disk_solution_sha(sandbox).as_deref()),
    );
    // Durable ledgers folded into the compaction head so a score-perfect-but-lossy
    // summary can't drop a hard-won build flag / recipe. The never-truncated
    // build-invariants note (item C1) rides AHEAD of the growing gotchas ledger
    // (item 3), each with its own re-inject preamble, so the most load-bearing
    // facts are read first and always in full.
    let gotchas = render_gotchas(sandbox).unwrap_or_default();
    let invariants = render_invariants(sandbox)
        .map(|body| format!("{}\n\n{body}", prompts::INVARIANTS_REINJECT_PREAMBLE))
        .unwrap_or_default();
    // The doc-survey index rides at the END of the ledger — design memory read
    // AFTER the never-re-derive build facts — with its own preamble so the block is
    // self-describing. Folding it in is the specific gap this closes: the compaction
    // preamble named the file but never carried its content across the reset.
    let doc_index = render_doc_index(sandbox)
        .map(|body| format!("{}\n\n{body}", prompts::DOC_INDEX_REINJECT_PREAMBLE))
        .unwrap_or_default();
    let ledger = fold_ledger(&invariants, &gotchas, &doc_index);
    let mut compacted = model.to_vec();
    compaction::compact(
        client,
        &mut compacted,
        &anchor,
        &ledger,
        compaction::KEEP_RECENT_TOKENS,
        context_window_tokens,
    )
    .await?;
    Ok(compacted)
}

/// SHA-256 of the agent's live working `solution/` (the actual sandbox files),
/// for the anchor's on-disk-vs-best line. Read the SAME way `evaluate` snapshots
/// a candidate (`sandbox.read_tree(SOLUTION_DIR)`), hashed with the SAME
/// [`types::fileset_sha256`] used to fill [`BestInfo::solution_sha256`], so the
/// comparison is exact by construction. `None` if the sandbox can't be read (the
/// anchor then omits the line rather than guessing).
fn on_disk_solution_sha(sandbox: &Sandbox) -> Option<String> {
    sandbox
        .read_tree(SOLUTION_DIR)
        .ok()
        .map(|files| types::fileset_sha256(&files))
}

/// The deterministic ground-truth anchor injected as a user note on every
/// episode start and stall rung, and folded into every compaction brief. Built
/// from the git lineage (never model-written) so it can NEVER drift: the
/// authoritative current confirmed best (geomean + node + solution sha), whether
/// the live sandbox `solution/` matches or differs from it (`on_disk_sha`, hashed
/// from the real sandbox by the caller), a short ranked committed trajectory, and
/// a legend explaining the metric. This is the specific fix for a run that lost
/// track of its own committed progress and blamed "GPU noise". Empty string when
/// nothing is confirmed yet (early run), so callers can skip injection.
fn ground_truth_anchor(
    search_tree: &SearchTree,
    current_node: Option<&TurnNodeId>,
    on_disk_sha: Option<&str>,
) -> String {
    // Scope the best + candidates to THIS episode's own lineage (root→current
    // node), not the global tree. Under beam search (p>1) the tree also holds
    // sibling branches; injecting their results here made the agent try to
    // reconcile a "best" it never produced and cannot load (run_1784328572 burned
    // turns on a sibling's 13.08x). With no current node (cold start) fall back to
    // the global best — there is only the seed lineage then anyway.
    let (best, candidates) = current_node.map_or_else(
        || (search_tree.current_best(), search_tree.ranked_candidates()),
        |node| (search_tree.lineage_best(node), search_tree.lineage_candidates(node)),
    );
    let Some(best) = best else {
        return String::new();
    };
    let best_geo = best.geomean_speedup.unwrap_or(1.0);
    let short = |s: &str| s.chars().take(12).collect::<String>();
    let mut out = String::new();
    out.push_str(
        "=== GROUND TRUTH (authoritative — from your OWN committed lineage; trust this over the summary and over your own recollection) ===\n",
    );
    let _ = writeln!(
        out,
        "Best confirmed on your lineage: geomean {best_geo:.4}x  |  node {}  |  solution sha {}",
        best.node_id,
        short(&best.solution_sha256),
    );
    // "Banked & safe" counter-fact (item B1), placed immediately under the best
    // number and (via the note assembly) directly beneath the remaining-time line,
    // so the reassurance sits next to the time pressure it answers. The run this
    // fixes kept deferring the one structural bet worth making because the wall felt
    // expensive; the truth is the confirmed best cannot regress, so a failed attempt
    // costs only time. Kept to its DISTINCT contribution (the EV/time reframe) — the
    // "kept safe on its own node" mechanic is already stated by the on-disk-differs
    // branch below and the LEGEND, so it is not repeated a third time here.
    out.push_str(
        "This best is BANKED and cannot go down: a failed, slow, or reverted attempt NEVER lowers it, so an \
         ambitious structural build costs you only time, never your score — spend the wall on the highest-EV \
         attempt, not on protecting this number.\n",
    );
    // On-disk-vs-best: does the agent's ACTUAL working `solution/` (its sandbox
    // files, hashed by the caller from `sandbox.read_tree(SOLUTION_DIR)`) equal the
    // confirmed best? Comparing the real sandbox — NOT the search tree's global
    // playhead leaf, which under `beam-width > 1` or right after a promotion points
    // at a different node than this episode's sandbox and made this line lie ("on
    // disk matches best" while the file was the 1.0x baseline). This is the exact
    // mis-attribution class from AUDIT_run_1784148617, so the fact must be measured
    // from disk, never inferred. `None` = couldn't read the sandbox: skip the line
    // rather than guess.
    match on_disk_sha {
        Some(sha) if sha == best.solution_sha256 => {
            out.push_str("On disk NOW: your working `solution/` matches the best confirmed on your lineage above.\n");
        }
        Some(_) => {
            out.push_str(
                "On disk NOW: your working `solution/` is NOT that best — this is EXPECTED, not an error: the search \
                 keeps every confirmed candidate safe on its own node, so a later turn reverting/regressing your \
                 working tree never loses the best on your line, and you never have to rebuild it from memory. To \
                 recover that best kernel (its `solution/` AND verified `artifacts/`), call `checkout` — it writes the \
                 confirmed best's source into a fresh `checkout/` dir (non-destructive: your live tree is untouched, so \
                 copy across what you need; rebuild before evaluating). Otherwise keep making evaluated improvements \
                 from what you have; if you beat the number above it is recorded automatically. \
                 (`search_view` shows your lineage's standings, read-only.)\n",
            );
        }
        None => {}
    }
    let confirmed: Vec<_> = candidates.iter().filter(|c| c.confirmed).take(8).collect();
    if !confirmed.is_empty() {
        out.push_str("Confirmed on your lineage (best first):\n");
        for c in &confirmed {
            let _ = writeln!(
                out,
                "  - turn {}: geomean {:.4}x  (node {})",
                c.turn,
                c.geomean,
                short(&c.node_id),
            );
        }
    }
    // Tried-and-rejected (item A): attempts the agent BUILT, ran, and MEASURED that
    // scored below its confirmed best and so were never promoted. These live on the
    // tree as non-confirmed `Timed` nodes (the UCB/beam policy even steers on them,
    // strategy.rs:261-273) — but the anchor used to hide them behind the
    // confirmed-only filter above. That is exactly how run_1784671694 re-derived a
    // scored structural attempt 4× and, at 85% wall, falsely concluded its
    // integration "was never started". Surfacing them as a LABEL (not a promotion — the
    // confirmation gate is untouched) prevents both, and reframes the remaining work
    // from "start from zero" to "resume the node you already built". Best-first
    // (the list is pre-sorted); capped so a long run can't crowd out context.
    // Exclude the node the agent is currently ON: it is not a separate attempt to
    // "checkout and RESUME from", and telling it to do so is nonsensical.
    let cur_id = current_node.map(super::exec::turn_tree::TurnNodeId::as_str);
    let rejected: Vec<_> = candidates
        .iter()
        .filter(|c| !c.confirmed && Some(c.node_id.as_str()) != cur_id)
        .take(6)
        .collect();
    if !rejected.is_empty() {
        out.push_str(
            "Tried & measured on your lineage (NOT promoted — recorded, not lost). \"NOT promoted\" means it \
             did not clear the paired re-timing confirmation gate, NOT that it is worthless:\n",
        );
        for c in &rejected {
            // A non-confirmed `Timed` node is NOT necessarily below the confirmed best:
            // a winner's-curse / within-noise confirmation REJECT keeps its raw (often
            // above-best) geomean and stays unconfirmed, and a still-pending leader is
            // unconfirmed-yet-above-best too. Hardcoding "below best" was therefore false
            // for exactly the highest-EV near-misses this list most wants the agent to
            // resume — so label each candidate honestly against `best_geo`.
            let tag = if c.geomean >= best_geo {
                "← at/ABOVE your best on a single eval, but NOT confirmed on paired re-timing (within-noise / winner's-curse) — a prime candidate to resume and re-run"
            } else {
                "← below your best"
            };
            let _ = writeln!(
                out,
                "  - turn {}: geomean {:.4}x  (node {})  {tag}",
                c.turn,
                c.geomean,
                short(&c.node_id),
            );
        }
        out.push_str(
            "You already BUILT, ran, and measured these — do NOT re-derive them from scratch. `checkout` the \
             node to recover its actual source (`solution/` + verified `artifacts/`) and RESUME from it. Only \
             revisit an approach if you can change what actually made it lose — a materially different design \
             or schedule — not to re-measure the same one.\n",
        );
    }
    out.push_str(
        "LEGEND: geomean = speedup over the first-correct baseline (1.0x), scoped to YOUR lineage \
         (the root→current-node path) — not other search branches, which are the orchestrator's to \
         manage, not yours to reconstruct. 'confirmed' = paired re-timing, so deltas between confirmed \
         candidates are REAL, not GPU noise. Your on-disk `solution/` may be BEHIND your lineage best \
         — a later turn can have reverted it — but the orchestrator always keeps that best safe, so \
         your work is NEVER lost and never has to be reconstructed: re-orient from this anchor (or \
         `search_view`) instead of concluding a score was lost. Keep improving from what you have; \
         `evaluate` (stage full) every candidate worth measuring.",
    );
    out
}

/// Re-inject the branch's persistent working plan (`notes/todo.json`) as an
/// EPHEMERAL user note, so the agent's memory of what it has already tried
/// survives context resets. Like [`ground_truth_anchor`], it is (a) rendered
/// FRESH from the file each call (never from stored node messages, so it always
/// reflects on-disk truth and never accumulates in the tree) and (b) folded in
/// via [`ProtocolClient::inject_user_note`], never a raw push, so it can't
/// create a fatal second-consecutive user turn on Anthropic. No-op when the
/// branch has no plan yet.
fn reinject_todos<C: ProtocolClient>(client: &C, transcript: &mut Vec<C::Message>, sandbox: &Sandbox) {
    if let Some(rendered) = todo::render_for_injection(sandbox) {
        client.inject_user_note(transcript, format!("{}\n\n{rendered}", prompts::TODO_REINJECT_PREAMBLE));
    } else {
        // Cold-start: the plan is empty, so there is nothing to re-show — and the
        // old code returned silently here, so the agent was never reminded to
        // START a plan (the whole feature went unused). Nudge it to seed one;
        // self-limiting, since this branch stops firing once an item exists.
        client.inject_user_note(transcript, prompts::TODO_SEED_PROMPT.to_string());
    }
}

/// Branch-workspace path of the durable solved-gotchas ledger (item 3). Under
/// `notes/` — snapshot-captured like [`todo::TODO_PATH`], so it survives
/// compaction/resume/branch expansion — and deliberately OUTSIDE `solution/`, so
/// editing it never perturbs the scored `read_tree(SOLUTION_DIR)` sha the anchor
/// compares against. Maintained by the agent with the normal `write`/`edit` file
/// tools (like `notes/DOC_INDEX.md`); there is no dedicated tool.
const GOTCHAS_PATH: &str = "notes/SOLVED_GOTCHAS.md";
/// Char cap on the injected/compaction-folded gotchas body (à la
/// [`todo::TODO_RENDER_CHARS`]). Kept generous — these are load-bearing recipes
/// re-derivation would cost hours — but bounded so a runaway ledger can't crowd
/// out context; on overflow the OLDEST entries are elided (newest kept).
const GOTCHAS_RENDER_CHARS: usize = 5_000;

/// Read the branch's solved-gotchas ledger for injection, capped to
/// [`GOTCHAS_RENDER_CHARS`] (newest tail kept, older head elided with a marker).
/// Returns the raw body (no preamble) so callers frame it as they need —
/// [`prompts::GOTCHAS_REINJECT_PREAMBLE`] at episode start, its own section in
/// the compaction head. `None` when the file is missing or empty. Rendered FRESH
/// from disk each call (never from stored node messages), mirroring
/// [`reinject_todos`] / [`ground_truth_anchor`].
fn render_gotchas(sandbox: &Sandbox) -> Option<String> {
    let text = sandbox.read(GOTCHAS_PATH).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() <= GOTCHAS_RENDER_CHARS {
        return Some(trimmed.to_string());
    }
    // Overflow: keep the newest tail (most recent gotchas), on a char boundary.
    let cut = trimmed.len().saturating_sub(GOTCHAS_RENDER_CHARS);
    let start = (cut..=trimmed.len())
        .find(|&i| trimmed.is_char_boundary(i))
        .unwrap_or(trimmed.len());
    Some(format!(
        "[…older ledger entries elided — `read` `{GOTCHAS_PATH}` for the full list, and use `checkout` to recover a verified kernel's source (solution/ + artifacts/) rather than re-deriving it…]\n{}",
        trimmed.get(start..).unwrap_or(trimmed)
    ))
}

/// Re-inject the branch's solved-gotchas ledger as an ephemeral user note at
/// episode start, so a hard-won fix (build flag, error→fix, working recipe) is
/// never re-derived after a context reset dropped it. No-op when the ledger is
/// empty. Injected SAFELY via [`ProtocolClient::inject_user_note`].
fn reinject_gotchas<C: ProtocolClient>(client: &C, transcript: &mut Vec<C::Message>, sandbox: &Sandbox) {
    if let Some(body) = render_gotchas(sandbox) {
        client.inject_user_note(transcript, format!("{}\n\n{body}", prompts::GOTCHAS_REINJECT_PREAMBLE));
    }
}

/// Branch-workspace path of the always-surfaced build-invariants note (item C1).
/// A DELIBERATELY SMALL, uncapped companion to [`GOTCHAS_PATH`]: the gotchas
/// ledger grows and is truncated to its newest tail ([`GOTCHAS_RENDER_CHARS`]),
/// which is how `run_1784671694` lost a load-bearing build flag off the top and
/// re-diagnosed it 13×. This file holds ONLY load-bearing invariants that must
/// never be re-derived (exact build/arch/compiler flags, hard hardware
/// constraints) and is re-shown VERBATIM and in FULL every reset.
const INVARIANTS_PATH: &str = "notes/BUILD_INVARIANTS.md";

/// Generous char cap on the injected/compaction-folded invariants body. The note
/// is TINY by contract (one line per invariant), but the file is agent-writable and
/// this body is folded VERBATIM into the compaction head with no downstream size
/// guard — so an unbounded note could push the post-compaction context OVER the
/// window and kill the run on the reactive retry-once path. Capped defensively: far
/// above any legitimate invariants note (≈3× the gotchas cap), but bounded. Unlike
/// gotchas (which keeps the newest TAIL), overflow here keeps the HEAD — the
/// load-bearing build flags are written first and are exactly what must never scroll
/// off. The cap is a backstop against misuse, not a license to grow the note.
const INVARIANTS_RENDER_CHARS: usize = 16_000;

/// Read the branch's build-invariants note for injection — the full body, capped at
/// [`INVARIANTS_RENDER_CHARS`] (a much higher bar than [`render_gotchas`], and
/// head-kept rather than tail-kept). Rendered fresh from disk each call (never from
/// stored node messages). `None` when the file is missing or empty.
fn render_invariants(sandbox: &Sandbox) -> Option<String> {
    let text = sandbox.read(INVARIANTS_PATH).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() <= INVARIANTS_RENDER_CHARS {
        return Some(trimmed.to_string());
    }
    // Overflow ⇒ the note is being misused as a scratchpad. Keep the HEAD on a char
    // boundary (the invariants that must never be lost) and tell the agent to trim it.
    let end = (0..=INVARIANTS_RENDER_CHARS)
        .rev()
        .find(|&i| trimmed.is_char_boundary(i))
        .unwrap_or(0);
    Some(format!(
        "{}\n[…`{INVARIANTS_PATH}` exceeded {INVARIANTS_RENDER_CHARS} chars and was truncated — it must stay TINY (one line per invariant); move recipes/logs/error dumps to `{GOTCHAS_PATH}` and trim this file back to load-bearing invariants…]",
        trimmed.get(..end).unwrap_or(trimmed)
    ))
}

/// Combine the (preamble-prefixed) invariants block, the gotchas block, and the
/// (preamble-prefixed) doc-survey index into the single "ledger" section folded
/// into the compaction head. Order = invariants → gotchas → doc index:
/// never-truncated build facts read first, hard-won fixes next, design memory last
/// (read after the load-bearing toolchain facts). Empty sides drop cleanly so there
/// is no stray blank separator. Pure, so the ordering + empty-handling are
/// unit-testable without a client or sandbox.
fn fold_ledger(invariants: &str, gotchas: &str, doc_index: &str) -> String {
    [invariants, gotchas, doc_index]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Re-inject the branch's build-invariants note as an ephemeral user note at
/// episode start, in FULL, so a load-bearing build flag is never dropped by the
/// gotchas-ledger truncation. No-op when the file is empty. Injected SAFELY via
/// [`ProtocolClient::inject_user_note`], mirroring [`reinject_gotchas`].
fn reinject_invariants<C: ProtocolClient>(client: &C, transcript: &mut Vec<C::Message>, sandbox: &Sandbox) {
    if let Some(body) = render_invariants(sandbox) {
        client.inject_user_note(
            transcript,
            format!("{}\n\n{body}", prompts::INVARIANTS_REINJECT_PREAMBLE),
        );
    }
}

/// Branch-workspace path of the agent's doc-survey index. Under `notes/` alongside
/// the other durable-memory ledgers (gotchas, invariants, plan) — NOT `solution/`,
/// so editing it does not perturb the scored `read_tree(SOLUTION_DIR)` sha (it is
/// design memory, never part of the scored kernel). It is snapshot-captured there
/// (see `snapshot_store`) so it survives compaction/resume/branch expansion. It is
/// the agent's distilled DESIGN memory — a one-line note per doc in the `docs/`
/// tree (reference docs AND papers): which fit this problem and the fast-path it is
/// driving toward — so it is re-surfaced like the ledgers rather than left on disk
/// to be re-read (or forgotten) after a reset.
const DOC_INDEX_PATH: &str = "notes/DOC_INDEX.md";

/// Char cap on the injected/compaction-folded doc-survey index. Generous — a
/// one-line note per doc across the whole `docs/` tree, plus per-paper design
/// notes, runs longer than the gotchas ledger — but bounded so it can't crowd the
/// post-compaction context. Overflow keeps the HEAD (like [`render_invariants`]):
/// the most-relevant docs and the flagged fast-path lead the file per the system
/// prompt's format, so the head is the load-bearing design direction; a marker
/// points at the file for the rest.
const DOC_INDEX_RENDER_CHARS: usize = 10_000;

/// Read the branch's doc-survey index for injection, capped to
/// [`DOC_INDEX_RENDER_CHARS`] (HEAD kept, head-truncated like [`render_invariants`]).
/// Returns the raw body (no preamble) so callers frame it —
/// [`prompts::DOC_INDEX_REINJECT_PREAMBLE`] at episode start, its own section in the
/// compaction head. Rendered FRESH from disk each call (never from stored node
/// messages). `None` when the file is missing or empty.
fn render_doc_index(sandbox: &Sandbox) -> Option<String> {
    let text = sandbox.read(DOC_INDEX_PATH).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() <= DOC_INDEX_RENDER_CHARS {
        return Some(trimmed.to_string());
    }
    // Overflow: keep the HEAD (most-relevant docs + flagged fast-path) on a char
    // boundary, and point at the file for the remaining, lower-priority entries.
    let end = (0..=DOC_INDEX_RENDER_CHARS)
        .rev()
        .find(|&i| trimmed.is_char_boundary(i))
        .unwrap_or(0);
    Some(format!(
        "{}\n[…`{DOC_INDEX_PATH}` exceeded {DOC_INDEX_RENDER_CHARS} chars and was truncated to its head — `read` `{DOC_INDEX_PATH}` for the full index…]",
        trimmed.get(..end).unwrap_or(trimmed)
    ))
}

/// Re-inject the branch's doc-survey index as an ephemeral user note at episode
/// start, so the flagged fast-path direction survives a context reset that dropped
/// it — rather than the agent re-surveying the docs tree or drifting to local
/// tuning. No-op when the file is empty. Injected SAFELY via
/// [`ProtocolClient::inject_user_note`], mirroring [`reinject_gotchas`].
fn reinject_doc_index<C: ProtocolClient>(client: &C, transcript: &mut Vec<C::Message>, sandbox: &Sandbox) {
    if let Some(body) = render_doc_index(sandbox) {
        client.inject_user_note(
            transcript,
            format!("{}\n\n{body}", prompts::DOC_INDEX_REINJECT_PREAMBLE),
        );
    }
}

/// Fold a [`LedgerDelta`] emitted by an episode into the loop's [`RunMeta`].
/// Mirrors what `record_turn` + the per-turn eval-accounting lines used to do
/// inside the episode; called loop-side after each [`EpisodeEvent::Turn`].
fn apply_delta(meta: &mut RunMeta, delta: &LedgerDelta) {
    meta.turns = meta.turns.saturating_add(1);
    meta.turns_since_best_improved = meta.turns_since_best_improved.saturating_add(1);
    meta.ledger.turns = meta.turns;
    let l = &mut meta.ledger;
    l.input_tokens = l.input_tokens.saturating_add(u64::from(delta.usage.input_tokens));
    l.output_tokens = l.output_tokens.saturating_add(u64::from(delta.usage.output_tokens));
    l.cache_read_tokens = l
        .cache_read_tokens
        .saturating_add(u64::from(delta.usage.cache_read_tokens));
    l.cache_write_tokens = l
        .cache_write_tokens
        .saturating_add(u64::from(delta.usage.cache_write_tokens));
    l.reasoning_tokens = l
        .reasoning_tokens
        .saturating_add(u64::from(delta.usage.reasoning_tokens));
    l.evaluations = l.evaluations.saturating_add(delta.evaluations);
    l.candidates_archived = l.candidates_archived.saturating_add(delta.candidates_archived);
    meta.last_context_tokens = delta.last_context_tokens;
}

/// Confirm a challenger against the incumbent best with paired re-timing (the
/// winner's-curse/drift defense, formerly inside `submit`, now run by the loop on
/// the policy's `Reevaluate`). Re-times both node snapshots and tests the
/// drift-cancelled improvement; on a clear win records the confirmation that
/// crowns the challenger as the new best. Returns the confirmed geomean on
/// promotion, `None` on a statistical reject, `Err` on infra failure (best left
/// unchanged).
///
/// The two re-timings run **concurrently on separate GPU pool leases** (so with
/// N>1 devices the paired confirm costs one eval-batch of wall, not two): the
/// challenger is staged on the loop's scratch `sandbox` and the incumbent on a
/// throwaway sandbox spawned for this confirm. On a single-GPU pool the two leases
/// serialize, preserving the old behavior. The scratch sandbox is left holding the
/// challenger (the incumbent never touches it), so no post-confirm restore is
/// needed. The incumbent needs only the always-mounted `_trusted/` evaluator plus
/// its own `solution/`, so its sandbox spawns with no workspace base.
async fn confirm_leader(
    sandbox: &Sandbox,
    queue: &GpuPool,
    eval_config: &EvalConfig,
    search_tree: &SearchTree,
    samples: u64,
    challenger: &TurnNodeId,
    incumbent: &TurnNodeId,
) -> Result<Option<f64>, String> {
    let n = usize::try_from(samples).unwrap_or(usize::MAX).max(2);
    let manager = search_tree.manager();
    let challenger_files = search_tree.read_solution_at(challenger)?;
    let incumbent_files = search_tree.read_solution_at(incumbent)?;

    // Stage the challenger on the shared scratch sandbox and the incumbent on a
    // fresh throwaway one, so the two re-timings never clobber each other's
    // `solution/` and can run at the same time on different cards.
    sandbox
        .write_tree(SOLUTION_DIR, &challenger_files)
        .map_err(|e| format!("staging challenger failed: {e}"))?;
    let incumbent_sb = manager
        .spawn(None)
        .map_err(|e| format!("infrastructure error: spawning incumbent confirm sandbox failed: {e}"))?;
    incumbent_sb
        .write_tree(SOLUTION_DIR, &incumbent_files)
        .map_err(|e| format!("staging incumbent failed: {e}"))?;

    // Both re-timings concurrently; each `run_evaluator_n` takes its own device
    // lease via `acquire_any`, so they land on different GPUs when the pool has
    // them and serialize when it doesn't.
    let (ch_res, inc_res) = tokio::join!(
        run_evaluator_n(sandbox, queue, eval_config, types::Stage::Full, n),
        run_evaluator_n(&incumbent_sb, queue, eval_config, types::Stage::Full, n),
    );
    let ch = ch_res?;
    let inc = inc_res?;

    let baseline = eval_config.baseline();
    let cg = scored_geomeans(&ch, &baseline);
    let hg = scored_geomeans(&inc, &baseline);
    let c = strategy::confirm_improvement(&cg, &hg, strategy::MIN_IMPROVEMENT_FRAC, strategy::CONFIRM_Z);
    if c.samples < 2 {
        // Inconclusive (flaky re-eval). The challenger already cleared the Stage-1
        // screen in `observe`; promote on the freshest challenger sample we got,
        // else leave the best unchanged so infra noise never crowns a phantom.
        if let Some(&g) = cg.first() {
            search_tree.record_confirmation(challenger, g, c.samples)?;
            return Ok(Some(g));
        }
        return Ok(None);
    }
    if c.promote {
        search_tree.record_confirmation(challenger, c.challenger_geomean, c.samples)?;
        Ok(Some(c.challenger_geomean))
    } else {
        Ok(None)
    }
}

/// Geomean **speedup-over-baseline** samples from the correct, scored runs of a
/// re-eval batch. The evaluator emits absolute per-config latencies; each sample
/// is scored against the frozen baseline (the first correct candidate). The
/// baseline cancels in the challenger-vs-incumbent ratio, so the confirmation
/// test is unchanged.
fn scored_geomeans(samples: &[types::EvalMetrics], baseline: &std::collections::BTreeMap<String, f64>) -> Vec<f64> {
    samples
        .iter()
        .filter(|m| m.correct && !m.per_config.is_empty())
        .map(|m| types::speedup_vs_baseline(baseline, &m.per_config))
        .filter(|g| *g > 0.0)
        .collect()
}

/// Seed the stub root workspace (fresh) or restore the best candidate into the
/// sandbox (resume). Returns the current best view if one exists — `None` on a
/// fresh run, where the performance baseline is established later by the agent's
/// FIRST correct candidate. The reference is the correctness oracle only; the
/// seed is a deliberately-incomplete stub the agent must implement, so it is
/// never benchmarked or scored here.
fn init_baseline(sandbox: &Sandbox, search_tree: &SearchTree, cfg: &AvoConfig) -> Result<Option<BestInfo>, String> {
    if cfg.resume {
        // On resume the session tree is the source of truth (re-derived on open());
        // restore the best node's solution into the sandbox and continue from there.
        if let Some(best) = search_tree.current_best() {
            let files = search_tree
                .read_solution_at(&best.node_id)
                .map_err(|e| format!("infrastructure error: restoring best node {} failed: {e}", best.node_id))?;
            sandbox
                .write_tree(SOLUTION_DIR, &files)
                .map_err(|e| format!("infrastructure error: restoring {SOLUTION_DIR}/ failed: {e}"))?;
            return Ok(Some(best));
        }
        // resume requested but nothing to resume — fall through to fresh seed.
    }

    // Fresh run: materialize the stub seed and snapshot it as the UNSCORED root.
    // No baseline exists yet — the agent's first correct `evaluate` freezes it.
    sandbox.write(&solution_entrypoint(), &cfg.seed_src).map_err(|e| {
        format!(
            "infrastructure error: writing seed {} failed: {e}",
            solution_entrypoint()
        )
    })?;
    let files = sandbox
        .read_tree(SOLUTION_DIR)
        .map_err(|e| format!("infrastructure error: reading seed {SOLUTION_DIR}/ failed: {e}"))?;
    search_tree.seed_root(&files)?;
    println!("seeded stub root; baseline will be set by the agent's first correct candidate");
    Ok(search_tree.current_best())
}

fn budget_summary(meta: &RunMeta) -> String {
    let l = &meta.ledger;
    format!(
        "turns={}, commits={}, evals={}, tokens={} (in {} / out {} / cache_r {} / cache_w {}), wall={}s (gpu-busy {}s)",
        l.turns,
        l.commits,
        l.evaluations,
        l.total_tokens(),
        l.input_tokens,
        l.output_tokens,
        l.cache_read_tokens,
        l.cache_write_tokens,
        l.wall_clock_secs,
        crate::domain::convert::f64_to_u64_saturating(l.queue_busy_secs),
    )
}

fn append_message_only_node<M>(
    search_tree: &SearchTree,
    beam_width: usize,
    turn: u64,
    workspace: &std::path::Path,
    new_messages: Vec<M>,
    explicit_parent: Option<&TurnNodeId>,
) -> Option<TurnNodeId>
where
    M: Clone + Serialize + DeserializeOwned,
{
    match search_tree.append_typed_turn_node_from(
        explicit_parent,
        turn,
        beam_width.max(1),
        new_messages,
        workspace,
        false,
        None,
        None,
    ) {
        Ok(id) => Some(TurnNodeId::new(id)),
        Err(e) => {
            eprintln!("[history] failed to append message-only trajectory node: {e}");
            None
        }
    }
}

fn workspace_digest(root: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut paths = Vec::new();
    collect_workspace_files(root, root, &mut paths).map_err(|e| e.to_string())?;
    paths.sort();
    let mut hasher = Sha256::new();
    for path in paths {
        let rel = path
            .strip_prefix(root)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        if crate::exec::snapshot_store::should_exclude_workspace_path(&rel) {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        hasher.update(u64::try_from(rel.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(rel.as_bytes());
        hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(crate::domain::types::sha256_hex(&hasher.finalize()))
}

fn collect_workspace_files(
    root: &std::path::Path,
    dir: &std::path::Path,
    out: &mut Vec<std::path::PathBuf>,
) -> std::io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if crate::exec::snapshot_store::should_exclude_workspace_path(&rel) {
            continue;
        }
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_workspace_files(root, &path, out)?;
        } else if ft.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

mod manifest;
mod session;
use manifest::{curve_point, run_params, write_manifest};
use session::load_or_init_session;

#[cfg(test)]
mod tests;
