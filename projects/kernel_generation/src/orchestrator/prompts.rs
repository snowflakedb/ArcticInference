//! Compile-time-embedded prompt text.
//!
//! Prompts live as editable Markdown under `src/prompts/` (one file per prompt) and
//! are baked into the binary with `include_str!` at build time, so the binary is
//! self-contained: there is no runtime file layout to resolve or ship, and a
//! missing/renamed prompt is a compile error rather than a runtime failure.
//! Editing a prompt requires a rebuild. The path is resolved against
//! `CARGO_MANIFEST_DIR`, so it is independent of which source file refers to it.

/// `include_str!` a file under the crate's `src/prompts/` directory.
macro_rules! prompt {
    ($file:literal) => {
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/prompts/", $file))
    };
}

/// The agent's standing instructions (AVO framing, tools, workflow).
pub const SYSTEM: &str = prompt!("system_prompt.md");

/// The compaction request handed to the model.
///
/// Asked of the model when the context window nears full: it replies with a
/// structured brief that replaces the OLDER prefix of the transcript (the recent
/// tail is kept verbatim). Pairs with [`COMPACT_TEMPLATE`].
pub const COMPACT: &str = prompt!("compact.md");

/// The compaction brief's fixed section structure, appended to [`COMPACT`].
///
/// A stable template makes rolling updates (preserve/merge on repeat compaction)
/// reliable and keeps exact paths/symbols/hashes in known slots.
pub const COMPACT_TEMPLATE: &str = prompt!("compact_template.md");

/// Prepended to the model's own brief after compaction.
pub const COMPACT_PREAMBLE: &str = prompt!("compact_preamble.md");

/// Supervisor: injected after several submissions are rejected in a row.
pub const SUPERVISOR_FAILED_COMMITS: &str = prompt!("supervisor_failed_commits.md");

/// Supervisor: injected after a long stretch with no committed improvement.
pub const SUPERVISOR_STAGNATION: &str = prompt!("supervisor_stagnation.md");

/// Supervisor: injected when the agent has edited/profiled for several turns
/// without a single `evaluate` — it's likely mistaking its own timings for the
/// score.
pub const SUPERVISOR_NO_EVAL: &str = prompt!("supervisor_no_eval.md");

/// Supervisor (active review): the reviewing model's instructions.
///
/// Handed to the cheaper supervisor model, prepended to the rendered lineage
/// trajectory + stall signal. It reviews the trajectory and returns concrete
/// directions to inject.
pub const SUPERVISOR_REVIEW: &str = prompt!("supervisor_review.md");

/// Prepended to the supervisor model's returned directions before injecting
/// them into the main agent's transcript, so the agent knows the source.
pub const SUPERVISOR_REVIEW_PREAMBLE: &str = "Your supervisor paused to review the run so far and suggests these optimization directions. \
     Consider them, pick the most promising, and pursue it — then `submit` the next correct improvement:";

/// Preamble for the re-injected per-branch working plan.
///
/// Prepended to the branch's persistent working plan when it is re-injected at
/// episode start and after every context reset. Framed like the ground-truth
/// anchor / supervisor preamble so the agent treats the plan as its own durable
/// memory (what it already tried) rather than a fresh instruction.
pub const TODO_REINJECT_PREAMBLE: &str = "Your working plan for THIS solution branch (kept via `todo_write`, re-shown after every context reset). \
     It records what you are trying, completed, ruled out by an evaluation (`cancelled` — a dead end, do not re-explore), \
     or parked for time (`deferred` — still viable, NOT disproven). Trust it over your recollection: pursue the open items. \
     Before re-opening a `deferred` lever, check the GROUND TRUTH 'Tried & measured' list and your gotchas/invariants: if you \
     ALREADY built or scored it, RESUME from that node with `checkout` — do NOT re-derive it — and if a structural build is \
     already in progress, see it through rather than re-litigating the keep-or-cut. Start a `deferred` lever fresh only when it \
     is neither already measured nor already committed. Update the plan as evals settle each direction:";

/// Cold-start nudge to seed a working plan when the branch has none.
///
/// Injected when the branch has NO working plan yet, so the agent is actually
/// reminded to seed one (the re-inject above only re-shows a plan that already
/// exists, so an unseeded plan was never advertised again). Self-limiting: stops
/// firing as soon as one item exists.
pub const TODO_SEED_PROMPT: &str = "You have not started a working plan for this branch yet. Seed one now with `todo_write` — \
     one item per optimization DIRECTION you intend to pursue (not micro-steps): e.g. the docs and papers you flagged \
     relevant in `notes/DOC_INDEX.md` and the fast-path build you are driving toward. It is re-shown \
     after every context reset, so it becomes your durable memory of what you've tried, are trying, and ruled out.";

/// Preamble for the re-injected solved-gotchas ledger.
///
/// Prepended to the branch's durable solved-gotchas ledger (`notes/SOLVED_GOTCHAS.md`)
/// when it is re-injected at episode start. Framed like [`TODO_REINJECT_PREAMBLE`]
/// so the agent treats it as its own hard-won memory (exact build flags,
/// error→fix mappings, working recipes) rather than a fresh instruction — the
/// specific loss this fixes is re-deriving a solved toolchain gotcha after a
/// compaction dropped it.
pub const GOTCHAS_REINJECT_PREAMBLE: &str = "Your durable solved-gotchas ledger for this branch (`notes/SOLVED_GOTCHAS.md`, re-shown after every context reset). \
     These are hard-won fixes you already paid for — exact build flags, error-string→fix mappings, working code recipes. \
     Trust them over re-derivation and do NOT re-litigate a gotcha recorded here as solved; append new ones as you solve them. \
     A confirmed candidate's verified source — its `solution/` AND standalone `artifacts/` kernels — is recoverable with `checkout` (it writes them into a fresh `checkout/` dir, non-destructively) instead of rebuilding from scratch:";

/// Preamble for the branch's always-surfaced build-invariants note (`notes/BUILD_INVARIANTS.md`).
///
/// Prepended at episode start and folded into the compaction head. Unlike
/// [`GOTCHAS_REINJECT_PREAMBLE`], this note is shown in FULL every time (never
/// truncated to a tail), so the preamble insists it stay SHORT — it is reserved
/// for the few invariants whose loss is catastrophic (the 13× re-diagnosed build
/// flag in `run_1784671694` was exactly this class).
pub const INVARIANTS_REINJECT_PREAMBLE: &str = "Your BUILD INVARIANTS for this branch (`notes/BUILD_INVARIANTS.md`, shown in FULL after every context reset — never truncated). \
     These are the few load-bearing facts that are catastrophic to re-derive: the exact build/arch/compiler flags that finally compiled, hard hardware constraints, and settings a reset must not drop. \
     Trust them verbatim and do NOT re-diagnose anything recorded here. Keep this file SHORT — one line per invariant; put longer recipes and error→fix mappings in `notes/SOLVED_GOTCHAS.md`:";

/// Preamble for the re-injected doc-survey index.
///
/// Prepended to the branch's doc-survey index (`notes/DOC_INDEX.md`) when it is
/// re-injected at episode start and folded into the compaction head. Framed like
/// [`GOTCHAS_REINJECT_PREAMBLE`] so the agent treats it as its own durable DESIGN
/// memory — a one-line note per doc in the `docs/` tree (reference docs AND papers):
/// which it judged fit this problem, and the fast-path it is driving toward — rather
/// than a fresh instruction. The loss this fixes: a context reset drops the flagged
/// direction, after which the agent silently drifts to local tuning or re-surveys
/// the whole docs tree from scratch (the compaction preamble already NAMED the file
/// but never folded its content in). Deliberately generic (no problem-specific
/// terms): the survey convention applies to any optimization problem, not one domain.
pub const DOC_INDEX_REINJECT_PREAMBLE: &str = "Your doc-survey index for this branch (`notes/DOC_INDEX.md`, re-shown after every context reset). \
     This is your own one-line-per-doc survey of the `docs/` tree — which reference docs and papers fit this problem, when to use each, and the fast-path design you are driving toward — design memory you already paid to build. \
     Trust it: pursue the direction it commits to, and before re-deriving a fast path from memory, re-read the relevant sections of the doc or paper you flagged for it rather than re-surveying the tree from scratch. Keep it current as your design understanding sharpens:";

/// Nudge after the model ends a turn without finishing the work.
pub const CONTINUE: &str = prompt!("continue.md");

/// Nudge after the model's output is cut off at the per-turn token cap.
pub const TRUNCATED: &str = prompt!("truncated.md");

/// Error `tool_result` returned in place of executing a tool call whose
/// completion was cut off at the output-token cap — the arguments may be
/// incomplete/garbled, so the call is NOT run.
pub const TRUNCATED_TOOL: &str = "Your previous turn was cut off at the output-token limit, so this tool call's arguments \
     may be incomplete or invalid — it was NOT executed. Re-issue the tool call with complete arguments \
     (split the work into smaller steps if your output is hitting the limit).";

/// Beam-only backstop nudge for an episode that never scored.
///
/// Injected when an expansion episode has run `EPISODE_TURN_CAP` turns without
/// producing a scorable (timed) evaluation, so a wandering episode is forced to
/// yield a rankable candidate the policy can act on. AVO does not use this — its
/// stalls are covered by the supervisor rungs.
pub const EPISODE_YIELD: &str = "You have run many turns in this episode without producing a scored evaluation. \
     Stop exploring and run `evaluate` (stage `full`) on your current best change now, so the search has a \
     ranked candidate to work from. If the current change does not evaluate, revert to a known-good state and evaluate that.";
