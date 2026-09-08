//! Session/journal loading and kickoff-prompt construction for a run.

use crate::ai::ProtocolClient;
use crate::domain::run_state::RunMeta;
use crate::domain::types::{BudgetLedger, RunId, Seeds};
use crate::harness::skills::Skill;

use super::{AvoConfig, BestInfo, SOLUTION_DIR, SearchTree, TRUSTED_DIR, prompts, solution_entrypoint};

/// Reconstruct the model-visible context from the session tree on resume, or
/// start a fresh session. `restored_meta` is the journal's `RunMeta` (already
/// loaded by the caller so the eval seed and the meta agree); `seeds` is used
/// only for a fresh session.
pub(super) fn load_or_init_session<C: ProtocolClient>(
    client: &C,
    cfg: &AvoConfig,
    search_tree: &SearchTree,
    best: Option<&BestInfo>,
    seeds: Seeds,
    restored_meta: Option<RunMeta>,
) -> (Vec<C::Message>, RunMeta) {
    if cfg.resume {
        let model = search_tree
            .current_node_id()
            .and_then(|id| search_tree.context::<C::Message>(&id).ok())
            .unwrap_or_default();
        let meta =
            restored_meta.unwrap_or_else(|| RunMeta::new(run_id_from_dir(cfg), BudgetLedger::new(), seeds.clone()));
        if !model.is_empty() {
            println!("resumed: {} messages, {} turns", model.len(), meta.turns);
            return (model, meta);
        }
        eprintln!("resume requested but no session-tree history found; starting fresh");
    }
    let messages = vec![client.user_message(kickoff_prompt(cfg, best, !client.uses_dedicated_instructions()))];
    (messages, RunMeta::new(run_id_from_dir(cfg), BudgetLedger::new(), seeds))
}

pub(super) fn run_id_from_dir(cfg: &AvoConfig) -> RunId {
    RunId(
        cfg.run_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("run")
            .to_string(),
    )
}

pub(super) fn kickoff_prompt(cfg: &AvoConfig, best: Option<&BestInfo>, include_system: bool) -> String {
    let system = include_system
        .then_some(format!("{}\n\n", prompts::SYSTEM))
        .unwrap_or_default();
    let skills = if cfg.resume {
        String::new()
    } else {
        render_skills_instructions(&cfg.skills)
    };
    // The reference is the correctness oracle only. The performance baseline is
    // the agent's FIRST correct kernel (scored 1.0x); until it exists there is no
    // speedup to report and the seed is a stub the agent must implement.
    let baseline_line = best.map_or_else(
        || {
            "There is no baseline yet: you are starting from a stub, not a working kernel. Write a first \
                 *correct* custom kernel and `evaluate` (full) it — that first correct kernel becomes the 1.0x \
                 performance baseline, and every later kernel is scored as speedup over it."
                .to_string()
        },
        |b| {
            format!(
                "The current best solution scores geomean {:.4}x over your first-correct baseline.",
                b.geomean_speedup.unwrap_or(1.0)
            )
        },
    );
    format!(
        "{system}{skills}--- task ---\nProblem: {}.\n\n## This machine (probed at startup — authoritative)\n{}\nWrite code for THIS GPU only. Ground every hardware-specific claim in this block plus the mounted docs, pulling from `docs/nvidia/`. First read `docs/nvidia/cuda.md` sections `5.1.2.1. Architecture-Specific Features`, `5.1.2.2. Family-Specific Features`, and `5.1.2.3. Feature Set Compiler Targets`, plus the detected architecture's compatibility guide; then use the probed `compute_capability` / `cuda_arch` to search the docs for the newest architecture-specific features supported by this GPU generation, especially features not supported on previous generations, before choosing an implementation path. Do not assume a previous-generation architecture-specific feature is available here unless the docs explicitly list this run's exact `cuda_arch` or a compatible family target. `docs/papers/` holds a shared pool of algorithm papers (not all relevant to every problem); triage them by abstract and read any that fit this problem — an algorithm is backend-independent, so read it regardless of which GPU you're on.\n\
         {} Your editable solution lives in the \
         `{}/` directory — the entrypoint `{}` exports `Solution`, and you may add helper modules or \
         kernel-source files beside it (they are versioned together). The trusted evaluator and \
         `problem.py` are READ-ONLY under `{}/`. Begin with an upfront discovery pass (required orientation, not stalling): `ls -R docs/`, `markdown_get_toc` each markdown doc, read the CUDA feature-set compiler target sections, grep/read for the detected backend, GPU model, `compute_capability`, `cuda_arch`, exact-target compatibility, and latest architecture features available only on this GPU generation, and **survey the docs tree**: skim each doc cheaply (`markdown_get_toc` for a reference doc; the title + `## Abstract` of each `docs/papers/*/paper.md`, text only — do NOT `view` figures during the survey) and record a one-line note per doc in `notes/DOC_INDEX.md` — `path · relevant y/n · what it covers / when to use it`, most-relevant first — design from the docs and any relevant paper rather than memory; deep-read a flagged doc's or paper's relevant sections (and `view` only the specific figure a section actually needs, when you need it) on demand. Then read the current solution and the problem, \
         profile if useful, then improve it and run `evaluate` (full) on each candidate worth measuring. Land a first *correct* custom kernel early — it can be simple and slow, it does not need any exotic hardware feature or the state-of-the-art algorithm to start — and `evaluate` it to get a scored candidate on the board before deep optimization. There is no separate submit step: every scored evaluation becomes a candidate, the search keeps the best, and you'll be handed a strong candidate to continue improving.\n\n\
         ## Time budget\n\
         This run has a hard wall-clock budget of ~{}. There is no per-turn limit, but the run ENDS at the wall and only SCORED candidates survive — an unscored build in flight when the wall hits is lost. Call `time_left` anytime to see how much is left (the remaining time is also shown in your re-orientation notes). Once a correct baseline is scored, the long structural rewrite onto the newest architecture-specific features is the MAIN EVENT, not a detour — do not retreat to grinding safe incremental margins to pad the score. Everything you write under the workspace (notes, sources, validated standalone test kernels, backups) persists across turns and context resets, and the search always keeps your confirmed best, so nothing you build is ever lost and the rewrite is pure upside. Pace it against the wall: get each validated build to one scored `evaluate full` before the wall so it counts — use `time_left` to pace the build, never to justify a safe retreat.",
        cfg.problem_name,
        cfg.hardware,
        baseline_line,
        SOLUTION_DIR,
        solution_entrypoint(),
        TRUSTED_DIR,
        render_wall_budget(cfg.max_wall_clock_secs),
    )
}

/// Human-readable wall-clock budget for the kickoff (`~1000h`, `~650h`, `~75h`).
/// Coarser than [`crate::tool::time_left::render_time_left`] — the kickoff states
/// the total; the tool reports the live remainder. Dilated by
/// [`crate::tool::time_left::DISPLAYED_TIME_DILATION`] to match the tool (see
/// that constant): the displayed budget is inflated so the agent's human
/// time-intuition maps to its real throughput; the run still ends at the TRUE
/// wall.
fn render_wall_budget(secs: u64) -> String {
    let secs = secs.saturating_mul(crate::tool::time_left::DISPLAYED_TIME_DILATION);
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    match (h, m) {
        (0, m) => format!("{m}m"),
        (h, 0) => format!("{h}h"),
        (h, m) => format!("{h}h {m}m"),
    }
}

pub(super) fn render_skills_instructions(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }

    let catalog = skills
        .iter()
        .map(|skill| {
            format!(
                "- {}: {} (file: {})",
                skill.name,
                skill.description,
                skill.sandbox_skill_md()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "<skills_instructions>\n\
Repo-local skills are read-only instruction/assets bundles mounted in the sandbox. Available skills:\n\
{catalog}\n\n\
If a listed skill is relevant to the task, read its full `SKILL.md` via the existing `read` tool before applying it. Resolve files referenced by a skill relative to that skill directory and read them from the corresponding `/workspace/skills/...` path. Do not modify skill files.\n\
</skills_instructions>\n\n"
    )
}
