//! The single-lineage admission rule — the paper's monotonic hill-climb, made
//! statistically rigorous for a noisy, thermally-drifting single machine.
//!
//! Promotion is a two-stage ranking-and-selection decision (cheap screen, then
//! a rigorous confirmation), defending against the three failure modes that
//! actually corrupt a multi-day search on one GPU:
//!
//! 1. **Measurement noise** — never decide on a single timing; the confirmation
//!    takes repeated samples and tests a confidence interval on the *difference*.
//! 2. **Winner's curse / selection bias** — taking the max over hundreds of
//!    noisy candidates drifts the recorded best upward. A confirmation re-run
//!    (the "selected" challenger must prove itself again) defeats this.
//! 3. **Thermal/temporal drift** — the champion was timed earlier, the
//!    challenger now. The confirmation re-times *both, back-to-back* (paired) so
//!    machine drift cancels out of the comparison.
//!
//! The [`MIN_IMPROVEMENT_FRAC`] floor is the ranking-and-selection
//! *indifference zone* δ\*: differences smaller than it aren't worth detecting,
//! and (given that a false *promotion* is sticky and stalls the hill-climb,
//! while a false *rejection* just makes the agent retry) a conservative floor is
//! exactly the right bias.
//!
//! Stage 1 ([`is_new_global_best`]) is the cheap screen run on the single full
//! eval a candidate already has. Stage 2 ([`confirm_improvement`]) is the paired,
//! repeated-sample confirmation. For `k = 2` (single lineage) this is the
//! specialization of OCBA / the Kim–Nelson indifference-zone procedure; the
//! general `k > 2` OCBA budget allocator is reserved for the future
//! tree/beam frontier (see `avo_parallelization_design.md`).
//!
//! AVO exposes this admission rule as policy-owned accepted-best state on top of
//! the session tree; runner context is selected by `TurnNodeId`.

use crate::domain::convert::{u64_to_f64_lossy, usize_to_f64_lossy};
use crate::domain::types::{Evaluation, fileset_sha256};
use crate::exec::turn_tree::TurnNodeId;
use crate::orchestrator::search_tree::SearchTree;

/// Stable policy identifier recorded in the run manifest.
pub const POLICY_AVO: &str = "avo";
pub const POLICY_BEAM: &str = "beam";
/// UCB tree-search policy: beam's frontier machinery, but node selection is a
/// UCB1 confidence bound over a broad candidate pool instead of top-k by score.
pub const POLICY_UCB: &str = "ucb";
/// Backward-compatible alias for call sites/tests that still say strategy.
pub const STRATEGY_ID: &str = POLICY_AVO;

/// One scheduled unit of search work.
///
/// The policy emits actions; the orchestrator
/// maps each to a resource: an [`Expand`](Action::Expand) needs an agent worker
/// (an episode of turns from that node, plus its eventual evaluation), while a
/// [`Reevaluate`](Action::Reevaluate) needs only an `GpuPool` slot (a re-timing
/// of an already-materialized node, no LLM turn). Splitting them lets a scheduler
/// spend cheap re-measurements without tying up agents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Run an agent episode expanding this node.
    Expand(TurnNodeId),
    /// Re-time an existing node to shrink its confidence interval. The execution
    /// wiring lands in Stage 2 (confirmation lifted out of the `submit` tool);
    /// today no policy emits this variant.
    Reevaluate(TurnNodeId),
}

/// The search policy: an action-emitting scheduler over the session tree.
///
/// Not `select -> one node` (that is AVO-brained). A policy decides, at each
/// scorable boundary, what work to run next — expansions of frontier nodes and/or
/// re-evaluations of existing ones — up to a resource `budget`. This shape fits
/// linear AVO (width-1), beam (top-k), and — later — MCTS/OCBA, whose selection is
/// a stateful traversal or a mixed expand/re-evaluate budget allocation rather
/// than a flat ranking.
///
/// The policy only *reads* [`SearchTree`] views (frontier, tree, current leaf);
/// it performs no git/IO. The store stays the snapshot/accepted authority.
pub trait SearchPolicy {
    /// What to run now, up to `budget` resource slots. Empty ⇒ converged/stop.
    fn schedule(&mut self, budget: usize) -> Vec<Action>;
    /// Integrate a freshly evaluated (or re-evaluated) node: insert / re-rank /
    /// backprop / cull, as the policy sees fit.
    fn observe(&mut self, node: &TurnNodeId, eval: &Evaluation);
    /// Stable identifier recorded in the run manifest.
    fn id(&self) -> &'static str;
}

/// Beam-diversity configuration for UCB selection (the frontier-collapse fix).
///
/// `dedup` never places two byte-identical kernels (same [`fileset_sha256`]) in
/// the same expansion round. Applied inside [`BeamUcb::schedule`].
#[derive(Clone)]
pub struct DiversityConfig {
    pub dedup: bool,
}

impl DiversityConfig {
    /// Diversity fully off — the pre-diversity UCB behavior. Used by tests and
    /// the beam/avo policies (which ignore diversity).
    #[must_use]
    pub const fn disabled() -> Self {
        Self { dedup: false }
    }
}

/// Build the policy for a run from its manifest `policy_id`. Unknown ids fall back
/// to AVO (the caller has already run [`validate_policy`] at startup).
#[must_use]
pub fn make_policy(
    policy_id: &str,
    search_tree: SearchTree,
    beam_width: usize,
    _local_cap: usize,
    ucb_c: f64,
    diversity: DiversityConfig,
) -> Box<dyn SearchPolicy> {
    match policy_id {
        // beam and ucb are one policy: a scored search over ALL nodes. beam is
        // just UCB with no exploration term (`c = 0` ⇒ pure exploitation, top-K by
        // score).
        POLICY_BEAM | POLICY_UCB => {
            let c = if policy_id == POLICY_UCB {
                if ucb_c > 0.0 { ucb_c } else { UCB_C_DEFAULT }
            } else {
                0.0
            };
            Box::new(BeamUcb {
                search_tree,
                width: beam_width.max(1),
                c,
                visits: std::collections::BTreeMap::new(),
                total_visits: 0,
                pending_leader: None,
                diversity,
                sha_cache: std::collections::BTreeMap::new(),
            })
        }
        _ => Box::new(AvoPolicy {
            search_tree,
            pending_leader: None,
        }),
    }
}

/// Linear AVO — the documented **beam-width-1** hill-climb.
///
/// `schedule` expands the
/// latest node (linear, no fork), except when a challenger has out-scored the
/// confirmed best on the Stage-1 screen: then it schedules the paired
/// `Reevaluate(challenger)` + `Reevaluate(incumbent)` confirmation before the
/// challenger can be crowned (winner's-curse/drift defense, formerly inside the
/// `submit` tool). `observe` runs the screen; the loop runs the re-timing and, on
/// a clear win, records the confirmation that makes the challenger the new best.
pub struct AvoPolicy {
    pub search_tree: SearchTree,
    /// A node whose raw geomean cleared the Stage-1 screen over the confirmed best
    /// and is awaiting paired confirmation. `schedule` drains it into `Reevaluate`s.
    pending_leader: Option<TurnNodeId>,
}

impl SearchPolicy for AvoPolicy {
    fn schedule(&mut self, _budget: usize) -> Vec<Action> {
        // A pending challenger takes priority: confirm it (paired re-timing) before
        // it can be crowned. Order is [challenger, incumbent] — the loop re-times
        // the first against the second.
        if let Some(leader) = self.pending_leader.take()
            && let Some(incumbent) = self.search_tree.current_best().map(|b| b.node_id)
            && incumbent != leader
        {
            return vec![Action::Reevaluate(leader), Action::Reevaluate(incumbent)];
        }
        // Otherwise expand the current linear leaf (no-op fork for AVO).
        self.search_tree
            .current_node_id()
            .map(Action::Expand)
            .into_iter()
            .collect()
    }

    fn observe(&mut self, node: &TurnNodeId, eval: &Evaluation) {
        // Stage-1 screen: a Timed candidate that clears the confirmed best by the
        // noise/floor bar becomes the pending challenger to confirm.
        if let Evaluation::Timed {
            geomean_speedup,
            noise_margin,
            ..
        } = eval
        {
            let best_g = self
                .search_tree
                .current_best()
                .and_then(|b| b.geomean_speedup)
                .unwrap_or(0.0);
            if is_new_global_best(Some(*geomean_speedup), true, *noise_margin, best_g) {
                self.pending_leader = Some(node.clone());
            }
        }
    }

    fn id(&self) -> &'static str {
        POLICY_AVO
    }
}

/// Default exploration weight `c` when `--ucb-c` is unset/zero.
///
/// With the
/// exploitation term normalized to `[0, 1]` (geomean / best), `0.5` keeps the
/// search exploiting the strong lineages while still re-broadening onto
/// under-visited competitive nodes — the frontier-collapse fix.
pub const UCB_C_DEFAULT: f64 = 0.5;

/// UCB1 acquisition value for one candidate.
///
/// The exploitation term is the node's
/// geomean normalized to the current best (so it lives in `(0, 1]` and the
/// constant `c` behaves consistently as absolute speedups grow over the run);
/// the exploration term is the standard confidence bonus that decays as a node
/// is expanded more often.
#[must_use]
pub fn ucb_value(geomean: f64, best_geomean: f64, visits: u64, total: u64, c: f64) -> f64 {
    let exploit = if best_geomean > 0.0 {
        geomean / best_geomean
    } else {
        0.0
    };
    let explore = c * (u64_to_f64_lossy(total).ln_1p() / (u64_to_f64_lossy(visits) + 1.0)).sqrt();
    exploit + explore
}

/// The unified branching search — `beam` and `ucb` are the same policy, differing
/// only in one score term. Each scheduling round scores **every** `Timed` node by
///
/// ```text
/// score = geomean/best  +  c·√(ln T / n)
///         └ exploit ┘     └── explore ──┘
/// ```
///
/// and greedily takes the K best. `c = 0` ⇒ `beam` (pure exploitation, top-K by
/// score); `c > 0` ⇒ `ucb` (exploration bonus pulls budget back to under-visited,
/// often structurally different branches). No bounded frontier: dominated nodes
/// never win, so the whole tree is the candidate set. Reuses the same paired
/// confirmation gate as AVO.
pub struct BeamUcb {
    pub search_tree: SearchTree,
    pub width: usize,
    /// Exploration weight `c` in the confidence bonus.
    pub c: f64,
    /// Lifetime selection (expansion) count per node — the UCB visit count `n`.
    visits: std::collections::BTreeMap<String, u64>,
    /// Total selections `T` so far (the confidence-bound numerator).
    total_visits: u64,
    /// A challenger awaiting paired confirmation (same role as beam/AVO).
    pending_leader: Option<TurnNodeId>,
    /// Beam-diversity settings (frontier-collapse fix).
    diversity: DiversityConfig,
    /// Cache: `node_id` → its solution content hash ([`fileset_sha256`]). Bounds
    /// git reads to once per node over the run.
    sha_cache: std::collections::BTreeMap<String, String>,
}

impl BeamUcb {
    /// The content hash of a candidate's solution files (its tier-1 dedup key),
    /// cached per node. `None` if the node's workspace can't be read.
    fn content_sha(&mut self, id: &str) -> Option<String> {
        if let Some(s) = self.sha_cache.get(id) {
            return Some(s.clone());
        }
        let files = self.search_tree.read_solution_at(&TurnNodeId::new(id)).ok()?;
        let sha = fileset_sha256(&files);
        self.sha_cache.insert(id.to_string(), sha.clone());
        Some(sha)
    }
}

impl SearchPolicy for BeamUcb {
    fn schedule(&mut self, budget: usize) -> Vec<Action> {
        // Same paired-confirmation gate as beam/AVO: confirm a pending challenger
        // (re-time challenger vs incumbent) before it can be crowned best.
        if let Some(leader) = self.pending_leader.take()
            && let Some(incumbent) = self.search_tree.current_best().map(|b| b.node_id)
            && incumbent != leader
        {
            return vec![Action::Reevaluate(leader), Action::Reevaluate(incumbent)];
        }
        let width = self.width.max(1);
        let batch = budget.max(1).min(width);
        // Candidate pool: all Timed nodes (score-desc), capped at `pool_size`.
        let ranked = self.search_tree.ranked_candidates();
        if ranked.is_empty() {
            return Vec::new();
        }
        let best_g = ranked.first().map_or(1.0, |c| c.geomean);
        // Candidate set = EVERY `Timed` node in the tree — no cap, no floor. The
        // score fn ranks the whole set and we take the K best; dominated nodes
        // simply never win, so pruning is unnecessary.
        //
        // Diversity pre-pass: resolve each pool candidate's content hash (the
        // dedup key). Done before the selection loop so the loop borrows `self`
        // only immutably. Skipped entirely when dedup is off.
        let dedup = self.diversity.dedup;
        let pool: Vec<(String, f64, Option<String>)> = ranked
            .iter()
            .map(|c| {
                let id = c.node_id.clone();
                let sha = if dedup { self.content_sha(&id) } else { None };
                (id, c.geomean, sha)
            })
            .collect();
        // Pick `batch` distinct nodes by descending UCB value. Incrementing the
        // visit counts as we go both records the selection and (via `chosen`) keeps
        // one round's expanders on different nodes; dedup additionally keeps them on
        // different *kernels* (distinct content hash).
        let mut actions = Vec::with_capacity(batch);
        let mut chosen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut chosen_shas: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        while actions.len() < batch {
            let value = |id: &str, geomean: f64| {
                ucb_value(
                    geomean,
                    best_g,
                    *self.visits.get(id).unwrap_or(&0),
                    self.total_visits,
                    self.c,
                )
            };
            let pick = pool
                .iter()
                .filter(|(id, _, sha)| {
                    // never expand a kernel identical to one already chosen this round.
                    !(chosen.contains(id) || dedup && sha.as_ref().is_some_and(|s| chosen_shas.contains(s)))
                })
                .max_by(|a, b| {
                    value(&a.0, a.1)
                        .partial_cmp(&value(&b.0, b.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _, sha)| (id.clone(), sha.clone()));
            match pick {
                Some((id, sha)) => {
                    let visits = self.visits.entry(id.clone()).or_insert(0);
                    *visits = visits.saturating_add(1);
                    self.total_visits = self.total_visits.saturating_add(1);
                    chosen.insert(id.clone());
                    if let Some(s) = sha {
                        chosen_shas.insert(s);
                    }
                    actions.push(Action::Expand(TurnNodeId::new(id)));
                }
                // Pool smaller than the batch: run what we have this round.
                None => break,
            }
        }
        actions
    }

    fn observe(&mut self, node: &TurnNodeId, eval: &Evaluation) {
        // Same Stage-1 screen as beam/AVO.
        if let Evaluation::Timed {
            geomean_speedup,
            noise_margin,
            ..
        } = eval
        {
            let best_g = self
                .search_tree
                .current_best()
                .and_then(|b| b.geomean_speedup)
                .unwrap_or(0.0);
            if is_new_global_best(Some(*geomean_speedup), true, *noise_margin, best_g) {
                self.pending_leader = Some(node.clone());
            }
        }
    }

    fn id(&self) -> &'static str {
        // beam is the c==0 (no-exploration) parameterization of this same policy.
        if self.c > 0.0 { POLICY_UCB } else { POLICY_BEAM }
    }
}

/// Reject an unsupported `--policy` id before the run starts.
///
/// # Errors
///
/// Fails when `policy` is not one of `avo`, `beam`, or `ucb`; the message lists
/// the supported ids.
pub fn validate_policy(policy: &str) -> Result<(), String> {
    if matches!(policy, POLICY_AVO | POLICY_BEAM | POLICY_UCB) {
        Ok(())
    } else {
        Err(format!(
            "unsupported search policy {policy:?}; supported policies: avo, beam, ucb"
        ))
    }
}

#[must_use]
pub fn is_beam(policy: &str) -> bool {
    policy == POLICY_BEAM
}

#[must_use]
pub fn is_ucb(policy: &str) -> bool {
    policy == POLICY_UCB
}

/// Whether the policy is a branching frontier search (beam or ucb) — as opposed
/// to linear AVO. Used by the loop for beam-style wiring (episode backstop) that
/// both frontier policies share.
#[must_use]
pub fn uses_beam_frontier(policy: &str) -> bool {
    matches!(policy, POLICY_BEAM | POLICY_UCB)
}

/// Minimum fractional improvement a promotion must show, whatever the noise.
///
/// Floor on the fractional improvement a strict-beat demands, regardless of
/// measured timing noise — the ranking-and-selection *indifference zone* δ\*, so
/// a vanishingly small (and likely spurious) win cannot be promoted as a new
/// global best. Mirrored into the run manifest's `params`.
pub const MIN_IMPROVEMENT_FRAC: f64 = 0.01;

/// One-sided normal quantile for the confirmation confidence interval
/// (≈95%: z₀.₀₅ = 1.645).
///
/// The challenger is promoted only when the *lower* bound
/// of the improvement interval clears the indifference zone — i.e. we are
/// ~95% confident the true improvement exceeds the floor, not merely that the
/// point estimate does.
pub const CONFIRM_Z: f64 = 1.645;

/// Whether a candidate is a correct, strict, noise-clearing improvement over the
/// current best — the cheap Stage-1 screen run on the single full eval a
/// candidate already has.
///
/// A candidate must beat `best_score` by more than the larger of
/// the candidate's measured `noise_margin` and the floor; an incorrect or
/// unscored (shallow-stage) candidate is never a best. Passing this screen only
/// makes a candidate *eligible* for the Stage-2 [`confirm_improvement`] test
/// (when confirmation is enabled); it is not by itself a promotion.
#[must_use]
pub fn is_new_global_best(candidate_score: Option<f64>, correct: bool, noise_margin: f64, best_score: f64) -> bool {
    let margin = noise_margin.max(MIN_IMPROVEMENT_FRAC);
    correct && candidate_score.is_some_and(|s| s > best_score * (1.0 + margin))
}

/// Outcome of the Stage-2 paired confirmation test.
#[derive(Debug, Clone)]
pub struct Confirmation {
    /// Promote? True iff the lower CI bound on the improvement clears the floor.
    pub promote: bool,
    /// Point estimate of the challenger-vs-champion improvement (e.g. `0.08` =
    /// +8%), from the ratio of the two geometric-mean speedups.
    pub point_improvement: f64,
    /// Lower bound of the one-sided confidence interval on that improvement.
    pub ci_low_improvement: f64,
    /// Indifference zone the lower bound had to clear (the floor).
    pub min_effect: f64,
    /// Re-measured geometric-mean speedup of the challenger and champion (paired,
    /// same thermal window), for the human-facing explanation.
    pub challenger_geomean: f64,
    pub champion_geomean: f64,
    /// Usable (scored) sample count actually used on the smaller side.
    pub samples: usize,
}

/// Paired confirmation test on repeated geomean-speedup samples of the
/// challenger and the (re-timed) champion.
///
/// The work is done in **log space**:
/// the geometric mean is the exponential of the mean log-speedup, so a
/// difference of mean logs is the log of the speedup *ratio* — exactly the
/// drift-cancelled challenger-vs-champion improvement (both sides divide by the
/// same frozen reference, so the ratio is `champion_latency / challenger_latency`).
///
/// Returns a [`Confirmation`] whose `promote` is true iff the lower bound of the
/// `z`-confidence interval on the log-ratio exceeds `ln(1 + min_effect)` — i.e.
/// we are confident the *true* improvement clears the indifference zone, not
/// just the noisy point estimate.
///
/// Both slices must already be filtered to scored, correct samples. With fewer
/// than two usable samples on either side the test is *inconclusive*
/// (`promote = false`, `samples` reports the shortfall); the caller decides how
/// to treat an inconclusive result (the orchestrator falls back to the Stage-1
/// screen so a flaky re-eval never blocks an otherwise-clear win).
#[must_use]
pub fn confirm_improvement(challenger: &[f64], champion: &[f64], min_effect: f64, z: f64) -> Confirmation {
    let lc: Vec<f64> = challenger.iter().filter(|x| **x > 0.0).map(|x| x.ln()).collect();
    let lh: Vec<f64> = champion.iter().filter(|x| **x > 0.0).map(|x| x.ln()).collect();
    let samples = lc.len().min(lh.len());
    if lc.len() < 2 || lh.len() < 2 {
        return Confirmation {
            promote: false,
            point_improvement: 0.0,
            ci_low_improvement: 0.0,
            min_effect,
            challenger_geomean: lc.first().map_or(0.0, |m| m.exp()),
            champion_geomean: lh.first().map_or(0.0, |m| m.exp()),
            samples,
        };
    }
    let mc = mean(&lc);
    let mh = mean(&lh);
    let se = (sample_var(&lc, mc) / usize_to_f64_lossy(lc.len()) + sample_var(&lh, mh) / usize_to_f64_lossy(lh.len()))
        .sqrt();
    let diff = mc - mh; // log of the speedup ratio
    let ci_low_log = z.mul_add(-se, diff);
    let threshold_log = min_effect.ln_1p();
    Confirmation {
        promote: ci_low_log > threshold_log,
        point_improvement: diff.exp_m1(),
        ci_low_improvement: ci_low_log.exp_m1(),
        min_effect,
        challenger_geomean: mc.exp(),
        champion_geomean: mh.exp(),
        samples,
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / usize_to_f64_lossy(v.len())
}

/// Unbiased (n−1) sample variance; `0.0` for fewer than two samples.
fn sample_var(v: &[f64], m: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    // `len() >= 2` from the guard above, so the saturating step is exact.
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / usize_to_f64_lossy(v.len().saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotes_only_strict_beats_clearing_the_noise_floor() {
        // +0.5% does not clear the 1% floor when measured noise is tiny.
        assert!(
            !is_new_global_best(Some(1.005), true, 0.0, 1.0),
            "+0.5% must not clear the 1% floor"
        );
        // +2% clears the 1% floor.
        assert!(
            is_new_global_best(Some(1.02), true, 0.0, 1.0),
            "+2% clears the 1% floor"
        );
        // A noise margin above the floor dominates.
        assert!(
            !is_new_global_best(Some(1.04), true, 0.05, 1.0),
            "+4% must not clear a 5% noise margin"
        );
        assert!(
            is_new_global_best(Some(1.06), true, 0.05, 1.0),
            "+6% clears a 5% noise margin"
        );
        // Equal is not a strict beat.
        assert!(
            !is_new_global_best(Some(1.0), true, 0.0, 1.0),
            "equal must not be promoted"
        );
        // Incorrect is never a best, however fast.
        assert!(
            !is_new_global_best(Some(2.0), false, 0.0, 1.0),
            "incorrect is never promoted"
        );
        // No score (a shallow stage produced no timing) is never a best.
        assert!(
            !is_new_global_best(None, true, 0.0, 1.0),
            "an unscored candidate is never promoted"
        );
    }

    #[test]
    fn policy_ids_validate_and_classify() {
        assert!(validate_policy("avo").is_ok());
        assert!(validate_policy("beam").is_ok());
        assert!(validate_policy("ucb").is_ok());
        assert!(validate_policy("mcts").is_err());
        assert!(is_beam(POLICY_BEAM));
        assert!(!is_beam(POLICY_AVO));
        assert!(is_ucb(POLICY_UCB));
        assert!(uses_beam_frontier(POLICY_BEAM));
        assert!(uses_beam_frontier(POLICY_UCB));
        assert!(!uses_beam_frontier(POLICY_AVO));
    }

    #[test]
    fn confirm_promotes_a_clear_tight_win() {
        // Challenger ~+10% with tight, repeatable samples; champion tight at ~1.0.
        let challenger = [1.10, 1.11, 1.09, 1.10];
        let champion = [1.00, 1.01, 0.99, 1.00];
        let c = confirm_improvement(&challenger, &champion, MIN_IMPROVEMENT_FRAC, CONFIRM_Z);
        assert!(c.promote, "a tight +10% win must clear the CI lower bound: {c:?}");
        assert!(
            c.ci_low_improvement > MIN_IMPROVEMENT_FRAC,
            "lower bound must exceed the floor: {c:?}"
        );
        assert!(
            (c.point_improvement - 0.10).abs() < 0.02,
            "point improvement ~+10%: {c:?}"
        );
    }

    #[test]
    fn confirm_rejects_a_within_noise_difference() {
        // Means are essentially equal; the apparent edge is pure jitter.
        let challenger = [1.01, 0.99, 1.02, 0.98];
        let champion = [1.00, 1.01, 0.99, 1.00];
        let c = confirm_improvement(&challenger, &champion, MIN_IMPROVEMENT_FRAC, CONFIRM_Z);
        assert!(!c.promote, "a within-noise difference must not promote: {c:?}");
    }

    #[test]
    fn confirm_rejects_a_high_variance_apparent_win() {
        // High point mean but huge variance -> the CI lower bound dips below the
        // floor, so we cannot be confident the true gain clears the indifference
        // zone (defeats the winner's curse on a lucky-but-noisy candidate).
        let challenger = [1.30, 0.90, 1.40, 0.80];
        let champion = [1.00, 1.00, 1.00, 1.00];
        let c = confirm_improvement(&challenger, &champion, MIN_IMPROVEMENT_FRAC, CONFIRM_Z);
        assert!(!c.promote, "a noisy apparent win must not clear the CI bound: {c:?}");
    }

    #[test]
    fn confirm_rejects_a_real_but_sub_floor_win() {
        // A tight, real +0.5% win — statistically significant, but below the 1%
        // indifference zone, so it is deliberately not worth promoting.
        let challenger = [1.005, 1.005, 1.004, 1.006];
        let champion = [1.000, 1.000, 1.000, 1.000];
        let c = confirm_improvement(&challenger, &champion, MIN_IMPROVEMENT_FRAC, CONFIRM_Z);
        assert!(!c.promote, "a sub-floor win must not promote even if real: {c:?}");
    }

    #[test]
    fn confirm_is_inconclusive_with_too_few_samples() {
        let c = confirm_improvement(&[1.10], &[1.00], MIN_IMPROVEMENT_FRAC, CONFIRM_Z);
        assert!(!c.promote, "fewer than two samples is inconclusive, not a promotion");
        assert!(c.samples < 2, "reports the sample shortfall: {c:?}");
    }

    #[test]
    fn ucb_exploration_bonus_favors_the_under_visited() {
        // Two candidates, equal score. The one expanded fewer times must win —
        // that is the anti-collapse behavior.
        let hi_visits = ucb_value(25.0, 25.0, 20, 100, UCB_C_DEFAULT);
        let lo_visits = ucb_value(25.0, 25.0, 1, 100, UCB_C_DEFAULT);
        assert!(
            lo_visits > hi_visits,
            "fewer visits ⇒ higher UCB: {lo_visits} vs {hi_visits}"
        );
    }

    #[test]
    fn ucb_exploits_a_clearly_better_node_when_visits_are_equal() {
        // With equal visit counts the confidence bonus cancels, so the higher
        // (normalized) score wins — UCB still exploits.
        let strong = ucb_value(25.0, 25.0, 3, 100, UCB_C_DEFAULT);
        let weak = ucb_value(13.0, 25.0, 3, 100, UCB_C_DEFAULT);
        assert!(strong > weak, "equal visits ⇒ higher score wins: {strong} vs {weak}");
    }

    #[test]
    fn ucb_bonus_eventually_rescues_a_much_weaker_unvisited_node() {
        // A strong node expanded many times can be overtaken by a decent,
        // never-expanded node — the re-broadening the collapse fix relies on.
        let overworked_best = ucb_value(25.0, 25.0, 40, 300, UCB_C_DEFAULT);
        let fresh_contender = ucb_value(20.0, 25.0, 0, 300, UCB_C_DEFAULT);
        assert!(
            fresh_contender > overworked_best,
            "{fresh_contender} vs {overworked_best}"
        );
    }
}
