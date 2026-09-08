use super::*;
use crate::orchestrator::selftest::{ScriptedClient, ScriptedMessage};
use crate::orchestrator::session::render_skills_instructions;
use tempfile::TempDir;

/// An empty search tree in a throwaway dir: `current_node_id()` is `None`, so
/// a resume falls back to the fresh kickoff. The `TempDir` must outlive the
/// store, so callers bind it.
fn empty_search_tree() -> (TempDir, SearchTree) {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    (dir, store)
}

fn test_skill() -> Skill {
    Skill {
        name: "cuda-tuning".to_string(),
        description: "CUDA tuning instructions".to_string(),
        host_dir: PathBuf::from("/repo/agent_ressources/skills/cuda-tuning"),
        sandbox_dir: "/workspace/skills/cuda-tuning".to_string(),
    }
}

fn test_config(resume: bool, skills: Vec<Skill>) -> AvoConfig {
    AvoConfig {
        run_dir: PathBuf::from("runs/test"),
        problem_name: "attention".to_string(),
        model_name: "test-model".to_string(),
        policy_id: strategy::POLICY_AVO.to_string(),
        beam_width: 1,
        beam_local_cap: 1,
        ucb_c: 0.0,
        python: "python3".to_string(),
        hardware: "test hardware".to_string(),
        max_wall_clock_secs: 21_600,
        resume,
        seed_src: SEED_STUB.to_string(),
        eval_timeout_secs: 1,
        seed: None,
        supervisor_model: None,
        diversity_dedup: false,
        context_window_tokens: 1_000_000,
        skills,
    }
}

fn test_best() -> BestInfo {
    BestInfo {
        node_id: crate::exec::turn_tree::TurnNodeId::new("n0"),
        geomean_speedup: Some(1.0),
        confirmed: true,
        solution_sha256: "test-sha".to_string(),
    }
}

#[test]
fn skills_catalog_uses_sandbox_paths_only() {
    let rendered = render_skills_instructions(&[test_skill()]);
    assert!(rendered.contains("<skills_instructions>"));
    assert!(
        rendered.contains("- cuda-tuning: CUDA tuning instructions (file: /workspace/skills/cuda-tuning/SKILL.md)")
    );
    assert!(rendered.contains("read-only instruction/assets bundles"));
    assert!(rendered.contains("existing `read` tool"));
    assert!(!rendered.contains("/repo/agent_ressources"));
}

#[test]
fn skills_catalog_is_omitted_when_empty() {
    assert_eq!(render_skills_instructions(&[]), "");
}

#[test]
fn fresh_session_injects_skills_catalog() {
    let client = ScriptedClient::new();
    let cfg = test_config(false, vec![test_skill()]);
    let (_tmp, search_tree) = empty_search_tree();
    let (messages, _) = load_or_init_session(
        &client,
        &cfg,
        &search_tree,
        Some(&test_best()),
        Seeds::from_base(1),
        None,
    );
    let ScriptedMessage::User(text) = &messages[0] else {
        panic!("expected first fresh message to be user kickoff");
    };
    assert!(text.contains("<skills_instructions>"));
    assert!(text.contains("/workspace/skills/cuda-tuning/SKILL.md"));
}

#[test]
fn resume_session_does_not_refresh_skills_catalog() {
    let client = ScriptedClient::new();
    let cfg = test_config(true, vec![test_skill()]);
    let (_tmp, search_tree) = empty_search_tree();
    let (messages, _) = load_or_init_session(
        &client,
        &cfg,
        &search_tree,
        Some(&test_best()),
        Seeds::from_base(1),
        None,
    );
    let ScriptedMessage::User(text) = &messages[0] else {
        panic!("expected fallback kickoff message");
    };
    assert!(!text.contains("<skills_instructions>"));
    assert!(!text.contains("/workspace/skills/cuda-tuning/SKILL.md"));
}

/// Append a node with solution `{solution.py: body}` and confirm it at `geomean`;
/// return its id. Enough to make `current_best()` non-empty for anchor tests.
fn append_confirmed(store: &SearchTree, turn: u64, body: &str, geomean: f64) -> String {
    let ws = TempDir::new().unwrap();
    std::fs::write(ws.path().join("solution.py"), body).unwrap();
    let id = store
        .append_typed_turn_node(
            turn,
            1,
            vec![serde_json::json!({ "assistant": body })],
            ws.path(),
            true,
            None,
            None,
        )
        .unwrap();
    store
        .record_confirmation(&crate::exec::turn_tree::TurnNodeId::new(id.clone()), geomean, 3)
        .unwrap();
    id
}

/// Like [`append_confirmed`] but parents the node at an explicit `parent`, so
/// tests can build FORKS (sibling branches) rather than one linear chain.
fn append_confirmed_child(store: &SearchTree, turn: u64, body: &str, geomean: f64, parent: &str) -> String {
    let ws = TempDir::new().unwrap();
    std::fs::write(ws.path().join("solution.py"), body).unwrap();
    let parent_id = crate::exec::turn_tree::TurnNodeId::new(parent.to_string());
    let id = store
        .append_typed_turn_node_from(
            Some(&parent_id),
            turn,
            1,
            vec![serde_json::json!({ "assistant": body })],
            ws.path(),
            true,
            None,
            None,
        )
        .unwrap();
    store
        .record_confirmation(&crate::exec::turn_tree::TurnNodeId::new(id.clone()), geomean, 3)
        .unwrap();
    id
}

#[test]
fn anchor_flags_on_disk_matches_best() {
    let (_tmp, store) = empty_search_tree();
    let id = append_confirmed(&store, 1, "print('A')", 2.0);
    let node = crate::exec::turn_tree::TurnNodeId::new(id);
    // On-disk sha == the lineage best's sha ⇒ "matches".
    let best_sha = store.lineage_best(&node).unwrap().solution_sha256;
    let anchor = ground_truth_anchor(&store, Some(&node), Some(&best_sha));
    assert!(anchor.contains("GROUND TRUTH"));
    assert!(anchor.contains("Best confirmed on your lineage: geomean 2.0000x"));
    assert!(anchor.contains("On disk NOW: your working `solution/` matches the best confirmed on your lineage"));
    assert!(!anchor.contains("is NOT that best"));
}

#[test]
fn anchor_flags_on_disk_differs_from_best() {
    let (_tmp, store) = empty_search_tree();
    let id = append_confirmed(&store, 1, "print('A')", 2.0);
    let node = crate::exec::turn_tree::TurnNodeId::new(id);
    // Sandbox holds a different sha than the lineage best (in-progress/reverted edit).
    let anchor = ground_truth_anchor(
        &store,
        Some(&node),
        Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
    );
    assert!(anchor.contains("Best confirmed on your lineage: geomean 2.0000x"));
    // Framed as an EXPECTED state (not a "discrepancy").
    assert!(anchor.contains("is NOT that best"));
    assert!(anchor.contains("EXPECTED"));
    assert!(anchor.contains("search_view"));
    assert!(!anchor.contains("matches the best confirmed"));
    // Reassures the best is recorded/kept safe, and points the agent at the
    // `checkout` tool to recover its source (solution/ + artifacts/) rather than
    // re-deriving it (N2 + I10).
    assert!(anchor.contains("recorded automatically"));
    assert!(anchor.contains("checkout"));
    assert!(!anchor.contains("recoverable verbatim"));
}

#[test]
fn anchor_omits_on_disk_line_when_sandbox_unreadable() {
    let (_tmp, store) = empty_search_tree();
    let id = append_confirmed(&store, 1, "print('A')", 2.0);
    let node = crate::exec::turn_tree::TurnNodeId::new(id);
    // None (sandbox couldn't be read) ⇒ omit the on-disk line rather than guess.
    let anchor = ground_truth_anchor(&store, Some(&node), None);
    assert!(anchor.contains("Best confirmed on your lineage: geomean 2.0000x"));
    assert!(!anchor.contains("On disk NOW"));
}

/// Empty tree ⇒ no confirmed best ⇒ empty anchor (callers skip injection).
#[test]
fn anchor_is_empty_before_first_confirmation() {
    let (_tmp, store) = empty_search_tree();
    assert!(ground_truth_anchor(&store, None, None).is_empty());
}

/// Lineage scoping (the beam-search fix): the anchor for a node on one branch
/// must report THAT branch's best, never a sibling branch's higher score. This is
/// the `run_1784328572` confusion — a sibling's 13.08x leaking into an episode
/// working a weaker line.
#[test]
fn anchor_is_scoped_to_its_own_lineage_not_the_global_best() {
    let (_tmp, store) = empty_search_tree();
    // Branch A: root(1.5x) → child(3.0x). Branch B forks off root with a higher 9.0x.
    let root = append_confirmed(&store, 1, "print('root')", 1.5);
    let root_node = crate::exec::turn_tree::TurnNodeId::new(root.clone());
    let a_child = append_confirmed_child(&store, 2, "print('A2')", 3.0, &root);
    let sibling = append_confirmed_child(&store, 2, "print('B_fast')", 9.0, &root);
    let a_node = crate::exec::turn_tree::TurnNodeId::new(a_child);
    let sibling_node = crate::exec::turn_tree::TurnNodeId::new(sibling);

    // Global best is the 9.0x sibling; but branch A's lineage best is 3.0x.
    assert_eq!(store.current_best().unwrap().geomean_speedup, Some(9.0));
    assert_eq!(store.lineage_best(&a_node).unwrap().geomean_speedup, Some(3.0));
    assert_eq!(store.lineage_best(&sibling_node).unwrap().geomean_speedup, Some(9.0));
    assert_eq!(store.lineage_best(&root_node).unwrap().geomean_speedup, Some(1.5));

    // The anchor for a node on branch A shows 3.0x and NEVER the sibling's 9.0x.
    let anchor = ground_truth_anchor(&store, Some(&a_node), None);
    assert!(
        anchor.contains("Best confirmed on your lineage: geomean 3.0000x"),
        "{anchor}"
    );
    assert!(!anchor.contains("9.0000"), "sibling branch must not leak in: {anchor}");
}

/// Append a node carrying an explicit `Timed` eval (so it joins the ranked
/// candidate list `candidates_ranked` reads), optionally parented at `parent`.
/// Unlike [`append_confirmed`], the node is NOT promoted unless the caller then
/// calls `record_confirmation` — so a bare `append_timed` renders in the anchor's
/// "Tried & measured (NOT promoted)" section. Returns its id.
fn append_timed(store: &SearchTree, turn: u64, body: &str, geomean: f64, parent: Option<&str>) -> String {
    let ws = TempDir::new().unwrap();
    std::fs::write(ws.path().join("solution.py"), body).unwrap();
    let eval = crate::domain::types::Evaluation::Timed {
        stage: crate::domain::types::Stage::Full,
        geomean_speedup: geomean,
        noise_margin: 0.0,
        per_config: Vec::new(),
    };
    let parent_id = parent.map(|p| crate::exec::turn_tree::TurnNodeId::new(p.to_string()));
    store
        .append_typed_turn_node_from(
            parent_id.as_ref(),
            turn,
            1,
            vec![serde_json::json!({ "assistant": body })],
            ws.path(),
            true,
            None,
            Some(eval),
        )
        .unwrap()
}

/// Fix A (flagship) + B1: the anchor surfaces measured-but-unpromoted attempts as
/// a LABEL (not hidden behind the confirmed-only filter), and states the confirmed
/// best is BANKED. This is the `run_1784671694` case: a scored prefill below the
/// confirmed best was a non-confirmed `Timed` node on the lineage; the old anchor
/// hid it, so the agent re-derived it 4× and falsely concluded "integration was
/// never started".
///
/// Also pins the two correctness fixes found in review:
///  - a non-confirmed candidate ABOVE the confirmed best (a winner's-curse /
///    within-noise confirmation reject) must NOT be labelled "below your best";
///  - the node the agent is currently ON is excluded from the "resume from it" list.
#[test]
fn anchor_surfaces_measured_but_unpromoted_candidates() {
    let (_tmp, store) = empty_search_tree();
    // Lineage: best(3.0x, CONFIRMED) → above(4.0x, unconfirmed) → below(2.0x,
    // unconfirmed) → cur(2.5x, unconfirmed, the CURRENT node).
    let best = append_timed(&store, 1, "print('best')", 3.0, None);
    store
        .record_confirmation(&crate::exec::turn_tree::TurnNodeId::new(best.clone()), 3.0, 3)
        .unwrap();
    // A measured attempt that scored ABOVE the confirmed best but failed the paired
    // re-timing (or is still pending) — stays unconfirmed at its raw 4.0x.
    let above = append_timed(&store, 952, "print('above')", 4.0, Some(&best));
    // A measured attempt genuinely below the best.
    let below = append_timed(&store, 951, "print('below')", 2.0, Some(&above));
    // The node the episode is currently working on (also an unconfirmed Timed node).
    let cur = append_timed(&store, 953, "print('cur')", 2.5, Some(&below));
    let node = crate::exec::turn_tree::TurnNodeId::new(cur);

    let anchor = ground_truth_anchor(&store, Some(&node), None);
    // Best is still the confirmed 3.0x, framed as banked/safe (B1).
    assert!(
        anchor.contains("Best confirmed on your lineage: geomean 3.0000x"),
        "{anchor}"
    );
    assert!(anchor.contains("BANKED and cannot go down"), "{anchor}");
    // Confirmed section renders the promoted best.
    assert!(anchor.contains("Confirmed on your lineage (best first):"), "{anchor}");
    assert!(anchor.contains("turn 1: geomean 3.0000x"), "{anchor}");
    // Fix A: unpromoted attempts are surfaced (not hidden) and reframed toward RESUME.
    assert!(
        anchor.contains("Tried & measured on your lineage (NOT promoted"),
        "{anchor}"
    );
    assert!(anchor.contains("do NOT re-derive"), "{anchor}");
    assert!(anchor.contains("checkout"), "{anchor}");
    // Per-candidate label, computed against the confirmed best (NOT hardcoded):
    // the 4.0x node is flagged as at/above-best-but-unconfirmed, never "below".
    let above_line = anchor
        .lines()
        .find(|l| l.contains("turn 952: geomean 4.0000x"))
        .unwrap_or_else(|| panic!("above-best line missing:\n{anchor}"));
    assert!(above_line.contains("ABOVE your best"), "{above_line}");
    assert!(!above_line.contains("below your best"), "{above_line}");
    // ...and the genuinely-below node reads "below your best".
    let below_line = anchor
        .lines()
        .find(|l| l.contains("turn 951: geomean 2.0000x"))
        .unwrap_or_else(|| panic!("below-best line missing:\n{anchor}"));
    assert!(below_line.contains("below your best"), "{below_line}");
    // Current node is excluded from the "resume from it" list (nonsensical there).
    assert!(
        !anchor.contains("turn 953: geomean 2.5000x"),
        "current node must be excluded:\n{anchor}"
    );
    // Genericity: no CUDA/attention-kernel jargon leaks into this generic harness text.
    assert!(!anchor.contains("warp specialization"), "{anchor}");
    assert!(!anchor.contains("ping-pong"), "{anchor}");
}

/// Review fix: `render_invariants` is capped (`INVARIANTS_RENDER_CHARS`) so an
/// agent-writable note can't blow the compaction head over the window. Unlike
/// gotchas it keeps the HEAD (load-bearing flags are written first) and flags the
/// truncation so the agent trims the file back.
#[test]
fn invariants_over_cap_keep_head_and_flag_truncation() {
    let sandbox = Sandbox::new().unwrap();
    let head = "-flag-that-must-survive-verbatim";
    let body = format!("{head}\n{}", "y".repeat(INVARIANTS_RENDER_CHARS + 4_000));
    sandbox.write(INVARIANTS_PATH, &body).unwrap();
    let rendered = render_invariants(&sandbox).expect("invariants present");
    assert!(rendered.len() < body.len(), "over-cap body must be truncated");
    assert!(
        rendered.starts_with(head),
        "load-bearing head must be kept: {}",
        rendered.get(..80).unwrap_or(&rendered)
    );
    assert!(rendered.contains("must stay TINY"), "truncation marker present");
}

/// `fold_ledger` orders invariants → gotchas → doc index (read-first = most
/// load-bearing build facts; design memory last) and drops empty sides cleanly (no
/// stray blank separator). Pins the compaction-head ordering that
/// `brief_orders_preamble_anchor_gotchas_summary` structurally can't reach (all
/// three are pre-concatenated into the single `ledger` arg by the caller).
#[test]
fn fold_ledger_orders_invariants_gotchas_doc_index_and_drops_empties() {
    assert_eq!(fold_ledger("", "", ""), "");
    assert_eq!(fold_ledger("INV", "", ""), "INV");
    assert_eq!(fold_ledger("", "GOT", ""), "GOT");
    assert_eq!(fold_ledger("", "", "DOC"), "DOC");
    // A missing middle side collapses without leaving a doubled separator.
    assert_eq!(fold_ledger("INV", "", "DOC"), "INV\n\nDOC");
    let all = fold_ledger("INV", "GOT", "DOC");
    assert_eq!(all, "INV\n\nGOT\n\nDOC");
    assert!(
        all.find("INV").unwrap() < all.find("GOT").unwrap(),
        "invariants read first"
    );
    assert!(
        all.find("GOT").unwrap() < all.find("DOC").unwrap(),
        "doc index (design memory) read last"
    );
}

/// The doc-survey index is re-surfaced like the ledgers, HEAD-kept on overflow
/// (the most-relevant docs + flagged fast-path lead the file, so — unlike the
/// tail-kept gotchas ledger — the head is exactly what must survive), with a marker
/// pointing at the file for the rest. The reinject preamble names the file and stays
/// domain-agnostic (the survey convention is not tied to any one problem).
#[test]
fn doc_index_is_head_kept_on_overflow_and_preamble_is_generic() {
    let sandbox = Sandbox::new().unwrap();
    // Full body under cap: returned verbatim (trimmed), no marker.
    let small = "docs/nvidia/ptx.md · relevant y · PTX ISA — instruction spellings & release notes";
    sandbox.write(DOC_INDEX_PATH, small).unwrap();
    assert_eq!(render_doc_index(&sandbox).expect("doc index present"), small);

    // Over-cap body (comfortably past cap + marker length): the load-bearing HEAD
    // (flagged fast-path) is kept; the overflow tail is dropped with a marker pointing
    // back at the file.
    let head = "docs/nvidia/ptx.md · relevant y · THE FAST PATH";
    let body = format!("{head}\n{}", "x".repeat(DOC_INDEX_RENDER_CHARS + 3_000));
    assert!(body.len() > DOC_INDEX_RENDER_CHARS);
    sandbox.write(DOC_INDEX_PATH, &body).unwrap();
    let doc_index = render_doc_index(&sandbox).expect("doc index present");
    assert!(doc_index.len() < body.len(), "over-cap body must be truncated");
    assert!(
        doc_index.starts_with(head),
        "load-bearing head (flagged fast-path) must survive"
    );
    assert!(doc_index.contains("truncated to its head"), "truncation marker present");

    // Preamble names the file and carries no problem-specific vocabulary.
    let p = prompts::DOC_INDEX_REINJECT_PREAMBLE;
    assert!(p.contains("notes/DOC_INDEX.md"), "preamble names the doc-survey file");
    for banned in ["CUDA", "attention", "FlashAttention", "tcgen05", "MPS", "Metal"] {
        assert!(
            !p.contains(banned),
            "reinject preamble must stay domain-agnostic (found `{banned}`)"
        );
    }
}

/// The harness is domain-agnostic and built for *future* architectures: the shared
/// agent-facing prompts must never bake in one problem's answer (today's fast path
/// happens to be an attention/tcgen05 kernel, but the prompts must not name it) — the
/// agent discovers the fast path from docs + probed hardware each run. This locks the
/// scrub: none of the problem/algorithm/instruction-as-answer tokens may appear in the
/// shipped prompts. The BACKEND/task-class contract is explicitly NOT scrubbed, so we
/// also assert `CUDA`/`PTX` survive in `SYSTEM` — a future over-scrub that strips the
/// raw-CUDA+PTX contract must fail here too.
#[test]
fn shipped_prompts_are_problem_and_algorithm_agnostic() {
    // Problem/algorithm/instruction-as-answer vocabulary that must never leak into the
    // generic machinery. NOT in this list: `CUDA`/`PTX`/`nvcc`/`-gencode` — those are the
    // backend/task-class contract, which the user explicitly wants kept.
    const BANNED: &[&str] = &[
        "attention",
        "flashattention",
        "flash-attention",
        "tcgen05",
        "wgmma",
        "tmem",
        "blackwell",
        "sm_100a",
        "sm_103a",
    ];
    let prompts: &[(&str, &str)] = &[
        ("SYSTEM", prompts::SYSTEM),
        ("SUPERVISOR_NO_EVAL", prompts::SUPERVISOR_NO_EVAL),
        ("SUPERVISOR_STAGNATION", prompts::SUPERVISOR_STAGNATION),
        ("SUPERVISOR_REVIEW", prompts::SUPERVISOR_REVIEW),
        ("SUPERVISOR_FAILED_COMMITS", prompts::SUPERVISOR_FAILED_COMMITS),
        ("COMPACT_PREAMBLE", prompts::COMPACT_PREAMBLE),
    ];
    for (name, text) in prompts {
        let lower = text.to_lowercase();
        for banned in BANNED {
            assert!(
                !lower.contains(banned),
                "`{name}` leaks problem/algorithm-specific token `{banned}` — the harness must \
                 stay domain-agnostic; say \"the newest architecture-specific features\" / \"the \
                 state-of-the-art algorithm from the relevant paper\" instead of naming it",
            );
        }
    }
    // Boundary guard: the backend/task-class contract is KEPT, not scrubbed. If a future
    // edit over-scrubs and removes the raw-CUDA+PTX contract, fail loudly here.
    assert!(
        prompts::SYSTEM.contains("CUDA"),
        "SYSTEM must keep the CUDA backend contract"
    );
    assert!(
        prompts::SYSTEM.contains("PTX"),
        "SYSTEM must keep the PTX backend contract"
    );
}

/// Fix C2's whole point is to route a re-opened `deferred` lever through the GROUND
/// TRUTH "Tried & measured" list and the `checkout`/resume path. If the anchor's
/// label text drifts, this preamble's cross-reference goes stale — pin the coupling.
#[test]
fn deferred_lever_preamble_cross_references_the_anchor_tried_and_measured_list() {
    assert!(prompts::TODO_REINJECT_PREAMBLE.contains("Tried & measured"));
    assert!(prompts::TODO_REINJECT_PREAMBLE.contains("checkout"));
}

/// Fix C1: the build-invariants note is surfaced in FULL, never truncated —
/// contrasted here against the gotchas ledger, which caps at
/// `GOTCHAS_RENDER_CHARS` newest-tail. `run_1784671694` lost the load-bearing
/// `-gencode arch=compute_100a,code=sm_100a` flag off the TOP of the growing
/// gotchas ledger and re-diagnosed it 13×; a dedicated uncapped note prevents that.
#[test]
fn invariants_are_surfaced_in_full_unlike_truncated_gotchas() {
    let sandbox = Sandbox::new().unwrap();
    // A body well over the gotchas truncation cap, with the load-bearing flag at
    // the HEAD (where the gotchas tail-truncation drops it).
    let flag = "-gencode arch=compute_100a,code=sm_100a";
    let body = format!("# BUILD INVARIANTS\n{flag}\n{}", "x".repeat(6_000));
    assert!(body.len() > GOTCHAS_RENDER_CHARS);

    // render_invariants: full body, never truncated, flag intact.
    sandbox.write(INVARIANTS_PATH, &body).unwrap();
    let invariants = render_invariants(&sandbox).expect("invariants present");
    assert_eq!(invariants, body.trim());
    assert!(invariants.contains(flag), "load-bearing flag must survive in full");
    assert!(!invariants.contains("older ledger entries elided"));

    // Contrast: the SAME oversized body through the gotchas renderer is truncated
    // to its newest tail, so the head-of-file flag falls off (the exact failure).
    sandbox.write(GOTCHAS_PATH, &body).unwrap();
    let gotchas = render_gotchas(&sandbox).expect("gotchas present");
    assert!(gotchas.len() < body.trim().len(), "gotchas must be truncated");
    assert!(gotchas.contains("older ledger entries elided"));
    assert!(
        !gotchas.contains(flag),
        "head-of-file flag is dropped by tail-truncation"
    );
}
