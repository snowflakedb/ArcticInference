use super::helpers::*;
use super::*;
use crate::domain::types::{Evaluation, Stage};
use tempfile::TempDir;

fn files(body: &str) -> SolutionFiles {
    std::iter::once(("solution.py".to_string(), body.to_string())).collect()
}

/// A Timed evaluation with an explicit geomean (speedup over the baseline) and
/// stage. The search layer consumes `geomean_speedup`/`stage` directly, so tests
/// inject them without going through the baseline-aware scorer.
fn timed(stage: Stage, geomean: f64) -> Evaluation {
    Evaluation::Timed {
        stage,
        geomean_speedup: geomean,
        noise_margin: 0.0,
        per_config: Vec::new(),
    }
}

/// Append an evaluated candidate node carrying an explicit Timed eval plus a
/// workspace snapshot (so it can join the beam frontier).
fn append_scored(store: &SearchTree, ws: &std::path::Path, turn: u64, beam_width: usize, eval: Evaluation) -> String {
    store
        .append_typed_turn_node(
            turn,
            beam_width,
            vec![serde_json::json!({"scored": turn})],
            ws,
            true,
            None,
            Some(eval),
        )
        .unwrap()
}

#[test]
fn appends_message_only_nodes() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let id = store
        .append_turn_node(1, 1, serde_json::json!({"assistant":"hi"}), ws.path(), false, None)
        .unwrap();
    let nodes = load_nodes(&dir.path().join(HISTORY_DIR)).unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id.as_str(), id);
    assert!(nodes[0].new_workspace.is_none());
    assert!(dir.path().join(HISTORY_DIR).join(NODES_FILE).exists());
}

#[test]
fn appends_workspace_snapshot_nodes() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    fs::write(ws.path().join("solution.py"), "print(1)").unwrap();
    fs::create_dir(ws.path().join("docs")).unwrap();
    fs::write(ws.path().join("docs/ignore.md"), "ignore").unwrap();
    store
        .append_turn_node(1, 1, serde_json::json!({"assistant":"edit"}), ws.path(), true, None)
        .unwrap();
    let nodes = load_nodes(&dir.path().join(HISTORY_DIR)).unwrap();
    let commit = nodes[0].new_workspace.as_ref().unwrap().as_str();
    let restored = store
        .manager()
        .load_workspace_snapshot(&WorkspaceSnapshotId::new(commit))
        .unwrap();
    assert!(restored.contains_key("solution.py"));
    assert!(!restored.keys().any(|k| k.contains("docs/ignore.md")));
}

#[test]
fn binary_artifacts_survive_snapshot_roundtrip_byte_exact() {
    // A profiler output (non-UTF-8) must round-trip through the git snapshot store
    // unchanged — the whole point of the bytes-typed workspace path. Under the old
    // String::from_utf8_lossy read path these bytes would come back as U+FFFD.
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    fs::create_dir_all(ws.path().join("artifacts")).unwrap();
    fs::create_dir_all(ws.path().join("solution")).unwrap();
    let raw: &[u8] = b"SQLite format 3\x00\x01\x80\xff\xfe binary profiler bytes";
    fs::write(ws.path().join("artifacts/prof.sqlite"), raw).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "class Solution: pass").unwrap();
    let id = store
        .append_turn_node(
            1,
            1,
            serde_json::json!({ "assistant": "profiled" }),
            ws.path(),
            true,
            None,
        )
        .unwrap();

    // Extract the node's tree to a fresh dir (the real restore/checkout path) and
    // read the binary artifact straight off disk — must be byte-identical.
    let out = TempDir::new().unwrap();
    store.extract_workspace_into(&TurnNodeId::new(id), out.path()).unwrap();
    let got = fs::read(out.path().join("artifacts/prof.sqlite")).unwrap();
    assert_eq!(
        got, raw,
        "binary profiler bytes must be preserved exactly through git extraction"
    );
}

#[test]
fn extract_workspace_recovers_artifacts_not_just_solution() {
    // I10: a confirmed node's verified standalone kernels (artifacts/*.cu) must be
    // recoverable, not just solution/. extract_workspace_into materializes both; the
    // solution-only read surfaces only solution/.
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    fs::create_dir_all(ws.path().join("solution")).unwrap();
    fs::create_dir_all(ws.path().join("artifacts")).unwrap();
    fs::create_dir_all(ws.path().join("notes")).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "class Solution: pass").unwrap();
    fs::write(ws.path().join("artifacts/tc.cu"), "// verified UMMA gemm").unwrap();
    fs::write(ws.path().join("notes/SOLVED_GOTCHAS.md"), "- flag X").unwrap();
    let id = store
        .append_turn_node(1, 1, serde_json::json!({"assistant": "edit"}), ws.path(), true, None)
        .unwrap();
    let node = TurnNodeId::new(id);

    let out = TempDir::new().unwrap();
    store.extract_workspace_into(&node, out.path()).unwrap();
    assert!(out.path().join("solution/solution.py").exists());
    assert!(
        out.path().join("artifacts/tc.cu").exists(),
        "artifacts must be recoverable"
    );
    assert!(out.path().join("notes/SOLVED_GOTCHAS.md").exists());

    let sol = store.read_solution_at(&node).unwrap();
    // solution-only read is scoped to solution/ (prefix stripped) — no artifacts.
    assert!(sol.contains_key("solution.py"));
    assert!(
        !sol.keys().any(|k| k.contains("tc.cu")),
        "solution read must not carry artifacts"
    );
}

#[test]
fn evaluated_files_snapshot_wins_over_current_workspace() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    fs::write(ws.path().join("solution.py"), "print('restored')").unwrap();
    let evaluated = files("print('evaluated')");
    let evaluated_sha = fileset_sha256(&evaluated);
    store
            .append_turn_node(
                1,
                2,
                serde_json::json!({"evaluations":[{"stage_reached":"full","ok":true,"correct":true,"per_config":[{"name":"c0","latency_ms":1.0}]}]}),
                ws.path(),
                false,
                Some(&evaluated),
            )
            .unwrap();
    let nodes = load_nodes(&dir.path().join(HISTORY_DIR)).unwrap();
    assert!(nodes[0].new_messages[0].get("evaluations").is_some());
    assert!(
        matches!(
            nodes[0].evaluation,
            Some(crate::domain::types::Evaluation::Timed { .. })
        ),
        "evaluated node should carry a Timed evaluation"
    );
    let state = load_history_state(&dir.path().join(HISTORY_DIR)).unwrap();
    assert_eq!(
        state
            .frontier
            .iter()
            .find(|e| e.node_id == nodes[0].node_id.0)
            .and_then(|e| e.workspace_sha256.as_deref()),
        Some(evaluated_sha.as_str()),
        "frontier entry should carry the evaluated fileset hash"
    );
    let commit = nodes[0].new_workspace.as_ref().unwrap().as_str();
    let restored = store
        .manager()
        .load_solution_snapshot(&WorkspaceSnapshotId::new(commit))
        .unwrap();
    assert_eq!(restored["solution.py"], "print('evaluated')");
}

#[test]
fn restores_legacy_root_relative_evaluated_snapshot() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let mut entries = BTreeMap::new();
    entries.insert("solution.py".to_string(), b"print('legacy')".to_vec());
    entries.insert("kernels/k.cu".to_string(), b"// legacy".to_vec());
    let snapshot = store.manager().save_snapshot("legacy-node", None, &entries).unwrap();
    let restored = store.manager().load_solution_snapshot(&snapshot).unwrap();
    assert_eq!(restored["solution.py"], "print('legacy')");
    assert_eq!(restored["kernels/k.cu"], "// legacy");
}

#[test]
fn evaluated_snapshot_preserves_full_mutable_workspace() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    fs::write(ws.path().join("notes.txt"), "research").unwrap();
    fs::create_dir_all(ws.path().join("solution/kernels")).unwrap();
    fs::write(ws.path().join("solution/old.py"), "stale").unwrap();
    fs::create_dir_all(ws.path().join("_trusted")).unwrap();
    fs::write(ws.path().join("_trusted/problem.py"), "secret").unwrap();
    let mut evaluated = files("print('evaluated')");
    evaluated.insert("kernels/k.cu".to_string(), "// kernel".to_string());
    store
            .append_turn_node(
                1,
                2,
                serde_json::json!({"evaluations":[{"stage_reached":"full","ok":true,"correct":true,"per_config":[{"name":"c0","latency_ms":1.0}]}]}),
                ws.path(),
                false,
                Some(&evaluated),
            )
            .unwrap();
    let nodes = load_nodes(&dir.path().join(HISTORY_DIR)).unwrap();
    let snapshot = WorkspaceSnapshotId::new(nodes[0].new_workspace.as_ref().unwrap().as_str());
    let restored = store.manager().load_workspace_snapshot(&snapshot).unwrap();
    assert_eq!(String::from_utf8_lossy(&restored["notes.txt"]), "research");
    assert_eq!(
        String::from_utf8_lossy(&restored["solution/solution.py"]),
        "print('evaluated')"
    );
    assert_eq!(String::from_utf8_lossy(&restored["solution/kernels/k.cu"]), "// kernel");
    assert!(!restored.contains_key("solution/old.py"));
    assert!(!restored.contains_key("_trusted/problem.py"));
}

#[test]
fn core_node_json_has_no_policy_state_fields() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let evaluated = files("print('evaluated')");
    store
            .append_turn_node(
                1,
                2,
                serde_json::json!({"evaluations":[{"stage_reached":"full","ok":true,"correct":true,"per_config":[{"name":"c0","latency_ms":1.0}]}]}),
                ws.path(),
                false,
                Some(&evaluated),
            )
            .unwrap();
    let line = fs::read_to_string(dir.path().join(HISTORY_DIR).join(NODES_FILE)).unwrap();
    let node_json: Value = serde_json::from_str(line.trim()).unwrap();
    // `evaluation` is a first-class node field by design; policy-only derivations
    // (frontier ranking, acceptance, commit outcome) must stay out of the node.
    for forbidden in ["frontier", "score", "accepted_version", "commit_outcome", "policy"] {
        assert!(
            node_json.get(forbidden).is_none(),
            "core node leaked policy field {forbidden}: {node_json}"
        );
    }
    assert!(
        node_json.get("evaluation").is_some(),
        "evaluated node should carry its evaluation"
    );
    let state_json: Value =
        serde_json::from_slice(&fs::read(dir.path().join(HISTORY_DIR).join(STATE_FILE)).unwrap()).unwrap();
    assert!(state_json.get("frontier").is_some());
}

#[test]
fn resolves_nearest_ancestor_workspace_snapshot() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();
    fs::write(ws.path().join("a.txt"), "a").unwrap();
    let n1 = store
        .append_turn_node(1, 1, serde_json::json!({"n":1}), ws.path(), true, None)
        .unwrap();
    let n2 = store
        .append_turn_node(2, 1, serde_json::json!({"n":2}), ws.path(), false, None)
        .unwrap();
    let c1 = load_nodes(&dir.path().join(HISTORY_DIR))
        .unwrap()
        .into_iter()
        .find(|node| node.node_id.as_str() == n1)
        .unwrap()
        .new_workspace
        .map(|id| id.0);
    assert_eq!(store.nearest_workspace_commit(Some(&n2)).unwrap(), c1);
    assert!(store.nearest_workspace_commit(Some(&n1)).unwrap().is_some());
}

#[test]
fn seed_root_is_unscored() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    assert!(store.current_best().is_none(), "no best before the seed");
    let f0 = files("class Solution: pass");
    let seed = store.seed_root(&f0).unwrap();
    // The seed is a stub, not a scored candidate — the reference is the
    // correctness oracle only. No confirmed best exists until the loop crowns
    // the first correct candidate as the 1.0x baseline.
    assert!(store.current_best().is_none(), "the stub seed is unscored");
    assert!(store.confirmed_score(&seed).is_none());
    // Reopen: still unscored (derived from the persisted, un-annotated node).
    let reopened = SearchTree::open(dir.path()).unwrap();
    assert!(reopened.current_best().is_none());
}

#[test]
fn record_confirmation_crowns_a_new_best() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();

    // The first correct candidate is crowned by the loop as the 1.0x baseline.
    fs::create_dir_all(ws.path().join("solution")).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "base").unwrap();
    let base = append_scored(&store, ws.path(), 1, 1, timed(Stage::Full, 1.0));
    store
        .record_confirmation(&TurnNodeId::new(base.clone()), 1.0, 1)
        .unwrap();
    assert_eq!(
        store.current_best().unwrap().node_id.as_str(),
        base,
        "first correct candidate is the baseline"
    );

    // A faster candidate with a raw Timed eval is NOT best until confirmed.
    fs::write(ws.path().join("solution/solution.py"), "fast").unwrap();
    let cand = append_scored(&store, ws.path(), 2, 1, timed(Stage::Full, 1.5));
    assert_eq!(
        store.current_best().unwrap().geomean_speedup,
        Some(1.0),
        "an unconfirmed candidate must not be crowned best (winner's-curse defense)"
    );
    store
        .record_confirmation(&TurnNodeId::new(cand.clone()), 1.48, 3)
        .unwrap();
    let best = store.current_best().unwrap();
    assert_eq!(best.node_id.as_str(), cand);
    assert_eq!(
        best.geomean_speedup,
        Some(1.48),
        "confirmed geomean crowns the new best"
    );
}

#[test]
fn beam_frontier_keeps_top_scored_nodes() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    fs::create_dir_all(ws.path().join("solution")).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "a=1").unwrap();
    append_scored(&store, ws.path(), 1, 2, timed(Stage::Correctness, 1.1));
    fs::write(ws.path().join("solution/solution.py"), "a=2").unwrap();
    append_scored(&store, ws.path(), 2, 2, timed(Stage::Full, 1.02));
    let frontier = store.frontier();
    assert_eq!(frontier.len(), 2);
    assert_eq!(frontier[0].stage_rank, 3, "full evidence outranks lower-stage evidence");
    assert!(frontier.first().unwrap().workspace_snapshot.is_some());
}

#[test]
fn beam_schedule_is_empty_until_a_candidate_is_scored() {
    use crate::orchestrator::strategy::Action;
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();

    // An un-scored edit is not a frontier candidate; with an empty frontier beam
    // has nothing to fork to, so `schedule` returns no action — the loop then runs
    // the bootstrap episode from the current leaf (the seed).
    fs::write(ws.path().join("solution.py"), "candidate").unwrap();
    store
        .append_turn_node(1, 4, serde_json::json!({"edit": true}), ws.path(), true, None)
        .unwrap();
    let mut policy = strategy::make_policy(
        strategy::POLICY_BEAM,
        store,
        4,
        1,
        0.0,
        strategy::DiversityConfig::disabled(),
    );
    assert!(
        !policy.schedule(1).iter().any(|a| matches!(a, Action::Expand(_))),
        "beam has no frontier candidate to expand yet"
    );
}

#[test]
fn beam_excludes_the_unscored_seed_from_the_frontier() {
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();

    fs::create_dir_all(ws.path().join("solution")).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "candidate").unwrap();
    let scored = append_scored(&store, ws.path(), 1, 4, timed(Stage::Full, 1.0001));

    assert_eq!(
        store.frontier().len(),
        1,
        "the turn-0 stub seed is never a beam frontier candidate"
    );
    let mut policy = strategy::make_policy(
        strategy::POLICY_BEAM,
        store,
        4,
        1,
        0.0,
        strategy::DiversityConfig::disabled(),
    );
    assert_eq!(
        policy.schedule(1),
        vec![crate::orchestrator::strategy::Action::Expand(TurnNodeId::new(scored))],
        "beam expands the sole scored candidate, not the excluded seed"
    );
}

#[test]
fn avo_policy_schedules_expand_of_the_current_leaf() {
    use crate::orchestrator::strategy::Action;
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();

    let mut policy = strategy::make_policy(
        strategy::POLICY_AVO,
        store.clone(),
        1,
        1,
        0.0,
        strategy::DiversityConfig::disabled(),
    );
    let leaf = store.current_node_id().unwrap();
    assert_eq!(
        policy.schedule(1),
        vec![Action::Expand(leaf)],
        "AVO expands exactly the current linear leaf"
    );
    // After a new turn node lands, the schedule tracks the new leaf — never forks
    // to a different branch (width-1 has no other branch).
    fs::write(dir.path().join("dummy"), "x").unwrap();
    let n1 = store
        .append_turn_node(1, 1, serde_json::json!({"n":1}), dir.path(), false, None)
        .unwrap();
    assert_eq!(policy.schedule(1), vec![Action::Expand(TurnNodeId::new(n1))]);
}

#[test]
fn beam_policy_schedules_expand_of_the_higher_scored_branch() {
    use crate::orchestrator::strategy::Action;
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();
    let seed = store.current_node_id().unwrap();

    // Two evaluated branches off the seed; the faster one must be scheduled, and
    // its branch-local context differs from the slower one's.
    fs::create_dir_all(ws.path().join("solution")).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "slow").unwrap();
    let slow = append_scored(&store, ws.path(), 1, 2, timed(Stage::Full, 1.10));
    // Return to the seed so the second branch is a sibling of the first, not a child.
    store.checkout_node(&seed).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "fast").unwrap();
    let fast = append_scored(&store, ws.path(), 2, 2, timed(Stage::Full, 1.50));

    let mut policy = strategy::make_policy(
        strategy::POLICY_BEAM,
        store.clone(),
        2,
        1,
        0.0,
        strategy::DiversityConfig::disabled(),
    );
    // The active in-progress child is the just-appended `fast` leaf; beam expands it.
    let Action::Expand(target) = policy.schedule(1).into_iter().next().unwrap() else {
        panic!("beam must emit an Expand action");
    };
    assert_eq!(target.as_str(), fast, "beam expands the active leaf");
    // Forking to either branch yields a distinct branch-local context.
    let fast_ctx = store.context::<Value>(&TurnNodeId::new(fast)).unwrap();
    let slow_ctx = store.context::<Value>(&TurnNodeId::new(slow)).unwrap();
    assert_ne!(fast_ctx, slow_ctx, "sibling branches have distinct contexts");
}

/// The `local_cap` knob (§14.5) redistributes a fixed batch across the frontier:
/// `1` spreads one expansion per distinct node (tree coverage); a higher cap lets
/// the top-ranked node absorb multiple expansions before the batch moves down
/// (local widening). Same two-sibling frontier (fast=1.50 outranks slow=1.10).
#[test]
fn ucb_fans_out_to_width_even_when_scores_straddle_the_floor() {
    use crate::orchestrator::strategy::Action;
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();
    let seed = store.current_node_id().unwrap();

    // Four scored siblings whose geomeans straddle the 0.5*best score floor
    // (best=10.0 => floor=5.0; three candidates sit below it). UCB must still keep
    // the top-`width` in its pool and fan out to `width` DISTINCT expanders —
    // otherwise the round serializes to a single expander (the parallelism bug).
    fs::create_dir_all(ws.path().join("solution")).unwrap();
    let mut ids = Vec::new();
    for (i, g) in [10.0_f64, 2.0, 1.5, 1.0].iter().enumerate() {
        store.checkout_node(&seed).unwrap();
        fs::write(ws.path().join("solution/solution.py"), format!("cand{i}")).unwrap();
        ids.push(append_scored(
            &store,
            ws.path(),
            u64::try_from(i + 1).unwrap(),
            4,
            timed(Stage::Full, *g),
        ));
    }

    let mut policy = strategy::make_policy(
        strategy::POLICY_UCB,
        store,
        4,
        1,
        0.5,
        strategy::DiversityConfig::disabled(),
    );
    let picks: Vec<String> = policy
        .schedule(4)
        .into_iter()
        .map(|a| match a {
            Action::Expand(id) => id.as_str().to_string(),
            other @ Action::Reevaluate(_) => panic!("expected Expand, got {other:?}"),
        })
        .collect();
    assert_eq!(picks.len(), 4, "UCB must fan out to `width` expanders, not serialize");
    let uniq: std::collections::BTreeSet<_> = picks.iter().collect();
    assert_eq!(uniq.len(), 4, "the `width` expanders must be on distinct nodes");
    for id in &ids {
        assert!(picks.contains(id), "candidate {id} must be reachable despite the floor");
    }
}

#[test]
fn ucb_dedup_excludes_byte_identical_siblings_from_one_round() {
    use crate::orchestrator::strategy::Action;
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();
    let seed = store.current_node_id().unwrap();
    fs::create_dir_all(ws.path().join("solution")).unwrap();

    // Two byte-identical kernels (same source ⇒ same fileset_sha256) plus a
    // distinct one. This is exactly the frontier-collapse shape the audit found.
    store.checkout_node(&seed).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "IDENTICAL").unwrap();
    let dup_a = append_scored(&store, ws.path(), 1, 4, timed(Stage::Full, 1.30));
    store.checkout_node(&seed).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "IDENTICAL").unwrap();
    let dup_b = append_scored(&store, ws.path(), 2, 4, timed(Stage::Full, 1.20));
    store.checkout_node(&seed).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "DIFFERENT").unwrap();
    let other = append_scored(&store, ws.path(), 3, 4, timed(Stage::Full, 1.10));

    let diversity = strategy::DiversityConfig { dedup: true };
    let mut policy = strategy::make_policy(strategy::POLICY_UCB, store, 2, 1, 0.5, diversity);
    let picks: Vec<String> = policy
        .schedule(2)
        .into_iter()
        .map(|a| match a {
            Action::Expand(id) => id.as_str().to_string(),
            o @ Action::Reevaluate(_) => panic!("expected Expand, got {o:?}"),
        })
        .collect();

    assert_eq!(
        picks.len(),
        2,
        "both expansion slots should still be filled, with distinct kernels"
    );
    assert!(
        picks.contains(&dup_a) ^ picks.contains(&dup_b),
        "exactly one of the byte-identical pair may be expanded in a round, not both"
    );
    assert!(picks.contains(&other), "the distinct kernel fills the other slot");
}

#[test]
fn beam_expands_top_k_distinct_by_score() {
    use crate::orchestrator::strategy::Action;
    let dir = TempDir::new().unwrap();
    let store = SearchTree::open(dir.path()).unwrap();
    let ws = TempDir::new().unwrap();
    let f0 = files("class Solution: pass");
    store.seed_root(&f0).unwrap();
    let seed = store.current_node_id().unwrap();

    fs::create_dir_all(ws.path().join("solution")).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "slow").unwrap();
    let slow = append_scored(&store, ws.path(), 1, 2, timed(Stage::Full, 1.10));
    store.checkout_node(&seed).unwrap();
    fs::write(ws.path().join("solution/solution.py"), "fast").unwrap();
    let fast = append_scored(&store, ws.path(), 2, 2, timed(Stage::Full, 1.50));

    // beam == the c=0 (no-exploration) BeamUcb: score every node and greedily take
    // the K best *distinct* nodes, highest first — no bounded frontier.
    let mut beam = strategy::make_policy(
        strategy::POLICY_BEAM,
        store,
        2,
        1,
        0.0,
        strategy::DiversityConfig::disabled(),
    );
    let picks: Vec<String> = beam
        .schedule(2)
        .into_iter()
        .map(|a| match a {
            Action::Expand(id) => id.as_str().to_string(),
            other @ Action::Reevaluate(_) => panic!("expected Expand, got {other:?}"),
        })
        .collect();
    assert_eq!(
        picks,
        vec![fast, slow],
        "beam picks the top-K distinct nodes, best first"
    );
}
