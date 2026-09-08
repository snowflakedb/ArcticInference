use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use kernelguy::ai::provider;
use kernelguy::ai::{AnthropicMessagesClient, AuthMethod, CompletionProtocol, OpenaiResponsesClient};
use kernelguy::domain::types::RunId;
use kernelguy::env;
use kernelguy::exec::cuda_preflight;
use kernelguy::exec::queue::GpuPool;
use kernelguy::exec::sandbox_manager::{MountSpec, SandboxManager};
use kernelguy::harness::skills::{Skill, load_repo_skills};
use kernelguy::orchestrator::{self, AvoConfig, SearchTree, prompts};

/// Per-evaluation execution timeout, a backstop for a *hung* candidate only.
/// `evaluate.py` benchmarks adaptively (time-budgeted iters; slow forwards take
/// one sample), so a merely-slow kernel finishes well under this and gets a real
/// score; the full stage is ~tens of seconds even for sluggish candidates. This
/// large cap exists to kill true hangs (infinite loops in a single forward).
const EVAL_TIMEOUT_SECS: u64 = 300;
/// How long a tool waits for the execution queue before giving up. Generous:
/// uncontended with a single agent, but the same lease serializes any future
/// concurrent workers.
const QUEUE_ACQUIRE_TIMEOUT_SECS: u64 = 1800;

#[tokio::main]
async fn main() {
    // A real run mints an access token by running the provider's auth command at
    // startup (see [`resolve_model`]). Providers live in
    // `~/.kernelguy/providers/*.json`, seeded from `src/default_providers/` on first
    // launch. The headless self-test needs no secrets, so a missing .env is not
    // fatal.
    let _ = dotenv::dotenv();

    match Cli::parse().command {
        Command::Avo(args) => {
            if let Err(e) = run_avo_mode(args).await {
                eprintln!("avo run failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// kernelguy — an agent that writes state-of-the-art GPU kernels.
#[derive(Parser)]
#[command(name = "kernelguy", version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the AVO optimization loop on a problem.
    Avo(AvoArgs),
}

#[derive(Args)]
struct AvoArgs {
    /// Path to the problem, e.g. `problems/attention.py`. Optional with
    /// `--resume` (inferred from the run manifest) or `--selftest`.
    problem: Option<PathBuf>,

    /// Resume a previous run by id; the problem is re-derived from its manifest.
    #[arg(long, value_name = "RUN_ID")]
    resume: Option<String>,

    /// Drive the loop with the scripted client (no API key / network).
    #[arg(long)]
    selftest: bool,

    /// Provider that serves `--model`, named after its file in
    /// `~/.kernelguy/providers/` (without the `.json`). Required: there is no
    /// default, so which file answered is never implicit.
    #[arg(long, value_name = "NAME")]
    provider: String,

    /// Model slug to drive the agent. Must be one `--provider` serves; the listing
    /// endpoint determines both the wire protocol and the context window, and an
    /// unknown slug is an error listing what that provider does serve. Required:
    /// there is no default.
    #[arg(long, value_name = "SLUG")]
    model: String,

    /// Force the wire protocol rather than letting the model's listing endpoint
    /// decide. Only useful when a provider lists one model id under several
    /// endpoints — by default the highest-priority protocol wins.
    #[arg(long, value_name = "PROTOCOL")]
    protocol_override: Option<CompletionProtocol>,

    /// Search policy: `avo` (accepted incumbent only) or `beam` (evidence-ranked frontier).
    #[arg(long, default_value = "avo", value_name = "POLICY")]
    policy: String,

    /// Number of active frontier nodes retained by `--policy beam`.
    #[arg(long, default_value_t = 4, value_name = "N")]
    beam_width: usize,

    /// Per-node local expansion cap for `--policy beam`: how many concurrent
    /// expansions one frontier node may receive in a single batch. `1` (default)
    /// spreads the batch across distinct frontier nodes (tree coverage); higher
    /// lets a dominant node get multiple expansions of its own child space (local
    /// widening). Ignored by `--policy avo`.
    #[arg(long, default_value_t = 1, value_name = "N")]
    beam_local_cap: usize,

    /// Exploration constant `c` for `--policy ucb`: the weight on the confidence
    /// bonus in `norm_geomean + c*sqrt(ln T / n)`. Higher = more re-broadening
    /// (expand under-visited nodes over the raw top scorers); `0` uses the policy
    /// default (0.5). Ignored by `--policy avo`/`beam`.
    #[arg(long, default_value_t = 0.5, value_name = "C")]
    ucb_c: f64,

    /// Disable beam-diversity tier-1 exact de-dup for `--policy ucb` (on by
    /// default). Tier-1 stops two byte-identical kernels from occupying the same
    /// expansion round — the frontier-collapse fix. Free (no model).
    #[arg(long)]
    no_diversity_dedup: bool,

    /// Wall-clock budget: stop after this many seconds [default: 21600 = 6h].
    /// This is the run's budget knob; a large token backstop guards runaways.
    #[arg(long, value_name = "SECS")]
    max_wall_secs: Option<u64>,

    /// Override the serving model's real input-token window (tokens) used to
    /// decide WHEN to compact (`context_window - reserve`). Defaults to the
    /// model's catalog entry. Set at or below the endpoint's actual input limit — a
    /// standard 200k Claude deployment should pass `--context-window-tokens 200000`.
    #[arg(long, value_name = "N")]
    context_window_tokens: Option<u32>,

    /// Fixed base seed for a reproducible run (drives the evaluator-input
    /// stream). Omit for a fresh random seed.
    #[arg(long, value_name = "N")]
    seed: Option<u64>,

    /// Enable the stall supervisor (off by default). On a stall it reviews the
    /// committed lineage and injects concrete next directions. Bare `--supervised`
    /// uses the agent's own model; `--supervised=<slug>` names a specific one
    /// (must be reached over the same protocol as `--model`).
    #[arg(long, value_name = "MODEL", num_args = 0..=1, require_equals = true)]
    #[expect(
        clippy::option_option,
        reason = "clap three-state flag: absent / bare --supervised / --supervised=slug"
    )]
    supervised: Option<Option<String>>,

    /// Disable repo-local startup skills from `agent_ressources/skills/`.
    #[arg(long)]
    no_skills: bool,
}

// ─── provider ────────────────────────────────────────────────────────────────

/// What resolving `--provider` + `--model` against the provider files yields.
struct ResolvedModel {
    protocol: CompletionProtocol,
    /// Full URL for [`Self::protocol`] on that provider.
    endpoint_url: String,
    context_size: u32,
    auth: AuthMethod,
    /// The provider's headers with this endpoint's overrides applied, already
    /// validated. A supervisor model reuses the *main* model's map — see
    /// `resolve_supervisor_model`, which resolves only to validate.
    headers: reqwest::header::HeaderMap,
}

/// Resolve `model` on the provider named `provider_name`, seeding the shipped
/// defaults first if `~/.kernelguy/providers/` is new.
///
/// There are no providers in the code — each is a JSON file, so retargeting a
/// host, adding a model, or swapping the auth command is an edit there rather than
/// a rebuild. Nothing here reads the environment: the file *is* the configuration
/// surface. How the auth command decides what to authenticate against is the
/// command's business, not ours.
///
/// Naming the provider is required rather than searching all of them, so which
/// file answered is never positional — two files listing the same model id cannot
/// shadow each other. Within one provider, a model listed under several endpoints
/// resolves by protocol priority; `protocol_override` pins one instead.
///
/// # Errors
///
/// Returns an error if the directory cannot be seeded or read, any file is
/// malformed, no provider is named `provider_name` (the message lists the ones
/// that exist), or that provider does not serve `model` over the protocol in force
/// (the message lists the models it does serve).
fn resolve_model(
    provider_name: &str,
    model: &str,
    protocol_override: Option<CompletionProtocol>,
) -> Result<ResolvedModel, String> {
    provider::seed_defaults().map_err(|e| e.to_string())?;
    let loaded = provider::load_all().map_err(|e| e.to_string())?;
    let dir = provider::dir().map_or_else(|e| e.to_string(), |d| d.display().to_string());

    let named = loaded.iter().find(|n| n.name == provider_name).ok_or_else(|| {
        let names: Vec<&str> = loaded.iter().map(|n| n.name.as_str()).collect();
        format!("unknown provider `{provider_name}`; {dir} holds: {}", names.join(", "))
    })?;

    let found = protocol_override.map_or_else(
        || named.provider.resolve(model),
        |p| named.provider.resolve_with_protocol(p, model),
    );
    let (endpoint, entry) = found.ok_or_else(|| {
        let over = protocol_override.map_or_else(String::new, |p| format!(" over {p:?}"));
        format!(
            "provider `{provider_name}` does not serve model `{model}`{over}; it serves: {}",
            named.provider.model_ids().join(", ")
        )
    })?;

    // `resolve` found the endpoint on this provider, so it serves the protocol and
    // `endpoint_url` cannot be None.
    let endpoint_url = named
        .provider
        .endpoint_url(endpoint.protocol)
        .ok_or_else(|| format!("provider `{provider_name}` does not serve {:?}", endpoint.protocol))?;

    let headers = named
        .provider
        .headers_for(endpoint)
        .map_err(|e| format!("provider `{provider_name}`: {e}"))?;

    Ok(ResolvedModel {
        protocol: endpoint.protocol,
        endpoint_url,
        context_size: entry.context_size,
        auth: endpoint.auth.clone(),
        headers,
    })
}

// ─── AVO mode ────────────────────────────────────────────────────────────────

#[expect(
    clippy::too_many_lines,
    reason = "one linear sequence; splitting trades length for state plumbing"
)]
#[expect(
    clippy::future_not_send,
    reason = "ProtocolClient::Stream is deliberately !Send; awaited on one LocalSet task"
)]
async fn run_avo_mode(args: AvoArgs) -> Result<(), String> {
    let max_wall_clock_secs = args.max_wall_secs.unwrap_or(6 * 3600);
    let seed = args.seed;
    let selftest = args.selftest;
    let resume = args.resume;
    let policy = args.policy.clone();
    orchestrator::strategy::validate_policy(&policy)?;
    let beam_width = args.beam_width.max(1);
    let beam_local_cap = args.beam_local_cap.max(1);
    let ucb_c = args.ucb_c;
    let model = args.model.clone();
    let provider_name = args.provider.clone();
    let supervised = args.supervised.clone();
    let protocol_override = args.protocol_override;
    let no_skills = args.no_skills;
    let diversity_dedup = !args.no_diversity_dedup;
    let context_window_tokens_arg = args.context_window_tokens;
    let problem_path = args.problem.unwrap_or_default();

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let skills = if no_skills {
        Vec::new()
    } else {
        load_repo_skills(&manifest, "/workspace/skills").map_err(|e| format!("skill loading failed: {e}"))?
    };

    // Self-test defaults to the attention problem; the scripted client ends the
    // run itself (via StopReason::Done) once its script is exhausted.
    let problem_path = if selftest && problem_path.as_os_str().is_empty() {
        manifest.join("problems/attention.py")
    } else {
        problem_path
    };

    // Resolve run dir + problem path (resume re-derives the problem path).
    let (run_dir, problem_path, is_resume) = if let Some(id) = &resume {
        let dir = PathBuf::from("runs").join(id);
        if !dir.exists() {
            return Err(format!("cannot resume: {} does not exist", dir.display()));
        }
        if !SearchTree::has_tree_history(&dir) {
            return Err(format!(
                "cannot resume {}: no session-tree history found (missing history/nodes.jsonl); older runs are not resumable",
                dir.display()
            ));
        }
        let pp = if problem_path.as_os_str().is_empty() {
            problem_from_manifest(&dir).unwrap_or_else(|| manifest.join("problems/attention.py"))
        } else {
            problem_path
        };
        (dir, pp, true)
    } else {
        if problem_path.as_os_str().is_empty() {
            return Err("--avo requires a problem path, e.g. --avo problems/attention.py".to_string());
        }
        let id = RunId::generate();
        let dir = PathBuf::from("runs").join(id.to_string());
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        (dir, problem_path, false)
    };

    // Both `--provider` and `--model` are required, so nothing is inferred. On a
    // resume the manifest records what the run used; warn rather than fail when the
    // flags now name something else, since switching model mid-run is legitimate
    // but doing it silently is not.
    if is_resume
        && let Some(previous) = model_from_manifest(&run_dir)
        && previous != model
    {
        eprintln!("kernelguy: warning: run was started with model {previous}, resuming with {model}");
    }

    // Look the slug up on the named provider. Reading and parsing the files costs
    // no network and no credential, so it runs before the self-test early-return
    // and gives both paths the same answer. Protocol comes from the *enclosing
    // endpoint*, so it cannot disagree with the context window the way the old slug
    // tests did.
    let resolved = resolve_model(&provider_name, &model, protocol_override)?;
    let protocol = resolved.protocol;
    eprintln!("kernelguy: model {model} on provider {provider_name}");

    // Resolve + validate the supervisor model. The supervisor is off unless
    // `--supervised` is given; `--supervised` alone uses the agent's own model,
    // `--supervised=<slug>` names another reached over the same protocol. `None` ⇒
    // no supervisor at all.
    let supervisor_model = resolve_supervisor_model(selftest, supervised, &model, &provider_name, protocol)?;

    // Real input window used to decide when to compact: explicit flag wins, else
    // the catalog's figure for this model (overridable because the true serving
    // limit is deployment-specific).
    let context_window_tokens = context_window_tokens_arg.unwrap_or(resolved.context_size);

    let problem_name = problem_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("problem")
        .to_string();

    // A problem is a single self-contained file: v0 is its naive `Reference`
    // baseline, materialized as `solution/solution.py` from this stub (no seed
    // file). The agent may grow the solution into a multi-file tree from there.
    let seed_src = orchestrator::SEED_STUB.to_string();

    let hardware = env::hardware::probe_hardware();
    // The SandboxManager owns the snapshot git repo (at run_dir/repo) + the mount
    // shell; it is the sole minter of sandboxes. Spawn the initial (shared, for now)
    // sandbox from it.
    let mounts = build_mount_spec(&problem_path, &skills).map_err(|e| format!("sandbox setup failed: {e}"))?;
    let manager =
        SandboxManager::new(run_dir.join("repo"), mounts).map_err(|e| format!("sandbox manager setup failed: {e}"))?;
    let sandbox = Arc::new(manager.spawn(None).map_err(|e| format!("sandbox spawn failed: {e}"))?);
    if env::cuda::should_validate_nvidia_tools(&hardware) {
        eprintln!("kernelguy: validating NVIDIA/CUDA agent toolchain inside sandbox...");
        let summary = cuda_preflight::validate_nvidia_agent_tooling(&sandbox)
            .await
            .map_err(|e| format!("NVIDIA/CUDA startup validation failed; refusing to start run.\n{e}"))?;
        eprintln!(
            "kernelguy: NVIDIA/CUDA agent toolchain validation passed:\n{}",
            summary.format_for_startup()
        );
    }
    let search_tree = SearchTree::with_manager(&run_dir, manager)?;
    // One lease per *job-sized group* of GPUs. The size is the problem's NUM_GPUS, asked
    // for once here because it is the same for every job in the run, so the pool can decide
    // its slots at construction and acquiring stays a single lock. A single-GPU problem gets
    // one lease per card exactly as before; a TP4 problem on 4 cards gets one slot, and the
    // evals stop overlapping — which is inherent, not a regression.
    let gpu_count = env::hardware::gpu_count();
    let gpus_per_job = problem_gpus_per_job(&problem_path);
    if gpus_per_job > gpu_count {
        return Err(format!(
            "problem needs {gpus_per_job} GPUs but only {gpu_count} are visible"
        ));
    }
    let queue = GpuPool::new(gpu_count, gpus_per_job, Duration::from_secs(QUEUE_ACQUIRE_TIMEOUT_SECS));
    if gpus_per_job > 1 {
        eprintln!(
            "kernelguy: {gpus_per_job} GPUs per job, {} concurrent slot(s)",
            queue.slot_count()
        );
    }

    let cfg = AvoConfig {
        run_dir,
        problem_name,
        model_name: model.clone(),
        policy_id: policy.clone(),
        beam_width,
        beam_local_cap,
        ucb_c,
        python: "python3".to_string(),
        hardware,
        max_wall_clock_secs,
        resume: is_resume,
        seed_src,
        eval_timeout_secs: EVAL_TIMEOUT_SECS,
        seed,
        supervisor_model: supervisor_model.clone(),
        diversity_dedup,
        context_window_tokens,
        skills,
    };

    println!(
        "AVO {}{} run on {} with {} ({})",
        if selftest { "self-test " } else { "" },
        if is_resume { "resuming" } else { "starting" },
        cfg.problem_name,
        cfg.model_name,
        cfg.run_dir.display()
    );
    // Provenance: which commit built this binary, and how old the binary is.
    // The age is the authoritative staleness signal — a multi-hour run launched
    // against a forgotten `target/release` binary shows up here immediately.
    println!("  build: {} ({})", env!("KERNELGUY_GIT"), binary_age());
    let openai_prompt_cache_prefix = format!("kernelguy:{}", cfg.run_dir.to_string_lossy().replace('/', ":"));

    if selftest {
        let run_dir = cfg.run_dir.clone();
        orchestrator::run_avo(
            orchestrator::selftest::ScriptedClient::new(),
            None,
            sandbox,
            search_tree,
            queue,
            cfg,
        )
        .await?;
        orchestrator::selftest::verify_run(&run_dir)?;
        println!("self-test OK: tree history and AVO manifest instrumentation present.");
        return Ok(());
    }

    // A real run resolves the endpoint's credential now, so a broken helper fails
    // at startup rather than mid-run. The command owns storage and refreshing; it
    // is re-run as the short-lived token nears expiry, so a multi-hour run
    // doesn't die mid-flight.
    let auth = resolved.auth.resolve().await.map_err(|e| e.to_string())?;

    // The supervisor reviewer (when configured) is a second client of the SAME
    // protocol as the main agent — `resolve_supervisor_model` guarantees the slug
    // matches — sharing the same refreshing auth.
    let url = resolved.endpoint_url;
    // Both clients on a protocol arm share the endpoint's headers: the supervisor
    // has no endpoint of its own (see `resolve_supervisor_model`).
    let headers = resolved.headers;
    match protocol {
        CompletionProtocol::AnthropicMessages => {
            let supervisor = match &supervisor_model {
                Some(sm) => Some(
                    AnthropicMessagesClient::with_auth_provider(&url, sm, auth.clone(), &headers)
                        .map_err(|e| format!("anthropic supervisor client: {e}"))?,
                ),
                None => None,
            };
            let client = AnthropicMessagesClient::with_auth_provider(&url, &model, auth, &headers)
                .map_err(|e| format!("anthropic client: {e}"))?;
            orchestrator::run_avo(client, supervisor, sandbox, search_tree, queue, cfg).await
        }
        CompletionProtocol::OpenaiResponses => {
            let supervisor = match &supervisor_model {
                Some(sm) => Some(
                    OpenaiResponsesClient::with_auth_provider(&url, sm, auth.clone(), &headers)
                        .map(|client| client.with_instructions(prompts::SYSTEM))
                        .map(|client| {
                            client.with_prompt_cache_key(format!("{openai_prompt_cache_prefix}:supervisor:{sm}"))
                        })
                        .map_err(|e| format!("openai supervisor client: {e}"))?,
                ),
                None => None,
            };
            let client = OpenaiResponsesClient::with_auth_provider(&url, &model, auth, &headers)
                .map(|client| client.with_instructions(prompts::SYSTEM))
                .map(|client| client.with_prompt_cache_key(format!("{openai_prompt_cache_prefix}:main")))
                .map_err(|e| format!("openai client: {e}"))?;
            orchestrator::run_avo(client, supervisor, sandbox, search_tree, queue, cfg).await
        }
        // In `CompletionProtocol` because it is a real protocol, but no
        // `ProtocolClient` implements it yet. Reachable only via a provider that
        // declares such an endpoint, which the built-in one does not.
        CompletionProtocol::OpenaiChatCompletions => {
            Err("no client implements the OpenAI Chat Completions protocol yet".to_string())
        }
    }
}

/// Resolve the active supervisor model from `--supervised`. The supervisor is
/// off by default (`None`); a bare `--supervised` enables it using the agent's
/// own model, and `--supervised=<slug>` names a specific one. The supervisor
/// client is the same concrete type as the main agent's, so a slug reached over a
/// *different protocol* is a hard error — the two share one monomorphized
/// `ProtocolClient` and therefore one wire message type. Always `None` under
/// `--selftest` (the scripted client has no model).
#[expect(
    clippy::option_option,
    reason = "clap three-state flag: absent / bare --supervised / --supervised=slug"
)]
fn resolve_supervisor_model(
    selftest: bool,
    supervised: Option<Option<String>>,
    main_model: &str,
    provider_name: &str,
    main_protocol: CompletionProtocol,
) -> Result<Option<String>, String> {
    if selftest {
        return Ok(None);
    }
    let slug = match supervised {
        None => return Ok(None),              // flag absent → supervisor off
        Some(None) => main_model.to_string(), // `--supervised` → the agent's own model
        Some(Some(s)) => s,                   // `--supervised=<slug>`
    };
    // Same provider as the agent, and the same protocol — the two share one
    // monomorphized `ProtocolClient` and therefore one wire message type. Forcing
    // the protocol here turns a mismatch into "does not serve <slug> over <p>"
    // rather than a comparison that could drift from the reason for it.
    resolve_model(provider_name, &slug, Some(main_protocol)).map_err(|e| {
        format!(
            "--supervised {slug}: {e}; the supervisor must share --model {main_model}'s protocol ({main_protocol:?})"
        )
    })?;
    Ok(Some(slug))
}

/// Human-readable age of the running executable (its mtime → "how long ago was
/// this binary built"). This is the authoritative freshness signal — independent
/// of any compiled-in timestamp — so a run launched against a stale
/// `target/release` binary is obvious in the startup banner. Best-effort:
/// returns `"build age unknown"` if the mtime can't be read.
fn binary_age() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.elapsed().ok())
        .map_or_else(
            || "build age unknown".to_string(),
            |d| format!("built {} ago", humanize_age(d.as_secs())),
        )
}

/// `90s` / `5m` / `2h07m` / `3d4h` — coarsening as the span grows, since a stale
/// binary only needs to be obviously stale, not precisely dated.
fn humanize_age(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86_400 {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d{}h", secs / 86_400, (secs % 86_400) / 3600)
    }
}

/// Best-effort: read the problem name back from a prior run's manifest so
/// `--resume <id>` doesn't need `--avo`.
fn problem_from_manifest(run_dir: &Path) -> Option<PathBuf> {
    let bytes = std::fs::read(run_dir.join("manifest.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let name = v.get("problem")?.as_str()?;
    Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("problems/{name}.py")))
}

/// Best-effort: read the model slug back from a prior run's manifest so
/// `--resume <id>` keeps driving the same model it started with (unless
/// `--model` overrides it).
fn model_from_manifest(run_dir: &Path) -> Option<String> {
    let bytes = std::fs::read(run_dir.join("manifest.json")).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some(v.get("model")?.as_str()?.to_string())
}

/// The mount shell for an AVO run's sandboxes: docs mounted read-only, the
/// trusted evaluator + `problem.py` staged read-only under `_trusted/` so a
/// candidate cannot forge its own score, and host-python readable/PATH. The
/// [`SandboxManager`] holds this and stamps it onto every sandbox it spawns; the
/// agent's editable solution lives in the writable `solution/` tree, scratch at
/// the workspace root. The `_trusted` staging tempdir is owned by the returned
/// [`MountSpec`] so it outlives every sandbox that mounts it read-only.
fn build_mount_spec(problem_path: &Path, skills: &[Skill]) -> std::io::Result<MountSpec> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `agent_ressources/docs` is mounted read-only at `/workspace/docs/`, so the
    // algorithm papers under `docs/papers/` surface at the exact path the
    // system prompt + supervisor nudges tell the agent to read.
    let docs_dir = manifest.join("agent_ressources/docs");
    warn_if_papers_missing(&docs_dir);
    let skills_dir = (!skills.is_empty()).then(|| manifest.join("agent_ressources/skills"));

    // Stage the trusted files into a persistent temp dir; the MountSpec owns it.
    let scripts_dir = manifest.join("scripts");
    let staging = tempfile::Builder::new().prefix("kernelguy-trusted-").tempdir()?;
    for f in ["evaluate.py", "problem_loader.py", "run_kernel.py"] {
        std::fs::copy(scripts_dir.join(f), staging.path().join(f))?;
    }
    std::fs::copy(problem_path, staging.path().join("problem.py"))?;

    let python_readable = detect_host_python_paths();
    let python_bin = detect_host_python_bin_dir();
    // Which interpreter the agent gets decides whether CUDA `torch` is
    // importable at all, so state it up front instead of letting a torch-less
    // pick surface later as an opaque startup-validation failure.
    match &python_bin {
        Some(bin) => eprintln!("kernelguy: agent python3 from {}", bin.display()),
        None => eprintln!("kernelguy: warning: no python3 found; the agent will have no interpreter"),
    }

    Ok(MountSpec {
        docs_dir,
        skills_dir,
        trusted_staging: staging,
        python_readable,
        python_bin,
    })
}

/// The agent's prompts (system prompt, supervisor nudges, kickoff) point at
/// `docs/papers/` as a shared pool of algorithm papers to triage, so a run is
/// degraded if the pool is empty. [`Sandbox::mount_docs`] stages `.md` (and
/// image) files recursively, so each paper reaches the agent as
/// `docs/papers/<Name>/paper.md`; this counts those readable papers and
/// surfaces a "papers not fetched yet" misconfiguration up front, rather than
/// letting the agent discover an empty `docs/papers/` mid-run.
fn warn_if_papers_missing(docs_dir: &Path) {
    let papers_dir = docs_dir.join("papers");
    let papers = count_papers(&papers_dir);
    if papers == 0 {
        eprintln!(
            "kernelguy: warning: no readable papers under {} (expected <Name>/paper.md) — the \
             agent is prompted to triage the papers in docs/papers/; run \
             agent_ressources/.generate/generate.sh to fetch them",
            papers_dir.display()
        );
    } else {
        eprintln!("kernelguy: mounted {papers} reference paper(s) read-only at docs/papers/ (as paper.md)");
    }
}

/// Count readable papers under `dir`: each `<dir>/<Name>/paper.md` — the form
/// [`Sandbox::mount_docs`] stages and the prompts tell the agent to read. (The
/// old check counted top-level `*.pdf`, which never matched the nested
/// `paper.md` layout and so always warned "no PDFs".) A missing or unreadable
/// directory yields `0`.
fn count_papers(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| e.path().join("paper.md").is_file())
        .count()
}

/// How many GPUs one job of `problem_path` needs, from the problem itself.
///
/// Runs `evaluate.py --describe <problem>` and reads `num_gpus`. Python has to answer:
/// `NUM_GPUS` is a module attribute a problem may compute, so matching a pattern in the
/// source would report 1 for a problem needing 4 — and that failure is a hang, because the
/// other ranks never get launched.
///
/// Falls back to 1 on any failure (no interpreter, probe error, unparseable output). A
/// single-GPU problem is the overwhelming case and is also what the pool did before this
/// existed, so degrading that way changes nothing for them; a multi-GPU problem instead
/// fails loudly later, when `evaluate.py` finds fewer visible devices than it asked for.
fn problem_gpus_per_job(problem_path: &Path) -> usize {
    let Some(python) = detect_host_python_bin() else {
        return 1;
    };
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/evaluate.py");
    let out = match ProcessCommand::new(&python)
        .arg(&script)
        .arg("--describe")
        .arg(problem_path)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return 1,
    };
    serde_json::from_slice::<serde_json::Value>(&out.stdout)
        .ok()
        .and_then(|v| v.get("num_gpus").and_then(serde_json::Value::as_u64))
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n >= 1)
        .unwrap_or(1)
}

/// Locate the `python3` whoever launched kernelguy is using, then ask
/// that interpreter for `sys.executable` / `sys.prefix` / `sys.base_prefix`.
/// Returns the deduped set of host paths the sandbox needs read access to
/// for the interpreter to start AND import its standard library.
///
/// Returns an empty vec if `python3` isn't on the harness's `PATH`, or if
/// the interpreter probe fails — in which case the agent simply won't
/// have python.
fn detect_host_python_paths() -> Vec<PathBuf> {
    let Some(bin) = detect_host_python_bin() else {
        return Vec::new();
    };
    let probe = "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)";
    let out = match ProcessCommand::new(&bin).args(["-c", probe]).output() {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let mut paths: Vec<PathBuf> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| PathBuf::from(l.trim()))
        .filter(|p| !p.as_os_str().is_empty())
        .collect();
    if let Some(parent) = bin.parent() {
        paths.push(parent.to_path_buf());
    }
    paths.sort();
    paths.dedup();
    paths
}

fn detect_host_python_bin_dir() -> Option<PathBuf> {
    detect_host_python_bin()?.parent().map(PathBuf::from)
}

/// Resolve the interpreter the agent should get, in descending priority:
///
///   1. `$VIRTUAL_ENV` — an explicitly activated venv is a deliberate choice.
///   2. `<repo>/.venv` — the project's own venv, so a run behaves identically
///      whether or not the launching shell activated it.
///   3. `python3` on `PATH` — a system interpreter.
///
/// Steps 1 and 2 matter because the CUDA-enabled `torch` lives in the venv, not
/// in `/usr/bin/python3`. Relying on `PATH` alone means forgetting to activate
/// silently selects a torch-less interpreter and the run dies in NVIDIA startup
/// validation rather than at the point of the mistake.
fn detect_host_python_bin() -> Option<PathBuf> {
    if let Some(venv) = std::env::var_os("VIRTUAL_ENV").map(PathBuf::from) {
        let bin = venv.join("bin/python3");
        if bin.is_file() {
            return Some(bin);
        }
    }
    let project = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".venv/bin/python3");
    if project.is_file() {
        return Some(project);
    }
    ProcessCommand::new("which")
        .arg("python3")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
        .filter(|p| !p.as_os_str().is_empty())
}
