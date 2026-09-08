//! `SandboxManager` — the git-backed sandbox minter.
//!
//! Owns the content-addressed workspace-snapshot store (a git object store on a
//! host path) **and** the mount-shell config (docs / `_trusted` / skills / CUDA /
//! host-python) that every sandbox is stamped with. It is the **sole creator of
//! sandboxes**: [`spawn`](SandboxManager::spawn) mints a fresh live worktree
//! (a tempdir the subprocess runs against; `Drop` cleans it) materialized from a
//! snapshot hash (or empty), and [`snapshot`](SandboxManager::snapshot) commits a
//! worktree back to a hash. The search tree (`orchestrator::search_tree`) holds a
//! clone of the manager and delegates all filesystem/snapshot work to it, keeping
//! only the node-tree metadata.
//!
//! This is git's own model: object store ↔ worktree. The snapshot store is keyed
//! by an opaque `node_id_hint` used as a ref label (`refs/kg/nodes/<hint>`) + the
//! commit message; the manager knows nothing about the search tree above it.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use tempfile::TempDir;

use crate::domain::types::SolutionFiles;
use crate::exec::sandbox::Sandbox;
use crate::exec::snapshot_store::{
    SKILLS_DIR, SOLUTION_DIR, TRUSTED_DIR, WorkspaceFiles, should_exclude_workspace_path,
};
use crate::exec::turn_tree::WorkspaceSnapshotId;

const SOLUTION_SUBDIR: &str = SOLUTION_DIR;
const NODE_REF_PREFIX: &str = "refs/kg/nodes";

/// The read-only mount shell every spawned sandbox is stamped with.
///
/// Built once by
/// the run's setup (host-python detection + `_trusted` staging), then held by the
/// manager and re-applied to each `spawn`. Owns the `_trusted` staging tempdir so
/// it outlives every spawned sandbox that mounts it read-only.
pub struct MountSpec {
    /// Host dir mounted read-only at `docs/` (papers + markdown assets).
    pub docs_dir: PathBuf,
    /// Host dir mounted read-only at `skills/`, if any.
    pub skills_dir: Option<PathBuf>,
    /// Staged trusted evaluator + `problem.py`, mounted read-only at `_trusted/`.
    pub trusted_staging: TempDir,
    /// Extra host paths made readable (host toolchains, e.g. a pyenv python).
    pub python_readable: Vec<PathBuf>,
    /// Extra `PATH` entry (host python bin dir), if detected.
    pub python_bin: Option<PathBuf>,
}

struct Inner {
    /// Git object store backing the workspace snapshots.
    repo: PathBuf,
    /// Serializes snapshot-authoring git operations (index writes, ref updates).
    write: Mutex<()>,
    mounts: MountSpec,
}

/// Cheaply-cloneable handle to the git-backed sandbox minter.
#[derive(Clone)]
pub struct SandboxManager {
    inner: Arc<Inner>,
}

impl SandboxManager {
    /// Open (or init) the snapshot repo at `repo_root` and hold the mount shell.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if `repo_root` cannot be created, or if the repo has
    /// no `.git` and `git init` fails (the git failure string is wrapped via
    /// [`io::Error::other`]).
    pub fn new(repo_root: impl Into<PathBuf>, mounts: MountSpec) -> io::Result<Self> {
        let repo = repo_root.into();
        fs::create_dir_all(&repo)?;
        if !repo.join(".git").exists() {
            run_git_io(&repo, &["init", "-q", "-b", "main"], None, &[]).map_err(io::Error::other)?;
        }
        Ok(Self {
            inner: Arc::new(Inner {
                repo,
                write: Mutex::new(()),
                mounts,
            }),
        })
    }

    /// A manager with an empty mount shell — for the snapshot/tree logic in tests
    /// and for callers that only save/load snapshots and never [`spawn`](Self::spawn)
    /// a real sandbox. Spawning off a bare manager yields a mount-less sandbox.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the throwaway `_trusted` staging tempdir cannot be
    /// created, or for any reason [`SandboxManager::new`] fails.
    pub fn bare(repo_root: impl Into<PathBuf>) -> io::Result<Self> {
        let repo = repo_root.into();
        let mounts = MountSpec {
            docs_dir: repo.clone(),
            skills_dir: None,
            trusted_staging: TempDir::with_prefix("kernelguy-trusted-bare-")?,
            python_readable: Vec::new(),
            python_bin: None,
        };
        Self::new(repo, mounts)
    }

    /// Mint a fresh live sandbox with the standard mounts, its workspace
    /// materialized from `base` (or empty when `None`). The returned `Sandbox`
    /// owns a tempdir that `Drop` cleans up.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the sandbox tempdir cannot be created, if staging
    /// any read-only mount (`docs`, `skills`, `_trusted`) fails, or if `base` is
    /// `Some` and git cannot extract that snapshot's tree into the fresh workspace.
    pub fn spawn(&self, base: Option<&WorkspaceSnapshotId>) -> io::Result<Sandbox> {
        let mut sb = Sandbox::new()?;
        let m = &self.inner.mounts;
        sb.mount_docs(&m.docs_dir)?;
        if let Some(skills) = &m.skills_dir {
            sb.mount_ro_dir(skills, SKILLS_DIR)?;
        }
        sb.mount_ro_dir(m.trusted_staging.path(), TRUSTED_DIR)?;
        for path in &m.python_readable {
            sb.add_readable(path.clone());
        }
        // CUDA before the host python bin, because `PATH` is ordered and a pip-installed
        // `nvidia-cuda-nvcc` wheel puts an `nvcc` in the venv. The startup preflight
        // validates whichever `nvcc` resolves first, so the toolchain that gets validated
        // should also be the one the agent compiles with.
        let cuda = crate::env::cuda::cuda_toolchain_access();
        for dir in cuda.bin_dirs {
            sb.add_path_entry(dir);
        }
        for root in cuda.roots {
            sb.add_readable(root);
        }
        if let Some(bin) = &m.python_bin {
            sb.add_path_entry(bin.clone());
        }
        // The sandbox is confined to its lease by which device nodes it can open, so
        // resolve them here — the argv builder binds what it is handed and knows
        // nothing about NVIDIA.
        let nodes = crate::env::hardware::gpu_device_nodes();
        sb.gpu_shared_devices.clone_from(&nodes.shared);
        sb.gpu_devices.clone_from(&nodes.per_device);
        sb.gpu_env = crate::env::cuda::cuda_env_passthrough();
        // Forwarded whenever the operator SET it, empty included: an empty
        // `CUDA_VISIBLE_DEVICES` means "no GPUs" to CUDA, and it resolves to zero
        // visible devices here, which lands in the unconfined fallback — so skipping
        // it would hand a job every card on the machine in answer to a request for
        // none. An unset mask is a different case (a host with no driver reads the
        // same way) and stays untouched.
        if nodes.per_device.is_empty()
            && let Ok(mask) = std::env::var("CUDA_VISIBLE_DEVICES")
        {
            // No per-device nodes means no node-level confinement on this host, so the
            // operator's mask is the only thing keeping a job off cards they excluded:
            // forward it as a ceiling. It is identical for every job and carries no lease
            // information. It must stay absent whenever the nodes DO confine, because the
            // leased cards renumber from 0 inside the jail.
            sb.gpu_env.push(("CUDA_VISIBLE_DEVICES".to_string(), mask));
        }
        if let Some(id) = base {
            // Fresh sandbox ⇒ empty workspace, so a direct git extraction gives an
            // exact mirror of the node's tree (byte-exact, incl. binary artifacts).
            self.extract_tree_into(id.as_str(), sb.workspace())
                .map_err(io::Error::other)?;
        }
        Ok(sb)
    }

    // ── snapshot store (git object store) ────────────────────────────────────

    /// Commit `files` as a workspace snapshot (delta against `parent`), label it
    /// `refs/kg/nodes/<node_id_hint>`, and return the commit id. The hint is an
    /// opaque label — the manager is agnostic to the search tree above it.
    ///
    /// # Errors
    ///
    /// Returns the failing git invocation's stderr if any step of the commit fails:
    /// creating the throwaway index, `hash-object` for a blob, `update-index`,
    /// `write-tree`, `commit-tree` (e.g. `parent` names an unknown commit), or the
    /// `update-ref` that labels the result.
    pub fn save_snapshot(
        &self,
        node_id_hint: &str,
        parent: Option<&WorkspaceSnapshotId>,
        files: &BTreeMap<String, Vec<u8>>,
    ) -> Result<WorkspaceSnapshotId, String> {
        let _write = crate::domain::util::lock(&self.inner.write);
        let commit = self.author_workspace_commit(node_id_hint, parent.map(WorkspaceSnapshotId::as_str), files)?;
        self.set_node_ref(node_id_hint, &commit)?;
        Ok(WorkspaceSnapshotId::new(commit))
    }

    /// Load only the `solution/` subtree of a snapshot (rel paths under `solution/`).
    ///
    /// # Errors
    ///
    /// Returns the failing git invocation's stderr if `snapshot_id` names no commit,
    /// or if an `ls-tree`/`show` fails. Note this path decodes blobs *lossily*, so a
    /// non-UTF-8 file is mangled rather than rejected — use
    /// [`load_workspace_snapshot`](Self::load_workspace_snapshot) for byte-exact reads.
    pub fn load_solution_snapshot(&self, snapshot_id: &WorkspaceSnapshotId) -> Result<SolutionFiles, String> {
        let commit = snapshot_id.as_str();
        let listing = self.git(&["ls-tree", "-r", "-z", "--name-only", commit, "--", SOLUTION_SUBDIR])?;
        let prefix = format!("{SOLUTION_SUBDIR}/");
        let mut out = SolutionFiles::new();
        for path in listing.split('\0').filter(|s| !s.is_empty()) {
            let content = self.git(&["show", &format!("{commit}:{path}")])?;
            let rel = path.strip_prefix(&prefix).unwrap_or(path).to_string();
            out.insert(rel, content);
        }
        if out.is_empty() {
            let legacy_listing = self.git(&["ls-tree", "-r", "-z", "--name-only", commit])?;
            for path in legacy_listing.split('\0').filter(|s| !s.is_empty()) {
                if path == "sidecar.json" || path.starts_with(&prefix) {
                    continue;
                }
                let content = self.git(&["show", &format!("{commit}:{path}")])?;
                out.insert(path.to_string(), content);
            }
        }
        Ok(out)
    }

    /// Load the whole workspace subtree of a snapshot (excluding read-only mounts).
    ///
    /// # Errors
    ///
    /// Returns the failing git invocation's stderr if `snapshot_id` names no commit,
    /// or if the `ls-tree` listing or any per-blob `show` fails.
    pub fn load_workspace_snapshot(&self, snapshot_id: &WorkspaceSnapshotId) -> Result<WorkspaceFiles, String> {
        let commit = snapshot_id.as_str();
        let listing = self.git(&["ls-tree", "-r", "-z", "--name-only", commit])?;
        let mut out = WorkspaceFiles::new();
        for path in listing.split('\0').filter(|s| !s.is_empty()) {
            if should_exclude_workspace_path(path) || path == "sidecar.json" {
                continue;
            }
            // Blob content read as raw bytes — the workspace may hold binary
            // artifacts (profiler dumps, etc.) that a String decode would corrupt.
            let content = self.git_bytes(&["show", &format!("{commit}:{path}")])?;
            out.insert(path.to_string(), content);
        }
        Ok(out)
    }

    /// Materialize a snapshot's tree directly into `dest_dir` using git's own
    /// tree→worktree extraction — `read-tree` into a throwaway index, then
    /// `checkout-index -a -f --prefix=<dest>/`. Two git processes total regardless
    /// of file count (vs `1 + N` for the per-blob `git show` path), and byte-exact
    /// by construction: git writes blob bytes raw, so binary artifacts (profiler
    /// dumps, etc.) survive intact without ever passing through a `String`.
    ///
    /// The committed tree is already source-filtered at snapshot time (see
    /// [`WorkspaceSnapshot::from_dir`]), so no read-side exclusion is needed.
    /// `checkout-index` overwrites but does not delete pre-existing files, so the
    /// caller must ensure `dest_dir` is empty/wiped when an exact mirror is wanted
    /// (a fresh sandbox workspace already is). Returns the number of files written.
    ///
    /// # Errors
    ///
    /// Returns a message if `dest_dir` cannot be created, if the throwaway index
    /// tempdir cannot be made inside the repo, or if `read-tree` (e.g. `commit` names
    /// no tree), `checkout-index` (a path in `dest_dir` is unwritable), or the final
    /// `ls-files` count fails.
    pub fn extract_tree_into(&self, commit: &str, dest_dir: &Path) -> Result<usize, String> {
        std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
        let tmp = tempfile::tempdir_in(&self.inner.repo).map_err(|e| e.to_string())?;
        let index_path = tmp.path().join("index").to_string_lossy().to_string();
        self.git_index(&index_path, &["read-tree", commit])?;
        // `--prefix` is prepended verbatim to each index path, so it must end in `/`
        // to land files under the directory; checkout-index creates leading dirs.
        let prefix = format!("--prefix={}/", dest_dir.to_string_lossy());
        self.git_index(&index_path, &["checkout-index", "-a", "-f", &prefix])?;
        let listed = self.git_index(&index_path, &["ls-files", "-z"])?;
        Ok(listed.split('\0').filter(|s| !s.is_empty()).count())
    }

    // ── git plumbing ─────────────────────────────────────────────────────────

    fn git(&self, args: &[&str]) -> Result<String, String> {
        run_git_io(&self.inner.repo, args, None, &[])
    }

    /// Byte-exact git read (blob content that may be binary). See [`run_git_bytes`].
    fn git_bytes(&self, args: &[&str]) -> Result<Vec<u8>, String> {
        run_git_bytes(&self.inner.repo, args, None, &[])
    }

    fn git_stdin(&self, args: &[&str], bytes: &[u8]) -> Result<String, String> {
        run_git_io(&self.inner.repo, args, Some(bytes), &[])
    }

    fn git_index(&self, index: &str, args: &[&str]) -> Result<String, String> {
        run_git_io(&self.inner.repo, args, None, &[("GIT_INDEX_FILE", index)])
    }

    fn set_node_ref(&self, node_id: &str, commit: &str) -> Result<(), String> {
        self.git(&["update-ref", &format!("{NODE_REF_PREFIX}/{node_id}"), commit])
            .map(|_| ())
    }

    fn author_workspace_commit(
        &self,
        node_id: &str,
        parent_commit: Option<&str>,
        files: &BTreeMap<String, Vec<u8>>,
    ) -> Result<String, String> {
        let tree = self.write_tree_from_entries(files)?;
        self.commit_tree(&tree, parent_commit, &format!("history node {node_id}"))
    }

    fn write_tree_from_entries(&self, entries: &BTreeMap<String, Vec<u8>>) -> Result<String, String> {
        let tmp = tempfile::tempdir_in(&self.inner.repo).map_err(|e| e.to_string())?;
        let index = tmp.path().join("index");
        let index_path = index.to_string_lossy().to_string();
        self.git_index(&index_path, &["read-tree", "--empty"])?;
        for (path, bytes) in entries {
            let blob = self
                .git_stdin(&["hash-object", "-w", "--stdin"], bytes)?
                .trim()
                .to_string();
            self.git_index(
                &index_path,
                &["update-index", "--add", "--cacheinfo", "100644", &blob, path],
            )?;
        }
        self.git_index(&index_path, &["write-tree"])
            .map(|s| s.trim().to_string())
    }

    fn commit_tree(&self, tree: &str, parent_commit: Option<&str>, msg: &str) -> Result<String, String> {
        let mut args = vec!["commit-tree", tree];
        if let Some(parent) = parent_commit {
            args.push("-p");
            args.push(parent);
        }
        args.push("-m");
        args.push(msg);
        self.git(&args).map(|s| s.trim().to_string())
    }
}

/// Run git and return raw stdout **bytes** (no UTF-8 decoding). Use this for blob
/// content that may be binary (e.g. profiler outputs, any non-UTF-8 artifact) — a
/// filesystem holds bytes, not strings, and lossy decoding would corrupt them.
pub(crate) fn run_git_bytes(
    repo: &Path,
    args: &[&str],
    stdin_bytes: Option<&[u8]>,
    envs: &[(&str, &str)],
) -> Result<Vec<u8>, String> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(repo).args(args);
    cmd.env("GIT_AUTHOR_NAME", "kernelguy");
    cmd.env("GIT_AUTHOR_EMAIL", "kernelguy@local");
    cmd.env("GIT_COMMITTER_NAME", "kernelguy");
    cmd.env("GIT_COMMITTER_EMAIL", "kernelguy@local");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    if stdin_bytes.is_some() {
        cmd.stdin(Stdio::piped());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("git spawn failed: {e}"))?;
    // `stdin` is piped above exactly when `stdin_bytes.is_some()`, so the second
    // binding always succeeds; matching on it beats re-unwrapping the same Option.
    if let Some(bytes) = stdin_bytes
        && let Some(stdin) = child.stdin.as_mut()
    {
        stdin.write_all(bytes).map_err(|e| format!("git stdin failed: {e}"))?;
    }
    let out = child.wait_with_output().map_err(|e| format!("git wait failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

/// String-returning git (paths, ids, listings). Lossy-decodes stdout — never use
/// for blob *content* that could be binary; use [`run_git_bytes`] there.
pub(crate) fn run_git_io(
    repo: &Path,
    args: &[&str],
    stdin_bytes: Option<&[u8]>,
    envs: &[(&str, &str)],
) -> Result<String, String> {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(repo).args(args);
    cmd.env("GIT_AUTHOR_NAME", "kernelguy");
    cmd.env("GIT_AUTHOR_EMAIL", "kernelguy@local");
    cmd.env("GIT_COMMITTER_NAME", "kernelguy");
    cmd.env("GIT_COMMITTER_EMAIL", "kernelguy@local");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    if stdin_bytes.is_some() {
        cmd.stdin(Stdio::piped());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("git spawn failed: {e}"))?;
    // See `run_git_bytes`: piped iff there are bytes to write.
    if let Some(bytes) = stdin_bytes
        && let Some(stdin) = child.stdin.as_mut()
    {
        stdin.write_all(bytes).map_err(|e| format!("git stdin failed: {e}"))?;
    }
    let out = child.wait_with_output().map_err(|e| format!("git wait failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}
