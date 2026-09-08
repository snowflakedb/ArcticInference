//! Process and filesystem sandboxing for tool calls.
//!
//! The agent's workspace tools (`bash` and the upcoming `read` / `write`)
//! all bottom out at `bwrap` (bubblewrap): it spawns the command in fresh
//! mount / IPC / UTS / network / cgroup namespaces, binds `/workspace` to
//! a per-session host directory, and bind-mounts a curated set of
//! read-only system paths so binaries and shared libraries load cleanly.
//!
//! The contract:
//! - Standard runtime filesystems are accessible read-only (binaries +
//!   shared libraries can be loaded).
//! - The workspace and a per-sandbox `/tmp` directory are read-write and
//!   persist across [`Sandbox::run`] / [`Sandbox::run_streaming`] calls (so
//!   build caches and intermediates survive between `bash()` invocations).
//! - [`Sandbox::run_streaming`] launches the command in a fresh process
//!   group and `killpg(-pid, SIGKILL)`s the entire group on timeout — bwrap's
//!   `--die-with-parent` only cascades to the immediate child, leaving
//!   grandchildren under `sh -c` to escape an otherwise normal kill. A command
//!   that starts a new session escapes the group kill too, so the timeout then
//!   sweeps the sandbox's whole mount namespace, and stops draining output
//!   rather than waiting on a pipe a process it could not reach still holds.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

mod argv_linux;
use argv_linux as backend;

#[cfg(not(target_os = "linux"))]
compile_error!("kernelguy::exec::sandbox supports only linux (bwrap)");

/// How long a timed-out command's output is still drained after the kill.
///
/// Bounded on purpose: a process that escaped the kill keeps the pipe's write end
/// open, so EOF may never arrive, and waiting for it would stall the caller — and
/// with it the run's wall-clock enforcement. Long enough for a reaped child's
/// buffered output to arrive, short enough to be invisible against any real timeout.
const POST_KILL_DRAIN_GRACE: Duration = Duration::from_secs(5);

/// Environment variable tagging every process inside a sandbox.
///
/// Set unconditionally by the argv builder under `--clearenv`, so it is present in
/// exactly the sandbox's own process tree and inherited across fork, exec, `setsid`
/// and reparenting to init. That inheritance is what makes it usable for cleanup: it
/// still identifies an orphan after every process that could have anchored a
/// process-tree or namespace lookup has exited.
pub(crate) const SANDBOX_ID_VAR: &str = "KG_SANDBOX_ID";

/// Retry cadence and budget for resolving the sandbox's mount namespace.
///
/// `bwrap` has not unshared yet at spawn, and a silent command generates no other
/// event to piggyback on, so the resolution is retried on its own short timer until it
/// lands — then the arm disables itself. The budget bounds the `/proc` scans on a host
/// where it can never resolve (no procfs).
const NS_RESOLVE_INTERVAL: Duration = Duration::from_millis(50);
const NS_RESOLVE_ATTEMPTS: u32 = 40;

/// A bind-mount entry.
///
/// Becomes a `bwrap` `--ro-bind` / `--bind` argument.
#[derive(Debug, Clone)]
pub struct Mount {
    /// Source path on the host.
    pub host: PathBuf,
    /// Destination path inside the sandbox. Must start with
    /// `/workspace/...` for the overlay logic in [`Sandbox::host_path`] to
    /// pick it up.
    pub guest: String,
}

/// Outcome of a [`Sandbox::run`] / [`Sandbox::run_streaming`] call.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// Process exit code. `124` indicates we killed the process group on
    /// timeout (mirrors `coreutils timeout(1)` and the Python reference).
    /// `-1` indicates termination by a signal we didn't send.
    pub returncode: i32,
    /// Merged stdout + stderr captured during the run, with a trailing
    /// newline after each line. Lines are preserved in the order they hit
    /// our pipes; cross-stream interleaving may differ slightly from
    /// kernel-level merging.
    pub stdout: String,
    /// True iff [`Self::returncode`] is `124` because we killed the process
    /// group on timeout.
    pub timed_out: bool,
}

/// A per-session sandbox.
///
/// Owns its workspace directory (and a sibling
/// `tmp/` for the in-sandbox `/tmp`), and knows how to spawn commands
/// inside the OS-native sandboxing primitive (`bwrap`).
///
/// Cheap to wrap in `Arc`. All run / read / write methods take `&self`;
/// only mount setup needs `&mut self`.
pub struct Sandbox {
    root: PathBuf,
    workspace: PathBuf,
    tmp: PathBuf,
    /// Read-only bind mounts, in declaration order. Last-mount-wins semantics
    /// (see [`Sandbox::host_path`]).
    pub ro: Vec<Mount>,
    /// Read-write bind mounts, in declaration order.
    pub rw: Vec<Mount>,
    /// Extra host paths the sandbox should be able to *read*. Each becomes a
    /// passthrough `--ro-bind <path> <path>` so the path is visible at
    /// the same location inside the sandbox. Used to expose host
    /// toolchains (e.g. a pyenv-installed python) without granting
    /// blanket access to the user's home directory.
    pub readable: Vec<PathBuf>,
    /// Extra entries prepended to the agent's `PATH`. The argv builder
    /// otherwise hardcodes `/usr/local/bin:/usr/bin:/bin` and would never
    /// see e.g. a pyenv shim dir without this knob.
    pub path_entries: Vec<PathBuf>,
    /// Device nodes bound into every GPU-enabled run, whichever devices it
    /// leases. Driver control and fabric interfaces live here.
    ///
    /// Populated at creation from [`crate::env::hardware::gpu_device_nodes`];
    /// the argv builder only binds what it is given and knows nothing about
    /// NVIDIA.
    pub gpu_shared_devices: Vec<PathBuf>,
    /// Device nodes owned by one device each, indexed by device pool slot — so a
    /// lease on slot `i` binds `gpu_devices[i]`.
    ///
    /// Empty means no per-device confinement is available on this host, in which
    /// case every card's node is in `gpu_shared_devices` instead and a run sees
    /// all of them.
    pub gpu_devices: Vec<PathBuf>,
    /// Extra environment applied to GPU-enabled runs only (`CUDA_HOME` and
    /// friends), as `(key, value)`.
    ///
    /// Emitted *after* the argv builder's own `--setenv` block, and bwrap lets
    /// the last one win — so an entry keyed `PATH` or `HOME` would override the
    /// harness's. Do not put one here.
    pub gpu_env: Vec<(String, String)>,
    /// Default per-call timeout for [`Sandbox::run`] / [`Sandbox::run_streaming`].
    /// Callers can override with the `timeout` parameter.
    pub timeout: Duration,
    /// When `true`, the on-disk root survives Drop (callers can resume into
    /// the same workspace later). Defaults to `false` for tempdir-backed
    /// sandboxes; `with_root()` always persists by design.
    pub keep: bool,
    /// `Some` when the root is a tempdir owned by us. Drop cleans it up
    /// (unless `keep == true`, in which case we forget the handle so the
    /// directory persists).
    tmpdir: Option<TempDir>,
    /// Per-canonical-path async locks, created on demand. A `write`/`edit`
    /// holds the lock for its file across the whole read-modify-write, so two
    /// concurrent mutations of the same path (e.g. a `gpu_job`-driven write
    /// racing an `edit`) can't interleave and lose an update.
    locks: Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>,
}

impl Sandbox {
    /// Create a sandbox rooted at a fresh tempdir. The directory is removed
    /// on Drop unless [`Sandbox::keep`] is set to `true`. Crate-internal: the
    /// `SandboxManager` is the sole creator of sandboxes in production; tests
    /// construct bare sandboxes directly.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the tempdir cannot be created, or for any reason
    /// [`Sandbox::init`] fails (canonicalization, `workspace/`+`tmp/` creation, or
    /// the backend availability check).
    pub(crate) fn new() -> io::Result<Self> {
        let tmp = TempDir::with_prefix("kernelguy-sandbox-")?;
        let root = tmp.path().to_path_buf();
        Self::init(&root, Some(tmp))
    }

    /// Create a sandbox rooted at `root`. The directory is created if it
    /// doesn't exist and persists across drops (callers that want cleanup
    /// should use [`Sandbox::new`] instead).
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if `root` cannot be created or canonicalized, if
    /// `workspace/` or `tmp/` cannot be created under it, or if the platform
    /// sandboxing binary (`bwrap`) is missing.
    pub fn with_root(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Self::init(&root, None)
    }

    fn init(root: &Path, tmpdir: Option<TempDir>) -> io::Result<Self> {
        // Canonicalize once so workspace/tmp paths use the resolved form.
        let root = std::fs::canonicalize(root)?;
        let workspace = root.join("workspace");
        let tmp = root.join("tmp");
        std::fs::create_dir_all(&workspace)?;
        std::fs::create_dir_all(&tmp)?;
        backend::check_available()?;
        Ok(Self {
            root,
            workspace,
            tmp,
            ro: Vec::new(),
            rw: Vec::new(),
            readable: Vec::new(),
            path_entries: Vec::new(),
            gpu_shared_devices: Vec::new(),
            gpu_devices: Vec::new(),
            gpu_env: Vec::new(),
            timeout: Duration::from_mins(1),
            keep: false,
            tmpdir,
            locks: Mutex::new(HashMap::new()),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Value this sandbox tags its processes with, via [`SANDBOX_ID_VAR`].
    ///
    /// The root directory's name: unique per sandbox (it is a fresh tempdir), stable for
    /// the sandbox's life, and not a host path — so exposing it inside the jail reveals
    /// nothing about the host layout. Falls back to the whole path if the root has no
    /// final component, which only a filesystem root would.
    pub(crate) fn marker(&self) -> &str {
        self.root
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_else(|| self.root.to_str().unwrap_or("kernelguy-sandbox"))
    }
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }
    pub fn tmp(&self) -> &Path {
        &self.tmp
    }

    /// Add a read-only bind mount.
    pub fn add_ro(&mut self, host: impl Into<PathBuf>, guest: impl Into<String>) {
        self.ro.push(Mount {
            host: host.into(),
            guest: guest.into(),
        });
    }

    /// Add a read-write bind mount.
    pub fn add_rw(&mut self, host: impl Into<PathBuf>, guest: impl Into<String>) {
        self.rw.push(Mount {
            host: host.into(),
            guest: guest.into(),
        });
    }

    /// Make `path` (and everything beneath it) readable from inside the
    /// sandbox at the same host path. Use this to expose host toolchains
    /// without granting blanket read access to wider parents (typical case:
    /// a pyenv-installed Python under `~/.pyenv` that the agent should be
    /// able to invoke).
    ///
    /// Emits a passthrough `--ro-bind path path` so bwrap exposes the host
    /// tree at the same location inside the sandbox.
    pub fn add_readable(&mut self, path: impl Into<PathBuf>) {
        self.readable.push(path.into());
    }

    /// Prepend `bin_dir` to the agent's `PATH`, and expose it as readable
    /// (calls [`Self::add_readable`] internally). Use when a tool's
    /// directory needs to be on `PATH` for the agent's `bash` to find it
    /// without the agent typing an absolute path.
    ///
    /// `add_readable` plus a record kept in `path_entries` that the argv
    /// builder prepends to the hardcoded `/usr/local/bin:/usr/bin:/bin`.
    pub fn add_path_entry(&mut self, bin_dir: impl Into<PathBuf>) {
        let p = bin_dir.into();
        self.path_entries.push(p.clone());
        self.add_readable(p);
    }

    /// Stage every `.md` / `.pdf` file, plus markdown-displayable image
    /// assets, under `dir` as a read-only copy at
    /// `/workspace/docs/<rel>`. Idempotent: a prior call's staging is wiped
    /// first so re-mounting picks up host-side additions / removals cleanly.
    /// If `dir` doesn't exist, this is a no-op.
    ///
    /// Layout (see [`Self::docs_host`] for the accessor): copies land under
    /// `<sandbox.root>/_ro_docs/<rel>`, a staging dir *outside* the writable
    /// workspace bind. The argv builder emits a single
    /// `--ro-bind <_ro_docs> /workspace/docs`, so the kernel itself enforces
    /// read-only.
    ///
    /// The staged copy is a true filesystem snapshot, not
    /// a symlink — so the agent's view of the file never reveals the
    /// upstream host path, and edits to the upstream after `mount_docs`
    /// returns are not visible mid-session.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the previous staging tree cannot be removed, if
    /// walking `dir` fails, or if creating a staging subdirectory or copying a file
    /// into it fails. A `dir` that is not a directory is not an error — it is a
    /// no-op.
    pub fn mount_docs(&mut self, dir: impl AsRef<Path>) -> io::Result<()> {
        // Guest mount path derived from the canonical `DOCS_DIR` so it can't drift
        // from the snapshot-exclusion predicate.
        let docs_guest = format!("/workspace/{}", crate::exec::snapshot_store::DOCS_DIR);
        let docs_guest_prefix = format!("{docs_guest}/");
        self.ro
            .retain(|m| m.guest != docs_guest && !m.guest.starts_with(&docs_guest_prefix));

        // Wipe the previous staging tree so a re-mount doesn't leave behind
        // copies whose upstream sources have since been deleted.
        let staging = self.docs_host();
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }

        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Ok(());
        }

        let mut files = Vec::new();
        walk_files(dir, &mut files)?;
        files.sort();
        let mut copied_any = false;
        for file in files {
            let suffix = file.extension().and_then(|s| s.to_str()).map(str::to_ascii_lowercase);
            if !matches!(
                suffix.as_deref(),
                Some("md" | "pdf" | "png" | "jpg" | "jpeg" | "gif" | "webp" | "svg")
            ) {
                continue;
            }
            let rel = file
                .strip_prefix(dir)
                .map_err(|_| invalid(format!("walk yielded {} outside {}", file.display(), dir.display())))?;
            let dst = staging.join(rel);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&file, &dst)?;
            copied_any = true;
        }

        // Single overlay entry covers the whole docs dir. The argv builder's
        // ro-bind keys off this Mount; `host_path`
        // and `host_path_for_write` resolve through the directory-prefix
        // branch in `overlay_for`.
        if copied_any {
            self.add_ro(&staging, docs_guest);
        }
        Ok(())
    }

    /// Where the `mount_docs` staging tree lives on the host: a sibling of
    /// `workspace/` so it sits *outside* the writable bind.
    fn docs_host(&self) -> PathBuf {
        self.root.join("_ro_docs")
    }

    /// Stage every file under `src` as a read-only snapshot at
    /// `/workspace/<guest_subdir>/<rel>`. This is the general form of
    /// [`Self::mount_docs`] (which is doc/image-only under `docs`): the AVO
    /// orchestrator uses it to mount the trusted evaluator (`evaluate.py`,
    /// `problem.py`, `problem_loader.py`, `run_kernel.py`) read-only so a
    /// candidate cannot tamper with its own score.
    ///
    /// Enforcement matches `mount_docs`: a `--ro-bind`, with staging outside
    /// the writable workspace. Idempotent: a prior staging of the same
    /// `guest_subdir` is wiped and re-copied.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the previous staging tree cannot be removed, if
    /// walking `src` fails, or if creating a staging subdirectory or copying a file
    /// into it fails. A `src` that is not a directory is a no-op, not an error.
    pub fn mount_ro_dir(&mut self, src: impl AsRef<Path>, guest_subdir: &str) -> io::Result<()> {
        let name = guest_subdir.trim_matches('/');
        let guest = format!("/workspace/{name}");
        let guest_prefix = format!("{guest}/");
        self.ro
            .retain(|m| m.guest != guest && !m.guest.starts_with(&guest_prefix));

        let staging = self.ro_staging(name);
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }

        let src = src.as_ref();
        if !src.is_dir() {
            return Ok(());
        }
        let mut files = Vec::new();
        walk_files(src, &mut files)?;
        files.sort();
        let mut copied_any = false;
        for file in files {
            let rel = file
                .strip_prefix(src)
                .map_err(|_| invalid(format!("walk yielded {} outside {}", file.display(), src.display())))?;
            let dst = staging.join(rel);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(&file, &dst)?;
            copied_any = true;
        }
        if copied_any {
            self.add_ro(&staging, guest);
        }
        Ok(())
    }

    /// Host staging location for a [`Self::mount_ro_dir`] guest subdir, placed
    /// outside the writable workspace bind.
    fn ro_staging(&self, name: &str) -> PathBuf {
        self.root.join(format!("_ro_{}", name.replace('/', "_")))
    }

    /// Resolve a workspace-relative path to its placeholder location under
    /// the host workspace dir. Accepts both styles for the same target:
    /// workspace-relative (`"notes/foo.md"`) or in-sandbox absolute
    /// (`"/workspace/notes/foo.md"`). Returns `Err(InvalidInput)` if the
    /// path tries to escape the workspace via `..` components or is
    /// otherwise not a valid workspace-relative path.
    ///
    /// NOTE: For paths covered by an `ro`/`rw` overlay, the placeholder may
    /// not contain the live content — use [`Sandbox::host_path`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `name` is absolute (after the
    /// optional `/workspace` prefix is stripped), contains a root/prefix component,
    /// or uses `..` to rise above the workspace root.
    pub fn path(&self, name: &str) -> io::Result<PathBuf> {
        let s = name
            .strip_prefix("/workspace")
            .map_or(name, |s| s.trim_start_matches('/'));

        let p = Path::new(s);
        if p.is_absolute() {
            return Err(invalid(format!("absolute path not allowed: {name}")));
        }

        let mut depth: usize = 0;
        let mut out = self.workspace.clone();
        for comp in p.components() {
            match comp {
                Component::Normal(s) => {
                    out.push(s);
                    depth = depth.saturating_add(1);
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    if depth == 0 {
                        return Err(invalid(format!("path escapes workspace: {name}")));
                    }
                    depth = depth.saturating_sub(1);
                    out.pop();
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(invalid(format!("absolute path not allowed: {name}")));
                }
            }
        }
        Ok(out)
    }

    /// Resolve a workspace path to where its actual contents live on the
    /// host. For paths covered by a bind-mount overlay, returns the bound
    /// source path (so host-side reads see the same content the sandboxed
    /// shell sees through the bind). For everything else, returns the
    /// placeholder under the workspace.
    ///
    /// # Errors
    ///
    /// Propagates [`Sandbox::path`]'s [`io::ErrorKind::InvalidInput`] for a `name`
    /// that is absolute or escapes the workspace.
    pub fn host_path(&self, name: &str) -> io::Result<PathBuf> {
        let placeholder = self.path(name)?;
        Ok(self
            .overlay_for(&self.guest_path(&placeholder))
            .map(|(host, _)| host)
            .unwrap_or(placeholder))
    }

    /// Like [`Self::host_path`], but errors if the path is covered by a
    /// read-only overlay — writing to the placeholder would either corrupt
    /// the upstream source or be silently shadowed by the bind-mount, so
    /// neither is what the caller meant.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `name` is not a valid
    /// workspace-relative path (see [`Sandbox::path`]), or if it is covered by a
    /// read-only overlay.
    pub fn host_path_for_write(&self, name: &str) -> io::Result<PathBuf> {
        let placeholder = self.path(name)?;
        match self.overlay_for(&self.guest_path(&placeholder)) {
            None => Ok(placeholder),
            Some((_, "ro")) => Err(invalid(format!("{name} is read-only (covered by an ro overlay)"))),
            Some((host, _)) => Ok(host),
        }
    }

    /// Read a workspace file as UTF-8.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] for a `name` outside the workspace,
    /// or the underlying [`io::Error`] if the file is missing, unreadable, or not
    /// valid UTF-8.
    pub fn read(&self, name: &str) -> io::Result<String> {
        std::fs::read_to_string(self.host_path(name)?)
    }

    /// Write `data` to a workspace file, creating parent directories as
    /// needed. Returns the host path that was written.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `name` is outside the workspace or
    /// covered by a read-only overlay, or the underlying [`io::Error`] if the parent
    /// directories or the file itself cannot be written.
    pub fn write(&self, name: &str, data: &str) -> io::Result<PathBuf> {
        let p = self.host_path_for_write(name)?;
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&p, data)?;
        Ok(p)
    }

    /// Acquire the per-path lock for `name`, returning a guard to hold across a
    /// read-modify-write (`edit`) or a `write`. Serializes concurrent mutations
    /// of the *same* file without a hidden `&mut` — the caller holds the guard
    /// for exactly the critical section:
    ///
    /// ```ignore
    /// let _g = sandbox.lock_path(&path).await?;
    /// let cur = sandbox.read(&path)?;      // read-modify-write, now atomic
    /// sandbox.write(&path, &edited)?;
    /// ```
    ///
    /// The key is the canonicalized host path, so both accepted spellings
    /// (`"foo.py"` and `"/workspace/foo.py"`) and overlay-resolved paths share
    /// one lock. A not-yet-created file falls back to its resolved host path
    /// (canonicalization only succeeds for existing paths).
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `name` is not a valid
    /// workspace-relative path (see [`Sandbox::path`]).
    pub async fn lock_path(&self, name: &str) -> io::Result<tokio::sync::OwnedMutexGuard<()>> {
        let host = self.host_path(name)?;
        let key = std::fs::canonicalize(&host).unwrap_or(host);
        let mutex = {
            let mut map = crate::domain::util::lock(&self.locks);
            map.entry(key)
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        Ok(mutex.lock_owned().await)
    }

    /// Read every file under workspace `subdir` into a map keyed by path
    /// relative to that subdir (forward-slash separated). A missing `subdir`
    /// yields an empty map. Files are read as UTF-8 text. This is how the
    /// orchestrator snapshots the agent's whole (multi-file) solution.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `subdir` is outside the workspace,
    /// or the underlying [`io::Error`] if the walk or a file read fails. Non-UTF-8
    /// files are skipped with a warning rather than erroring (see the comment in the
    /// body).
    pub fn read_tree(&self, subdir: &str) -> io::Result<BTreeMap<String, String>> {
        let root = self.host_path(subdir)?;
        let mut out = BTreeMap::new();
        if !root.is_dir() {
            return Ok(out);
        }
        let mut files = Vec::new();
        walk_files(&root, &mut files)?;
        files.sort();
        for file in files {
            let rel = file
                .strip_prefix(&root)
                .map_err(|_| invalid(format!("walk yielded {} outside {}", file.display(), root.display())))?
                .to_string_lossy()
                .replace('\\', "/");
            if crate::exec::snapshot_store::should_exclude_workspace_path(&rel) {
                continue;
            }
            // Skip non-UTF-8 files instead of failing the whole read. The agent's
            // workflow (methodology step 6) is to compile standalone test kernels,
            // which drop ELF binaries (`test_flash`, `nvcc -o test`, `.ncu-rep`,
            // `.cubin`, …) into `solution/` — extensionless or otherwise not caught
            // by the canonical exclusion predicate. Those binaries are not part of the
            // scored source (they rebuild on eval) and would `read_to_string`-error out
            // the entire snapshot, blocking `evaluate` and destabilizing the on-disk sha.
            // Skipping them keeps the snapshot to real source and lets scoring proceed.
            let bytes = std::fs::read(&file)?;
            match String::from_utf8(bytes) {
                Ok(text) => {
                    out.insert(rel, text);
                }
                Err(_) => {
                    eprintln!(
                        "[read_tree] skipping non-UTF-8 file {rel} (binary artifact; not part of the source snapshot)"
                    );
                }
            }
        }
        Ok(out)
    }

    /// Replace the contents of workspace `subdir` with `files` (keyed by path
    /// relative to the subdir). Any existing tree under `subdir` is removed
    /// first, so a restore is exact — no stale files from a previous candidate
    /// linger. Used to materialize a committed/seed solution into the sandbox.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `subdir` is outside the workspace
    /// or covered by a read-only overlay, or the underlying [`io::Error`] if wiping
    /// the old tree, creating directories, or writing a file fails.
    pub fn write_tree<V: AsRef<[u8]>>(&self, subdir: &str, files: &BTreeMap<String, V>) -> io::Result<()> {
        let root = self.host_path_for_write(subdir)?;
        if root.exists() {
            std::fs::remove_dir_all(&root)?;
        }
        std::fs::create_dir_all(&root)?;
        for (rel, content) in files {
            let dst = root.join(rel);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&dst, content.as_ref())?;
        }
        Ok(())
    }

    /// Run a GPU command inside the sandbox and collect its merged
    /// stdout+stderr. `timeout` overrides [`Sandbox::timeout`] when `Some`.
    /// `devices` are pool lease indices; only those devices' nodes are bound, so the
    /// subprocess cannot open any other card. `None` binds every device this sandbox
    /// was given (used by one-time startup CUDA validation, which runs before the
    /// pool exists).
    ///
    /// # Errors
    ///
    /// See [`Sandbox::run_streaming`] — an empty `cmd`, a failure building the
    /// backend argv, or a spawn/IO failure. A non-zero exit or a timeout is *not* an
    /// error; it is reported in [`RunOutput`].
    pub async fn run(
        &self,
        cmd: &[&str],
        timeout: Option<Duration>,
        devices: Option<&[usize]>,
    ) -> io::Result<RunOutput> {
        self.run_streaming(cmd, timeout, true, devices, |_| {}).await
    }

    /// Convenience: `run(&["sh", "-c", cmd], …)`.
    ///
    /// Note: NOT a login shell (no `-l`). Sourcing the host user's
    /// `/etc/profile` + `~/.profile` is undesirable — the sandbox should run
    /// in a controlled env, not pick up arbitrary shell customizations.
    ///
    /// # Errors
    ///
    /// See [`Sandbox::run_streaming`]. A non-zero exit from `cmd` is reported in
    /// [`RunOutput`], not as an error.
    pub async fn shell(&self, cmd: &str, timeout: Option<Duration>) -> io::Result<RunOutput> {
        self.run(&["sh", "-c", cmd], timeout, None).await
    }

    /// Run a command, streaming each output line through `on_line` as it
    /// arrives. On timeout, kills the entire process group via
    /// `killpg(-pid, SIGKILL)`.
    ///
    /// Lines are passed to `on_line` without the trailing newline. The
    /// returned [`RunOutput::stdout`] reconstructs newlines for caller
    /// convenience.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `cmd` is empty or the backend
    /// produced an empty argv, and the underlying [`io::Error`] if the runtime cache
    /// dirs cannot be prepared, the backend argv cannot be built, the sandbox binary
    /// cannot be spawned, the child exposes no usable pid or piped stdio, or reading
    /// a line from the child fails. A non-zero exit or a timeout kill is reported in
    /// [`RunOutput`] rather than as an error.
    pub async fn run_streaming<F>(
        &self,
        cmd: &[&str],
        timeout: Option<Duration>,
        gpu: bool,
        devices: Option<&[usize]>,
        mut on_line: F,
    ) -> io::Result<RunOutput>
    where
        F: FnMut(&str),
    {
        if cmd.is_empty() {
            return Err(invalid("empty command"));
        }

        self.prepare_runtime_caches()?;

        let argv = backend::argv(self, cmd, gpu, devices);
        let timeout_d = timeout.unwrap_or(self.timeout);

        // `split_first` keeps "argv is non-empty" local: the backends always emit the
        // sandbox binary as argv[0], and this preserves the exact program/args split
        // the indexing form had.
        let (program, rest) = argv
            .split_first()
            .ok_or_else(|| invalid("sandbox backend produced an empty argv"))?;
        let mut command = Command::new(program);
        command
            .args(rest)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // bwrap sets the cwd via `--chdir` and `$WORKSPACE` via `--setenv`.

        // Put the child in a fresh process group so we can `killpg` the
        // entire tree (bwrap + grandchildren) on timeout.
        #[cfg(unix)]
        command.process_group(0);

        // Kill the child if this future is dropped while the runtime stays alive
        // (an in-process cancellation — task abort, an outer `select!`/timeout,
        // structured shutdown — as opposed to a whole-process SIGKILL, which the
        // OS already reaps via the process group). Without this a dropped eval
        // would detach and keep holding the GPU. `killpg`-on-timeout still covers
        // the grandchildren; this covers the drop path for the group leader.
        command.kill_on_drop(true);

        let mut child = command.spawn()?;
        // `id()` is `Some` until the child is reaped, and a pid always fits in `i32`
        // on both supported targets. Convert once here so the timeout arm below is
        // infallible; erroring out (rather than panicking) also lets `kill_on_drop`
        // reap the child we just spawned.
        let pid = child
            .id()
            .and_then(|raw| i32::try_from(raw).ok())
            .ok_or_else(|| invalid("spawned child exposed no usable pid"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| invalid("child stdout was not piped"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| invalid("child stderr was not piped"))?;
        let mut stdout = BufReader::new(stdout).lines();
        let mut stderr = BufReader::new(stderr).lines();

        let now = tokio::time::Instant::now();
        // `Instant + Duration` panics on overflow. Any real timeout is minutes, so
        // `checked_add` always succeeds; the fallback keeps this total without a
        // panic, and degrading to "already expired" errs toward killing a sandboxed
        // command rather than letting it run unbounded.
        let deadline = now.checked_add(timeout_d).unwrap_or(now);
        let mut output = String::new();
        let mut stdout_done = false;
        let mut stderr_done = false;
        let mut timed_out = false;
        let mut status = None;
        // Only read once `timed_out` is set; until then its arm is disabled.
        let mut give_up = deadline;
        // The sandbox's mount namespace, resolved once and then cached, because the
        // process it is resolved *from* need not outlive the timeout: `bwrap` and its
        // child can exit while a detached grandchild holds the pipe, and after that the
        // namespace is unreachable from the process tree even though processes are still
        // inside it. Resolved only while the child is unreaped, so the pid is certainly
        // still ours and cannot have been recycled onto someone else's sandbox.
        let mut sandbox_ns: Option<String> = None;
        let mut ns_attempts: u32 = 0;
        let mut ns_retry = now;

        while status.is_none() || !stdout_done || !stderr_done {
            tokio::select! {
                line = stdout.next_line(), if !stdout_done => {
                    stdout_done = absorb_line(line?, &mut output, &mut on_line);
                }
                line = stderr.next_line(), if !stderr_done => {
                    stderr_done = absorb_line(line?, &mut output, &mut on_line);
                }
                wait = child.wait(), if status.is_none() => {
                    status = Some(wait?);
                }
                // Resolve the namespace early and keep retrying briefly until it lands: a
                // silent command produces no other event, so waiting for one would leave
                // the deadline arm with nothing cached. Disables itself once resolved,
                // once the child is reaped, or once the attempts are spent.
                () = tokio::time::sleep_until(ns_retry), if sandbox_ns.is_none()
                    && status.is_none()
                    && ns_attempts < NS_RESOLVE_ATTEMPTS => {
                    ns_attempts = ns_attempts.saturating_add(1);
                    sandbox_ns = sandbox_mount_ns(pid);
                    let at = tokio::time::Instant::now();
                    ns_retry = at.checked_add(NS_RESOLVE_INTERVAL).unwrap_or(at);
                }
                () = tokio::time::sleep_until(deadline), if !timed_out => {
                    // SIGKILL the process group, then everything left in the sandbox's
                    // mount namespace: a command that starts a new session escapes the
                    // group and would otherwise keep running with the GPU held. The
                    // `if !timed_out` guard disables this arm so we don't kill twice.
                    timed_out = true;
                    // Signal the group ONLY while its leader is unreaped. Once reaped the
                    // pid can be recycled, and `kill(-pid)` would then land on an
                    // unrelated process group. The namespace sweep covers the group's
                    // members either way.
                    let swept = kill_timed_out(
                        status.is_none().then_some(pid),
                        sandbox_ns.as_deref(),
                        self.marker(),
                    );
                    if swept > 0 {
                        on_line(&format!(
                            "[sandbox] timeout: killed {swept} process(es) still inside the sandbox"
                        ));
                    }
                    // Drain what is already buffered, but do not wait on EOF: a pipe
                    // inherited by a process we could not reach never closes, and
                    // waiting for it blocks this call — and with it the run's
                    // wall-clock check — indefinitely.
                    let after_kill = tokio::time::Instant::now();
                    give_up = after_kill.checked_add(POST_KILL_DRAIN_GRACE).unwrap_or(after_kill);
                }
                () = tokio::time::sleep_until(give_up), if timed_out => {
                    break;
                }
            }
        }

        // A timeout reports 124 whether or not the child was reaped: after the kill we
        // stop waiting, so `status` may legitimately still be `None`.
        let returncode = if timed_out {
            124
        } else {
            status
                .ok_or_else(|| invalid("drain loop ended without a child exit status"))?
                .code()
                .unwrap_or(-1)
        };

        Ok(RunOutput {
            returncode,
            stdout: output,
            timed_out,
        })
    }

    fn prepare_runtime_caches(&self) -> io::Result<()> {
        for rel in ["torch_extensions", ".cache", "nv_compute_cache"] {
            let dir = self.tmp.join(rel);
            std::fs::create_dir_all(&dir)?;
        }
        remove_stale_torch_extension_locks(&self.tmp.join("torch_extensions"))?;
        remove_stale_torch_extension_locks(&self.workspace.join(".cache/torch_extensions"))?;
        Ok(())
    }

    /// Reverse of the placeholder mapping in [`Self::path`]: turn a host
    /// placeholder path under `self.workspace` back into its in-sandbox
    /// `/workspace/...` form. Used by the overlay lookup.
    fn guest_path(&self, host: &Path) -> String {
        let rel = host.strip_prefix(&self.workspace).unwrap_or_else(|_| Path::new(""));
        let s = rel.to_string_lossy();
        if s.is_empty() || s == "." {
            "/workspace".to_string()
        } else {
            format!("/workspace/{s}")
        }
    }

    /// Walk ro then rw mounts in declaration order (matches the bwrap argv
    /// ordering) and keep the LAST match. Mirrors bwrap's
    /// last-mount-wins semantics. Exact matches and paths underneath bound
    /// directories both resolve.
    fn overlay_for(&self, guest: &str) -> Option<(PathBuf, &'static str)> {
        let mut hit: Option<(PathBuf, &'static str)> = None;
        for (mounts, mode) in [(&self.ro, "ro"), (&self.rw, "rw")] {
            for m in mounts {
                if m.guest == guest {
                    hit = Some((m.host.clone(), mode));
                    continue;
                }
                let prefix = m.guest.trim_end_matches('/').to_string() + "/";
                // `strip_prefix` is the old `starts_with` guard and the
                // `&guest[prefix.len()..]` slice in one step — same short-circuit
                // order, same suffix, but the offset can no longer split a UTF-8
                // character.
                if let Some(suffix) = guest.strip_prefix(&prefix)
                    && m.host.is_dir()
                {
                    hit = Some((m.host.join(suffix), mode));
                }
            }
        }
        hit
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        if self.keep {
            // Forget the TempDir handle so its Drop doesn't wipe the directory.
            if let Some(td) = self.tmpdir.take() {
                let _ = td.keep();
            }
        }
        // Otherwise: TempDir's own Drop wipes a tempdir-backed sandbox;
        // with_root() has no tmpdir so the directory persists by design.
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// Recursive file-only walk. Cheap to do by hand; not worth a `walkdir`
/// dependency for our handful of doc files.
fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

fn remove_stale_torch_extension_locks(root: &Path) -> io::Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    remove_stale_torch_extension_locks_inner(root)
}

fn remove_stale_torch_extension_locks_inner(dir: &Path) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            remove_stale_torch_extension_locks_inner(&path)?;
        } else if entry.file_name() == "lock" {
            let _ = std::fs::remove_file(&path);
        }
    }
    Ok(())
}

/// Stream one drained line to the caller and append it to the captured output.
///
/// Returns `true` when the stream reached EOF, which is what the caller records as
/// that pipe being done.
fn absorb_line<F: FnMut(&str)>(line: Option<String>, output: &mut String, on_line: &mut F) -> bool {
    let Some(line) = line else {
        return true;
    };
    on_line(&line);
    output.push_str(&line);
    output.push('\n');
    false
}

/// Kill a timed-out command: its process group, then anything still alive in its
/// mount namespace. Returns how many the namespace sweep signalled.
///
/// `group` is `None` once the leader has been reaped, because its pid can then be
/// recycled and `kill(-pid)` would signal an unrelated group. The sweep does not
/// depend on it: a namespace covers the group's members too.
fn kill_timed_out(group: Option<i32>, ns: Option<&str>, marker: &str) -> usize {
    if let Some(pid) = group {
        kill_process_group(pid);
    }
    let by_marker = kill_by_sandbox_marker(marker);
    by_marker.saturating_add(ns.map_or(0, kill_mount_namespace))
}

/// SIGKILL every process tagged with this sandbox's [`SANDBOX_ID_VAR`], returning how
/// many were signalled.
///
/// The primary reach of a timeout, because it is the only identification that does not
/// depend on catching something alive at the right moment. A process-tree or
/// mount-namespace lookup has to be anchored on a live descendant, and the window can
/// be a few milliseconds wide — a foreground that exits immediately after detaching a
/// child leaves nothing to anchor on, while the detached child keeps running with a GPU
/// and the output pipe held. An inherited environment variable survives all of that.
///
/// Safe by construction: the value is a fresh tempdir's name that exists nowhere else,
/// and `--clearenv` means only this sandbox's own tree carries it, so the match set can
/// never include the harness, the host, or another sandbox. Refuses an empty marker
/// rather than matching everything.
#[cfg(unix)]
fn kill_by_sandbox_marker(marker: &str) -> usize {
    if marker.is_empty() {
        return 0;
    }
    let needle = format!("{SANDBOX_ID_VAR}={marker}");
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return 0;
    };
    let mut killed = 0usize;
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(pid) = name.parse::<i32>() else {
            continue;
        };
        // `environ` is NUL-separated; split so a marker cannot match a longer value that
        // merely starts with it.
        let Ok(environ) = std::fs::read(entry.path().join("environ")) else {
            continue;
        };
        if !environ.split(|b| *b == 0).any(|kv| kv == needle.as_bytes()) {
            continue;
        }
        // SAFETY: kill is signal-safe. `pid` carries this sandbox's unique marker, so it
        // is one of our own descendants; a racing exit yields ESRCH, dropped as above.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        killed = killed.saturating_add(1);
    }
    killed
}

#[cfg(not(unix))]
fn kill_by_sandbox_marker(_marker: &str) -> usize {
    0
}

/// The mount namespace the sandbox actually runs in, located via the direct child of
/// the spawned `bwrap`.
///
/// `bwrap` itself stays in the harness's namespace and only its child unshares, so
/// reading the spawned pid's own namespace names the host — which
/// [`is_foreign_namespace`] then refuses, leaving the sweep a silent no-op. Must be
/// called while the child is alive, i.e. before the kill.
///
/// Only ever returns a foreign namespace, so it cannot be used to target the host
/// even if a backend one day nests differently.
#[cfg(unix)]
fn sandbox_mount_ns(spawned: i32) -> Option<String> {
    for entry in std::fs::read_dir("/proc").ok()?.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(pid) = name.parse::<i32>() else {
            continue;
        };
        if parent_of(pid) != Some(spawned) {
            continue;
        }
        if let Some(ns) = mount_ns_of(pid)
            && is_foreign_namespace(&ns)
        {
            return Some(ns);
        }
    }
    None
}

/// Parent pid of `pid`, from `/proc/<pid>/stat`.
///
/// Split after the LAST `)`: the `comm` field is unquoted and may itself contain
/// spaces and parentheses, so field-counting from the left mis-parses any process
/// whose name does. The remainder starts at `state ppid ...`.
#[cfg(unix)]
fn parent_of(pid: i32) -> Option<i32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    stat.rsplit_once(')')?.1.split_whitespace().nth(1)?.parse().ok()
}

/// Whether `ns` is a namespace other than the harness's own, i.e. safe to sweep.
///
/// **Fail-closed, and the most safety-critical line here.** A sandboxed child shares
/// this process's namespace until `bwrap` unshares, so a namespace read too early
/// names the host — and sweeping the host namespace SIGKILLs every process on the
/// machine. If our own namespace cannot be determined we therefore refuse, because we
/// cannot prove `ns` is not ours.
///
/// Kept a separate predicate so it is testable without invoking the killer.
#[cfg(unix)]
fn is_foreign_namespace(ns: &str) -> bool {
    i32::try_from(std::process::id())
        .ok()
        .and_then(mount_ns_of)
        .is_some_and(|own| own != ns)
}

/// Identity of the mount namespace `pid` runs in, e.g. `mnt:[4026532801]`.
///
/// Must be read while the sandboxed child is alive. It is the one handle that says
/// "inside this sandbox" without guessing at process names, parentage, or process
/// groups — all of which a sandboxed command can change.
#[cfg(unix)]
fn mount_ns_of(pid: i32) -> Option<String> {
    std::fs::read_link(format!("/proc/{pid}/ns/mnt"))
        .ok()
        .map(|target| target.to_string_lossy().into_owned())
}

/// SIGKILL every process inside the sandbox's mount namespace `ns`, returning how
/// many were signalled.
///
/// [`kill_process_group`] is not sufficient on its own. `torchrun` starts its
/// workers with `start_new_session=True`
/// (`torch/distributed/elastic/multiprocessing/subprocess_handler`), so each worker
/// holds its own session and process group and survives a group kill — left running
/// with the GPU held and the command's stdout pipe open, which blocks the drain loop
/// and keeps a device busy for every later lease.
///
/// Namespace identity is exact — only processes in this sandbox share it — and needs
/// no knowledge of what was run, so `setsid`, a new process group, or a detached
/// grandchild are all covered without naming them.
///
/// It is **not** a total guarantee: a sandboxed process holds `CAP_SYS_ADMIN` in
/// bwrap's user namespace and can `unshare(CLONE_NEWNS)`, putting its descendants
/// outside `ns` where this will not find them. `setsid` does not change the mount
/// namespace, so the case that motivated this is covered; deliberate evasion is not.
/// `warn_if_devices_still_held` is the backstop. `--unshare-pid` would be airtight but
/// the nested kernels this runs on reject `mount -t proc` after `CLONE_NEWPID`.
/// # Safety of the target set
///
/// Refuses to sweep the harness's OWN namespace, and that refusal is load-bearing
/// rather than defensive: a sandboxed child shares it until `bwrap` unshares, so a
/// namespace read too early names the host, and sweeping the host namespace SIGKILLs
/// every process on the machine. Read the namespace late (once the child has
/// certainly exec'd) *and* keep this guard, so a future caller cannot reintroduce it.
#[cfg(unix)]
fn kill_mount_namespace(ns: &str) -> usize {
    if !is_foreign_namespace(ns) {
        return 0;
    }
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return 0;
    };
    let mut killed = 0usize;
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(other) = name.parse::<i32>() else {
            continue;
        };
        if mount_ns_of(other).as_deref() != Some(ns) {
            continue;
        }
        // SAFETY: kill is signal-safe. `other` is a pid read from /proc that shares
        // the sandbox's mount namespace, so it is one of our own descendants; a
        // racing exit just yields ESRCH, which is dropped like the group kill's.
        unsafe {
            libc::kill(other, libc::SIGKILL);
        }
        killed = killed.saturating_add(1);
    }
    killed
}

#[cfg(not(unix))]
fn kill_mount_namespace(_ns: &str) -> usize {
    0
}

#[cfg(not(unix))]
fn mount_ns_of(_pid: i32) -> Option<String> {
    None
}

#[cfg(unix)]
fn kill_process_group(pid: i32) {
    // `checked_neg` fails only for `i32::MIN`, which is not a representable pid; in
    // that impossible case there is no group to signal, so returning is the correct
    // no-op — and `-pid` would have overflowed.
    let Some(group) = pid.checked_neg() else { return };
    // SAFETY: kill is signal-safe; passing -pid sends SIGKILL to the entire
    // process group whose pgid == pid (which it is, since we passed
    // process_group(0) at spawn). Errors (ESRCH if the leader already
    // exited, EPERM if we lack permission) are dropped: there's nothing
    // useful we can do about them here.
    unsafe {
        libc::kill(group, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: i32) {
    // Only Linux is supported (see compile_error! at module top).
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The timeout's reach depends entirely on mount-namespace identity being
    /// readable and discriminating, so check both against this process rather than
    /// trusting the `/proc` layout.
    #[test]
    #[cfg(unix)]
    fn mount_namespace_identifies_a_process() {
        let me = std::process::id();
        let Ok(me) = i32::try_from(me) else {
            return;
        };
        let Some(ns) = mount_ns_of(me) else {
            // No `/proc/<pid>/ns` on this host (e.g. macOS under the local fixture):
            // the sweep degrades to the process-group kill, which is the old
            // behaviour, so there is nothing to assert.
            return;
        };
        assert!(ns.starts_with("mnt:["), "unexpected namespace form: {ns}");
        // Reading it twice must agree, or it cannot be used to match peers.
        assert_eq!(mount_ns_of(me).as_deref(), Some(ns.as_str()));
        // A pid that cannot exist yields nothing rather than a false match — the
        // sweep would otherwise signal unrelated processes.
        assert_eq!(mount_ns_of(i32::MAX), None);
    }

    /// The marker is what a timeout identifies its processes by, so it has to actually be
    /// set, be unique, and refuse to degenerate into "match everything".
    #[test]
    fn the_sandbox_marker_is_set_unique_and_never_empty() {
        let Ok(a) = Sandbox::new() else {
            return;
        };
        assert!(!a.marker().is_empty(), "an empty marker would match no process at all");
        let argv = argv_linux::argv(&a, &["true"], false, None);
        let tagged = argv
            .windows(3)
            .any(|w| w[0] == "--setenv" && w[1] == SANDBOX_ID_VAR && w[2] == a.marker());
        assert!(tagged, "every sandbox must tag its processes: {argv:?}");
        if let Ok(b) = Sandbox::new() {
            assert_ne!(a.marker(), b.marker(), "two sandboxes must not share a marker");
        }
        // Guard the degenerate case directly: an empty marker must sweep nothing.
        assert_eq!(kill_by_sandbox_marker(""), 0);
    }

    /// `/proc/<pid>/stat` holds an unquoted `comm` that can contain spaces and
    /// parentheses, so counting fields from the left resolves the wrong parent for any
    /// process whose name does — and the sweep would then look for a child of the
    /// wrong pid and find nothing.
    #[test]
    #[cfg(unix)]
    fn parent_is_parsed_after_the_last_paren() {
        let Ok(me) = i32::try_from(std::process::id()) else {
            return;
        };
        // No procfs on this host (macOS under the local fixture): there is nothing to
        // parse, and the sweep is unavailable for the same reason.
        if mount_ns_of(me).is_none() {
            return;
        }
        let parent = parent_of(me);
        assert!(parent.is_some_and(|p| p > 0), "own parent should resolve: {parent:?}");
        assert_eq!(parent_of(i32::MAX), None);
    }

    /// Sweeping our own namespace SIGKILLs every process on the machine, and a
    /// namespace read before `bwrap` unshares names exactly that — so this predicate
    /// is the difference between killing a sandbox and killing the host.
    ///
    /// Tests the predicate rather than `kill_mount_namespace`, deliberately: a test
    /// that called the killer would take the machine down with it if the guard ever
    /// regressed, which is precisely the failure it exists to prevent.
    #[test]
    #[cfg(unix)]
    fn the_sweep_refuses_its_own_namespace() {
        let Some(own) = i32::try_from(std::process::id()).ok().and_then(mount_ns_of) else {
            // Cannot read our own namespace, so nothing may be swept at all.
            assert!(!is_foreign_namespace("mnt:[4026531840]"));
            return;
        };
        assert!(!is_foreign_namespace(&own), "must refuse its own namespace");
        // Some other namespace is sweepable; a stale/absent one simply matches nothing.
        assert!(is_foreign_namespace("mnt:[0]"));
    }

    /// The regression that wedged a run for 16h past its budget: a command whose
    /// child leaves the process group survives `killpg`, keeps the stdout pipe open,
    /// and the drain loop then waits for an EOF that never comes.
    ///
    /// `setsid` is what `torchrun` does to every worker, and the inner `sleep`
    /// deliberately inherits stdout — redirecting it would hide the pipe half of the
    /// bug. The outer `timeout` converts a regression from "hangs the whole suite"
    /// into a readable failure.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_timeout_returns_even_when_a_child_leaves_the_process_group() {
        let Ok(sb) = Sandbox::new() else {
            return; // no bwrap on this host
        };
        let call = sb.run_streaming(
            &["sh", "-c", "setsid sleep 600 & sleep 600"],
            Some(Duration::from_secs(2)),
            false,
            None,
            |_| {},
        );
        let settled = tokio::time::timeout(Duration::from_mins(1), call).await;
        let out = settled
            .expect("run_streaming hung after the kill — an escaped child still holds the pipe")
            .expect("run_streaming failed");
        assert!(out.timed_out, "the command should have been timed out: {out:?}");
        assert_eq!(out.returncode, 124);
    }

    /// The shape where the namespace cannot be resolved at the deadline: the foreground
    /// exits immediately, so `bwrap` is reaped long before the timeout, while a detached
    /// grandchild keeps the pipe open. Resolving from the process tree at that point
    /// finds nothing — and worse, the reaped pid may have been recycled onto an unrelated
    /// process or a sibling sandbox. Only an early-cached namespace reaches the orphan.
    ///
    /// Passing means the call returned; if the sweep missed, it returns via the drain
    /// grace instead, so the assertion is on completion rather than on timing.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_timeout_reaches_an_orphan_after_the_sandbox_leader_exits() {
        let Ok(sb) = Sandbox::new() else {
            return; // no bwrap on this host
        };
        let call = sb.run_streaming(
            &["sh", "-c", "setsid sleep 600 & exit 0"],
            Some(Duration::from_secs(2)),
            false,
            None,
            |_| {},
        );
        let settled = tokio::time::timeout(Duration::from_mins(1), call).await;
        let out = settled
            .expect("run_streaming hung: an orphan held the pipe and was never reached")
            .expect("run_streaming failed");
        assert!(
            out.timed_out,
            "the orphan should have kept it open to the deadline: {out:?}"
        );
        assert_eq!(out.returncode, 124);
    }

    #[test]
    fn path_accepts_relative_and_absolute_workspace_form() {
        let sb = Sandbox::new().unwrap();
        let a = sb.path("notes/foo.md").unwrap();
        let b = sb.path("/workspace/notes/foo.md").unwrap();
        assert_eq!(a, b);
        assert_eq!(a, sb.workspace().join("notes/foo.md"));

        let workspace_self = sb.path("/workspace").unwrap();
        assert_eq!(workspace_self, sb.workspace());
    }

    #[test]
    fn path_rejects_escapes() {
        let sb = Sandbox::new().unwrap();
        assert!(sb.path("../escape").is_err());
        assert!(sb.path("foo/../../escape").is_err());
        assert!(sb.path("/etc/passwd").is_err());
    }

    #[test]
    fn read_write_roundtrip() {
        let sb = Sandbox::new().unwrap();
        sb.write("a/b/c.txt", "hello").unwrap();
        assert_eq!(sb.read("a/b/c.txt").unwrap(), "hello");
        assert_eq!(sb.read("/workspace/a/b/c.txt").unwrap(), "hello");
    }

    #[test]
    fn read_tree_skips_runtime_cache_artifacts() {
        let sb = Sandbox::new().unwrap();
        let root = sb.host_path_for_write("solution").unwrap();
        std::fs::create_dir_all(root.join(".cache/torch_extensions/ext")).unwrap();
        std::fs::write(root.join("solution.py"), "class Solution: pass\n").unwrap();
        std::fs::write(root.join("kernel.cu"), "extern \"C\" __global__ void k() {}\n").unwrap();
        std::fs::write(root.join(".cache/torch_extensions/ext/lock"), "").unwrap();
        std::fs::write(root.join(".cache/torch_extensions/ext/ext.so"), b"\x7fELF").unwrap();
        std::fs::write(root.join("kernel.cuda.o"), b"object").unwrap();
        std::fs::write(root.join("kernel.cuda.o.d"), "deps").unwrap();

        let files = sb.read_tree("solution").unwrap();

        assert_eq!(files.len(), 2);
        assert!(files.contains_key("solution.py"));
        assert!(files.contains_key("kernel.cu"));
    }

    #[test]
    fn read_tree_skips_non_utf8_binaries_without_failing() {
        // The agent compiles standalone test kernels (methodology step 6), leaving
        // extensionless ELF binaries (`test_flash`, `test_mma`) in solution/ that
        // `is_generated_artifact` can't pattern-match. read_tree must skip them —
        // NOT error the whole snapshot (which blocked `evaluate` in run_1784320595).
        let sb = Sandbox::new().unwrap();
        let root = sb.host_path_for_write("solution").unwrap();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("solution.py"), "class Solution: pass\n").unwrap();
        std::fs::write(root.join("test_flash.cu"), "// standalone test\n").unwrap();
        // Extensionless ELF-like binary + invalid UTF-8 bytes → must be skipped.
        std::fs::write(root.join("test_flash"), b"\x7fELF\x02\x01\x01\xff\xfe\x00\x80").unwrap();
        std::fs::write(root.join("test_mma"), [0xff, 0xfe, 0xfd]).unwrap();

        let files = sb.read_tree("solution").unwrap();

        assert_eq!(files.len(), 2, "only the two UTF-8 source files are kept");
        assert!(files.contains_key("solution.py"));
        assert!(files.contains_key("test_flash.cu"));
        assert!(!files.contains_key("test_flash"), "extensionless ELF binary skipped");
        assert!(!files.contains_key("test_mma"));
    }

    #[test]
    fn prepare_runtime_caches_removes_stale_torch_locks() {
        let sb = Sandbox::new().unwrap();
        let tmp_lock = sb.tmp.join("torch_extensions/py312/ext/lock");
        let workspace_lock = sb.workspace.join(".cache/torch_extensions/py312/ext/lock");
        std::fs::create_dir_all(tmp_lock.parent().unwrap()).unwrap();
        std::fs::create_dir_all(workspace_lock.parent().unwrap()).unwrap();
        std::fs::write(&tmp_lock, "").unwrap();
        std::fs::write(&workspace_lock, "").unwrap();

        sb.prepare_runtime_caches().unwrap();

        assert!(!tmp_lock.exists());
        assert!(!workspace_lock.exists());
        assert!(sb.tmp.join("torch_extensions").is_dir());
        assert!(sb.tmp.join(".cache").is_dir());
        assert!(sb.tmp.join("nv_compute_cache").is_dir());
    }

    #[test]
    fn ro_overlay_blocks_writes_and_redirects_reads() {
        let upstream = TempDir::new().unwrap();
        let upstream_file = upstream.path().join("readme.md");
        std::fs::write(&upstream_file, "from upstream").unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.add_ro(upstream.path(), "/workspace/docs");

        let read = sb.read("docs/readme.md").unwrap();
        assert_eq!(read, "from upstream");
        assert!(sb.write("docs/readme.md", "x").is_err());
        assert_eq!(
            std::fs::read_to_string(&upstream_file).unwrap(),
            "from upstream",
            "upstream must remain untouched"
        );
    }

    #[test]
    fn mount_docs_wires_md_files_under_workspace_docs() {
        let upstream = TempDir::new().unwrap();
        std::fs::create_dir_all(upstream.path().join("nested")).unwrap();
        std::fs::write(upstream.path().join("top.md"), "TOP").unwrap();
        std::fs::write(upstream.path().join("nested/inner.md"), "INNER").unwrap();
        // Non-doc/image files must NOT be mounted.
        std::fs::write(upstream.path().join("fig.png"), b"PNG").unwrap();
        std::fs::write(upstream.path().join("ignore.sh"), "#!/bin/sh").unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.mount_docs(upstream.path()).unwrap();

        assert_eq!(sb.read("docs/top.md").unwrap(), "TOP");
        assert_eq!(sb.read("docs/nested/inner.md").unwrap(), "INNER");
        assert_eq!(std::fs::read(sb.host_path("docs/fig.png").unwrap()).unwrap(), b"PNG");
        assert!(
            sb.read("docs/ignore.sh").is_err(),
            "non-doc/image files must not be mounted"
        );

        // Staging contents are real files, not symlinks (the snapshot
        // semantics that distinguish copy-and-deny-write from the older
        // symlink approach). Non-doc/image files must be absent from the staging.
        let staging = sb.docs_host();
        let top = staging.join("top.md");
        let inner = staging.join("nested/inner.md");
        let fig = staging.join("fig.png");
        assert!(
            std::fs::symlink_metadata(&top).unwrap().file_type().is_file(),
            "expected real file at {}",
            top.display()
        );
        assert!(std::fs::symlink_metadata(&inner).unwrap().file_type().is_file());
        assert!(std::fs::symlink_metadata(&fig).unwrap().file_type().is_file());
        assert!(!staging.join("ignore.sh").exists());

        // Calling again must not duplicate the ro overlay list. Single
        // dir-level Mount covers the whole staging tree.
        sb.mount_docs(upstream.path()).unwrap();
        let docs_count = sb
            .ro
            .iter()
            .filter(|m| m.guest == "/workspace/docs" || m.guest.starts_with("/workspace/docs/"))
            .count();
        assert_eq!(docs_count, 1, "expected exactly 1 docs overlay after re-mount");
    }

    /// `mount_docs` is a snapshot: edits to the upstream after the call
    /// returns must NOT change what the sandbox sees. (This is the
    /// behavioral contract that distinguishes copy-and-deny-write from a
    /// symlink/bind into the upstream.)
    #[test]
    fn mount_docs_is_a_snapshot_not_a_live_view() {
        let upstream = TempDir::new().unwrap();
        let upstream_file = upstream.path().join("note.md");
        std::fs::write(&upstream_file, "v1").unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.mount_docs(upstream.path()).unwrap();
        assert_eq!(sb.read("docs/note.md").unwrap(), "v1");

        // Mutate upstream after the mount.
        std::fs::write(&upstream_file, "v2").unwrap();
        assert_eq!(
            sb.read("docs/note.md").unwrap(),
            "v1",
            "mount_docs must capture a snapshot, not track upstream"
        );

        // Re-mounting picks up the new content.
        sb.mount_docs(upstream.path()).unwrap();
        assert_eq!(sb.read("docs/note.md").unwrap(), "v2");
    }

    /// End-to-end: after `mount_docs`, a sandboxed `bash` (i.e., what
    /// the Bash tool will use) can `ls docs/` and `cat docs/hello.md`
    /// and see upstream content through the bwrap ro-bind.
    ///
    /// We use workspace-relative paths because cwd is the workspace.
    #[tokio::test]
    async fn mount_docs_visible_to_sandboxed_shell() {
        let upstream = TempDir::new().unwrap();
        std::fs::write(upstream.path().join("hello.md"), "# Hello\nfrom upstream\n").unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.mount_docs(upstream.path()).unwrap();

        let out = sb
            .shell("ls docs/ && cat docs/hello.md", Some(Duration::from_secs(10)))
            .await
            .unwrap();
        assert_eq!(out.returncode, 0, "stdout: {:?}", out.stdout);
        assert!(
            out.stdout.contains("hello.md"),
            "ls didn't see the doc: {:?}",
            out.stdout
        );
        assert!(
            out.stdout.contains("from upstream"),
            "cat didn't read upstream: {:?}",
            out.stdout
        );
    }

    /// Writes to a path mounted via `mount_docs` must be denied by the
    /// sandbox itself: a sandboxed shell trying to overwrite or unlink a
    /// staged doc must fail, AND the upstream must remain untouched.
    ///
    /// Enforced by bwrap's `--ro-bind`.
    ///
    /// Note: the upstream deliberately lives in `$HOME` rather than a
    /// `TempDir` — the sandbox gets its own private `/tmp`, so an upstream
    /// under `/tmp` would not be reachable from inside at all and the test
    /// would not exercise the production boundary.
    #[tokio::test]
    async fn mount_docs_writes_are_denied() {
        let home = std::env::var("HOME").expect("HOME must be set");
        let upstream = tempfile::Builder::new()
            .prefix("kernelguy-mount-docs-write-test-")
            .tempdir_in(&home)
            .unwrap();
        let upstream_file = upstream.path().join("readme.md");
        std::fs::write(&upstream_file, "ORIGINAL").unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.mount_docs(upstream.path()).unwrap();

        let out = sb
            .shell(
                "echo HACKED > docs/readme.md 2>&1; echo --; rm -f docs/readme.md 2>&1; echo done",
                Some(Duration::from_secs(10)),
            )
            .await
            .unwrap();
        assert!(
            out.stdout.contains("done"),
            "shell didn't reach the end (sandbox killed it?) stdout: {:?}",
            out.stdout
        );
        // Upstream must be untouched: copy-and-deny-write makes it
        // structurally impossible for the agent to reach back into the
        // host (no symlink target to follow), and the in-sandbox copy
        // is itself protected by the sandbox.
        assert_eq!(
            std::fs::read_to_string(&upstream_file).unwrap(),
            "ORIGINAL",
            "upstream was modified (out: {:?})",
            out.stdout,
        );
        // The staged copy must also be unchanged: read it back via the
        // host path the overlay resolves to.
        let staged = sb.host_path("docs/readme.md").unwrap();
        assert_eq!(
            std::fs::read_to_string(&staged).unwrap(),
            "ORIGINAL",
            "staged copy was modified (out: {:?})",
            out.stdout,
        );
    }

    #[tokio::test]
    async fn run_echo_in_sandbox() {
        let sb = Sandbox::new().unwrap();
        let out = sb
            .shell("echo hello && echo world", Some(Duration::from_secs(10)))
            .await
            .unwrap();
        assert_eq!(out.returncode, 0);
        assert!(out.stdout.contains("hello"), "stdout: {:?}", out.stdout);
        assert!(out.stdout.contains("world"), "stdout: {:?}", out.stdout);
        assert!(!out.timed_out);
    }

    /// `$WORKSPACE` must be set to the agent's cwd on both platforms so a
    /// `cd "$WORKSPACE"` always succeeds — the agent never has to guess a
    /// `/workspace` path that only exists on Linux.
    #[tokio::test]
    async fn workspace_env_is_set_and_cd_able() {
        let sb = Sandbox::new().unwrap();
        let out = sb
            .shell(
                "test -n \"$WORKSPACE\" && cd \"$WORKSPACE\" && echo OK",
                Some(Duration::from_secs(10)),
            )
            .await
            .unwrap();
        assert_eq!(
            out.returncode, 0,
            "$WORKSPACE must be set and cd-able; stdout: {:?}",
            out.stdout
        );
        assert!(out.stdout.contains("OK"), "stdout: {:?}", out.stdout);
    }

    #[tokio::test]
    async fn timeout_kills_subtree() {
        let sb = Sandbox::new().unwrap();
        let out = sb.shell("sleep 30", Some(Duration::from_millis(200))).await.unwrap();
        assert!(out.timed_out, "expected timeout");
        assert_eq!(out.returncode, 124);
    }

    #[tokio::test]
    async fn writes_land_in_workspace() {
        let sb = Sandbox::new().unwrap();
        sb.shell("echo persistent > note.txt", Some(Duration::from_secs(10)))
            .await
            .unwrap();
        // The shell writes through the bind; we read back via
        // the host workspace dir to confirm it really hit the workspace.
        let body = std::fs::read_to_string(sb.workspace().join("note.txt")).unwrap();
        assert_eq!(body.trim(), "persistent");
    }

    #[tokio::test]
    async fn tmp_is_private_readable_and_writable() {
        let host_tmp_sentinel = format!("/tmp/kernelguy_host_tmp_should_not_leak_{}", std::process::id());
        let host_tmp_payload = format!("host tmp sentinel {}", std::process::id());
        std::fs::write(&host_tmp_sentinel, &host_tmp_payload).unwrap();

        let sb = Sandbox::new().unwrap();
        let cmd = format!(
            "set -eu; test -r /tmp; test -w /tmp; echo sandbox-payload > /tmp/sandbox-file; cat /tmp/sandbox-file; if test -e {host_tmp_sentinel}; then echo HOST_TMP_LEAKED; fi"
        );
        let out = sb.shell(&cmd, Some(Duration::from_secs(10))).await.unwrap();

        let _ = std::fs::remove_file(&host_tmp_sentinel);

        assert_eq!(out.returncode, 0, "stdout: {:?}", out.stdout);
        assert!(
            out.stdout.contains("sandbox-payload"),
            "sandbox /tmp was not readable after writing: {:?}",
            out.stdout
        );
        assert!(
            !out.stdout.contains("HOST_TMP_LEAKED"),
            "sandbox /tmp exposed the host /tmp: {:?}",
            out.stdout
        );
        assert_eq!(
            std::fs::read_to_string(sb.tmp().join("sandbox-file")).unwrap().trim(),
            "sandbox-payload",
            "sandbox /tmp writes should land in the sandbox tmp dir"
        );
    }

    #[tokio::test]
    async fn writes_outside_workspace_are_blocked() {
        let sb = Sandbox::new().unwrap();
        // Try to write to a sibling of the workspace — should be denied by
        // the bwrap mount layout (no bind there).
        let out = sb
            .shell(
                "echo nope > /etc/kernelguy_should_not_exist 2>&1; echo done",
                Some(Duration::from_secs(10)),
            )
            .await
            .unwrap();
        assert!(out.stdout.contains("done"), "stdout: {:?}", out.stdout);
        assert!(
            !std::path::Path::new("/etc/kernelguy_should_not_exist").exists(),
            "sandbox failed to block /etc write"
        );
    }

    /// `add_readable` must make a caller-supplied host path reachable
    /// inside the sandbox. We place a
    /// sentinel under $HOME, expose only that subdir, and confirm:
    ///   1) reading the sentinel inside `add_readable`'s subpath works,
    ///   2) reads of OTHER paths under $HOME (e.g. ~/.ssh) still fail, so
    ///      the carve-out is narrowly scoped.
    ///
    /// (1) is satisfied via `--ro-bind`, (2) by virtue of /home not being
    /// bound at all.
    #[tokio::test]
    async fn add_readable_exposes_a_caller_path_under_home() {
        let home = std::env::var("HOME").expect("HOME must be set");
        let exposed = tempfile::Builder::new()
            .prefix("kernelguy-add-readable-test-")
            .tempdir_in(&home)
            .unwrap();
        std::fs::write(exposed.path().join("hello.txt"), "VISIBLE").unwrap();

        let mut sb = Sandbox::new().unwrap();
        sb.add_readable(exposed.path());

        let cmd = format!(
            "cat {p}/hello.txt 2>&1; echo --; cat {h}/.ssh/id_ed25519 2>&1 || true; echo done",
            p = exposed.path().display(),
            h = home,
        );
        let out = sb.shell(&cmd, Some(Duration::from_secs(10))).await.unwrap();

        assert!(out.stdout.contains("done"), "stdout: {:?}", out.stdout);
        assert!(
            out.stdout.contains("VISIBLE"),
            "add_readable failed to expose the path: {:?}",
            out.stdout,
        );
        // Sibling under $HOME must still be denied (only the exposed subdir
        // got carved out) — guaranteed by /home not being bound. The denial
        // surfaces as one of these substrings.
        let lc = out.stdout.to_lowercase();
        let denied = lc.contains("operation not permitted")
            || lc.contains("permission denied")
            || lc.contains("no such file or directory");
        assert!(
            denied,
            "expected denial for ~/.ssh/id_ed25519 outside the exposed path; got: {:?}",
            out.stdout
        );
    }
}
