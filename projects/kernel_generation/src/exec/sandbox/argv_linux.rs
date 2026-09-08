//! Linux backend: build a `bwrap` argv that runs `cmd` in an isolated
//! mount / IPC / UTS / network namespace with `/workspace` bound to the
//! sandbox's host workspace directory.
//!
//! Mirrors the Python reference (`references/kernelguy/.../sandboxing.py`).
//! Where the two diverge, comments call out the reason (usually a quirk of
//! nested-container kernels or a CUDA requirement).

use std::collections::HashSet;
use std::io;
use std::path::Path;

use super::Sandbox;

pub fn check_available() -> io::Result<()> {
    if which("bwrap") {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "bubblewrap is required: install with `sudo apt-get install bubblewrap`",
        ))
    }
}

/// Build the `bwrap` argv. Infallible — every path that could fail (a missing
/// system dir, a device index this sandbox was not given) degrades by omitting a
/// bind rather than erroring, so the caller always gets a runnable argv.
///
/// Renders the [`Sandbox`] spec and nothing more: it has no notion of what a GPU
/// is and holds no vendor knowledge. Device nodes and GPU environment are resolved
/// at sandbox creation and read from `gpu_shared_devices` / `gpu_devices` /
/// `gpu_env`, with `devices` selecting which pool slots' nodes to bind.
///
/// Every path it is handed must already exist: `bwrap` fails the whole command on
/// a missing bind source rather than skipping it, so existence filtering belongs
/// to the producer.
#[expect(
    clippy::too_many_lines,
    reason = "one linear sequence; splitting trades length for state plumbing"
)]
pub fn argv(sandbox: &Sandbox, cmd: &[&str], gpu: bool, devices: Option<&[usize]>) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "bwrap".into(),
        "--die-with-parent".into(),
        // Don't pass --new-session: we already setsid via process_group(0)
        // on Command::spawn, so bwrap itself is the session leader (which
        // lets killpg(-pid) reach the whole tree on timeout). bwrap's
        // --new-session would then setsid AGAIN in the sandboxed child and
        // abort with EPERM ("setsid: Operation not permitted").
        "--unshare-ipc".into(),
        "--unshare-uts".into(),
        "--unshare-net".into(),
    ];

    // We deliberately do NOT pass --unshare-pid: the nested-container
    // kernels we sometimes run on (Cursor cloud agents, locked-mount EC2
    // images, ...) reject `mount -t proc` with EPERM after CLONE_NEWPID,
    // leaving us unable to mount /proc inside. Running in the host pid-ns
    // is weaker (host PIDs visible) but still functional.

    // --unshare-cgroup is best-effort; some nested envs reject it. Drop
    // this line if your kernel complains about cgroup namespaces.
    a.push("--unshare-cgroup".into());

    // Fresh procfs. `--ro-bind /proc /proc` would also work for most
    // things, but CUDA needs to write per-task entries (e.g.
    // /proc/self/task/<tid>/comm to set thread names) — a read-only bind
    // makes those EROFS and cuInit returns CUDA error 304.
    a.extend(["--proc".into(), "/proc".into()]);

    // Minimal Linux userspace. /sys is needed by CUDA/NVML for PCI +
    // topology discovery; /opt covers vendor software installs that
    // /usr/local/bin symlinks into (nsight-compute -> /opt/nvidia/...).
    for p in ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/sys", "/opt"] {
        if Path::new(p).exists() {
            a.extend(["--ro-bind".into(), p.into(), p.into()]);
        }
    }

    a.extend([
        "--dev".into(),
        "/dev".into(),
        "--bind".into(),
        path_str(sandbox.tmp()),
        "/tmp".into(),
        "--tmpfs".into(),
        "/var/tmp".into(),
    ]);

    // Auto-pass GPU device nodes when present so libcuda's cuInit doesn't
    // return Error 304 ("OS call failed") — but ONLY for a GPU-capable run.
    // A `gpu = false` run gets the fresh `--dev /dev` with no device nodes bound, so
    // `cuInit` fails and the command is effectively deviceless (this is how `bash` is
    // kept off the GPU while sharing the workspace fs). These binds must stay HERE:
    // after `--dev /dev` (which would otherwise wipe them) and before the `/workspace`
    // bind, because bwrap applies operations in order.
    if gpu {
        let already_bound: HashSet<&str> = sandbox
            .ro
            .iter()
            .chain(sandbox.rw.iter())
            .map(|m| m.guest.as_str())
            .collect();
        // A lease is enforced by binding only ITS device nodes: what a process cannot
        // open, it cannot use. `devices` names pool slots, so slot `i` gets
        // `gpu_devices[i]`; no lease means every device this sandbox was given.
        // Missing indices are skipped rather than faulted, keeping this infallible.
        let leased = devices.map_or_else(
            || sandbox.gpu_devices.iter().collect::<Vec<_>>(),
            |slots| slots.iter().filter_map(|&i| sandbox.gpu_devices.get(i)).collect(),
        );
        for node in sandbox.gpu_shared_devices.iter().chain(leased) {
            let node = path_str(node);
            if !already_bound.contains(node.as_str()) {
                a.extend(["--dev-bind".into(), node.clone(), node]);
            }
        }
    }

    // /workspace bind goes BEFORE user mounts so they can overlay paths
    // under it (a later /workspace bind would shadow any earlier mount
    // underneath it).
    a.extend(["--bind".into(), path_str(sandbox.workspace()), "/workspace".into()]);

    // Pre-create mount points under /workspace in the host workspace so
    // bwrap doesn't lazily create them inside the bind (which would
    // persist as empty dirs/files in the workspace after teardown).
    for m in sandbox.ro.iter().chain(sandbox.rw.iter()) {
        if let Some(rel) = m.guest.strip_prefix("/workspace/") {
            let target = sandbox.workspace().join(rel);
            if m.host.is_dir() {
                let _ = std::fs::create_dir_all(&target);
            } else if let Some(parent) = target.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
    }

    for m in &sandbox.ro {
        a.extend(["--ro-bind".into(), path_str(&m.host), m.guest.clone()]);
    }
    for m in &sandbox.rw {
        a.extend(["--bind".into(), path_str(&m.host), m.guest.clone()]);
    }
    // Caller-supplied passthrough exposures (`Sandbox::add_readable` /
    // `add_path_entry`). We bind each at the same path inside the sandbox
    // so the agent's PATH lookup finds binaries at exactly the same
    // location the harness saw them. Skip entries that fall under one of
    // the standard system ro-binds we already emitted above (they're
    // already visible) or under the workspace bind (would shadow it).
    let mut emitted: HashSet<String> = HashSet::new();
    for p in &sandbox.readable {
        let s = path_str(p);
        let already_covered = ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/sys", "/opt"]
            .iter()
            .any(|root| s == *root || s.starts_with(&format!("{root}/")));
        if already_covered {
            continue;
        }
        if s.starts_with("/workspace") {
            continue;
        }
        // Skip entries nested inside another passthrough entry. Binding both an
        // outer path and something beneath it makes bwrap fail while creating
        // the inner mount point ("Can't create file at <path>: No such file or
        // directory") — by then the outer read-only bind is already in place.
        // The same deep path binds fine on its own, so it is the overlap that
        // breaks, not the depth. A virtualenv hits this every time: `sys.prefix`,
        // its `bin/`, and `sys.executable` all arrive here, nested three deep.
        // Binding only the outermost path exposes the whole subtree at the same
        // location, which is all the agent's PATH lookup needs.
        if sandbox.readable.iter().any(|q| is_inside(p, q)) {
            continue;
        }
        // `add_path_entry` also calls `add_readable`, so the same dir can arrive
        // twice; bind it once.
        if !emitted.insert(s.clone()) {
            continue;
        }
        a.extend(["--ro-bind".into(), s.clone(), s]);
    }

    let mut path_parts: Vec<String> = sandbox
        .path_entries
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    for fixed in ["/usr/local/bin", "/usr/bin", "/bin"] {
        path_parts.push(fixed.into());
    }

    a.extend([
        "--chdir".into(),
        "/workspace".into(),
        "--clearenv".into(),
        // Tags every process in this sandbox, however it detaches. `--clearenv` means
        // nothing outside can carry it, and the environment is inherited across fork,
        // exec, `setsid` and reparenting to init — so it identifies what a timeout must
        // clean up even after the process that started it is gone. The value is the
        // sandbox root's directory name: unique per sandbox, and not a host path.
        "--setenv".into(),
        super::SANDBOX_ID_VAR.into(),
        sandbox.marker().into(),
        "--setenv".into(),
        "HOME".into(),
        "/workspace".into(),
        // Portable absolute base for the agent's bash, so `cd "$WORKSPACE"`
        // always works without the agent hard-coding `/workspace`.
        "--setenv".into(),
        "WORKSPACE".into(),
        "/workspace".into(),
        "--setenv".into(),
        "TMPDIR".into(),
        "/tmp".into(),
        "--setenv".into(),
        "XDG_CACHE_HOME".into(),
        "/tmp/.cache".into(),
        "--setenv".into(),
        "TORCH_EXTENSIONS_DIR".into(),
        "/tmp/torch_extensions".into(),
        "--setenv".into(),
        "CUDA_CACHE_PATH".into(),
        "/tmp/nv_compute_cache".into(),
        "--setenv".into(),
        "PATH".into(),
        path_parts.join(":"),
        "--setenv".into(),
        "LANG".into(),
        "C.UTF-8".into(),
    ]);

    // Deliberately last: bwrap lets the final `--setenv` for a key win, so this is a
    // caller override point. What belongs here is decided at sandbox creation.
    if gpu {
        for (key, value) in &sandbox.gpu_env {
            a.extend(["--setenv".into(), key.clone(), value.clone()]);
        }
    }

    a.push("--".into());

    a.extend(cmd.iter().map(|s| (*s).to_string()));
    a
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// True when `p` sits strictly below `other`. Compares path *components*, so
/// `/opt/nvidia-x` is not treated as living inside `/opt/nvidia`.
fn is_inside(p: &Path, other: &Path) -> bool {
    p != other && p.starts_with(other)
}

fn which(prog: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|p| p.join(prog).is_file()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A sandbox whose GPU spec is filled in by hand, standing in for what
    /// `SandboxManager::spawn` resolves. Two devices plus one shared node, so the
    /// per-slot selection below asserts something on every host, GPU box or not.
    fn sandbox_with_fake_gpu_spec() -> Sandbox {
        let mut sb = Sandbox::new().expect("sandbox");
        sb.gpu_shared_devices = vec![PathBuf::from("/dev/fake-ctl")];
        sb.gpu_devices = vec![PathBuf::from("/dev/fake0"), PathBuf::from("/dev/fake1")];
        sb.gpu_env = vec![("KG_FAKE_CUDA".to_string(), "yes".to_string())];
        sb
    }

    #[test]
    fn a_lease_binds_only_its_own_slots_nodes() {
        let sb = sandbox_with_fake_gpu_spec();

        // Slot 1's lease reaches slot 1's node and not slot 0's. This is the
        // confinement: what the process cannot open, it cannot use.
        let one = argv(&sb, &["true"], true, Some(&[1]));
        assert!(one.iter().any(|s| s == "/dev/fake1"), "{one:?}");
        assert!(!one.iter().any(|s| s == "/dev/fake0"), "{one:?}");
        // Shared nodes are not per-device and always bind.
        assert!(one.iter().any(|s| s == "/dev/fake-ctl"), "{one:?}");

        // No lease means every device this sandbox was given (the pre-pool path).
        let all = argv(&sb, &["true"], true, None);
        for node in ["/dev/fake0", "/dev/fake1", "/dev/fake-ctl"] {
            assert!(all.iter().any(|s| s == node), "{node} missing: {all:?}");
        }

        // An index the sandbox was never given is skipped, not faulted — argv is
        // infallible and a bogus slot must not produce an unrunnable argv.
        let over = argv(&sb, &["true"], true, Some(&[9]));
        assert!(
            !over.iter().any(|s| s.starts_with("/dev/fake0") || s == "/dev/fake1"),
            "{over:?}"
        );
        assert!(over.iter().any(|s| s == "/dev/fake-ctl"), "{over:?}");
    }

    #[test]
    fn gpu_env_is_gated_and_emitted_last() {
        let sb = sandbox_with_fake_gpu_spec();

        let gpu_argv = argv(&sb, &["true"], true, None);
        assert_setenv(&gpu_argv, "KG_FAKE_CUDA", "yes");
        // The sandbox's own /tmp cache contract is unconditional and stays in argv.
        assert_setenv(&gpu_argv, "XDG_CACHE_HOME", "/tmp/.cache");
        assert_setenv(&gpu_argv, "TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions");
        assert_setenv(&gpu_argv, "CUDA_CACHE_PATH", "/tmp/nv_compute_cache");

        // Emitted after argv's own --setenv block, since bwrap lets the last win.
        let gpu_env_at = gpu_argv.iter().position(|s| s == "KG_FAKE_CUDA").expect("gpu env");
        let path_at = gpu_argv.iter().position(|s| s == "PATH").expect("PATH");
        assert!(
            gpu_env_at > path_at,
            "gpu_env must come after the fixed block: {gpu_argv:?}"
        );

        // A deviceless run gets neither the nodes nor the GPU env — this is how
        // `bash` is kept off the GPU.
        let cpu = argv(&sb, &["true"], false, None);
        assert!(
            !cpu.windows(2).any(|w| w[0] == "--setenv" && w[1] == "KG_FAKE_CUDA"),
            "gpu=false must not pass gpu_env: {cpu:?}"
        );
        assert!(!cpu.iter().any(|s| s.starts_with("/dev/fake")), "{cpu:?}");
    }

    #[test]
    fn argv_holds_no_nvidia_knowledge() {
        // The builder discovers nothing itself: with an empty spec, even a gpu run binds
        // no device node and reads no CUDA variable. This is the guard on that.
        unsafe {
            std::env::set_var("CUDA_HOME", "/tmp/kg-cuda-home-test");
        }
        let sb = Sandbox::new().expect("sandbox");
        let bare = argv(&sb, &["true"], true, Some(&[0]));
        assert!(
            !bare.iter().any(|s| s.contains("/dev/nvidia")),
            "argv must not discover nvidia nodes itself: {bare:?}"
        );
        assert!(
            !bare.windows(2).any(|w| w[0] == "--setenv" && w[1] == "CUDA_HOME"),
            "argv must not read CUDA env itself: {bare:?}"
        );
        unsafe {
            std::env::remove_var("CUDA_HOME");
        }
    }

    #[test]
    fn argv_keeps_network_isolated() {
        let sb = Sandbox::new().expect("sandbox");
        let argv = argv(&sb, &["true"], true, None);
        assert!(argv.iter().any(|s| s == "--unshare-net"), "{argv:?}");
    }

    #[test]
    fn gpu_flag_gates_device_binds() {
        // The flag, not the host, decides. The spec is supplied so this asserts on every
        // machine rather than only on one with real device nodes.
        let sb = sandbox_with_fake_gpu_spec();
        let cpu = argv(&sb, &["true"], false, None);
        assert!(
            !cpu.iter().any(|s| s.starts_with("/dev/fake")),
            "gpu=false must not bind any device: {cpu:?}"
        );
        assert!(
            !cpu.iter().any(|s| s == "--dev-bind"),
            "gpu=false binds no device at all: {cpu:?}"
        );
        let gpu = argv(&sb, &["true"], true, None);
        assert!(
            gpu.iter().any(|s| s.starts_with("/dev/fake")),
            "gpu=true binds them: {gpu:?}"
        );
    }

    /// A virtualenv reaches the argv builder as three nested paths
    /// (`sys.prefix`, its `bin/`, and `sys.executable`). Only the outermost may
    /// be bound: bwrap cannot create the inner mount points underneath the
    /// read-only outer bind, and dies with "Can't create file at ...".
    #[test]
    fn argv_binds_only_the_outermost_passthrough_path() {
        let venv = PathBuf::from("/home/kg-test/.venv");
        let mut sb = Sandbox::new().expect("sandbox");
        sb.add_readable(venv.clone());
        sb.add_readable(venv.join("bin"));
        sb.add_readable(venv.join("bin/python3"));
        // `add_path_entry` re-adds the bin dir, so this also covers the dedup.
        sb.add_path_entry(venv.join("bin"));

        let argv = argv(&sb, &["true"], true, None);
        let bound: Vec<&String> = argv
            .windows(3)
            .filter(|w| w[0] == "--ro-bind" && w[1].starts_with("/home/kg-test"))
            .map(|w| &w[1])
            .collect();
        assert_eq!(
            bound,
            vec![&path_str(&venv)],
            "expected exactly one bind, of the venv root: {argv:?}"
        );
        // The bin dir must still reach PATH even though it is not bound itself.
        let path = argv
            .windows(3)
            .find(|w| w[0] == "--setenv" && w[1] == "PATH")
            .map(|w| w[2].clone())
            .expect("PATH setenv");
        assert!(
            path.split(':').any(|e| e == "/home/kg-test/.venv/bin"),
            "venv bin dir missing from PATH: {path}"
        );
    }

    /// Sibling paths that share a string prefix are independent mounts.
    #[test]
    fn argv_does_not_treat_string_prefix_as_nesting() {
        let mut sb = Sandbox::new().expect("sandbox");
        sb.add_readable(PathBuf::from("/home/kg-test/env"));
        sb.add_readable(PathBuf::from("/home/kg-test/env-2"));

        let argv = argv(&sb, &["true"], true, None);
        let bound = argv
            .windows(3)
            .filter(|w| w[0] == "--ro-bind" && w[1].starts_with("/home/kg-test"))
            .count();
        assert_eq!(bound, 2, "both siblings must bind: {argv:?}");
    }

    /// A lease must name its device by UUID so the value composes with an outer
    /// `CUDA_VISIBLE_DEVICES` mask instead of being re-resolved against the full
    /// device list.
    fn assert_setenv(argv: &[String], key: &str, value: &str) {
        let found = argv
            .windows(3)
            .any(|w| w[0] == "--setenv" && w[1] == key && w[2] == value);
        assert!(found, "missing --setenv {key} {value} in {argv:?}");
    }
}
