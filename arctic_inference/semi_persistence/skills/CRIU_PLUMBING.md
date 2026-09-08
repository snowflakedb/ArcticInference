# CRIU Plumbing: What Gets Cleaned Up and Why

## Overview

CRIU (Checkpoint/Restore In Userspace) dumps a process tree to disk and
restores it later, potentially on a different GPU.  A vLLM child process
is not a simple program — it has CUDA contexts, PyTorch distributed
backends, io_uring rings, POSIX semaphores, and hundreds of file
descriptors.  CRIU is strict: every FD, memory mapping, and file
reference in the image must be restorable at restore time, or it fails.

The dump is **destructive**: CRIU kills the child process after writing
the image to disk.  Every subsequent use of the model goes through
`load()` which creates a fresh process from the on-disk image.  This
avoids dangling processes from non-destructive (`-R`) dumps and
simplifies the state machine — after `save()`, the model is always
in `"saved"` state with no live process.

This document describes the complications we hit and how each is handled.

---

## Installation

CRIU install instructions (PPA path on Ubuntu 24.04, plus a
build-from-source fallback) have moved to [`INSTALL.md`](./INSTALL.md).
The plugin directory caveat referenced from *Complication 7* below is
covered there.

---

## The Pipeline

```
[Dump side — prepare_criu_dump in vllm_child.py + _worker_criu_save in worker.py]

  (in child)
  1. Drain in-flight engine requests
  2. Destroy PyTorch process group   (NCCL, TCPStore threads)
  3. Wait for store threads to exit  (poll /proc/<pid>/task, up to 2.5s)
  4. dup2 /dev/null over stdout/stderr (fd 1, fd 2)
  5. Walk /proc/<pid>/fd and close every FD that does not match the
     keep-list  (see "FD Policy" below); pipe_fd and fd 0/1/2 are skipped
  6. Munmap every "io_uring" region from /proc/<pid>/maps
  7. Remove /dev/shm/sem.*   (mappings stay live as anonymous memory)
  8. Audit /proc/<pid>/task for non-"python" threads (informational)

  (back in worker)
  9. Scan child AND descendant fds for socket:[ino] → --external unix[ino]
     (TP>1 worker subprocesses hold the multiproc-executor IPC sockets)
 10. Scan child fds for /dev/nvidia*   → record into meta.json
 11. → criu dump (destructive): image written, child killed

[Every use after dump — load from image]

  1. Pass pipe FD via Unix socket (SCM_RIGHTS) into sudo'd helper
  2. helper dup2's the pipe fd into place, then unshares a fresh PID
     namespace and forks: PID 1 is a reaper that unshares a mount ns with
     a private /proc and execs criu restore *inside* the new namespace
     (see Complication 10 — this frees the image's recorded PIDs)
  3. criu restore --inherit-fd fd[N]:pipe_resource
                  --inherit-fd fd[1]:stdout, fd[2]:stderr
                  --link-remap --tcp-close        (no --shell-job)
  4. worker finds the restored root's *host* PID via the inherited pipe
  5. cuda-checkpoint restore / driver API restores GPU context
```

---

## FD Policy at Dump Time

The child does not blanket-null its FDs.  `prepare_criu_dump`
walks `/proc/<pid>/fd/` and applies a **keep-list** based on the symlink
target; everything else is `os.close()`'d:

| FD class                | Action                                                     |
|-------------------------|------------------------------------------------------------|
| `fd 0` (stdin)          | left untouched (skipped: `<= 2`)                           |
| `fd 1`, `fd 2`          | `dup2(/dev/null, …)` first, then skipped                   |
| pipe fd to worker       | preserved (skipped explicitly via `pipe_fd` argument)      |
| `/dev/nvidia*`          | preserved — CUDA driver fds; restored by CRIU CUDA plugin  |
| `/dev/shm/*`            | preserved — POSIX shm, including pinned host buffers       |
| `anon_inode:*`          | preserved — eventfd, epoll, **and io_uring** (rings munmap'd separately) |
| `socket:[…]`            | preserved — worker tags unix sockets `--external unix[ino]`; TCP gets `--tcp-close` plus `SO_LINGER(1,0)` (Complication 12) |
| `pipe:[…]`              | preserved                                                  |
| Everything else         | closed (regular files, Triton `.so` opens, log files, etc.) |

The keep-list lives in `prepare_criu_dump`:

```python
keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:",
                 "socket:", "pipe:")
```

### What survives into the CRIU image

- The Python interpreter, the main thread, and any threads that aren't
  the NCCL/TCPStore set torn down above.
- All Python objects: the `LLM` engine, tokenizer, scheduler, KV cache
  manager, etc.
- Model weights and KV cache on the GPU (handled by the CUDA plugin at
  dump and by `cuCheckpointProcess*` / `cuda-checkpoint` at restore).
- `/dev/nvidia*` FDs (also recorded as `nvidia_fds` in `meta.json`).
- `/dev/shm` FDs and their mappings (anon pages now that `sem.*` files
  are deleted; pinned host buffers remain mapped).
- `anon_inode:` FDs (eventfd, epoll; io_uring FDs too — see Complication 2).
- `socket:` FDs marked `--external unix[<ino>]`, plus the worker pipe FD
  re-inherited via `--inherit-fd`.
- `pipe:` FDs.
- Triton JIT `.so` mappings, including ones already `(deleted)` on disk
  (see Complication 9 — these are intentionally *not* pre-handled and
  rely on `--link-remap` + the destructive dump to keep ghost-remap
  collisions rare).

---

## Complication 1: PyTorch Distributed (NCCL + TCPStore)

**Problem:** `dist.init_process_group()` spawns background threads
(`pt_tcpstore`, `pt_nccl_watchdg`, `pt_nccl_heartbt`) and opens TCP
sockets.  CRIU cannot restore TCP connections or threads that are
blocked on network I/O.

**Fix (dump side):** Call `dist.destroy_process_group()` before dump.
Then poll `/proc/<pid>/task/` until the store threads exit (up to 2.5s).
The TCPStore thread sometimes lingers — the dump proceeds with a
warning, and CRIU handles the remaining thread via `--tcp-close`.

`--tcp-close` covers the *connections*, not the ports they occupied:
CRIU rebinds each recorded local port before closing it, so the sockets
that survive into the image also need `SO_LINGER(1,0)` at dump time.  See
Complication 12.

---

## Complication 2: io_uring

**Problem:** Modern PyTorch/libtorch uses `io_uring` for async I/O.
CRIU cannot checkpoint `io_uring` instances — the kernel ring buffers
and submission queues are not serializable.  Both the FDs
(`anon_inode:[io_uring]`) and the memory mappings show up in
`/proc/<pid>/maps`.

**Fix (dump side):** Munmap every `io_uring` region found in
`/proc/<pid>/maps` (scanned line-by-line, `libc.munmap()` via ctypes).
The `anon_inode:[io_uring]` FDs themselves currently fall under the
`anon_inode:` keep-prefix and are *not* explicitly closed — in
practice this has worked because once the rings are unmapped, what
remains is a bare `anon_inode` that CRIU can dump.

> Caveat: if a future PyTorch/libtorch revision triggers a
> dump failure on `anon_inode:[io_uring]`, tighten the keep-list in
> `prepare_criu_dump` to close those FDs explicitly while keeping
> other `anon_inode:` FDs (eventfd, epoll, …).

---

## Complication 3: POSIX Semaphores (`/dev/shm/sem.*`)

**Problem:** Python's `multiprocessing` module creates POSIX named
semaphores in `/dev/shm/`.  These are memory-mapped into the process.
The semaphore file is unlinked shortly after creation (standard POSIX
pattern), so `/proc/<pid>/maps` shows the path as `(deleted)`.  CRIU
then needs to handle the deleted file — either via ghost remaps or
`link_remap`, both of which cause problems at restore time (see
Complication 9).

**Fix (dump side):** In `prepare_criu_dump` (child process), delete
all `/dev/shm/sem.*` files before the dump.  The live process retains
its existing mmap (the kernel keeps the inode alive).  CRIU then
captures the mapping as anonymous memory — no file reference, no ghost,
no `link_remap`.

**Side effect:** At process exit, Python's multiprocessing finalizers
try to `sem_unlink()` the already-removed files, producing harmless
`FileNotFoundError` warnings.  These are cosmetic only.

---

## Complication 4: stdout/stderr (`outt_test` file)

**Problem:** When running with `&>` or `| tee`, the child's fd 1/2
point to a regular file.  CRIU records the file path and size at dump
time.  Between dump and restore, more output gets written to the file,
changing its size.  CRIU's file validation rejects the restore:

```
File outt_test has bad size 17042 (expect 15267)
```

**Fix (dump side):** Redirect fd 1 and fd 2 to `/dev/null` via
`os.dup2()` before the dump.  CRIU captures `/dev/null` instead of
the log file — no size validation issues.

**Fix (restore side):** Pass `--inherit-fd fd[1]:stdout` and
`--inherit-fd fd[2]:stderr` to `criu restore`.  CRIU skips
restoring these FDs and inherits them from the restoring process.

---

## Complication 5: Pipe FD passing through `sudo`

**Problem:** The worker communicates with the child via a pipe.  CRIU
needs the pipe FD to be inherited by the restored process
(`--inherit-fd fd[N]:resource`).  But `criu restore` runs under `sudo`,
which closes all FDs >= 3.

**Original approach:** `sudo -C 1024` (raise the close-from limit).
Failed because the sudoers policy doesn't permit `-C`.

**Fix:** Use a Unix domain socket with `SCM_RIGHTS` to pass the pipe FD:
1. Worker creates a temporary Unix socket and listens
2. A background thread accepts and sends the pipe FD via `SCM_RIGHTS`
3. A Python helper script runs under `sudo`, connects to the socket,
   receives the FD, `dup2`s it into place, then `execvp`s `criu restore`

`sudo` is skipped when the worker is already root (`_is_root`), which
avoids leaving a long-lived `sudo` between the worker and the reaper —
it holds the terminal for the life of the namespace and leaves the tty
in a bad state when teardown SIGKILLs it.  The `SCM_RIGHTS` dance stays
either way: `subprocess` closes FDs >= 3 in the child regardless of
`sudo`.

---

## Complication 6: CUDA Context (driver API vs CLI)

**Problem:** After CRIU restore, the CUDA context needs to be
re-established on the target GPU.  The `cuda-checkpoint` CLI
(`sudo cuda-checkpoint --action restore`) works but spawns a subprocess
for every lock/checkpoint/restore/unlock call (~2-5s overhead each).

**Fix:** When running as root (`euid == 0`), use `libcuda.so` driver
API directly via ctypes:
- `cuCheckpointProcessLock(pid, NULL)`
- `cuCheckpointProcessCheckpoint(pid, NULL)`
- `cuCheckpointProcessRestore(pid, args)`  (with GPU UUID mapping for migration)
- `cuCheckpointProcessUnlock(pid, NULL)`

Non-root falls back to the `sudo cuda-checkpoint` CLI.  The driver API
path is 2-8x faster per operation.

---

## Destructive Save

The CRIU dump kills the child process after writing the image.  This
simplifies the lifecycle to a single code path:

- **First run (cache miss):** `init → ... → checkpoint → save` (child dies)
  → `load → restore → generate → teardown`
- **Every subsequent run (cache hit):** `load → restore → generate → teardown`

After `save()`, the model is in `"saved"` state with no live process.
The worker exits cleanly.  This eliminates dangling processes from
non-destructive dumps and avoids the POSIX semaphore futex deadlock
that occurred when remapping deleted shared-memory files.

---

## Image Metadata (`meta.json`)

`save()` writes a `meta.json` file alongside the CRIU image containing:

- **CRIU plumbing**: `child_pid`, `pipe_fd`, `pipe_resource`, `nvidia_fds`
- **Placement and hardware identity**: `rank`, `gpus` (the physical GPU
  list; single-element at TP=1), and `gpu_uuids` — the *capture* node's
  GPU UUIDs, which the restore device map needs as its "old" side so the
  image can be restored on a different node
- **Instance metadata**: `vllm_config` (which may include the reserved
  `_env` mapping of per-model env vars), `model_dir`, `total_gpu_bytes`,
  `pinned_cpu_bytes`, `n_gpus`, `max_pinned_bytes_per_worker`

On `criu_restore()`, the instance validates that the image's
`vllm_config` and `model_dir` both match.  A mismatch raises
`RuntimeError` immediately, before any worker is spawned or CRIU restore
attempted.  The `model_dir` check matters because the image bakes
absolute compile-cache paths (see
[semi-p_DESIGN.md](semi-p_DESIGN.md)).

The child's `os.environ` is itself captured inside the CRIU dump and
restored verbatim by `load()`, so env vars set during cold start
(both the hard-coded trio in `vllm_child_loop` and any
`vllm_config["_env"]` applied before `from vllm import LLM`) survive
restore without further work.  `_env` in `meta.json` is consulted only
on cold-start paths (orchestrator-side `register`) and by
client-side dedup; it is *not* re-applied on `load()`.

---

## Complication 7: CRIU Plugin Directory

**Problem:** The dump passes `--libdir /usr/lib/criu/empty` to prevent
CRIU from loading plugins (the CUDA plugin is loaded separately by the
CRIU CUDA infrastructure).  If the directory does not exist, CRIU
aborts at plugin initialization and the dump fails.

```
Error (criu/plugin.c:228): Unable to open directory /usr/lib/criu/empty: No such file or directory
```

**Fix:** `_worker_criu_save` creates the directory before every dump —
`os.makedirs`, falling back to `sudo mkdir -p` when the worker is not
root, since it lives under `/usr/lib`.  Neither the PPA package nor the
source build creates it, and it is easier to make than to require of
every node.  The `mkdir` in [`INSTALL.md`](INSTALL.md) is therefore
optional now.

Dump-only: no restore path passes `--libdir`, and nothing here is
TP-dependent — the same argv serves TP=1 and TP>1.

---

## Complication 8: PID Collisions at Restore

**Problem:** CRIU restores every task at its *recorded* PID (via
`clone3(set_tid)`), so the restore fails if that PID is already taken:

```
Can't fork for 47619: File exists
```

**It is task ids, not just PIDs.** A PID is only the id of a thread
group's leader, and leader ids and thread ids are allocated from a single
per-namespace counter.  CRIU restores *every* task, so the set that must
be free is the whole recorded task list — 915 ids across 3 leaders for a
TP2 image — and the kernel reports the two cases through different code
paths, neither of which names the occupant:

```
Can't fork for 47619: File exists              # a thread group leader
pie: 2103: Unable to create a thread: -17      # any other thread (EEXIST)
```

This is also why `ps` is useless for diagnosing it: threads appear only
under `/proc/<pid>/task`, so a 200-thread service can own hundreds of the
ids an image needs while showing a single unrelated PID.

Four ways they get taken:

1. **A zombie from the dump.** The destructive dump kills the child, but
   its still-live worker never reaps it, and a zombie holds its PID.
   This one is guaranteed, not incidental — the image's own `child_pid`
   is the PID that is occupied.
2. **Concurrent restores.** Two images captured on the same node with
   their child trees alive simultaneously carry adjacent/interleaved
   PIDs, so the second restore collides with the first.
3. **An unrelated host process** happening to hold a recorded id — with
   its threads, most likely, and a long-lived service never yields them.
4. **The restore's own launcher.** The counter is sequential, so a driver
   started when it happens to sit below the image's range lands *inside*
   it, together with its `sudo` wrappers and its interpreter's threads.
   Nothing can move an already-running process, so this one is only
   fixable by placing the image's ids elsewhere.

**Partial fix (retry loop):** `worker_loop` retries the load up to 5
times with 0.5s backoff, keyed on `_is_pid_collision` so it matches the
leader wording, the thread wording and the preflight's own marker alike.
This only helps for case 3 with a short-lived holder.  A zombie under a
live parent never goes away, so cases 1, 2 and 4 defeat it entirely.

**Diagnosis (reduced-capability path):** `_worker_criu_load_lowcap`
preflights `_pid_collision_report` before spawning criu — decode
`pstree.img`, walk `/proc/*/task`, group the intersection by occupying
process — and re-runs it if criu fails with either collision signature,
since an id can be claimed in between.  Cost is ~65 ms, essentially all
of it `crit decode`.  `scripts/pidcheck.py` is the same check standalone,
plus `--burn`/`--burn-to`, which advance the counter by forking so
*subsequently started* processes clear the range (it frees nothing;
`ns_last_pid` would do it in one write but is read-only on these nodes).

**Real fix:** restore each tree into its own PID namespace, where the
recorded ids are always free.  See Complication 10.  Where that needs
capabilities the node won't grant (Complication 11), the equivalent is to
put the ids somewhere nothing competes for: burn the counter to ~200000
*before* the cold start, so the image records ids no container reaches in
normal operation.

---

## Complication 9: Ghost Remap Race (CRIU 4.2)

**Problem:** When a memory-mapped file has been deleted from disk (the
path shows `(deleted)` in `/proc/<pid>/maps`), CRIU uses a *ghost
remap*: it embeds the file content in the image and creates a temporary
`.cr.<id>.ghost` hard link at restore time, then unlinks it afterwards.

With `--link-remap`, CRIU 4.2 has a race condition: when the restored
process has multiple threads (vLLM processes have 400+), each thread
can attempt to unlink the same `.ghost` file.  The first `unlink()`
succeeds; subsequent threads get `ENOENT`, causing CRIU to exit with
`rc=1` even though the restore actually completed:

```
Couldn't unlink remap /tmp/torchinductor_root/.../tmp12345.ghost: No such file or directory
```

Two sources of `(deleted)` mappings were identified:

1. **Triton JIT `.so` files** — Triton compiles kernel `.so` files,
   `dlopen()`s them, then `unlink()`s the file from disk.  Primarily
   affects FP8 models.
2. **POSIX semaphores** — Python's multiprocessing creates and
   immediately unlinks `/dev/shm/sem.*` files.

Tolerating `rc=1` was attempted but led to unstable processes — the
CUDA context was sometimes left in an inconsistent state, and
subsequent `cudaHostRegister` (repin) calls would crash the child.

Recreating deleted files from `/proc/map_files/` (worker side) was
also attempted, but creating a new file at the same path doesn't
clear the `(deleted)` flag on the existing mapping's inode.
Remapping via `mmap(MAP_FIXED)` inside the child corrupted semaphore
futex state, causing deadlocks.

**Fix:** The destructive dump (no `-R`) eliminates the ghost race by
a combination of two approaches:

1. **Semaphores:** Deleted in `prepare_criu_dump` before dump.  CRIU
   captures the mapping as anonymous memory — no file reference at all.
2. **Triton `.so` files:** These remain as `(deleted)` mappings, but
   with the destructive dump the number of ghost remaps is typically
   small (only triton kernels, no semaphores).  The race is still
   theoretically possible but far less likely with fewer ghosts.

If ghost remap races resurface, the remaining fix would be to also
handle triton `.so` files in `prepare_criu_dump` by `munmap` +
`mmap(MAP_FIXED)` from a recreated file (safe for `.so` mappings
which are `MAP_PRIVATE`, unlike semaphores which are `MAP_SHARED`).

---

## Complication 10: Per-Restore PID Namespace (and the tty it forced out)

**Problem:** See Complication 8 — the recorded PIDs are frequently
occupied, and CRIU 4.2 explicitly rejects `--join-ns pid:`, so we cannot
ask it to place the tree into a pre-made namespace.

**Fix: run criu inside a fresh PID namespace held open by a reaper.**
The sudo'd restore helper (`_worker_criu_load` in `worker.py`):

1. Receives the inherited pipe fd (SCM_RIGHTS, as in Complication 5),
   then `unshare(CLONE_NEWPID)` and `fork()`.
2. The host-namespace parent publishes **PID 1's host PID** to
   `reaper.pid` (in the image dir) and blocks — this keeps the namespace,
   and the `sudo` handle, alive.
3. **PID 1** unshares a mount namespace, mounts a private `/proc` (so
   criu sees the namespace's PID view), then forks `criu restore -d`
   into the namespace, records criu's exit code to `restore.rc`, and
   reaps forever. A live PID 1 is required both for the namespace to
   persist and for `clone3(set_tid)` of any non-1 PID to be permitted.
4. After criu detaches, the restored tree reparents onto PID 1. The
   worker discovers the restored root's **host** PID via the inherited
   pipe and drives the rest of the lifecycle with it (CUDA
   checkpoint/restore, `/proc` walks, and signals all use host PIDs,
   which are unaffected by the nesting) — `--pidfile` now holds the
   meaningless in-namespace PID.

Teardown SIGKILLs PID 1, which makes the kernel tear down the whole
namespace and the restored tree in one shot. An `atexit` fallback in
`worker_loop` catches Ctrl-C and unhandled exceptions, and a stale-reaper
sweep (reads a leftover `reaper.pid` and kills it) covers `SIGKILL`
deaths that cannot run cleanup.

**The tty this forced out.** A private PID namespace is incompatible
with a `--shell-job` image. The child inherited the interactive shell's
pts on fd 0, so it was captured as a shell job tied to an external
terminal. On restore inside the new namespace the session and pgrp
collapse to `1`, and CRIU's `TIOCSPGRP` on the host terminal fails:

```
tty: Restore inherited group 1
Error (criu/tty.c:689): tty: Failed to set group 1 on 0: Inappropriate ioctl for device
Error (criu/files.c:1221): Unable to open fd=0 id=0xec
```

**Fix (dump side):** the child sheds its controlling terminal at startup
(`vllm_child_loop`): point fd 0 at `/dev/null` and `os.setsid()` so the
captured tree owns its own session and holds no terminal. With no tty in
the image, `--shell-job` is unnecessary and is dropped from both `criu
dump` and `criu restore`. This matches the CRIU maintainers' guidance for
PID-namespace restore. **Images captured before this change (with a tty /
`--shell-job`) must be re-dumped.**

---

## Complication 11: Unprivileged Dump + Restore (`SEMIP_UNPRIVILEGED`)

**Problem:** the default dump and restore paths need `CAP_SYS_ADMIN` in
two places, so they only run on privileged pods:

1. CRIU's kerndat init builds a throwaway *network* namespace to probe
   kernel features; creating a netns needs `CAP_SYS_ADMIN`. Without it
   criu aborts with "Could not initialize kernel features detection" --
   on **both** dump and restore.
2. The restore path additionally wraps criu in a private PID namespace
   (Complication 10): `unshare(CLONE_NEWPID|CLONE_NEWNS)` + a private
   `/proc` mount, which also needs `CAP_SYS_ADMIN`.

Production pods are non-privileged: they grant only
`CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`, never `CAP_SYS_ADMIN`.

**Fix: `SEMIP_UNPRIVILEGED=1` -- one flag for both sides.**

- **Dump** (`_worker_criu_save`): append `--unprivileged`, which makes
  CRIU skip the netns kerndat probe. The dump has no namespace machinery
  of its own, so this one flag is the only change it needs; seizing the
  target still uses `CAP_SYS_PTRACE`.
- **Restore**: take `_worker_criu_load_lowcap` instead of
  `_worker_criu_load`. It runs `criu restore -d --unprivileged`
  **directly in the host PID namespace** -- no `unshare`, no reaper, no
  private `/proc`. The recorded PID is therefore the host PID, so
  `--pidfile` is authoritative (with a pipe-scan fallback), and teardown
  SIGKILLs the restored tree directly (holder
  `{"kind": "tree", "pid": host_pid}`) rather than collapsing a namespace.

Why the asymmetry (a flag on dump, a whole function on restore): the dump
had only dependency (1); the restore had (1) **and** (2), and (2) lives
in the sudo'd wrapper *around* criu, so it cannot be undone by a criu
flag -- it needs a different, wrapper-free restore procedure.

Because `-d` makes criu exit with the restore rc, the sudo process's exit
code *is* criu's rc, so this path needs none of the `reaper.pid` /
`restore.rc` side-channel files the namespace path uses. Its stdout and
stderr must go to `/dev/null` rather than pipes: with `-d` the restored
process inherits fd 1/2 and holds them open after criu exits, so a pipe
would never EOF and draining it would deadlock.

**Capability floor:** `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`, no
`CAP_SYS_ADMIN`. `clone3(set_tid)` at the recorded PIDs in the host PID
namespace is authorized by `CAP_CHECKPOINT_RESTORE`, so it does not need
to write the read-only `ns_last_pid` sysctl. Validate a node with
`sudo criu check --unprivileged` (a residual read-only `ns_last_pid`
complaint from the checker is expected and does not block restore).

**Trade-off (accepted):** without the private PID namespace every
recorded *task* id must be free on the host -- threads included, one
number space, ~900 ids for a TP2 image -- so at most **one live restore
per node**, and any unrelated process whose threads span the image's range
blocks it too. Fine for a one-job-per-pod layout. A collision surfaces as
CRIU "File exists" (leader) or `pie: Unable to create a thread: -17`
(thread); the path preflights for both and names the occupants, and the
retry loop recognizes either wording. Note this makes
`scripts/test_weights.py`, which restores two models concurrently,
incompatible with this mode. See Complication 8 for the id arithmetic and
for placing an image's ids out of contention at dump time.

**Restorable image requires shed capabilities.** CRIU's `restore_creds()`
calls `capset()` to reinstate each task's recorded caps; if the image
recorded caps the low-cap node can't grant, restore fails at
`criu/pie/restorer.c` ("Unable to restore capabilities"). So in
unprivileged mode the vLLM child zeroes its caps *before* `import torch`
-- torch and vLLM spawn many background threads and CRIU records
credentials **per thread**, so dropping first is what makes every task
record an empty cap set. This is **implied by `SEMIP_UNPRIVILEGED=1`**,
not a separate flag: the worker sets an internal
`_SEMIP_CHILD_DROP_CAPS` signal only across the child's spawn, because
the worker itself must keep its caps to run `sudo criu`.

Consequently **image portability is decided at dump time**: an image
dumped without the flag records real caps and cannot be restored on a
low-cap node without re-dumping.

### Precondition: everything the child reads must be world-accessible

The worker tree runs as root, and root bypasses file permission checks
via `CAP_DAC_OVERRIDE`.  Zeroing the capabilities takes that away while
leaving uid 0 in place, so from that instant the child can only read what
is reachable by *other* -- every directory on the path needs `o+x`, and
the files need `o+r`.

If the interpreter itself lives behind a private directory, the child
dies immediately after the drop, and the symptom does not look like a
permission problem.  `multiprocessing` spawn runs `spawn_main` ->
`_main` -> unpickle, and unpickling imports the target's module
(`vllm_child`), which is where the drop fires.  The next stdlib import
needed to finish that same unpickle then fails:

```
[semip] dropped capabilities for portable image (capset rc=0)
Traceback (most recent call last):
  File ".../multiprocessing/spawn.py", line 132, in _main
ModuleNotFoundError: No module named 'multiprocessing.popen_spawn_posix'
```

followed by `init FAILED (child pipe broken, ... state=Z (zombie),
exited_code=1)`.  A `ModuleNotFoundError` for a stdlib module right after
the cap-drop line always means this.

Observed with a conda interpreter under a `drwxr-x---` home directory.
Note a venv does not help by itself: it shares its base interpreter's
stdlib, so `/data-fast/myenv/bin/python` built on
`/home/user/miniconda3/envs/dev` still reads its stdlib from under the
private home -- switching between several such venvs changes nothing.
Check the base, not the venv:

```bash
namei -l "$(python -c 'import sys; print(sys.base_prefix)')"
```

and fix by granting traversal (`chmod o+x /home/<user>`, which permits
path traversal without allowing directory listing) or by using an
interpreter under a world-readable prefix such as `/usr/lib/python3.12`.

The operative variable is *where the interpreter lives*, not how it was
installed.  A pre-baked pod image that ships vLLM, CRIU and Python
system-wide satisfies this for free -- `/usr/lib/python3.12` and
`/usr/local/lib/python3.12/dist-packages` are `drwxr-xr-x`, so dropping
capabilities changes nothing that process could already read.  That is
why the mode appears to "just work" on such an image and then fails on a
development pod where the same code runs from a venv on a private home.
A venv built on `/usr/bin/python3` is equally fine; a venv built on a
conda install under `$HOME` is not.

`criu` itself is never implicated: it lives at `/usr/sbin/criu`
(`-rwxr-xr-x`) and runs in the worker, which keeps its capabilities.

**`criu_restore` itself is safe; what runs after it is not.**  The
restore maps an already-initialised process back into memory, so the
child resumes in-memory code, imports nothing, and performs no
DAC-checked read.  The file reads the restore *does* need (reopening
every file-backed mapping) are done by `criu` in the worker, which keeps
its capabilities -- only the child ever drops.

That makes `init` the obvious victim: it spawns a fresh interpreter, so
the child runs Python's import machinery, which is exactly where the drop
fires.  But **any post-restore primitive that spawns a process or imports
a not-yet-loaded module hits the same wall**, and at TP>1 `reinit_nccl`
does:

```
_reinit_nccl -> init_worker_distributed_environment
             -> init_distributed_environment -> _node_count
             -> in_the_same_node_as        (vllm/distributed/parallel_state.py)
             -> shared_memory.SharedMemory(create=True)
             -> multiprocessing.resource_tracker.ensure_running()   # spawns python
```

The tracker spawn cannot read its own stdlib, dies, and the write to its
pipe fails with `BrokenPipeError: [Errno 32] Broken pipe`.

**The symptom is a silent hang, not an error.**  vLLM wraps that block in
`contextlib.suppress(OSError)`, and `BrokenPipeError` *is* an `OSError`,
so nothing is logged at all -- not even the `Error ignored in
is_in_the_same_node` line, which only fires for non-`OSError`.  The
source rank skips its `broadcast_object_list` and proceeds to the
`barrier()` a few lines below, while every other rank waits forever in
the matching broadcast.  `py-spy dump` on the workers shows the two ranks
stopped at *different* lines of `in_the_same_node_as`, which is the
signature.  All you see in the log is `reinit_nccl` never returning and
vLLM repeating:

```
No available shared memory broadcast block found in 60 seconds.
```

So the precondition is not cold-start-only: an image dumped before it was
met keeps *restoring* fine and then deadlocks in `reinit_nccl`.  The fix
is the same (`chmod o+x` the home, or a world-readable interpreter
prefix) and needs no re-dump, since nothing about the image is wrong.

---

## Complication 12: `TIME_WAIT` vs the Recorded Local Port

**Problem:** CRIU records every inet socket's exact local port and
**rebinds it** at restore, *before* `--tcp-close` gets to close it.  The
dump is destructive, so every socket still open at dump time closes with
a FIN when the tree is killed and leaves its tuple in `TIME_WAIT` on the
host.  A restore that follows its own dump immediately then collides
with those tuples:

```
Error (criu/sk-inet.c): Can't bind inet socket (id 0x…): Address already in use
```

The bind cannot step over the tuple because CRIU restores `SO_REUSEADDR`
from the recorded value rather than forcing it on.  Decode an image to
see which sockets are exposed:

```bash
sudo crit decode -i <image_dir>/files.img --pretty \
  | jq -c '.entries[] | select(.type=="INETSK") | .isk
           | {state, src_port, reuseaddr: .opts.reuseaddr}'
```

Listeners — and the sockets accepted from them, which inherit the option
— carry `reuseaddr: true` and rebind fine.  The client-side connected
sockets carry `reuseaddr: false`, and those are the ones that fail.

**Deterministic at TP2+, a race at TP1.**  Every rank adds a TCPStore
connection, so at TP>1 there is always a colliding tuple.  A TP1 image
still contains such sockets (the store's self-connection, plus whatever
HTTPS connections the child left open), so TP1 is exposed too — it just
usually wins the race.

**Fix (dump side): `SO_LINGER(1,0)` before the NCCL teardown.**
`_mark_inet_sockets_rst` (`vllm_child.py`) walks the worker's
`/proc/<pid>/fd` and, for every `AF_INET`/`AF_INET6` `SOCK_STREAM`
socket, sets `SO_LINGER` with `l_onoff=1, l_linger=0` so the eventual
close sends an **RST** instead of a FIN and the tuple never enters
`TIME_WAIT`.  It runs at the top of `_destroy_nccl`, before anything is
closed, and it marks through a `dup()` so the wrapper's `close()`
releases only the duplicate: the live fd stays open for CRIU, and
because the option lives on the shared socket the RST fires whenever
teardown or CRIU's kill finally closes it.  `AF_UNIX` and non-stream
sockets are left alone.

**Time-bound, not image-bound — no re-dump needed.**  `TIME_WAIT` is
`TCP_TIMEWAIT_LEN` (60s, hard-coded in the kernel; `tcp_fin_timeout` is
a different timer), so the window closes a minute after the dump that
opened it.  Unlike Complications 10 and 11, **images dumped before this
change do not need re-dumping** — they restore fine once the window has
drained, which is exactly why the failure looks like it needs a
cooldown.

What a fresh dump does buy: CRIU stores the option in the image
(`SkOptsEntry.so_linger`) and replays it at restore, so a tree restored
from a new image also RSTs on its own teardown.  That keeps back-to-back
`criu_restore` → teardown → `criu_restore` cycles on one node clean,
where an old image re-blocks its ports on every teardown.

**Not covered (known, benign today):** the marking sits after the
`tp_size <= 1` early return in `_destroy_nccl`, so TP1 is never marked;
and `_destroy_nccl` is a `collective_rpc` target, so only the workers
are walked — the child's own sockets (e.g. leftover HTTPS connections to
the model hub, bound to the routable IP rather than loopback) still
leave `TIME_WAIT` behind.

---

## Complication 13: Scoping the Teardown Kill (no-namespace path)

**Problem:** the namespace path tears a restored tree down by SIGKILLing
the namespace's PID 1 and letting the kernel collapse everything inside
it — bounded by construction. The lowcap path (Complication 11) has no
namespace, so it has to name its victims. `criu restore -d` detaches the
tree, so tasks reparent onto PID 1 and escape a `_get_descendant_pids`
walk from the root. The original code recovered them with a process-group
kill:

```python
subprocess.run(["sudo", "kill", "-9", f"-{root_pid}"], capture_output=True)
```

That never became a group kill. `/usr/bin/kill` is procps-ng, which parses
arguments with `getopt_long` under `opterr=0`, so a multi-digit negative
pid is consumed as an *option cluster* and never reaches the pid-operand
loop. Its handler for that case derives the target from the first digit
alone (`pid = (long)('0' - optopt)`), silently discarding the rest. So
`sudo kill -9 -1181` ran `kill(-1, SIGKILL)`: as root, every process it is
permitted to signal. It killed the worker mid-sweep and PID 1's only
child, which exits the container with code 137 and burns a `backoffLimit`
retry. One pod exhausted all six and was terminated, returning its 8
H200s to the shared pool. (Long-standing procps bug; Ubuntu #1637026
describes the same failure wiping Hadoop nodes.)

Only the first digit matters, which is why this looked like a TP problem
and is not one: `tp2test` restores at 1181 → `kill(-1, ...)`, fatal, while
`qwen_27b`/`qwen_35b` restore at 3109/3111 → `kill(-3, ...)`, ESRCH on an
empty process group. A TP=1 image restoring at a pid beginning with `1`
would collapse the pod just as reliably.

**Why not scope it in the kernel.** `cgroup.kill` exists on these nodes
and is the ideal mechanism — write `1` and the kernel atomically SIGKILLs
exactly that cgroup, immune to pid reuse. It is unusable here:
`/sys/fs/cgroup` is mounted `ro` and cannot be remounted, because
`CAP_SYS_ADMIN` is absent from the bounding set. That is the same
constraint that forces the lowcap path to exist at all.

**Fix: snapshot identity at restore, verify every victim at teardown.**
`_tree_identity(root_pid)` records each task id with its start time (field
22 of `/proc/<pid>/stat`) while the tree is known-live, and becomes the
holder. `_kill_restored_tree` then kills only positive pids that are not
`<= 1`, not in `_own_ancestry()`, still live, and whose start time still
matches the snapshot — so a recycled pid is never hit. The live descendant
walk that catches tasks forked since the snapshot runs only while the root
itself still matches, since pids it discovers carry no recorded start time
to check against. The victim list is logged *before* the kills, because
the old code logged after and destroyed the evidence along with the
worker.

No re-dump: this reads live `/proc` after a restore has already succeeded,
so image format, `pstree.img`, and `meta.json` are untouched.

Full incident record, invariants, and test plan in
[`TEARDOWN_SCOPING.md`](TEARDOWN_SCOPING.md).

---

## Summary Table

| Resource           | Problem at dump time               | Dump-side fix                      | Restore-side fix                     |
|--------------------|-------------------------------------|------------------------------------|--------------------------------------|
| NCCL/TCPStore      | Background threads, TCP sockets; the tree-kill's FINs park the recorded local ports in `TIME_WAIT` | `destroy_process_group()` + poll; `SO_LINGER(1,0)` on every inet TCP socket so the kill RSTs instead | `--tcp-close` — but it closes only *after* the rebind, so it is not sufficient alone |
| io_uring           | Non-serializable kernel state       | Munmap rings (FDs kept via keep-list) | —                                 |
| POSIX semaphores   | `(deleted)` files → ghost/link remap| Delete sem files; captured as anon | —                                    |
| stdout/stderr      | File size changes between dump/load | Redirect to /dev/null              | `--inherit-fd fd[1]/fd[2]`           |
| Pipe FD            | sudo closes FDs >= 3                | —                                  | SCM_RIGHTS via Unix socket           |
| CUDA context       | GPU state not in CRIU image         | CRIU CUDA plugin at dump           | Driver API or cuda-checkpoint        |
| Plugin directory   | `--libdir` path missing             | `_worker_criu_save` creates `/usr/lib/criu/empty` before each dump | —                    |
| stdin / tty        | pts captured as `--shell-job`; can't reattach in a PID ns | fd 0 → `/dev/null` + `setsid()` at child start | drop `--shell-job` |
| PID collisions     | Recorded task ids (leaders *and* threads) already taken: dump zombie, concurrent restore, unrelated service, or the restore's own launcher | — | Restore each tree in its own PID namespace (reaper + private /proc); on the lowcap path a preflight that names the occupants + `scripts/pidcheck.py`, with dump-side id placement as the durable fix; retry loop as backstop |
| Privileged-only CRIU | dump + restore need `CAP_SYS_ADMIN` (netns kerndat probe; restore PID namespace) | `--unprivileged` (`SEMIP_UNPRIVILEGED=1`) skips the netns probe; caps shed in the child before `import torch` | lowcap path: `criu restore -d --unprivileged` in the host PID ns (no `unshare`), with a snapshot-verified teardown kill (Complication 13) |
| Ghost remap race   | `(deleted)` .so → ghost race        | Destructive dump + sem deletion    | `--link-remap`                       |
| Teardown scoping   | No namespace to collapse on the lowcap path, and `-d` reparents tasks off the root; a `kill -9 -<pid>` meant as a group kill becomes `kill(-1)` in procps-ng | — | Snapshot task ids + start times at restore (`_tree_identity`); kill only verified positive pids, never a negative one |

---

## CRIU-Related Commit History

Chronological summary of CRIU plumbing changes across the branch.

| Commit    | Date       | Summary |
|-----------|------------|---------|
| `586641e` | 2026-03-27 | **Initial semi_persistence package.** GPU sleep/wake lifecycle, `cuda-checkpoint` CLI integration. No CRIU yet. |
| `e8098d6` | 2026-03-31 | **Cross-GPU migration.** Add instance migration with `cuda-checkpoint --device-map`. |
| `b0524b7` | 2026-03-31 | **Orchestrator + multiplexing demo.** Multi-model orchestration, first end-to-end save/load with CRIU dump/restore. |
| `9a04880` | 2026-04-06 | **CRIU save/load, driver API, cross-GPU migration.** Replace `cuda-checkpoint` CLI with `libcuda.so` driver API (`cuCheckpointProcess*`). Add `--link-remap`, `--ext-unix-sk`, `--shell-job`, `--tcp-close`. Add pipe FD passing via SCM_RIGHTS. Add `/dev/shm/sem.*` cleanup pre-dump. Add `--leave-running` for non-destructive save. |
| `fc3a3db` | 2026-04-07 | **State machine ladder.** `saved ↔ checkpoint ↔ up` lifecycle; CRIU load integrated into orchestrator state transitions. |
| `ad31cca` | 2026-04-17 | **CRIU v4.2 build instructions.** Added build-from-source instructions for CRIU 4.2 with CUDA plugin support. |
| `ccaaf81` | 2026-04-20 | **CRIU restore robustness.** Init-time cleanup of stale ghost files and `/dev/shm` leftovers. PID collision retry loop (5 attempts). Orphan process killing on failed restores. Pipe error handling in `_child_thread`. Process settle wait before CUDA restore. `CUresult=401` tolerance for redundant CUDA restores. |
| `df7d82c` | 2026-04-21 | **Destructive dump + ghost remap fixes.** Switch from `--leave-running` to destructive dump (child killed after save). Delete `/dev/shm/sem.*` in `prepare_criu_dump` so CRIU captures semaphores as anonymous memory. Remove `_recover_deleted_mappings()`, `_clear_ghost_files()`, and debug instrumentation. Clean up orchestrator init. All models must be re-dumped. |
