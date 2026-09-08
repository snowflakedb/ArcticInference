# Model directory and image semantics

How a saved model is laid out on disk, what each part means, and the
rules an image binds itself to.  Read this before wiring semi-persistence
into a serving stack, or when debugging a restore that fails on paths,
config, or stale code.

For the primitive-by-primitive API see [reference.md](reference.md) and
[instance_DESIGN.md](instance_DESIGN.md); for CRIU internals see
[CRIU_PLUMBING.md](CRIU_PLUMBING.md); for tensor parallelism see
[tp_DESIGN.md](tp_DESIGN.md).

---

## 1. The `model_dir` contract

An `Instance` may be constructed with a per-model directory:

```python
inst = Instance(vllm_config, "/data-fast/image-cache/qwen_35b")
```

It holds everything a saved model needs, so `criu_dump()` and
`criu_restore()` take no path:

```
<model_dir>/
  compilation/   # per-model JIT/compile cache: Triton, torch inductor,
                 # vLLM torch.compile, FlashInfer
  image/         # CRIU image + meta.json
  weights/       # detachable weight shards: shard_NNNN.bin + weights_meta.json
                 # (per-rank subdirs weights/rank{R}/ at TP>1)
```

`model_dir` is optional.  Without it the primitives take explicit paths
(`criu_dump("/path/to/image")`), the weights directory defaults to a
`weights` sibling of the image path, and the compile caches keep their
node-local defaults.  That mode is correct for same-node restore and is
what the orchestrator uses.

`model_dir` threads `init(gpu)` -> `worker_loop` -> `vllm_child_loop`,
where the child points its compile-cache environment variables at
`<model_dir>/compilation` *before* importing vLLM.

**Suggested convention** when you do use it: derive the directory from
the model path, e.g. `f"{resolve_model_path(model_name)}_image"`.  It is
deterministic, colocated with the weights, human-readable, needs no
hashing, and gives CRIU a stable absolute path for the recorded JIT
mappings.

**One image and one weights set per `model_dir`.** There are no
per-config subdirectories and no path encoding of TP size, dtype, or
`max_model_len`.  Binding an image to a specific `vllm_config` in the
path is not attempted; the config check below does that job instead.

---

## 2. What a re-dump does to each directory

Verified behavior, and the three parts differ:

| Directory | On re-dump |
|---|---|
| `image/` | `rm -rf`'d and recreated. CRIU will not reliably overwrite a differing file set, so a stale image could otherwise corrupt the new one. |
| `weights/` | Fully overwritten (cleared and rewritten). Shard counts vary with model size, so a partial overwrite would leave a mix the manifest cannot detect. |
| `compilation/` | **Reused, never cleared.** It is a content-keyed cache, so re-dumping into the same directory skips recompilation. |

The `rm -rf` paths fall back to `sudo rm -rf` because an aborted run can
leave root-owned files behind.

---

## 3. What binds an image

An image is not portable across arbitrary configurations.  `criu_restore`
checks two things before spawning anything, and both raise immediately
rather than failing later inside CRIU.

**`vllm_config` must match exactly.**  The comparison is a full dict
`!=`, so the config passed to `Instance` must be exactly the baked dict,
with no extra keys.  Because `tensor_parallel_size` lives in the user's
`vllm_config`, it participates in this check — a TP=2 image cannot be
mistaken for a TP=1 one.

**`model_dir` must match.**  CRIU records the compile-cache `.so` and
cubin mappings by absolute path, so an image is bound to the directory it
was dumped under.  `criu_dump` records `model_dir` in `meta.json` and a
mismatch is rejected by name.

`meta.json` also carries the CRIU plumbing (`child_pid`, `pipe_fd`,
`pipe_resource`, `nvidia_fds`), the placement and hardware identity
(`rank`, `gpus`, `gpu_uuids`), and the budget inputs (`total_gpu_bytes`,
`pinned_cpu_bytes`, `n_gpus`, `max_pinned_bytes_per_worker`).

---

## 4. Constraints worth knowing before you debug

**Cross-node restore needs the same absolute path.** Copy the whole
`model_dir` to the target node at the identical path.  The compile cache
is the reason: its `dlopen`'d artifacts are recorded by absolute path and
must exist at restore.  Two of the other cross-node requirements are
handled automatically — `meta.json` carries the capture node's GPU UUIDs
so the CUDA restore device map can pair them against the local node's
(zero overlap between the two UUID sets is the normal case, not an
error), and the child pins vLLM's rendezvous and the NCCL/gloo bootstrap
sockets to loopback so CRIU does not bake a routable IP into the image.

**Cross-node restore also needs a byte-identical environment, and this
one is not checked up front.**  CRIU records every file-backed mapping by
absolute path *and* size, then re-validates the size when it reopens the
file at restore.  One mapped file of a different length aborts the entire
restore:

```
Error (criu/files-reg.c:2175): File <path> has bad size <local> (expect <image>)
Error (criu/mem.c:1467): `- Can't open vma
Error (criu/cr-restore.c:2331): Restoring FAILED.
```

The venv is where this bites in practice: two nodes that installed the
same requirements at different times end up with different builds of some
compiled extension, and CRIU aborts on the first one it hits rather than
reporting them all.  Unlike the `vllm_config` and `model_dir` checks
above, there is no early raise — the failure surfaces from inside CRIU.
Reinstalling from the same requirements is *not* sufficient, because
identical version specs routinely yield different bytes; copy the tree
itself (`rsync -a`, or a tarball — never `tar -h`, which dereferences the
venv's symlinks and changes sizes) to the identical absolute path.

**Check an image against the node before spending a restore attempt.**
`scripts/imgdiff.py <image_dir>` decodes the image's `files.img` and
reports every recorded mapping whose local size differs or whose file is
missing, so a cross-node environment mismatch takes seconds to diagnose
instead of a failed restore.  It also compares the ELF build-IDs CRIU
recorded, which catches a same-size-but-different-build library that
would pass CRIU's size check and then map the wrong text pages.  The one
entry it always lists and that is never a problem is the `/dev/shm/sem.*`
ghost file: unlinked at dump time, carried inside the image, recreated by
CRIU at restore.

**A restored child runs the code frozen in the image.** Editing
`vllm_child.py` has no effect on an existing image: the child resumes
in-memory code and re-imports nothing from disk.  New behavior only
appears after an offline re-dump with the updated code.  Keep dump-time
code and runtime code the same checkout.  This is a common source of
"my fix did nothing" confusion.

**Root is required.** CUDA checkpoint/restore uses the libcuda driver API
when `geteuid() == 0` and otherwise falls back to `sudo
cuda-checkpoint`; CRIU dump and restore always shell out via `sudo`.
`Instance` spawns its worker and vLLM child with `mp.spawn`, which
inherits the uid, so the whole process tree must run as root.

**Modules import their siblings by bare name.** `import semip_logging`,
`from instance import Instance`, and so on, so the package directory must
be on `sys.path`.  The lazy `__getattr__` in `__init__.py` installs that
entry on first attribute access, which is what makes
`from arctic_inference.semi_persistence import Instance` work while still
letting the spawned worker and child import flat.

**Concurrent restores are supported, deliberately.** Each restore runs
inside its own PID namespace, so the recorded PIDs cannot collide with a
sibling restore or with a zombie left by a destructive dump.  See
Complications 8 and 10 in [CRIU_PLUMBING.md](CRIU_PLUMBING.md).  Images
captured before that change carry a tty and must be re-dumped.

**`SEMIP_UNPRIVILEGED=1` trades that concurrency for a lower capability
floor.** It is the single switch for running on pods that grant only
`CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE` (no `CAP_SYS_ADMIN`): the dump
gains `--unprivileged`, the restore takes a path with no private PID
namespace, and the child sheds its capabilities so the image is
restorable where `capset()` cannot grant real ones.  Two consequences
worth planning around:

- **At most one live restore per node**, since the recorded PIDs must be
  free.  `scripts/test_weights.py` restores two models concurrently and
  is therefore incompatible with this mode.
- **Portability is decided at dump time.**  An image dumped *without* the
  flag records real capabilities and cannot be restored on a low-cap node
  at all; you have to re-dump with the flag set.  Setting it on a
  privileged pod is fine and is the normal way to produce a portable
  image -- `--unprivileged` only skips a probe you do not need.

See Complication 11 in [CRIU_PLUMBING.md](CRIU_PLUMBING.md), and validate
a target node with `sudo criu check --unprivileged`.

---

## 5. Known gaps

- **Weight sync is not implemented.** The RL weight-update path is
  deliberately deferred.
- **Multiplexing more than one job per GPU with semi-persistence is
  unvalidated.**
- **Config compatibility is exact, not semantic.** The flat dict equality
  above is strict: it rejects configs that differ only in engine-internal
  or harmless keys. A richer model — bake the full effective engine
  config at dump time and validate a meaningful subset — is the eventual
  fix.
- **`logprobs` are surfaced by the child** but not plumbed through every
  adapter path.
