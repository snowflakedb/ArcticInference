# Tensor Parallelism (TP>1) — Design

How the semi-persistence stack supports tensor-parallel instances, on top
of the single-GPU (TP=1) baseline, including dense TP and TP+EP (expert
parallel) MoE.

The guiding principle is **strict superset**: TP>1 is a set of extra
primitives and a few widened signatures, and every TP-specific step is a
**no-op at TP=1**.  The proven single-GPU path is left untouched, so
running with one GPU reproduces the original behavior exactly.

Scope: dense TP and TP+EP MoE. Ulysses / shift sequence-parallel is out
of scope (`vllm_child._is_arctic_parallel_worker` returns `False`).

---

## 1. Why TP>1 is not free

CRIU cannot snapshot a live distributed engine as-is, and CUDA graphs
bake absolute device addresses that move across a checkpoint/restore.
Three things break at TP>1 that simply do not exist at TP=1:

1. **NCCL communicators / TCPStore.** There is no communicator at TP=1.
   At TP>1, live NCCL comms and the torch `ProcessGroupNCCL` /
   `CustomAllreduce` IPC handles cannot survive a CRIU dump, so they must
   be torn down before checkpoint and rebuilt after restore.
2. **CUDA-graph baked addresses.** At TP=1 the captured decode graphs
   have no cross-rank IPC pointers.  At TP>1 (especially with
   CustomAllreduce) the graphs bake peer buffer addresses that go stale
   after `destroy_nccl` -> `reinit_nccl` reallocates them, so the graphs
   must be rebound (reuse) or recaptured (full).  See `ca_graph_rebind.py`.
3. **Physical GPU placement.** A TP group must span an arbitrary set of
   physical GPUs while keeping all of them visible (the cuda-checkpoint
   physical-GPU addressing requires it), rather than pinning one GPU via
   `CUDA_VISIBLE_DEVICES`.
4. **Where the weight-staging buffer lives.** At TP=1 vLLM uses the
   "uni" executor and the worker *is* the vllm_child process, so a
   buffer allocated there and captured by a `collective_rpc` closure is
   mutated in place.  At TP>1 the multiproc executor cloudpickles that
   closure into N subprocesses: each gets a *copy* of the buffer, writes
   into the copy, and the writes are discarded when the call returns --
   silently.  Each rank also owns a different shard of the parameters,
   so a single child-side buffer is the wrong size regardless.

These map to the four new primitives (`destroy_nccl`, `reinit_nccl`,
`cleargraph`, `recapture_graphs`), the `SemipGPUWorker` + `SEMIP_GPU_MAP`
placement mechanism, the worker-local staging primitives
(`worker._semip_*`), and a per-worker memory-budget correction.

---

## 2. Lifecycle chains, TP=1 vs TP>1

**TP is selected by the vLLM config**, not by the GPU count: set
`tensor_parallel_size=N` in `Instance(vllm_config)` (default 1 -> TP=1).
The `gpus=[...]` passed to `init` / `cuda_restore` is *physical placement
only* and must have exactly `tensor_parallel_size` entries (validated,
raises otherwise).  `n_gpus == tensor_parallel_size` is fixed at
construction.

The core pipeline is one shared sequence.  TP>1 interleaves NCCL and
graph steps; TP=1 omits them (they would be no-ops anyway).

```
                 TP=1                               TP>1 (tensor_parallel_size=N)
  save:          init(gpu)                          init(gpus=[...])  # len == N
                 generate                            generate
                 attach                              attach
                 [stage] save_weights                [stage] save_weights
                 detach                              detach
                 sleep                               sleep
                 cuda_checkpoint  ─────────────────  cuda_checkpoint
                                                       ├─ cleargraph(reuse)   ┐ auto-inserted
                                                       ├─ destroy_nccl(reuse) ┘ when n_gpus>1
                                                       └─ cuda_checkpoint
                 criu_dump                           criu_dump

  restore:       criu_restore                        criu_restore
                 cuda_restore(gpu)                   cuda_restore(gpus=[...])
                                                     reinit_nccl          <- NEW
                 wake_up_weights                     attach
                 attach                              load_weights
                 repin                               wake_up_weights
                 load_weights                        repin
                 plan_restore_weights                plan_restore_weights
                 restore_weights                     restore_weights
                 wake_up_kv_cache                    wake_up_kv_cache
                                                     recapture_graphs(reuse)  <- NEW
                 generate                            generate
```

`cleargraph`/`destroy_nccl` before the checkpoint and
`reinit_nccl`/`recapture_graphs` after the restore are the only
TP-specific caller-visible steps.  `cuda_checkpoint` inserts the first
pair itself; the caller must invoke `reinit_nccl` and `recapture_graphs`
on the restore side (see `scripts/test_tp2.py`).

---

## 3. What TP adds to `instance.py`

Every item below is additive or a widened signature; nothing was removed
from the TP=1 path.

### 3.1 Per-instance TP state

`gpus` is the physical GPU list; `n_gpus == tensor_parallel_size`.
Because TP is wired from the config, `n_gpus` is derived from
`vllm_config["tensor_parallel_size"]` at construction (authoritative) and
`gpus` is validated against it — never the other way around.
`max_pinned_bytes_per_worker` is the largest **per-worker** staging shard,
needed because at TP>1 the aggregate `pinned_cpu_bytes` overstates the
per-GPU budget ~N-fold.

### 3.2 `init(gpus=...)`: validate placement, don't infer TP

`init(gpus=None, gpu=None)` is back-compatible: a scalar `gpu=` or
positional `init(0)` still works for TP=1.  TP size is read from the
config, and the supplied GPU list is validated to match; `init` never
infers TP from `len(gpus)` nor injects `tensor_parallel_size` into the
config (the user already set it).  It does inject the internal
`worker_cls` for TP>1.

`total_gpu_bytes` is snapshotted from `gpus[0]` — every GPU in a TP group
has the same capacity, so it stays representative of the per-GPU
`gpu_memory_utilization` budget.

### 3.3 `cuda_checkpoint()`: graph-preserving NCCL teardown prologue

Before the checkpoint, TP>1 preserves the captured graphs
(`cleargraph(reuse)` is a no-op on the graphs themselves in reuse mode)
and unilaterally aborts NCCL (`destroy_nccl(reuse)`).  TP=1 keeps the
single `cuda_checkpoint` send.

### 3.4 Four new primitives

All four short-circuit to a no-op in the child when
`_semip_tp_size(worker) <= 1`, so they are safe to call unconditionally.

| Primitive | When | Child-side |
|---|---|---|
| `destroy_nccl(mode)` | inside `cuda_checkpoint` (TP>1) | `_destroy_nccl` — unilateral `ncclCommAbort` (reuse/EP) or collective destroy; closes CustomAllreduce IPC handles |
| `reinit_nccl()` | after `cuda_restore` | `_reinit_nccl` — fresh TCP port, `init_worker_distributed_environment`, rebind `tp:0`/`world:0` (+ `ep:0`/`dp:0` for MoE) group slots the graphs look up |
| `cleargraph(mode)` | inside `cuda_checkpoint` (TP>1); also `full` before recapture | `_semip_cleargraph` — no-op in `reuse`; in `full` drops exec handles, refreshes the shared graph pool, resets MoE aux stream / V2 graph manager |
| `recapture_graphs(mode)` | after `wake_up_kv_cache` | `_semip_recapture_graphs` — `reuse` rebinds preserved graphs' baked addresses (`ca_graph_rebind`); `full` clears + `capture_model()` |

`graph_mode` semantics: **`reuse`** preserves the cold-start captured
graphs across the checkpoint and only rewrites their stale device
addresses on restore (fast). **`full`** discards and recaptures via
`capture_model()` (slow; the fallback when reuse rebind fails).

### 3.5 `criu_dump()` / `criu_restore()`: TP shape in the image

`criu_dump` persists `n_gpus` and `max_pinned_bytes_per_worker` into
`meta.json` (alongside `gpus` and `gpu_uuids`, written worker-side).

The restore side has no live `init(gpus)` call.  TP size is authoritative
from this instance's `vllm_config["tensor_parallel_size"]` (already equal
to the image's, per the config mismatch check), so `n_gpus` is derived
from the config with `meta["n_gpus"]` kept only as a legacy fallback.
`gpus` / `max_pinned_bytes_per_worker` are hydrated from `meta.json` (with
a `rank` -> `[rank]` shim for legacy images), and the worker is spawned on
the restored GPU list rather than a hardcoded GPU 0.

### 3.6 `cuda_restore(gpus=...)`: place the group, validate against config

`cuda_restore(gpu=None, gpus=None)`.  With neither argument it falls back
to the `self.gpus` hydrated from `meta.json`, so a restore can re-place
the group on a different physical GPU set.  The GPU count is validated
against the config-derived `n_gpus` (TP size cannot change across a
restore) rather than redefining it.

### 3.7 `plan_restore_weights()`: per-GPU (not aggregate) chunk budget

This is the one correctness-critical numeric change.  The restore chunk
budget must be sized against the **per-GPU** staging shard.  Using the
TP-aggregate `pinned_cpu_bytes` would overstate the per-worker pinned
buffer ~N-fold, shrinking the chunk budget needlessly (or spuriously
failing the `param exceeds chunk_size` check):

```python
pinned = self.max_pinned_bytes_per_worker or self.pinned_cpu_bytes
mb = int(0.9 * min(pinned, allotment - pinned))
```

> **Why the budget matters (both TP=1 and TP>1).** `restore_weights`
> without a cached plan allocates a GPU staging buffer as large as the
> whole per-worker weight buffer, which torch's caching allocator keeps
> resident even after `resize_(0)`. `wake_up_kv_cache` then allocates via
> vLLM's `cumem` allocator straight from the driver and cannot reuse
> torch's cached block -> CUDA OOM. `plan_restore_weights()` bounds the
> staging buffer so enough free driver memory remains for the KV wake-up.
> Always call `plan_restore_weights()` before `restore_weights()`.
> Pass an explicit `max_buffer_bytes` for images dumped before the
> `empty_cache()` fix.

### 3.8 Result bookkeeping

`attach` and `plan_restore_weights` surface `max_pinned_bytes_per_worker`;
`criu_restore` / `cuda_restore` echo the resolved `gpus` back into
instance state; `detach` clears the per-worker figure.

---

## 4. Child-side counterparts

The `instance.py` surface is only the orchestration; the mechanism lives
in the child.  All of it degrades to a no-op at TP=1.

- **`_semip_worker.SemipGPUWorker`** (`vllm_config["worker_cls"]`, TP>1
  only). Two hooks:
  - `init_device` remaps vLLM `local_rank` -> physical GPU via
    `SEMIP_GPU_MAP` (set by the child before spawn), so the group spans
    an arbitrary GPU set with all GPUs visible.
  - `compile_or_warm_up_model` forces the cold-start CUDA-graph capture
    onto the CustomAllreduce copy path with `keep_graph=True`, so the
    preserved graph is reuse-friendly for `ca_graph_rebind`.  It returns
    whatever the base class returns, so it is agnostic to that method's
    return type across vLLM versions.
- **GPU visibility fork** (`vllm_child.vllm_child_loop`): TP>1 clears
  `CUDA_VISIBLE_DEVICES` and sets `SEMIP_GPU_MAP`; TP=1 pins the single
  GPU as device 0 (unchanged).
- **TP-only engine config** (`init`, `_tp >= 2`): disables fused
  allreduce+RMS, NVLS, symmetric-memory allreduce and the MoE
  shared-experts stream, all of which hold state CRIU cannot serialize
  or complicate graph capture.
- **`_destroy_nccl` / `_reinit_nccl`**: NCCL abort/rebuild; both
  early-out when `_semip_tp_size(worker) <= 1`.
- **`_prepare_worker_dump`**: per-worker CRIU dump prep (FD close,
  io_uring/IB munmap, PSM shm unlink), invoked only when `len(gpus) > 1`
  since at TP=1 the "worker" is the driver process itself.
- **`ca_graph_rebind.py`**: the graph-address rebind engine that lets
  preserved decode graphs replay after `destroy_nccl` -> `reinit_nccl`
  without a full recapture (the reuse path). This is the heaviest and
  most fragile piece of the TP work; it is entirely bypassed at TP=1.
- **Worker-local weight staging** (`_semip_attach`, `_semip_stage`,
  `_semip_repin`, `_semip_unpin`, `_semip_plan_load_weights`,
  `_semip_restore_weights`, `_semip_detach`, `_semip_save_weights`,
  `_semip_load_weights`): the staging buffer, param index and chunk plan
  live on the worker (`worker._semip_*`) for the reason in section 1.4.
  Unlike the primitives above this is *not* a TP-only addition -- it is
  the single path both TP sizes take, and the child only aggregates the
  per-worker results. `attach_pinned` is unsupported on it.
- **Per-rank weight shards**: `save_weights`/`load_weights` fan out to
  `weights/rank{R}/` at TP>1; TP=1 keeps the flat `weights/` layout.

### Worker-side (`worker.py`)

- `worker_loop` and `_child_thread` take a `gpus` list (a bare int is
  still accepted); `rank = gpus[0]`.
- `_gpu_migration_permutation` generalises the old 2-GPU swap into a full
  N-GPU bijection pinning `old_gpus[k] -> new_gpus[k]`, completing the
  permutation over the remaining GPUs so `cuCheckpointProcessRestore`
  gets a bijection over every visible device.
- Two-pass restore: restore **all** pids, then unlock **all** pids.  With
  a TP process tree an unlocked worker could otherwise touch a
  still-locked sibling over NCCL/IPC and race.
- The CRIU dump scans **descendant** PIDs (deduped by socket inode), not
  just the child, so the multiproc-executor Unix sockets are declared
  `--external`.

---

## 5. Invariants and gotchas

- **TP is config-driven.** Set `tensor_parallel_size` in the vLLM config;
  `init`/`cuda_restore` take `gpus` as placement only and require
  `len(gpus) == tensor_parallel_size` (raise otherwise). `n_gpus` is
  fixed at construction and never re-derived from a GPU count.
- **Superset invariant.** Any TP>1 primitive is a no-op at TP=1, and the
  TP=1 chain is exactly the original chain. Do not route TP=1 through
  the NCCL/graph machinery.
- **Caller drives NCCL/graph on restore.** `cuda_checkpoint` inserts
  `cleargraph`+`destroy_nccl` itself, but `reinit_nccl` (right after
  `cuda_restore`) and `recapture_graphs("reuse")` (right after
  `wake_up_kv_cache`) are the caller's responsibility.
- **`destroy_nccl` frees the ports too, not just the comms.** Before
  tearing the process groups down it marks every inet TCP socket in each
  worker `SO_LINGER(1,0)`, so the destructive dump's kill RSTs them and
  their local ports skip `TIME_WAIT`.  Without that, a restore inside 60s
  of its own dump fails in CRIU with `Address already in use`, because
  CRIU rebinds each recorded local port.  Every rank contributes a
  TCPStore connection, which is why the collision is deterministic at
  TP>1 and only a race at TP=1.  See Complication 12 in
  [`CRIU_PLUMBING.md`](CRIU_PLUMBING.md).
- **`reinit_nccl` ordering.** It must run after `cuda_restore`, and before
  anything that runs the model or replays a captured graph — in practice
  before `recapture_graphs` and `generate`.  `attach` and `load_weights`
  are *not* constrained by it: `collective_rpc` is the executor's
  Unix-socket fan-out, not a device collective, and the work each rank
  does is CPU-only (a host `torch.empty` for `_semip_attach`, then shard
  reads from `<weights_dir>/rankN` for `_semip_load_weights`).
  `test_tp2.py` runs both *before* `cuda_restore` for that reason.
  `restore_weights` is in between: its H2D copies need the restored
  context, but no NCCL.
- **Per-worker budget, not aggregate.** Size the restore chunk plan from
  `max_pinned_bytes_per_worker`; the aggregate `pinned_cpu_bytes` is
  ~N times too large per GPU.
- **`reuse` first, `full` as fallback.** Prefer `graph_mode="reuse"`
  (rebind); fall back to `full` (recapture) only when rebind fails.
- **Silent degradation.** `ca_graph_rebind` is written defensively with
  many guarded imports and `getattr` fallbacks, so a mismatch against a
  newer vLLM tends to *skip* a rebind path rather than raise. When
  bringing up a new vLLM version, check the logs for skipped paths
  instead of assuming success.
- **Image portability.** `meta.json` carries `n_gpus`, `gpus`,
  `gpu_uuids` and `max_pinned_bytes_per_worker`; legacy single-GPU images
  fall back via the `rank` -> `[rank]` shim and `n_gpus=1`. Because
  `tensor_parallel_size` lives in the user's `vllm_config`, it is
  persisted in `meta.json` and participates in the `criu_restore`
  mismatch check, so a TP=2 image is not mistaken for a TP=1 one.

---

## 6. Primitive delta summary

```
TP>1 = TP=1 baseline
     + { destroy_nccl, reinit_nccl, cleargraph, recapture_graphs }   # 4 new primitives
     + TP wired from vllm_config["tensor_parallel_size"]             # config-driven, not GPU-count
     + init(gpus=[...]) / cuda_restore(gpus=[...]) placement + check  # len(gpus) == tp validated
     + cuda_checkpoint auto-inserts cleargraph+destroy_nccl @ TP>1    # behavior change
     + plan_restore_weights uses per-worker pinned budget             # per-GPU correctness
     + meta.json carries n_gpus / gpus / max_pinned_bytes_per_worker  # image portability
     + weights/rank{R}/ shard layout                                  # per-rank shards
```

Plus one change that is shared rather than TP-gated: weight staging
lives on the workers (`worker._semip_*`) for both TP sizes, because a
child-side buffer cannot be written by workers at TP>1.

Nothing is removed from the TP=1 path; TP>1 is a strict superset.
