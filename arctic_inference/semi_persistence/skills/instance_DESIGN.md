# Semi-Persistent Instances -- Design

Semi-persistence allows vLLM instances to checkpoint and restore without
needing a cold start each time.  Checkpointing saves an instance's whole
CUDA context to CPU, leaving 0 GB trace on GPU.  Restoring wakes up the
model and loads the weights.

TTFT is directly impacted by the restore time.  A cold-start
initialization of a vLLM instance typically takes a minute.  Restore
takes a second or few.  We need to maintain hundreds of checkpointed
instances.

## Design Objective

Support cases such as below.  In the first case, multiple instances can
be checkpointed on a GPU yet only one instance can be alive.  This is
the case with large models.

```
   GPU 0     GPU 1
| inst 1  | inst 2* |
| inst 3* |         |
```

In the second case, multiple instances with small models (inst 4 and 5)
can be active on a GPU.

```
   GPU 0     GPU 1
| inst 1  | inst 2  |
| inst 3* | inst 4* |
|         | inst 5* |
```

There are many cases like these.

## Instance-Based Design

An instance represents a vLLM engine with a specific vLLM configuration.

```python
vllm_config_1 = {"model": "Qwen/Qwen3-1.7B"}
vllm_config_2 = {"model": "nvidia/Llama-3.1-70B-Instruct-FP8"}
vllm_config_3 = {"model": "Qwen/Qwen3-32B"}
```

A model is composed of an a) architecture, and b) weights.  For example,
Qwen3-32B and Qwen3-32B-Instruct have the same model architecture yet
different weights.

```python
instance_1 = Instance(vllm_config_1)
instance_2 = Instance(vllm_config_2)
```

Instances 1 and 2 initialize in parallel with real weights
(`load_format=auto`).  The GPU is specified at `init()` time:

```python
instance_1.init(gpu=0)
instance_2.init(gpu=1)
```

### Optional `model_dir`

`Instance(vllm_config, model_dir=None)` accepts a per-model directory
holding `{compilation, image, weights}`.  With it, `criu_dump()` and
`criu_restore()` need no path, the weight shards land in
`<model_dir>/weights`, and the JIT/compile caches move under
`<model_dir>/compilation` so the whole directory travels between nodes as
one unit.  Without it the primitives take explicit paths and the caches
keep their node-local defaults, which is what the orchestrator uses.
See [semi-p_DESIGN.md](semi-p_DESIGN.md).

### Tensor parallelism

TP size comes from `vllm_config["tensor_parallel_size"]`.  `init` and
`cuda_restore` then take a `gpus` list that is placement only and must
have exactly that many entries:

```python
inst = Instance({"model": "...", "tensor_parallel_size": 2}, model_dir)
inst.init(gpus=[2, 3])
```

Four additional primitives (`destroy_nccl`, `reinit_nccl`, `cleargraph`,
`recapture_graphs`) bracket the checkpoint and restore; all are no-ops at
TP=1.  See [tp_DESIGN.md](tp_DESIGN.md).

## Process Hierarchy

Each Instance owns one **worker process**, created when `init(gpu)` or
`criu_restore(filename)` is called.  Both the worker and the vLLM child are **spawned** via
`mp.get_context("spawn")`.  Spawning is safe to call from any thread
(e.g. from a `ThreadPoolExecutor` in the orchestrator), unlike fork
which can deadlock on glibc mutexes held by other threads.

- **Worker (spawned)**: runs the command loop, executes
checkpoint/restore via `cuCheckpointProcess`* ctypes.  Queues and
shared counters are passed as constructor args.
- **vLLM child (spawned)**: starts with a clean address space -- no
inherited CUDA contexts.  vLLM can freely use its own multiprocessing
internally.  Tensor parallelism works.
- **EngineCore (in-process)**: `VLLM_ENABLE_V1_MULTIPROCESSING=0` runs
the EngineCore inside the vLLM child process (no separate subprocess).
This avoids IPC serialization overhead during `restore_weights` (which
would otherwise pickle GPU tensors across processes, failing for
models >4 GiB and adding ~16s latency even for small models).
- **Per-model env vars**: `vllm_config["_env"]` (optional
`dict[str, str]`) is popped from a local copy of the config inside
the child's `init` handler and applied to `os.environ` *before*
`from vllm import LLM`, so flags vLLM reads at import time
(e.g. `VLLM_USE_DEEP_GEMM`, `VLLM_ATTENTION_BACKEND`) take effect.
The reserved trio `CUDA_VISIBLE_DEVICES` /
`VLLM_ENABLE_V1_MULTIPROCESSING` / `USE_LIBUV` is hard-set at the
top of the child loop and silently dropped if present in `_env`.
The on-registry / on-disk copies of `vllm_config` retain `_env` so
it participates in client-side dedup and is persisted in `meta.json`.

```
Main process
  |
  |-- Instance 1  (handle, main-process side)
  |     |
  |     `-- Worker process 1  (spawned)
  |           |-- [checkpoint/restore via cuCheckpointProcess ctypes]
  |           |
  |           `-- vLLM child process  (spawned)
  |                 |-- EngineCore (in-process, holds GPU memory)
  |                 |     `-- each vLLM worker owns a CPU staging buffer
  |                 |         (allocated on attach, pinned via repin).  At
  |                 |         TP=1 the worker is this same process; at TP>1
  |                 |         there is one worker subprocess (and one buffer)
  |                 |         per rank.
  |                 `-- resource_tracker  (no GPU, skipped during checkpoint)
  |
  |-- Instance 2  (handle, main-process side)
  |     |
  |     `-- Worker process 2  (spawned)
  |           `-- vLLM child process  (spawned)
  |                 |-- EngineCore (in-process)
  |                 `-- resource_tracker
  |
  `-- ...
```

**Important**: GPU memory queries use NVML (`pynvml`) instead of
`torch.cuda.mem_get_info` to avoid initializing CUDA in the main
process.

## Primitives

Primitives are the minimum number of scheduling operations for
describing complicated schemes.  In a compositional design, the
primitives must be:

1. **Non-blocking** (except `wait`) so that we can manage multiple
  instances at the same time.
2. **Optimized** so that their combination will also be optimized.
3. **Chainable** -- all primitives return `self`.


| Primitive               | What it does                                                 | Executed in                                |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| `init(gpu)`             | Cold start a model with real weights on the given GPU.  Spawns the worker process and vLLM child.  Applies `vllm_config["_env"]` (if present) to `os.environ` before importing vLLM. | Worker + Child                 |
| `wait()`                | Block the main process until all pending commands complete.  | Main process                               |
| `sleep()`               | `llm.sleep(level=2)` -- frees GPU memory for both main and drafter weights.  Main and drafter `named_buffers()` ride the stock vLLM / arctic CPU snapshot path; main and drafter `named_parameters()` come back via the pinned-buffer `stage` / `restore_weights` pair, so the per-sleep CPU snapshot of drafter parameters is suppressed (see *Arctic patch interaction*). | Child |
| `cuda_checkpoint()`          | Save CUDA state to CPU via `cuCheckpointProcess`*.  Instance becomes stateless (`gpu=None`). | Worker (ctypes) |
| `criu_dump(filename)`        | CRIU-dump the child process tree to disk (destructive).  The child is killed after the image is written.  Writes `meta.json` with `vllm_config` (including `_env` if set) and CRIU metadata. | Worker (child thread) |
| `criu_restore(filename)`        | Restore a process from a CRIU image on disk.  Validates that the image's `vllm_config` matches this instance.  Spawns a new worker and CRIU-restores the child.  Does *not* re-apply `_env` -- the child's `os.environ` is captured inside the CRIU image and restored verbatim. | Worker |
| `cuda_restore(gpu)`          | Restore checkpointed CUDA state onto the specified GPU.  `gpu` is required. | Worker (ctypes) |
| `attach()`              | Allocate unpinned CPU memory *per worker*, sized to the union of that rank's `main.named_parameters()` and (if present) `drafter.model.named_parameters()`.  Speculative-decoding drafters that expose a `.model` (Eagle / Medusa / DraftModel / ArcticProposer) contribute extra entries; non-model drafters (Ngram / Suffix) are skipped, in which case the layout collapses to main params only.  Reports `max_pinned_bytes_per_worker`. | Worker |
| `attach_pinned()`       | **Unsupported; raises.**  Use `attach()` -> `repin()` instead.  (It allocated a permanently-pinned buffer via torch `pin_memory=True` before staging moved onto the workers.) | Worker |
| `detach()`              | Free the CPU buffer on every worker.                         | Worker                                     |
| `repin()`               | `cudaHostRegister` each worker's buffer for DMA transfers.  Idempotent. | Worker                           |
| `unpin()`               | `cudaHostUnregister` each worker's buffer (data stays, CUDA registration removed).  Idempotent. | Worker          |
| `stage()`               | Snapshot main and drafter params (GPU -> that worker's CPU buffer) in vLLM's internal format. | Worker           |
| `plan_restore_weights(max_buffer_bytes=None)` | Self-compute the budget from `max_pinned_bytes_per_worker` (the per-GPU shard, not the TP-aggregate) and walk each worker's `index` once to build a chunk plan (each chunk packs whole params under the budget).  Cache the plan on the worker for the next `restore_weights()`.  An explicit `max_buffer_bytes` overrides the computation. | Instance + Worker |
| `restore_weights()`      | Pure execution against the cached chunk plan: per chunk, copy a slice of that worker's buffer to a single reused GPU staging buffer, then scatter into `main.named_parameters()` and (if present) `drafter.model.named_parameters()` in place using the namespaced index.  Falls back to a single chunk if no plan was cached.  Frees the staging buffer and calls `empty_cache()` before returning. | Worker |
| `save_weights()` / `load_weights()` | Write / read each worker's buffer as shards plus a `weights_meta.json` manifest.  Flat `weights/` at TP=1, per-rank `weights/rank{R}/` at TP>1. | Worker |
| `wake_up_weights()`     | Re-allocate weight tensors on GPU (main + drafter).  The arctic patch's disk reload of the main model is suppressed via `_skip_main_reload_on_wake`; both main and drafter parameters are populated by the subsequent `restore_weights()` from the worker's buffer.  Drafter `named_buffers()` are restored here from the per-sleep CPU snapshot. | Child |
| `wake_up_kv_cache()`    | Re-allocate KV cache on GPU.                                 | Child                                      |
| `generate(prompts, sp)` | Submit inference to the engine.  Assigns a unique `req_id`; result stored in `generate_results[req_id]` and `last_generate_result`. | Child (async engine loop) |
| `pause()`               | Freeze the engine and snapshot in-flight requests.  Sets `_paused`, captures every active sub-request's `(prompt_token_ids, output_token_ids_so_far, sampling_params)` into a child-local list, then `engine.abort_request(eids)` so subsequent `unpin`/`sleep`/`cuda_checkpoint` are safe.  Pending `generate_done` messages are deferred until `resume`. | Child |
| `resume()`              | Re-add saved requests via prefill and unfreeze the engine.  For each saved record, calls `engine.add_request(new_eid, TokensPrompt(prompt + output_so_far), SamplingParams(max_tokens=remaining, ...))`, repopulates `_active_reqs`, then clears `_paused`.  Original `req_id` continues seamlessly; eventual completion folds pre-pause `output_text` and token counts into the reported view. | Child |
| `teardown()`            | Tear down the instance, worker, and child.  Resets to created state, ready for `init(gpu)` again. | Worker + Child |
| `remove()`              | Deregister from the class-level registry (`Instance._all`).  Non-blocking and non-destructive; does not touch the worker process or pending commands.  Returns the `Instance` class so a chained `status()` resolves to the classmethod view. | Main process |
| `status()`              | Print all instances grouped by GPU with per-GPU memory usage.  Non-blocking: per-instance state is kept fresh in real time by each Instance's demuxer (the sole consumer of `_result_queue`), so no explicit sync step is needed before rendering.  Works as both `instance.status()` (returns `self` for chaining) and `Instance.status()` (returns the class). | Main process |


## CPU Buffer and Pin Management

### The buffer lives on the worker, not in the child process

All of the staging state -- the buffer, the param index, and the chunk
plan -- lives on each vLLM worker as `worker._semip_*`, and every step
runs there through `collective_rpc` (`_semip_attach`, `_semip_stage`,
`_semip_repin`, `_semip_unpin`, `_semip_restore_weights`,
`_semip_detach`, and the two weight-file primitives).

This is not incidental.  At TP>1 `collective_rpc` cloudpickles the
callable into every worker subprocess, so a buffer allocated in the
vllm_child process and captured by a closure would be copied by value
per worker and its writes discarded -- silently, with no error.  Each
rank also owns a *different shard* of the parameters, so one buffer in
the child would be the wrong size regardless.  At TP=1 the single
worker is this same process, so the identical code path just works.

The child aggregates the per-worker results: `attach` and
`plan_restore_weights` report `max_pinned_bytes_per_worker`, which is
what `Instance.plan_restore_weights` sizes the chunk budget from (the
TP-aggregate `pinned_cpu_bytes` would overstate the per-GPU figure).

### What each step does

`attach()` allocates a regular (unpinned) CPU buffer per worker via
`torch.empty(total_size, dtype=torch.uint8)`.  The buffer is sized to
the total bytes of `main.named_parameters()` plus, when speculative
decoding is configured with a model-bearing drafter,
`drafter.model.named_parameters()`.  An `index` dict maps each
*namespaced* parameter name -- `"main:p:<name>"` or
`"drafter:p:<name>"` -- to its `(offset, nbytes, dtype, shape)` in the
buffer.  Non-model drafters (Ngram / Suffix) contribute no entries; the
index then collapses to main params only and behavior matches the
pre-drafter pipeline byte for byte.

Pinning is a separate step: `repin()` calls `cudaHostRegister` (via
ctypes on `libcudart.so`) to register the buffer for DMA transfers.
`unpin()` calls `cudaHostUnregister` to remove the registration while
keeping the memory allocated and data intact.  Both are idempotent --
the attach buffer starts unpinned, and a double register or an
unregister of an unregistered buffer would hard-error.

`detach()` frees the CPU buffer entirely.

`attach_pinned()` is **not supported** on this path and raises; use
`attach()` followed by `repin()`.

### Why separate attach / repin / unpin

`cuCheckpointProcessRestore` must re-map all CUDA-registered (pinned)
host memory pages, which dominates restore latency (5-7s for large
models).  By unpinning before checkpoint and re-pinning after restore,
restore only needs to reconstruct the GPU context (~1s) and the pin
cost is paid separately via `cudaHostRegister` (~1-3s depending on
buffer size).  Net savings: ~1.5-2s per restore cycle for large models.

Using `cudaHostRegister` / `cudaHostUnregister` (instead of
`cudaHostAlloc` / `cudaHostFree`) is required because `cudaHostAlloc`
memory cannot be selectively unregistered -- `cudaHostUnregister` only
works on memory registered via `cudaHostRegister`.

### Standard sequences

- **Registration**: `attach() -> repin() -> stage() -> unpin() -> sleep() -> cuda_checkpoint()`
- **Save to disk**: `... -> cuda_checkpoint() -> criu_dump(filename)`
- **Load from disk**: `criu_restore(filename) -> plan_restore_weights() -> cuda_restore(gpu) -> ...`
- **Generate restore**: `cuda_restore(gpu) -> wake_up_weights() -> repin() -> restore_weights() -> wake_up_kv_cache() -> ...`
- **Generate checkpoint**: `... -> unpin() -> sleep() -> cuda_checkpoint()`
- **Pause checkpoint**: `pause() -> unpin() -> sleep() -> cuda_checkpoint()`
- **Pause restore**: `cuda_restore(gpu) -> repin() -> wake_up_weights() -> restore_weights() -> wake_up_kv_cache() -> resume()`

`plan_restore_weights()` is chained right after `criu_restore(filename)` because that
is when the instance has hydrated `total_gpu_bytes` and `pinned_cpu_bytes`
from `meta.json`.  The plan caches in the worker, survives `up <-> sleep`
cycles, and is rebuilt on each fresh `criu_restore(filename)`.  Cold start does not
need it (cold start never calls `restore_weights()`), and in-memory
checkpoint+restore paths that skip `criu_dump`/`criu_restore` rely on the single-chunk
fallback inside `restore_weights()`.

## stage / plan_restore_weights / restore_weights Pipeline

`stage()` (host capture) and `restore_weights()` (device populate) are an
inverse pair around the pinned CPU buffer.  Between `criu_restore(filename)` and
the first `restore_weights()`, the instance calls `plan_restore_weights()` to
build and cache a chunk plan in the worker.  `restore_weights()` then
executes the cached plan as pure I/O.

### stage (GPU main + drafter params -> worker CPU buffer)

`stage()` runs `_semip_stage` on every worker via `collective_rpc` (so
it can reach `worker.model_runner.drafter` -- `apply_model` only passes
the main `nn.Module`).  Each worker builds a unified
`name -> tensor.data` source table covering
`main.named_parameters()` keyed `"main:p:<name>"` plus, if
`drafter.model` exists, `drafter.model.named_parameters()` keyed
`"drafter:p:<name>"`.  It then walks its own `index` and copies each
entry's `.data` (contiguous, viewed as uint8) into its own buffer at
the recorded offset.  This captures weights in vLLM's post-processed
internal format (e.g. Marlin-packed for GPTQ, cutlass layout for FP8,
plain tensors for BF16) for both models in a single sweep.

`named_buffers()` are intentionally not staged.  Main buffers ride
stock vLLM's `_sleep_saved_buffers` snapshot; drafter buffers ride
arctic's `_save_module_state` snapshot, which is restored inside
`wake_up_weights()`.  The pinned-buffer pipeline is reserved for
parameters because parameters are static across the
sleep/checkpoint/restore/wake cycle and therefore safe to capture once
per registration.

### plan_restore_weights (cache the chunk plan)

`plan_restore_weights()` computes `max_buffer_bytes` from instance state
on the parent side:

```
allotment        = total_gpu_bytes * gpu_memory_utilization
max_buffer_bytes = min(pinned_cpu_bytes, allotment - pinned_cpu_bytes)
```

`total_gpu_bytes` is an NVML `.total` snapshot taken once in
`Instance.init(gpu)` (safe under the orchestrator contract that init
always takes a full L1 slot).  `pinned_cpu_bytes` is set by `attach()`
in the cold-start path or read from `meta.json` in the restore path.
When weights crowd the model's allotment, the budget shrinks; when
they fit comfortably (`pinned_cpu_bytes <= allotment / 2`) the budget
equals `pinned_cpu_bytes` and only one chunk is needed.

The worker then walks `index` in offset order and packs whole
parameters into chunks of `<= max_buffer_bytes` (no intra-parameter
splits).  If a single parameter exceeds the budget, the planner raises
`param X exceeds chunk_size`.  The plan is cached as
`(chunk_lo, chunk_hi, members)` triples plus `chunk_size`, and lives
on the worker until `detach()` resets it.

### restore_weights (cached plan -> GPU staging -> model params)

`_semip_restore_weights` runs entirely on each worker, so both the host
buffer and the GPU staging buffer are local to the rank that owns those
parameters.  It allocates one
`gpu_buf = torch.empty(chunk_size, dtype=torch.uint8, device=worker.device)`
before the loop, then for each chunk in the cached plan:

1. **Worker CPU buffer -> GPU staging buffer.**
   `gpu_buf[:n].copy_(buf[lo:hi], non_blocking=True)` followed by
   `torch.cuda.synchronize()`.
2. **GPU staging buffer -> main + drafter params (in place).**  The
   worker rebuilds the namespaced `name -> tensor.data` target table
   (mirror image of `stage`'s source table) and scatters this chunk's
   members; each src view is
   `gpu_buf[(off - lo):(off - lo) + nbytes].view(dtype).reshape(shape)`,
   copied into the corresponding `"main:p:<name>"` or
   `"drafter:p:<name>"` parameter via `target.copy_(src)`.  No
   `model.load_weights()` or `process_weights_after_loading()` is
   needed because the staged data is already in vLLM's internal format.

After the loop, `gpu_buf` is freed via `gpu_buf.storage().resize_(0)`
followed by `torch.cuda.empty_cache()`.  This releases memory through
PyTorch's normal caching allocator path, keeping allocator metadata
consistent for CRIU checkpoint/restore (see *Known Issues* below).

When `chunk_plan` is `None` (paths that never called
`plan_restore_weights`, such as in-memory checkpoint+restore tests that
skip `criu_dump`/`criu_restore`), the handler falls back to a single-chunk plan
covering the entire `index`, which is byte-identical to the
pre-chunking behavior.

### Arctic patch interaction

Arctic Inference patches `vllm.v1.worker.gpu_worker.Worker.sleep` and
`Worker.wake_up`.  At `level=2`, the upstream behavior is to drop main
weights from GPU and reload them from disk on wake; arctic adds a
per-sleep CPU snapshot of drafter parameters and buffers.  For
semi-persistence, both of these would conflict with the pinned-buffer
pipeline, so right after `LLM(...)` the child sets two opt-out flags
on every worker via `collective_rpc`:

| Flag                            | Effect                                                                                                                                  |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `_skip_main_reload_on_wake`     | `WorkerPatch.wake_up` does not call `GPUModelRunnerPatch._orig_reload_weights(self.model_runner)` -- main params come from the worker's staging buffer instead. |
| `_skip_drafter_param_snapshot`  | `WorkerPatch._save_module_state(drafter.model, skip_params=True)` skips `named_parameters()` -- drafter params come from the worker's staging buffer instead. |

Drafter `named_buffers()` are still snapshotted unconditionally and
restored inside `WorkerPatch.wake_up`; they are sub-MB and may carry
runtime state that is unsafe to reuse across a save/load.  Both flags
are read via `getattr(..., False)`, so default arctic behavior is
preserved for callers that do not opt in.

`GPUModelRunnerPatch.reload_weights` (which augments the standard
`reload_weights` with `drafter.load_model(self.model)`) is not gated:
semi-persistence never calls `model_runner.reload_weights` directly,
and the patched `Worker.wake_up` reaches the unpatched original via
`GPUModelRunnerPatch._orig_reload_weights`, so the drafter-load
augmentation is unreachable from this child.

### Staging buffer budget

The budget formula is the single source of truth and lives in
`Instance.plan_restore_weights`.  No safety margin, no minimum-budget
floor: pathological cases self-surface in the chunk planner with a
precise `param X exceeds chunk_size` message rather than via a
separate threshold.

`total_gpu_bytes` (NVML `.total` at `init`) and `pinned_cpu_bytes` are
written into `meta.json` at `criu_dump` time in the order
`{vllm_config, total_gpu_bytes, pinned_cpu_bytes}`.  Old images that
predate `total_gpu_bytes` are still loadable: `Instance.criu_restore` falls
back to the legacy `pinned_bytes` key for `pinned_cpu_bytes`, and a
missing `total_gpu_bytes` causes `plan_restore_weights` to send
`max_buffer_bytes=None`, which yields the single-chunk fallback in
`restore_weights`.

KV cache is not yet allocated when `restore_weights` runs (it is
`wake_up_kv_cache`'s job afterwards), so the `allotment - pinned`
slack is fully usable for the staging buffer.  No NVML calls happen
at runtime past the per-instance `init` snapshot.

## Pause and Resume

`pause` / `resume` are two CD-player-style verbs over in-flight
generation.  `pause` parks an instance mid-generation; `resume`
continues it from where it left off (with the same `req_id`s).  Both
are non-blocking, chainable, and idempotent.

### `pause()` -- freeze and snapshot

`pause` does two things in one step, both inside the child:

1. Sets the child's `_paused` flag so the main loop stops calling
   `engine.step()` even if there are unfinished requests.
2. Snapshots every active sub-request's
   `(prompt_token_ids, output_token_ids_so_far, sampling_params)`
   into a child-local list (`_saved_requests`) and calls
   `engine.abort_request(eids)` so the scheduler is empty.

After `pause`, the engine has no in-flight requests and no KV
blocks held on its behalf, so subsequent `unpin` / `sleep` /
`cuda_checkpoint` are safe.  The captured state lives as a plain
Python list, which CRIU dumps and restores for free across
`cuda_checkpoint` / `cuda_restore`, so no extra plumbing is needed.

The captured fields per sub-request `eid` come from the child's
own bookkeeping (populated incrementally in `_process_step_outputs`):

- `prompt_token_ids` -- exactly what the engine processed during prefill
- `output_token_ids` -- cumulative decoded tokens at the moment of pause
- `output_text` -- detokenized text for those output tokens
- `sampling_params` -- the dict captured at `generate()` submit time
- per-`req_id`: `t0`, `first_token_ts`, original `prompts` (for logging)

`pause` is **n=1 only** in this revision; n>1 raises so the caller
can fall back to the canonical "wait for completion" path.

### `resume()` -- re-prefill and unfreeze

`resume` is the inverse of `pause`.  For each saved record it calls
`engine.add_request(new_eid, TokensPrompt(prompt + output_so_far),
SamplingParams(max_tokens=remaining, ...))`, repopulates
`_active_reqs[req_id]` with the original `t0`, `first_token_ts`, and
`prompts` plus `pre_pause_completion` / `pre_pause_text` /
`original_prompt_tokens` per sub-request, then clears `_paused`.

When the engine eventually finalizes the resumed request,
`_process_step_outputs` folds those fields back in so the final
`generate_done`:

- carries the **original** `req_id`
- reports `prompt_tokens` from the original prompt only (not
  `prompt + output_so_far`)
- reports `completion_tokens` as `pre_pause + post_resume`
- splices `pre_pause_text + post-resume text` per sub-request

If a sub-request had already hit its `max_tokens` pre-pause,
`resume` emits a synthetic `generate_done` immediately rather than
re-submitting it.

### Design choice: re-prefill, not KV-block snapshot

`pause` does **not** move the KV cache to CPU.  It captures the
token-id state and aborts; the KV blocks are then freed by `sleep()`
along with the rest of GPU memory.  `resume` pays a one-shot prefill
instead of DMAing tens of GB of KV blocks back from CPU.

The trade-off is that re-prefill is **bit-exact only for greedy**
(`temperature=0`) decoding.  For `temperature>0` / non-zero `top_p`
paths the post-resume token trajectory will diverge from a hypothetical
un-paused run because the per-request RNG state is not captured.  If
you need lossless continuation across pause boundaries for stochastic
sampling, use a fixed `seed` per request, or stage the actual KV
blocks (not implemented here).

### Worker drain interaction

After `pause`, the child no longer produces `generate_done` messages,
so the worker would deadlock on `_drain_pipe_generates` -- which
synchronous commands like `sleep` / `cuda_checkpoint` call before
forwarding.  The worker mirrors `_paused` in `_worker_paused` (set on
a successful `pause` ack, cleared on `resume`) and turns the drain
into a no-op while paused.  `_pending_generates` itself is unchanged
across the cycle: the original `generate_done` arrives once the
resumed engine finishes the request, and the counter decrements
normally.  Both `pause` and `resume` are in the worker's drain-skip
set on the default forwarding path.

## Command Sequences

### Cold start and checkpoint

```python
instance_1 = Instance(vllm_config_1)
instance_2 = Instance(vllm_config_2)

instance_1.init(gpu=0).attach().repin().stage().unpin().sleep().cuda_checkpoint()
instance_2.init(gpu=1).attach().repin().stage().unpin().sleep().cuda_checkpoint()
```

`init()` loads real weights via `load_format=auto`, so vLLM runs
`process_weights_after_loading()` and the model is ready to use.
`attach()` allocates an unpinned CPU buffer sized to the model's
parameters.  `repin()` registers it with CUDA for DMA.  `stage()`
snapshots the post-processed GPU parameters into the buffer.  `unpin()`
removes the CUDA registration so that `cuda_checkpoint()` is fast (the CUDA
driver does not need to re-map pinned pages on restore).  The buffer
data survives `unpin()`, `sleep()`, and `cuda_checkpoint()` since it is CPU
memory.

### Save to disk

After checkpoint, `criu_dump()` writes a CRIU image to disk.  The dump is
destructive — the child process is killed after the image is written.
The worker exits and the instance returns to a clean state:

```python
inst = Instance(vllm_config)
inst.init(gpu=0).attach().repin().stage().unpin().sleep().cuda_checkpoint()
inst.criu_dump("/data-fast/image-cache/my_model").wait()
# child is dead, worker exits
```

### Load from disk

Every use after save goes through `criu_restore()`, which restores a fresh
process from the on-disk image.  The instance's `vllm_config` must
match the saved image's config (validated automatically):

```python
inst = Instance(vllm_config)
inst.criu_restore("/data-fast/image-cache/my_model").plan_restore_weights().wait()

inst.cuda_restore(gpu=0).wake_up_weights().repin().restore_weights().wake_up_kv_cache()
inst.generate(prompts, sampling_params).wait()
```

`plan_restore_weights()` is chained right after `criu_restore()` because that is
when the instance has hydrated `total_gpu_bytes` and `pinned_cpu_bytes`
from `meta.json`.

### Initialize a new instance after another finishes

We need to wait for instance 1's checkpoint on GPU 0 before
initializing instance 3 on the same GPU.

```python
instance_1.wait()

instance_3 = Instance(vllm_config_3)
instance_3.init(gpu=0).attach().repin().stage().unpin().sleep().cuda_checkpoint()

instance_3.wait()
instance_2.wait()
```

### Restore and generate (hot path)

After cold-start, weights are already staged in each worker's CPU buffer
(unpinned from CUDA).  Restore re-pins and moves them CPU→GPU:

```python
instance_1.cuda_restore(gpu=0)
instance_1.repin()
instance_1.wake_up_weights()
instance_1.restore_weights()
instance_1.wake_up_kv_cache()
instance_1.generate(prompts, sampling_params)
instance_1.wait()
result = instance_1.last_generate_result
```

To re-checkpoint after generate:

```python
instance_1.unpin().sleep().cuda_checkpoint().wait()
```

### Swap active model on a GPU

```python
instance_1.unpin().sleep().cuda_checkpoint().wait()

instance_3.cuda_restore(gpu=0).repin()
instance_3.wake_up_weights()
instance_3.restore_weights()
instance_3.wake_up_kv_cache()
instance_3.wait()
```

### Two instances active on the same GPU

```python
vllm_config_4 = {"model": "Qwen/Qwen2.5-1.5B", "gpu_memory_utilization": 0.4}
vllm_config_5 = {"model": "Qwen/Qwen3-1.7B", "gpu_memory_utilization": 0.4}

instance_4 = Instance(vllm_config_4)
instance_5 = Instance(vllm_config_5)

instance_2.sleep().detach().cuda_checkpoint().wait()

instance_4.init(gpu=1)
instance_4.wait()
instance_5.init(gpu=1)
instance_4.attach().repin().stage()
instance_5.attach().repin().stage()
instance_4.unpin().sleep().cuda_checkpoint()
instance_5.unpin().sleep().cuda_checkpoint()
instance_4.wait()
instance_5.wait()
```

Reload both on the same GPU at the same time:

```python
# In-memory checkpoint+restore (no save/load), so plan_restore_weights
# is not chained: restore_weights falls back to a single-chunk plan.
instance_4.cuda_restore(gpu=1).repin().wake_up_weights().restore_weights().wake_up_kv_cache()
instance_5.cuda_restore(gpu=1).repin().wake_up_weights().restore_weights().wake_up_kv_cache()

instance_4.wait()
instance_5.wait()
```

If on the same GPU, `init()` does not overlap.  It is the caller's
responsibility to serialize (e.g. by calling `wait()` between inits).

### Cross-GPU migration

Once checkpointed, an instance is stateless (`gpu=None`).  `cuda_restore(gpu)`
specifies which GPU to restore onto -- it can be the same or a different
GPU.

```python
instance = Instance(vllm_config)
instance.init(gpu=0).attach().repin().stage().unpin().sleep().cuda_checkpoint().wait()
# instance.gpu is now None

# Restore on GPU 1
instance.cuda_restore(gpu=1).repin()
instance.wake_up_weights().restore_weights().wake_up_kv_cache()
instance.wait()
# instance.gpu is now 1
```

### Pause across checkpoint

Park an instance mid-generation, swap it out, and continue later with
the same `req_id`.  The original `generate(...)` call's `wait()`
eventually receives the completed output, with timings and token
counts folded as if the pause never happened.

```python
instance.generate(prompts, sampling_params)  # long-running

# ... some time later, after partial decode ...
instance.pause().unpin().sleep().cuda_checkpoint().wait()
# instance.gpu is now None; KV blocks freed; only token-id state lives in CPU.

# ... model is swapped out, another instance runs on the GPU ...

instance.cuda_restore(gpu=0).repin().wake_up_weights().restore_weights() \
        .wake_up_kv_cache().resume().wait()
result = instance.last_generate_result   # original req_id, full output
```

`resume` re-prefills `prompt_token_ids + output_token_ids_so_far`
per sub-request and reduces `max_tokens` by the pre-pause output
length.  For `temperature=0` (greedy) sampling the post-pause
trajectory is bit-exact; for stochastic sampling it diverges (see
"Pause, Resume, Cancel").

## Concurrency Model

All primitives are **non-blocking** from the main process -- they
enqueue a command and return immediately.  Each Instance has its own
worker process, so **commands for different instances run concurrently**.
Commands for the same instance are serialized by its worker.

```
Main process          Worker              vLLM child
-----------          ------              ----------
inst.sleep() -----> cmd_queue
                     pipe.send("sleep") -----------> pipe.recv
                     pipe.recv("done")  <----------- pipe.send("done")
                     completed_counter += 1
                     result_queue.put()

inst.cuda_checkpoint()--> cmd_queue
                     _worker_checkpoint(pid)
                       enumerate descendants via psutil
                       checkpoint EngineCore (leaf first)
                       checkpoint vLLM child (parent last)
                       store checkpointed PID list
                     completed_counter += 1
                     result_queue.put()

demuxer  <----------- result_queue.get()
inst.wait() <-------- demuxer.wait_idle()  (condvar on _pending_count)
```

### Demuxer architecture

Each Instance owns a single per-instance `Demuxer` thread (see
[`demuxer.py`](../demuxer.py)) that is the **sole consumer** of
`_result_queue`.  The demuxer is created lazily by `_ensure_queues`
(at `init` / `criu_restore`) and torn down by `_close_queues` (at
`teardown` / `_reset`).  For every result it:

1. Calls `_apply_result` (with prompts pre-injected for generate
   cmds) so observable Instance state -- `state`, `gpu`, `pid`,
   `pinned_cpu_bytes`, `last_*`, `generate_results` -- is updated
   first.
2. Decrements `_pending_count` and `notify_all`s the condvar all
   `wait()` callers park on, latching the first error encountered
   so a subsequent `wait_idle()` re-raises and clears it.
3. Logs the cmd's outcome.
4. Dispatches to per-cmd listeners (e.g. the orchestrator's
   generate listener) and to any catch-all (`cmd=None`) listeners
   (e.g. the orchestrator's FIFO ack signaller).

Because there is exactly one consumer thread per instance,
concurrent `inst.wait()` callers are safe and cannot deadlock by
both racing for `_result_queue.get()`.

### `wait()` semantics

`instance.wait()` blocks until the demuxer reports `_pending_count
== 0` (a thread-safe condvar wait); it never reads from
`_result_queue` directly.  You can batch many commands and call
`wait()` once, including from many threads at once:

```python
instance.unpin().sleep().cuda_checkpoint().wait()
```

If any cmd in this batch failed at the worker, the first error is
re-raised by `wait()` (and cleared from the latch so the next
batch starts fresh).  Listeners still receive both successful and
failed results.

## Checkpoint / Restore Ordering

Checkpoint and restore walk the full process tree of the vLLM child
using `psutil`.  The `resource_tracker` process is skipped (no GPU).

With `VLLM_ENABLE_V1_MULTIPROCESSING=0`, the EngineCore runs in-process
so there is typically only the vLLM child process to checkpoint.  The
process tree walk still handles any descendants that may hold GPU state.

The list of checkpointed PIDs is stored so that restore uses the exact
same set.

### Cross-GPU Restore

`cuda_restore(gpu)` supports restoring onto a different GPU using the CUDA
driver's `CUcheckpointRestoreArgs` with `CUcheckpointGpuPair` UUID
mapping (requires driver 580+).  The GPU pair mapping must be a valid
permutation: old_gpu swaps with new_gpu, all others map to themselves.

## Instance Status

`Instance.status()` shows the state of all instances grouped by GPU,
including per-GPU memory usage and per-process GPU bytes (via NVML's
`nvmlDeviceGetComputeRunningProcesses`).  It uses a class-level
`WeakValueDictionary` registry (`Instance._all`) to auto-discover all
instances.

`status()` is dual-purpose: called as `instance.status()` it returns
`self` so it can be chained between primitives, and called as
`Instance.status()` it returns the `Instance` class.  `remove()`
returns the class as well, so chains like
`inst.teardown().wait(); inst.remove().status()` are valid.

No explicit pre-print sync step is needed: each Instance's demuxer
thread is the **sole consumer** of `_result_queue`, draining and
applying results in real time, so `status()` always observes a
fresh view without competing for `get()`.

GPU memory is queried via NVML (`pynvml`, not
`torch.cuda.mem_get_info`) to avoid initializing CUDA in the main
process.

## Architecture

Two models share the same **architecture** if they have the same
structure but potentially different weights.  Architecture is determined
by reading `config.json` from the model path and extracting:

```
(architectures, hidden_size, num_hidden_layers, num_attention_heads,
 num_key_value_heads, intermediate_size, head_dim, vocab_size)
```

## CRIU Installation (v4.2, from source)

Ubuntu archive mirrors may be unreachable on some machines.  The
reliable path is to build CRIU and its dependencies from source using
GitHub (which is always reachable).

### Dependencies (built from source)

```bash
mkdir -p /tmp/criu-build && cd /tmp/criu-build

# libcap
git clone --depth 1 --branch libcap-2.73 https://git.kernel.org/pub/scm/libs/libcap/libcap.git
cd libcap && make -j$(nproc) && sudo make install prefix=/usr && cd ..

# protobuf-c (requires protoc + libprotobuf-dev already installed)
#   If /usr/include/google/protobuf/compiler/ is missing, copy headers
#   from the protobuf source matching your installed protoc version:
#     git clone --depth 1 --branch v3.21.12 https://github.com/protocolbuffers/protobuf.git protobuf-src
#     sudo cp -r protobuf-src/src/google/protobuf/compiler /usr/include/google/protobuf/compiler
#   If libprotoc.so symlink is missing:
#     sudo ln -sf /usr/lib/x86_64-linux-gnu/libprotoc.so.32 /usr/lib/x86_64-linux-gnu/libprotoc.so
git clone --depth 1 --branch v1.5.0 https://github.com/protobuf-c/protobuf-c.git
cd protobuf-c && ./autogen.sh && ./configure --prefix=/usr && make -j$(nproc) && sudo make install && cd ..

# libnet
git clone --depth 1 --branch v1.3 https://github.com/libnet/libnet.git
cd libnet && ./autogen.sh && ./configure --prefix=/usr && make -j$(nproc) && sudo make install && cd ..

# uuid.h (if /usr/include/uuid/uuid.h is missing)
git clone --depth 1 --branch v2.40.4 https://github.com/util-linux/util-linux.git
sudo mkdir -p /usr/include/uuid
sudo cp util-linux/libuuid/src/uuid.h /usr/include/uuid/uuid.h
sudo ln -sf /usr/lib/x86_64-linux-gnu/libuuid.so.1 /usr/lib/x86_64-linux-gnu/libuuid.so
```

### CRIU 4.2

```bash
cd /tmp/criu-build
git clone --depth 1 --branch v4.2 https://github.com/checkpoint-restore/criu.git
cd criu
PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH" make -j$(nproc)
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-criu PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-lib PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-crit PREFIX=/usr

# Empty plugin directory (used by --libdir during dump; the dump creates
# it if missing, so this is optional)
sudo mkdir -p /usr/lib/criu/empty
```

### Verify

```
criu --version          # Version: 4.2
which crit              # /usr/local/bin/crit
ls -d /usr/lib/criu/empty
```

## File Structure

```
semi_persistence/
  instance.py        -- Instance class (GPU-agnostic handle, owns worker process)
  worker.py          -- Worker loop, child thread, checkpoint/restore/CRIU save/load
  vllm_child.py      -- vLLM child process (owns GPU, pinned memory, async engine loop)
  demuxer.py         -- Result-queue demultiplexer (keeps Instance state fresh)
  slots.py           -- Buddy allocator for fractional GPU slots (see slots_DESIGN.md)
  pipeline.py        -- Per-model Op pipeline (see pipeline_DESIGN.md)
  orchestrator.py    -- Orchestrator class (see orchestrator_DESIGN.md)
  orch_server.py     -- HTTP front end for the Orchestrator
  client.py          -- OrchestratorClient (see client_DESIGN.md)
  state_server.py    -- HTTP /state endpoint for the dashboard
  dashboard.py       -- Curses-based live dashboard (GPU/CPU tiers, requests)
  abstract.py        -- Abstract InstanceBase interface (reference only)
  semip_logging.py   -- Logging setup

  tests/             -- pytest suite (CPU-only, no GPU required)
    test_pipeline.py -- Pipeline primitives against fake Ops
    test_slots.py    -- Buddy-allocator unit tests

  scripts/           -- Imperative repro scripts (require real GPUs + vLLM)
    main_test.py     -- Integration walkthrough of the Instance primitives
    test_copy.py     -- Concurrent multi-model restore driven through Slots
    test_env.py      -- Per-model vllm_config["_env"] smoke test
    test_generate.py -- Orchestrator end-to-end register/generate/move
    test_image.py    -- CRIU save/load image cache test

  skills/            -- Agent skill + all documentation
    SKILL.md         -- Orientation map for the subsystem
    reference.md     -- Per-subsystem detail behind SKILL.md
    instance_DESIGN.md        -- This file
    orchestrator_DESIGN.md    -- Orchestrator state machine and API
    pipeline_DESIGN.md        -- Per-model Op pipeline
    slots_DESIGN.md           -- Buddy allocator
    client_DESIGN.md          -- OrchestratorClient job/model split
    async_generate_DETAILS.md -- Async generate implementation details
    CRIU_PLUMBING.md          -- CRIU dump/restore complications and fixes
    INSTALL.md                -- CRIU install + draft model sync
```

## Known Issues

### Staging buffer must be freed via `storage().resize_(0)`

When vLLM's sleep mode is enabled, the cumem pluggable allocator
intercepts all `torch.empty(..., device="cuda")` calls -- including the
GPU staging buffer allocated inside `restore_weights`.  The buffer **must**
be freed via `buf_gpu.storage().resize_(0)`, not
`caching_allocator_delete(ptr)`.

`caching_allocator_delete` tells PyTorch the memory was freed
externally, which corrupts the caching allocator's internal block
tracking.  When CRIU restores the process, the allocator uses stale
metadata and subsequent GPU allocations crash with
`c10::Error: invalid device pointer`.

`storage().resize_(0)` releases memory through the normal allocator
`free` path, keeping block metadata consistent across
checkpoint/restore cycles.

### Cold start is slower than dummy-weight init

`init()` uses `load_format=auto` to load real weights from disk and
run `process_weights_after_loading()`.  This is slower than the
previous `load_format=dummy` approach, but it only happens once per
instance.  The benefit is that `stage()` captures weights in vLLM's
internal format, so restore works correctly for all model types
including quantized models (GPTQ, FP8).
