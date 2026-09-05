# semi_persistence — subsystem reference

Deeper detail behind [SKILL.md](SKILL.md). Organised bottom-up: instance
primitives, slots, pipeline, orchestrator, client, CRIU, then the test and
tooling inventory.

---

## 1. Instance primitives

`Instance` is a main-process handle wrapping one vLLM config. Primitives are
designed to be **non-blocking** (except `wait`), **optimized** so combinations
stay optimal, and **chainable** — they all return `self`.

```python
inst = Instance({"model": "Qwen/Qwen3-8B-FP8", "enforce_eager": True})
inst.init(gpu=0).attach().repin().stage().unpin().sleep().cuda_checkpoint()
inst.criu_dump("/data-fast/image-cache/foo").wait()
```

### Lifecycle

| Primitive | Effect | Runs in |
|---|---|---|
| `init(gpus=[...])` | Cold start with real weights; spawns worker + child; applies `_env`. A scalar `gpu=` still works at TP=1 | Worker + child |
| `criu_restore(path=None)` | CRIU-restore from disk; validates the image's `vllm_config` and `model_dir` match. Defaults to `<model_dir>/image` | Worker |
| `criu_dump(path=None)` | CRIU-dump the child tree (**destructive**); writes `meta.json`. Defaults to `<model_dir>/image` | Worker |
| `teardown()` | Tear down instance, worker, child; resets to created state | Worker + child |
| `remove()` | Deregister from `Instance._all`; non-blocking, non-destructive | Main |
| `wait()` | Block until pending commands complete | Main |

`Instance(vllm_config, model_dir=None)`. With a `model_dir`, the dump and
restore paths default to `<model_dir>/image` and the compile cache moves
under `<model_dir>/compilation`; see [semi-p_DESIGN.md](semi-p_DESIGN.md).

### GPU residency

| Primitive | Effect |
|---|---|
| `sleep()` | `llm.sleep(level=2)` — frees GPU memory for main and drafter weights |
| `cuda_checkpoint()` | Save CUDA state to CPU via `cuCheckpointProcess`; `gpu` becomes `None`. At TP>1 also inserts `cleargraph` + `destroy_nccl` first |
| `cuda_restore(gpus=[...])` | Restore CUDA state onto specific GPU(s). Defaults to the placement recorded in the image; a scalar `gpu=` still works at TP=1 |
| `wake_up_weights()` | Re-allocate weight tensors on GPU (main + drafter) |
| `wake_up_kv_cache()` | Re-allocate the KV cache on GPU |

### Tensor parallel (no-ops at TP=1)

TP size comes from `vllm_config["tensor_parallel_size"]`; `gpus` is
placement only and must have exactly that many entries. See
[tp_DESIGN.md](tp_DESIGN.md).

| Primitive | Effect |
|---|---|
| `destroy_nccl(graph_mode="reuse")` | Tear down NCCL and CustomAllreduce IPC before a checkpoint. Also marks the worker's inet TCP sockets `SO_LINGER(1,0)` so their ports skip `TIME_WAIT` when the dump kills the tree (Complication 12) |
| `reinit_nccl()` | Rebuild NCCL on a fresh port. Must run after `cuda_restore` and before the model runs or a captured graph replays; `attach`/`load_weights` are CPU-only and unconstrained by it |
| `cleargraph(graph_mode="reuse")` | Drop CUDA-graph exec handles; `reuse` preserves them |
| `recapture_graphs(graph_mode="reuse")` | Rebind (`reuse`) or recapture (`full`) decode graphs, after `wake_up_kv_cache` |

### CPU buffer and weight transfer

| Primitive | Effect |
|---|---|
| `attach()` | Allocate unpinned CPU memory *per worker*, sized to that rank's main + drafter `named_parameters()` |
| `attach_pinned()` | **Unsupported; raises.** Use `attach()` -> `repin()` |
| `detach()` | Free each worker's CPU buffer |
| `repin()` / `unpin()` | `cudaHostRegister` / `cudaHostUnregister` each worker's buffer. Idempotent |
| `stage()` | Snapshot main + drafter params GPU -> that worker's CPU buffer |
| `save_weights()` | Write each worker's buffer to `<model_dir>/weights` as shards + `weights_meta.json`. Call after `stage()`, before `detach()`, so the image stays small |
| `load_weights()` | Read those shards back into the buffer. Requires a prior `attach()` on the restore side |
| `plan_restore_weights(max_buffer_bytes=None)` | Build and cache a chunk plan under a computed byte budget; pass an explicit cap for older images |
| `restore_weights()` | Execute the cached plan: buffer -> one reused GPU staging buffer -> scatter |

`save_weights` / `load_weights` are optional: without them the staged
weights stay inside the CRIU image, which is what the orchestrator does.
At TP>1 the shards fan out to `weights/rank{R}/`.

All of this state (buffer, param index, chunk plan) lives on the vLLM
workers as `worker._semip_*` and runs there via `collective_rpc`, not in
the child process. At TP>1 a buffer held in the child would be
cloudpickled by value into each worker subprocess and its writes lost;
each rank also owns a different shard. One code path serves both TP
sizes. See [instance_DESIGN.md](instance_DESIGN.md).

The drafter contributes extra parameter entries only when it exposes a `.model`
(Eagle / Medusa / DraftModel / ArcticProposer). Ngram and Suffix drafters are
skipped, collapsing the layout to main params only.

### Serving

| Primitive | Effect |
|---|---|
| `generate(prompts, sp)` | Submit inference; result lands in `generate_results[req_id]` |
| `pause()` | Freeze engine, snapshot in-flight requests, `abort_request` them |
| `resume()` | Re-add saved requests via prefill with `max_tokens=remaining`, unfreeze |
| `status()` | Print all instances grouped by GPU; works as instance or class method |

`pause` captures each active sub-request's `(prompt_token_ids,
output_token_ids_so_far, sampling_params)` so that `unpin` / `sleep` /
`cuda_checkpoint` are safe afterwards. `resume` replays them under a fresh
engine id while the caller's original `req_id` continues seamlessly; the final
result folds pre-pause text and token counts back in.

---

## 2. Slots — the buddy allocator

Pure bookkeeping in `slots.py`. No CUDA, no vLLM, no process state.

- **`Slot(gpu_id, level, index)`** — immutable and frozen. Level `L` covers
  `1 / 2^(L-1)` of a GPU; `0 <= index < 2^(level-1)`.
- **Node states** — every conceptual tree node is exactly one of `FREE` (in the
  level-`L` free list), `ALLOC` (handed out), or `SPLIT` (has live descendants,
  so it cannot be handed out whole).
- **Buddy** of `(g, L, i)` is `(g, L, i ^ 1)`; **parent** is `(g, L-1, i // 2)`;
  **children** are `(g, L+1, 2i)` and `(g, L+1, 2i+1)`.
- **Split** on allocation when no exact-size slot is free but a larger one is.
  **Coalesce** on deallocation when both buddies are `FREE`.
- **Waiters** queue FIFO. Auto-pick prefers the coldest GPU, ordered by
  `(_last_used, gpu_id)` ascending.

Key API: `init(gpus)`, `allocate(level, gpu=None)` (blocking),
`try_allocate(level, gpu=None)` (returns `None` rather than blocking),
`deallocate(slot)`, `status()`, `remove()`.

The behaviours worth preserving are pinned down by `tests/test_slots.py`:
`try_allocate` never poaches a different GPU than the one requested, FIFO order
holds under concurrent waiters, an L1 waiter is unblocked by an L2 coalesce, and
auto-pick really is coldest-first.

---

## 3. Pipeline — explicit ops per model

`pipeline.py` gives each `model_id` one worker thread plus one FIFO queue.
This replaced an implicit design built on `ThreadPoolExecutor`, `_futures[mid]`,
`_last_generate_future`, and `_inflight[mid]`, along with four 0.5 s polling
loops.

**Ops:** `RegisterOp`, `MoveOp`, `EvictForPeerOp`, `GenerateOp`, `PauseOp`,
`ResumeOp`, `RemoveOp`.

**Ordering** is the FIFO itself — back-to-back submissions cannot interleave,
because `Op.execute()` is atomic on the worker thread. This is what makes the
old eviction-mid-generate and move-vs-generate races structurally impossible
rather than merely fixed.

**Interrupts.** `InterruptFlag` replaces polling. A long-running op calls
`raise_if_set()` at yield points or parks in `wait_or_interrupt(ev, timeout)`,
which returns early and raises `Interrupted` when the flag fires. Pause is
`interrupt_now()` followed by `submit_front(PauseOp)`, so the pause jumps the
queue and the interrupted op is re-queued behind it.

**Cross-model work.** `submit_to_peer_and_wait(peer_pipe, op)` is how an
acquirer evicts an incumbent from a slot it wants. It carries cycle detection so
two pipelines evicting each other cannot deadlock.

Legacy names still appear throughout `orchestrator_DESIGN.md` as anchors for the
bug-fix discussion; that document has a translation table mapping each to its
post-migration counterpart.

---

## 4. Orchestrator

`orchestrator.py` maps human-readable `model_id`s to `Instance`s and drives the
state ladder. Public API:

| Method | Purpose |
|---|---|
| `init(image_cache, gpus=None)` | Point at an image cache dir, seed the GPU pool, discover saved models |
| `register(model_id, vllm_config)` | Cold-start and register a new model |
| `move(model_id, target, target_gpu=None)` | Walk the ladder up or down to `target` |
| `generate(model_id, prompts, sampling_params=None)` | Auto-up if needed, run, park slotless |
| `pause(model_id)` / `resume(model_id)` | Freeze / thaw in-flight work |
| `remove(model_id)` | Auto-move to `saved` if needed, then delete image and registry entry |
| `wait(model_id)` | Block until that model's queue drains |
| `models()` / `status()` | Registry listing and console view |
| `add(gpu)` / `sub(gpu)` / `wait_gpu(gpu)` | Runtime GPU pool resize |

Fan-out across every registered model is the caller's job — `models()` plus a
loop. Passing `target_gpu` to `move(..., "sleep", G)` produces the
**slotless-sleep** flavour: the model parks on a named GPU without consuming a
slot.

Eviction targets slotless `up` models, walking an incumbent down the ladder far
enough to free the slot the acquirer needs.

---

## 5. Client and serving

`orch_server.py` exposes the orchestrator over HTTP; `client.py` provides
`OrchestratorClient`, a classmethod-style client with a **two-layer naming
model**:

```
job_id (caller's vocabulary)  ->  _jobs dict (client-local)  ->  model_id (server)
"job 1"                            {"job 1": "model_1"}          "model_1"
```

Two jobs sharing a `vllm_config` **deduplicate to one model** on the server,
which is how a workload fans out over logical jobs while sharing one backing
model. Note that `vllm_config["_env"]` participates in that dedup, so two jobs
differing only in env vars stay distinct.

The per-job API (`generate`, `wait`, `remove`, `pause`, `resume`, `move`) takes
`job_id` first and uses **type-based dispatch** on that slot: `str` means a job
id, `int` means something method-appropriate, and `None` or omitted means fan
out across every registered job.

---

## 6. CRIU

Full detail in `CRIU_PLUMBING.md`. The dump is **destructive** by design: CRIU
kills the child after writing the image, so there are never dangling processes
from a non-destructive (`-R`) dump and the state machine stays simple.

### Dump sequence

In the child (`prepare_criu_dump` in `vllm_child.py`):

1. Drain in-flight engine requests.
2. Destroy the PyTorch process group (NCCL, TCPStore threads).
3. Wait for store threads to exit — poll `/proc/<pid>/task`, up to 2.5 s.
4. `dup2 /dev/null` over stdout and stderr.
5. Walk `/proc/<pid>/fd` and close everything off the keep-list.
6. Munmap every `io_uring` region found in `/proc/<pid>/maps`.
7. Remove `/dev/shm/sem.*` (the mappings stay live as anonymous memory).
8. Audit `/proc/<pid>/task` for non-`python` threads (informational only).

Then back in the worker (`_worker_criu_save` in `worker.py`): map child socket
FDs to `--external unix[ino]`, record `/dev/nvidia*` FDs into `meta.json`, and
run the destructive `criu dump`.

### Restore sequence

Pass the pipe FD through a Unix socket via `SCM_RIGHTS` into a helper (under
`sudo` only when the worker is not already root), which `dup2`s it into place,
unshares a private PID namespace, and runs `criu restore` inside it with
`--inherit-fd` for the pipe plus stdout/stderr, `--link-remap` and
`--tcp-close` (no `--shell-job` — the child holds no tty).  The CUDA context
comes back via `cuda-checkpoint restore`.

### The thirteen complications

| # | Complication | Shape of the fix |
|---|---|---|
| 1 | PyTorch distributed (NCCL + TCPStore) | Destroy process group, wait for store threads |
| 2 | io_uring | Munmap the rings before dump |
| 3 | POSIX semaphores (`/dev/shm/sem.*`) | Unlink the files, keep the mappings |
| 4 | stdout/stderr | `dup2 /dev/null`, restore with `--inherit-fd` |
| 5 | Pipe FD through `sudo` | Pass via `SCM_RIGHTS`, helper `dup2`s then `execvp`s |
| 6 | CUDA context | Driver API rather than the CLI |
| 7 | CRIU plugin directory | `--libdir` needs `/usr/lib/criu/empty`; the dump creates it |
| 8 | Task-id collisions at restore (threads count, not just PIDs) | Restore into a private PID namespace; on the no-namespace path a preflight that names the occupants (`scripts/pidcheck.py`) and dump-side id placement; retry loop as backstop |
| 9 | Ghost remap race (CRIU 4.2) | `--link-remap` handling |
| 10 | Per-restore PID namespace, and the tty it forced out | Reaper + private `/proc`; child `setsid`, `--shell-job` dropped |
| 11 | Unprivileged dump + restore | `SEMIP_UNPRIVILEGED=1`: `--unprivileged` on both sides, no-namespace restore, caps shed in the child |
| 12 | `TIME_WAIT` on the recorded local port | `SO_LINGER(1,0)` on the workers' inet TCP sockets, so the dump's kill RSTs instead of FINs |
| 13 | Teardown kill scoping on the no-namespace path (a `kill -9 -<pid>` meant as a group kill becomes `kill(-1)` in procps-ng, ending the container) | Snapshot task ids + start times at restore (`_tree_identity`); kill only verified positive pids, never a negative one. See [`TEARDOWN_SCOPING.md`](TEARDOWN_SCOPING.md) |

`meta.json` alongside the image holds the `vllm_config` (including `_env`) and
the CRIU metadata, which is what lets the orchestrator rediscover saved models
on reboot and lets `criu_restore` validate the image against the instance.

### Environment switches

| Variable | Effect |
|---|---|
| `SEMIP_UNPRIVILEGED=1` | Run dump and restore on a pod granting only `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`. Adds `--unprivileged` to both, takes the no-namespace restore path, and sheds the child's capabilities so the image is portable to a low-cap node. Costs concurrent restore (one live restore per node). See Complication 11 |
| `_SEMIP_CHILD_DROP_CAPS=1` | Internal only, set by the worker across the child's spawn. Not a user flag |

---

## 7. Tests, scripts, tools

### `tests/` — pytest, CPU-only

| File | Contents |
|---|---|
| `conftest.py` | `pipeline_mode` fixture, a compatibility shim now always returning `"pipeline"` — **currently unused by any test** |
| `test_pipeline.py` | 29 tests driving `ModelPipeline` with fake `Op` subclasses |
| `test_slots.py` | 6 tests for the buddy allocator; also has a standalone `__main__` runner |

35 tests, roughly 2.5 s, no GPU. This is the only part suitable for CI.

Each file starts with a `sys.path.insert` bootstrap. Be aware this front-loads
very generic module names — `client`, `worker`, `monitor`, `slots`, `pipeline`,
`compare` — onto `sys.path` for the whole session, which is a shadowing hazard
if these ever run alongside the wider repo suite.

### `scripts/` — imperative, need real GPUs

| File | What it exercises |
|---|---|
| `main_test.py` | Full `Instance` walkthrough: checkpoint, restore, GPU swap, two small models on one GPU |
| `test_copy.py` | 13 configs restored concurrently from images via threads, driven through `Slots` |
| `test_env.py` | The `vllm_config["_env"]` path: per-child env, reserved-key drops, `meta.json` persistence |
| `test_generate.py` | Orchestrator end-to-end: register 15 models, generate in a loop, move all to `saved` |
| `test_image.py` | CRIU image cache hit/miss, then restore onto a different GPU |

They keep the `test_` prefix for historical reasons but contain no pytest test
functions — each is a `main()`. A bare `pytest` run from the package directory
would import them and collect nothing.

**The eviction repro is missing.** `test_eviction.py` was a byte-identical copy
of the generate script and was deleted rather than left as phantom coverage;
`pipeline_DESIGN.md` describes what it should assert.

### Observability

`dashboard.py` is the only live view, and it sits with the library code rather
than in a separate CLI directory. It consumes the orchestrator's `/state`
endpoint served by `state_server.py`.

A `tools/` directory previously held `monitor.py` (plotext scatter and
utilization charts) and `compare.py` (side-by-side curses comparison of two
recordings). Both were dropped: `plotext` was never declared as a dependency,
and `monitor.py` rebuilt one of its functions via `exec()` of patched library
source, which the security scanner flags as arbitrary code execution.

---

## 8. Known gotchas

- **The package cannot be imported by package path.** 24 flat sibling imports
  across 15 files; `__init__.py` raises `ModuleNotFoundError` on
  `semip_logging`. Its `__all__` also advertises `Slots` and
  `OrchestratorClient`, neither of which it imports.
- **Staging buffer frees.** Use `buf_gpu.storage().resize_(0)`, never
  `caching_allocator_delete(ptr)` — vLLM's cumem pluggable allocator intercepts
  `torch.empty(..., device="cuda")` under sleep mode.
- **`_env` reserved trio** (`CUDA_VISIBLE_DEVICES`,
  `VLLM_ENABLE_V1_MULTIPROCESSING`, `USE_LIBUV`) is hard-set at the top of the
  child loop and silently dropped from `_env`, but stays in the on-disk copy.
- **`criu_restore` does not re-apply `_env`** — the environment is captured inside
  the CRIU image and restored verbatim.
- **NVML, not torch,** for GPU memory queries in the main process, to avoid
  initializing CUDA there.
- **Doc drift.** `orchestrator_DESIGN.md`'s file inventory and its output
  sections describe `demo.py` / `demo.ipynb`, and `slots_DESIGN.md` positions
  `Slots` as complementing `gpu_pool.py` / `gpu_slot.py`. None of those four
  files exist in the tree.
- **`tests/conftest.py` is dead weight.** Its `pipeline_mode` fixture has no
  consumers and its `sys.path` bootstrap duplicates one both test files already
  perform; the suite passes with the file deleted.
