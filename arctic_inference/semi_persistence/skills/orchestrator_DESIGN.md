# Orchestrator -- Design

The orchestrator is a higher-level API on top of `Instance` primitives.
It maps human-readable model IDs to `Instance` objects and manages a
state machine for each model.  GPU resources are managed via a single
buddy allocator (`Slots`) that hands out fractional **slots**; each
model has an intrinsic *level* (derived from its
`gpu_memory_utilization`) and acquires a slot of that level when it
needs to run.

> **Architecture note (2026-05).**  The orchestrator's concurrency
> model has been migrated from an implicit future-chaining design
> (`ThreadPoolExecutor` + `_futures[mid]` + per-model
> `_last_generate_future` + `_inflight[mid]`) to an explicit
> per-model pipeline (`pipeline.ModelPipeline`, one worker thread + FIFO
> queue per `model_id`, with operations encoded as `Op` subclasses --
> `RegisterOp`, `MoveOp`, `EvictForPeerOp`, `GenerateOp`, `PauseOp`,
> `ResumeOp`, `RemoveOp`).  See `pipeline_DESIGN.md` for the new
> design and `pipeline.py` for the implementation.  Most of this
> document is unchanged because the public API is unchanged and the
> state machine, slot allocation, HBM accounting, and Known Issues
> all transferred verbatim.  The "Non-Blocking Design" section
> below has been rewritten; legacy method names (`_register_sync`,
> `_move_sync`, `_generate_sync`, `_pause_sync`, `_resume_sync`,
> `_remove_sync`, `_evict_for_phase2`, `_get_generate_lock`,
> `_await_prev`, `_drain_inflight_generates`, `_pool`, `_futures`,
> `_last_generate_future`) appear in historical notes throughout
> the rest of the doc as anchors for the bug-fix discussion -- the
> table below maps each one to its post-migration counterpart.

### Legacy -> pipeline translation table

| Legacy symbol                    | Post-migration counterpart                                |
|----------------------------------|-----------------------------------------------------------|
| `Orchestrator._pool`             | `ModelPipeline` worker threads + ad-hoc daemon threads    |
| `Orchestrator._futures[mid]`     | `ModelPipeline._q` (FIFO of `_QueueItem`)                 |
| `Orchestrator._last_generate_future[mid]` | `Orchestrator._generate_futures` (user-thread Phase-3 daemon-thread futures) |
| `Orchestrator._generate_locks[mid]` | `entry["_gen_lock"]` (RLock on the registry entry)     |
| `Orchestrator._get_generate_lock(mid)` | `entry["_gen_lock"]`                                  |
| `Orchestrator._await_prev(...)`  | (gone; pipeline FIFO orders ops naturally)                |
| `Orchestrator._drain_inflight_generates(...)` | (gone; `MoveOp` lands behind in-flight `GenerateOp`s in the FIFO; pipeline drain replaces the manual snapshot+wait) |
| `Orchestrator._evict_for_phase2(...)` | `EvictForPeerOp`, submitted via `acquirer_pipe.submit_to_peer_and_wait(incumbent_pipe, ...)` |
| `Orchestrator._register_sync`    | `RegisterOp.execute`                                      |
| `Orchestrator._move_sync`        | `MoveOp.execute`                                          |
| `Orchestrator._generate_sync`    | `GenerateOp.execute` (Phase 1+2) + a daemon-thread `_wait_and_collect` (Phase 3) |
| `Orchestrator._pause_sync`       | `PauseOp.execute` (submitted via `submit_front` after `pipe.interrupt_now()`) |
| `Orchestrator._resume_sync`      | `ResumeOp.execute`                                        |
| `Orchestrator._remove_sync`      | `RemoveOp.execute` + a daemon-thread teardown that joins the pipeline worker |
| The four 0.5 s polling loops     | `InterruptFlag.raise_if_set()` / `wait_or_interrupt()` cooperative yield-points inside `Op.execute` bodies |

## State Machine

Models live on a single ordered ladder.  `move()` walks up or down,
executing each step in between.

```
                         ┌───────────────────────────┐
                         │       register()          │
                         │   (cold-start new model)  │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
  ┌───────────┐  load   ┌────────────┐  alloc+restore  ┌───────────┐  alloc+wake_up  ┌─────────┐  generate  ┌─────────┐
  │   saved   │ ──────► │ checkpoint │  repin          │   sleep   │  h2d+scatter    │   up    │ ─────────► │ running │
  │           │         │            │ ──────────────► │           │ ──────────────► │         │            │         │
  │ image     │         │ image +    │                 │ CUDA on   │                 │ image + │            │ weights │
  │ on disk   │ ◄────── │ live proc  │ ◄─────────────  │ GPU,      │ ◄─────────────  │ process │ ◄───────── │ on GPU, │
  │           │ teardown│            │  unpin+ckpt     │ slotted   │  sleep_weights  │ + slot  │  generate  │ slotted │
  └─────┬─────┘ remove  └────────────┘  +dealloc       │ or none   │  +dealloc held  └─────────┘  completes └─────────┘
        │                                              └───────────┘                       │
        ▼                                                                                  │
  ┌──────────┐                              ┌─────────────────────────────────────────────┘
  │ remove() │                              │
  │ delete   │                              ▼ slot deallocated when generate finishes
  │ image +  │                       ┌──────────────┐
  │ registry │                       │  up slotless │ (eligible for eviction)
  └──────────┘                       └──────────────┘


  move() walks the ladder:

      UP:    saved ──────► checkpoint ──────► sleep ──────► up
      DOWN:  saved ◄────── checkpoint ◄────── sleep ◄────── up

  generate() is transient:  (auto-up) ──► running ──► up (slotless)
```

### Upward sequence

```
saved
  load(image_dir)                          # CRIU restore from disk -> live process
checkpoint
  Slots.allocate(level=L)                  # block on FIFO queue, coldest-first auto-pick
  inst.restore(slot.gpu).repin()           # CUDA context on GPU, small footprint
sleep
  (slot already held)                      # tier A/B/C only kicks in if slotless
  inst.wake_up_weights().h2d()
  .scatter().wake_up_kv_cache()            # weights on GPU, ready to serve
up
  inst.generate(prompts)                   # transient sub-state of up
running
  (generate completes -> Slots.deallocate)
up (slotless)                              # parks here; can be evicted to make room
```

### Downward sequence

```
up (running or slotted)
  inst.sleep().wait()                      # free GPU mem (weights + KV cache)
  Slots.deallocate(slot) (if held)         # post-sleep reconcile; releases the slot
                                           #   that running -> up couldn't free because
                                           #   _pending_count saw a queued sleep as in-flight
sleep
  inst.unpin().checkpoint().wait()
  Slots.deallocate(slot) (if held)         # CUDA context torn down, slot released
checkpoint
  inst.teardown().wait().remove()          # kill process, keep image on disk
saved
  remove()                                 # delete image + registry entry
(gone)
```

### State definitions

| State        | Image on disk | Live process | Slot held       | CUDA context | Weights on GPU |
| ------------ | ------------- | ------------ | --------------- | ------------ | -------------- |
| `saved`      | yes           | no           | no              | no           | no             |
| `checkpoint` | yes           | yes          | no              | no           | no             |
| `sleep`      | yes           | yes          | usually (\*)    | yes (small)  | no             |
| `up`         | yes           | yes          | usually (\*\*)  | yes          | yes            |
| `running`    | yes           | yes          | **always**      | yes          | yes            |

(\*) `sleep` is normally slotted (allocated at `checkpoint -> sleep`
and held until either `running -> up` or `sleep -> checkpoint`
releases it).  It is **slotless** only when the user explicitly pins
a target GPU via `move(model_id, "sleep", target_gpu=G)` (see
"Slotless-sleep flavour" below).

(\*\*) `up` is slotted while running and immediately after climbing
from sleep.  It becomes **slotless** after `running -> up` deallocates
the slot, allowing other models to acquire it.  Slotless `up` models
are eligible for eviction back to `sleep`.

`running` is a transient sub-state of `up` that exists only during a
`generate()` call.  It is **not** a valid target for `move()` — the
move-validator raises `ValueError` if requested.

`wait` is a second transient state, published only while the model
is enqueued in the `Slots` FIFO during a `checkpoint -> sleep`
transition (i.e. blocked inside `Slots.allocate`).  It is not part
of the state ladder and not a valid `move()` target — it is purely a
dashboard signal and is replaced by `sleep` as soon as the slot is
acquired.

## Slot Allocation (`Slots`)

`Slots` is a singleton buddy allocator (`slots.py`).  Each GPU starts
as one whole slot at level 1.  Splitting a level-L slot yields two
level-(L+1) buddies that each cover half the area.  A level-L slot
covers `1 / 2**(L-1)` of a GPU.

Public API:

| Method                                  | Semantics |
| --------------------------------------- | --------- |
| `Slots.init(gpu_ids)`                   | Initialise pools, one root slot per GPU. |
| `Slots.allocate(level, gpu=None)`       | **Blocking, FIFO.**  Picks coldest GPU when `gpu is None`. |
| `Slots.try_allocate(level, gpu=None)`   | **Non-blocking.**  Returns `None` if not satisfiable now.  Bypasses FIFO. |
| `Slots.deallocate(slot)`                | Release and coalesce buddies upward. |

Coldest-first auto-pick is implemented inside `_try_allocate`: it
sorts GPUs by `(last_used_time, gpu_id)` and returns the first one
with a satisfiable subtree.  FIFO is enforced by a single
`_waiters` deque guarded by a condition variable.

### Per-model intrinsic level

Each registry entry carries a `level` property derived once at
discovery / registration time:

```python
def _pick_level(util):
    """Smallest L with 1 / 2**(L-1) >= util."""
    if util <= 0.0 or util >= 1.0:
        return 1
    return max(1, ceil(log2(1.0 / util)))
```

Boundaries:

| `gpu_memory_utilization`  | level | slot fraction |
| ------------------------- | ----- | ------------- |
| `(0.5, 1.0]`              | L1    | 1.0  (whole)  |
| `(0.25, 0.5]`             | L2    | 0.5  (half)   |
| `(0.125, 0.25]`           | L3    | 0.25 (quarter)|
| `(0.0625, 0.125]`         | L4    | 0.125 (eighth)|
| ...                       | ...   | ...           |

The level is a static property of the model, computed once from its
`vllm_config.gpu_memory_utilization` and stored as `entry["level"]`.
All future allocations for the model use this level.

### Slot lifecycle on the ladder

| Transition                | Slot action                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `saved -> checkpoint`     | none                                                           |
| `checkpoint -> sleep`     | publish `wait` + `Slots.allocate(level)` (blocking, FIFO)      |
| `sleep -> up`             | tier A/B/C if slotless; else no-op (Phase 1) + evict slotless squatters until HBM fits (Phase 2) |
| `up -> running`           | tier A if slotless (else no-op); see "Slot acquisition at `up -> running`" |
| `running -> up`           | `Slots.deallocate(slot)`  → slotless `up` (primary release)    |
| `up -> sleep`             | `Slots.deallocate(slot)` if held (post-sleep reconcile; see "Slot release at `up -> sleep`") |
| `sleep -> checkpoint`     | `Slots.deallocate(slot)` if held                               |
| `checkpoint -> saved`     | none                                                           |

Allocation happens at exactly one point on the ladder
(`checkpoint -> sleep`); deallocation has two release points:
`running -> up` is the **primary** one (fires on every healthy
generate completion), and `up -> sleep` is a **secondary
reconcile** that closes a race where the primary release was
suppressed by a queued peer-eviction `sleep` (see below).  The
asymmetry between the two release points lets one slotless model
wait at `up` for traffic without holding a slot, while another
sleeper takes that slot.

### Slot acquisition at `up -> running`

`running` always holds a slot.  A slotless `up` model (left there by
a previous `running -> up`) must reacquire a slot before publishing
`running` on the next generate:

1. **Tier A** -- `Slots.try_allocate(level, gpu=home_gpu)`.  If the
   home GPU is free, claim the slot in place.  Weights stay in HBM,
   no movement.  Common case for the second generate on a model
   that nobody has displaced.
2. **Fallback (migration / FIFO)** -- if Tier A fails, the model
   retreats to `sleep` (`_step_down(up, sleep)` frees HBM) and
   climbs back up via `_step_up(sleep, up)`, which uses the full
   tier-A/B/C path below (including Phase-2 eviction and the
   blocking FIFO `Slots.allocate`).

After this, the published state transitions `up -> running` with the
slot held.

### Slot release at `up -> sleep`

`_step_down(up, sleep)` is called from three places:

1. **`_acquire_slot_for_running` retreat** -- a slotless `up`
   model couldn't claim a slot on its home GPU and falls back
   through `sleep` to climb the full Tier A/B/C path.  Enters
   with `slot is None`.
2. **Phase-2 eviction** -- another model's `_step_up` validated
   this model as slotless+`up` on the same GPU and is freeing
   our HBM.  Enters with `slot is None`.
3. **`_move_sync` ladder walk** -- explicit `move(target="sleep")`
   from `up`.  Enters with whatever slot the registry holds.

Every caller's `_send_cmd_with_ack("sleep")` parks on the
worker's ack, and during that park the model's *own* thread can
race all the way back up to `running` with a fresh slot (e.g. a
new generate landed, walked the ladder, and the worker queued
the late `sleep` behind the new `generate`).  The worker drains
the `generate` first, then runs the `sleep` -- so by the time
the ack lands, the engine is asleep but the registry reads
`running` with the slot still held.

`_step_down(up, sleep)` resolves this by **reconciling the
registry to the post-sleep worker** instead of trusting the
caller's pre-sleep snapshot: after `_send_cmd_with_ack("sleep")`
returns, it unconditionally deallocates any slot the registry
holds and flips state to `sleep` (under `model_lock`).  For
callers (1) and (2) the slot is already `None`, so this is a
no-op `Slots.deallocate` and a same-`sleep`/`sleep` `_set_state`
that doesn't log.  For caller (3) and for the racing-back-to-
running case it is the **only** code path that frees the
orphaned slot -- `_on_generate_done`'s `_pending_count == 0`
guard saw the queued `sleep` as still in flight and declined to
release.

This is the second release point referenced in the slot
lifecycle table.  Together with `running -> up`, it ensures
`Slots.deallocate` matches every `Slots.allocate` even under
the cross-thread cmd-queue interleavings the worker permits.

### Tier A/B/C: slotless `sleep -> up`

If a sleeper has lost its slot (e.g. via the slotless-sleep flavour),
`_step_up` reacquires opportunistically before falling back to FIFO:

1. **Tier A** -- `Slots.try_allocate(level, gpu=home_gpu)`.  Fast
   path: same GPU, no migration.
2. **Tier B** -- `Slots.try_allocate(level)`.  Any free GPU.  If the
   returned slot is on a different GPU, migrate via
   `inst.unpin().checkpoint()` then `inst.restore(new_gpu).repin()`.
3. **Tier C** -- nothing free.  Retreat: `_step_down(sleep,
   checkpoint)` (no-op for slot, since slotless), then
   `_step_up(checkpoint, sleep)` which calls the blocking
   `Slots.allocate(level)` and respects FIFO.

### Phase 2: HBM eviction

After the slot is held on `home_gpu`, `_step_up` frees just enough
HBM for the new model's weights by evicting slotless `up`
incumbents on the same GPU, oldest first by `state_since`.  The
amount evicted is computed from the GPU's HBM budget, not a fixed
"evict one" or "evict all" rule:

- Each model's HBM footprint is its level share, `1 / 2**(L-1)`.
- `slotted_others` is the sum of slot shares on `home_gpu` for
  every *other* slotted model, **regardless of state**.  We use
  the slot share rather than live `up`/`running` state because slot
  allocation is what serialises concurrent wake-ups: a slotted
  `sleep` model whose Phase 3 is already in flight on the worker
  will be HBM-resident by the time *our* Phase 3 commands run,
  even though its registry state still reads `sleep`.  Counting
  every slot is the only race-free upper bound; `slotted_others`
  may slightly over-account when a slotted sleeper is genuinely
  not waking, but that just causes a small over-eviction, never
  an OOM.  "On `home_gpu`" is keyed off `slot.gpu_id` (the buddy
  allocator's source of truth, mirrored by `Slots._live`), not the
  registry's `entry["gpu"]`: during a peer's Tier B migration the
  slot flips to its new GPU before `entry["gpu"]` does, and only
  the slot reflects where the share is actually owed.
- `new_share` is the slot share we just allocated for this model.
  After Phase 3 those weights will be resident, so they consume
  `new_share` of HBM.
- `slack = 1.0 - slotted_others - new_share` is the HBM the buddy
  allocator has *not* handed out — exactly the budget available
  for slotless squatters.  Because the buddy allocator keeps the
  sum of all slot shares on a GPU at `<= 1.0`, `slack` is always
  `>= 0`.

`_step_up` then sorts slotless `up` incumbents on `home_gpu` by
`state_since` (oldest first) and evicts them one by one back to
`sleep` until the remaining slotless share fits inside `slack`.
This evicts the minimum number of incumbents needed and prefers
the coldest victims.  Phase 3
(`wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()`)
has no try/except, so under-eviction would OOM and crash the
orchestrator; the formula above is the precise condition for
Phase 3 to fit within the GPU under any concurrent wake-up
pattern the buddy allocator allows.

Each eviction first calls
`Orchestrator._drain_inflight_generates(incumbent, "phase-2
evict")` to wait for the incumbent's in-flight generates to
drain on the worker pipe before issuing
`_step_down(incumbent, "up", "sleep")`.  Without that drain,
the eviction's `sleep` cmd could land between two generate
cmds on the worker pipe (a follow-up `inst.generate(...)`
that races with the eviction sees `state="running"` under
`gen_lock` and Phase 2 enqueues directly), letting the worker
drain `generate -> sleep -> generate` and drop the trailing
generate onto a dormant engine -- which the `_saved_requests`
deferral path absorbs without an ack and wedges the worker
on the next sync cmd's `_drain_pipe_generates`.  See
"Eviction-mid-generate dormant-engine wedge" in Known Issues
for the full root-cause analysis and the model-16 demo-run
trace.

Worked example: an L2 model (`new_share=0.5`) wakes up on a GPU
hosting two slotless L3 squatters (each `0.25`), one slotless L2
squatter (`0.5`), and no other slotted models (`slotted_others=0`).
Then `slack=0.5` and the slotless total is `1.0`.  If the two L3
squatters are oldest, evicting both frees `0.5`, leaving the
slotless L2 in place — exactly enough to fit.

Concurrent-wake example: two L2 models `A` and `B` are both being
woken on the same GPU.  After Phase 1 the GPU's slots are
`A`(0.5) + `B`(0.5) = 1.0.  When `B`'s Phase 2 runs, `A` may
still be in registry state `sleep` (its Phase 3 commands are
queued but `_set_state(up)` hasn't fired yet).  Counting only
"up/running slotted" would yield `slack=0.5` and skip evicting a
slotless squatter `C`, then `A` and `C` and `B` would all try to
fit (1.5 total → OOM).  Counting every slot gives
`slotted_others=0.5`, `slack=0`, forcing `C` to be evicted.

### Slotless-sleep flavour

`move(model_id, "sleep", target_gpu=G)` parks a model on GPU `G`
**without allocating a slot**:

- `_step_up`'s `checkpoint -> sleep` skips `Slots.allocate` when
  `target_gpu` is given, and just calls `inst.restore(G).repin()`.
- A tail in `_move_sync` deallocates any slot the model still holds
  after the ladder walk (e.g. coming from `up` on the same GPU, or
  already sleeping but slotted).

The model competes for HBM but not slots.  When woken later via
`generate()` or `move("up")`, the standard tier-A/B/C logic
re-acquires a slot.

### What `register()` allocates

Cold-start always uses a full GPU regardless of the model's eventual
level: `Slots.allocate(level=1)`.  This `register_slot` is local to
`_register_sync` and is released inline once the image is saved.
The model's intrinsic `level` (used in subsequent
`checkpoint -> sleep` transitions) is computed from its
`gpu_memory_utilization` at the same time.

## API

```python
Orchestrator.init(image_cache="/data-fast/image-cache")

Orchestrator.register("qwen3-8b",
                      {"model": "Qwen/Qwen3-8B",
                       "gpu_memory_utilization": 0.4})  # -> intrinsic level L2
Orchestrator.wait("qwen3-8b")

# Per-model env vars: vllm_config["_env"] is applied to os.environ
# in the vLLM child before `from vllm import LLM`, so flags vLLM
# reads at import time take effect.
Orchestrator.register("qwen3-8b-deepgemm",
                      {"model": "Qwen/Qwen3-8B",
                       "gpu_memory_utilization": 0.4,
                       "_env": {"VLLM_USE_DEEP_GEMM": "1"}})
Orchestrator.wait("qwen3-8b-deepgemm")

# generate auto-transitions to 'up' if needed; leaves model in 'up'
# (slotless) after running.
fut = Orchestrator.generate("qwen3-8b", "Hello, how are you?")
results = fut.result()

# explicit state control
Orchestrator.move("qwen3-8b", "checkpoint")
Orchestrator.wait("qwen3-8b")

# slotless-sleep flavour: park on a specific GPU without using a slot
Orchestrator.move("qwen3-8b", "sleep", target_gpu=5)
Orchestrator.wait("qwen3-8b")

# remove auto-transitions to 'saved' if needed
Orchestrator.remove("qwen3-8b")
Orchestrator.wait("qwen3-8b")

# fan-out across every registered model is the caller's job:
for mid in Orchestrator.models():
    Orchestrator.wait(mid)

# grow / shrink the GPU pool at runtime
Orchestrator.add(7)            # synchronous; pool is now {0..7}
Orchestrator.sub(3)            # non-blocking drain of GPU 3
Orchestrator.wait_gpu(3)       # block until drain completes

Orchestrator.status()
```

## `init(image_cache, gpus=None)`

- Discovers GPUs via NVML (`pynvml`) without initializing CUDA in the
  main process.
- Resets `Slots` if it was already initialised (test re-entry safety),
  then calls `Slots.init(gpu_ids)`.
- Scans `image_cache` for subdirectories containing `meta.json`.
  Each entry is registered in `saved` state and tagged with its
  intrinsic `level` (computed from
  `meta.json -> vllm_config.gpu_memory_utilization`).
- Initializes the `ThreadPoolExecutor`.

## `register(model_id, vllm_config)`

Non-blocking.  Submits `_register_sync` to the thread pool.

1. **Acquire a full GPU** via `Slots.allocate(level=1)` (blocks on
   FIFO).
2. **Cold-start sequence** --
   `Instance(vllm_config).init(gpu).attach().repin().stage().unpin().sleep().checkpoint().wait()`.
3. **Save image** -- `inst.save(image_dir).wait()` writes the CRIU
   image to disk.  The dump is destructive — the child is killed.
4. **Release slot** -- inline `Slots.deallocate(register_slot)`.
5. **Store in registry** with intrinsic `level` derived from
   `gpu_memory_utilization`; state set to `saved` (image on disk,
   no live process, no slot).

With N GPUs, up to N models cold-start in parallel.

### Reserved keys in `vllm_config`

`vllm_config` is otherwise passed through verbatim to `LLM(**cfg)` in
the child.  The orchestrator reserves a single key inside it:

* **`_env`** -- optional `dict[str, str]` mapping env-var names to
  values.  Popped from a local copy of `vllm_config` in the vLLM
  child (so the on-registry / on-disk copies retain it) and applied
  to `os.environ` *before* `from vllm import LLM`, so flags vLLM
  reads at module import time (e.g. `VLLM_USE_DEEP_GEMM`,
  `VLLM_ATTENTION_BACKEND`) take effect.  The trio
  `CUDA_VISIBLE_DEVICES` / `VLLM_ENABLE_V1_MULTIPROCESSING` /
  `USE_LIBUV` is hard-set at the top of the child loop and silently
  dropped if present in `_env`.

Because `_env` is part of `vllm_config`, it participates in the
client-side dedup check (`OrchestratorClient.register` compares the
dict against existing models on the orchestrator): two registrations
of the same model with different `_env` correctly resolve to two
distinct backing models.

`_env` is persisted into `meta.json` alongside the rest of
`vllm_config` (via `criu_dump`'s `meta_extra`), so it survives orch
reboots: on the next `Orchestrator.init`, the saved model is
rediscovered with `_env` intact and dedup remains honest.

CRIU restore captures the child's `os.environ` directly into the
image, so `criu_restore` paths inherit the dump-time env without
re-applying `_env` from `meta.json`.  `_env` is therefore consulted
only on cold-start paths (`register` itself).

## `move(model_id, target, target_gpu=None)`

Walks the state ladder from the current state to the target.  Valid
targets: `"saved"`, `"checkpoint"`, `"sleep"`, `"up"`.  Submits a
`MoveOp` onto the model's pipeline; also reused internally by
`GenerateOp`, `ResumeOp`, and `RemoveOp` to walk the ladder.
Non-blocking.

- **Going up / down**: walks the ladder one step at a time via
  `_step_up` / `_step_down`.
- **Already at target**: no-op (unless `target_gpu` is given, see
  "Slotless-sleep flavour" above).
- **Model in `running` state**: raises `RuntimeError` (wait for
  generate to finish first).
- **`target == "running"`**: rejected with `ValueError` (running is
  only reachable via `generate()`).
- **`target_gpu` set with `target != "sleep"`**: rejected.
- **Model not registered**: prints a warning and returns (no error).

### Step details

| Transition                | Instance primitives + slot action                                                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `saved -> checkpoint`     | `Instance(config).load(image_dir).wait()`                                                                |
| `checkpoint -> sleep`     | (alloc) `Slots.allocate(level)` + `inst.restore(slot.gpu).repin().wait()`                                |
| `sleep -> up`             | Phase 1 tier A/B/C (alloc if needed, possibly migrate); Phase 2 evict oldest slotless incumbents on same GPU until HBM fits (`slack = 1 - active_others - new_share`); Phase 3 `inst.wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()` |
| `up -> sleep`             | `inst.sleep().wait()` + `Slots.deallocate(slot)` if held (post-sleep reconcile; see "Slot release at `up -> sleep`")  |
| `sleep -> checkpoint`     | `inst.unpin().checkpoint().wait()` + `Slots.deallocate(slot)` if held; clear `entry["gpu"]`              |
| `checkpoint -> saved`     | `inst.teardown().wait().remove()`                                                                        |
| `up -> running`           | (occurs inside `_move_sync` announce branch) Tier A `Slots.try_allocate`, else fallback through `sleep -> up` |
| `running -> up`           | (occurs inside the demuxer's generate listener, not `_step_down`) `Slots.deallocate(slot)` after the last inflight generate's ack and `_pending_count == 0` |

The `move(.., "sleep", target_gpu=G)` flavour is implemented as a
combination of (a) skipping `Slots.allocate` in `checkpoint -> sleep`
when `target_gpu` is set and (b) a tail in `_move_sync` that
deallocates any leftover slot.

## `generate(model_id, prompts, sampling_params=None)`

Non-blocking.  Submits a `GenerateOp` onto the model's pipeline and
returns a `Future[list]`.  Can be called from **any** state.

1. **Set t0 (first generate only)** -- anchors `t0` for relative
   request timestamps (`submit_rel_s`, etc.).
2. **Auto-transition to `up`** -- `GenerateOp.execute` Phase 1 calls
   `_step_up` with `announce_state="running"` (skipping the
   observable `up` window) so the ladder walk and the state flip
   are atomic.
3. **State becomes `running`** -- atomic with the final `_step_up`.
4. **Submit to engine** -- Phase 2 takes `entry["_gen_lock"]`,
   sends `inst.generate(...)`, appends `(rid, q_rec, done_event)`
   to `_inflight[mid]`, and returns a `PendingRequest`.  The
   pipeline worker is then free to dequeue the next op.
5. **Wait for result on a daemon thread** -- `submit_generate`
   spawns `generate-wait[mid:N]`, which calls `done_event.wait()`
   and resolves the user-visible `Future[list]` with the engine's
   outputs.  Phase 3 therefore does **not** block the pipeline
   worker.  The user-future is parked in
   `Orchestrator._generate_futures` so `wait()` can join it.
6. **Generate listener** (registered once per Instance on the
   per-instance demuxer; see [`instance_DESIGN.md`](instance_DESIGN.md)
   "Demuxer architecture").  Fires for every `cmd="generate"` ack.
   Resolves the matching inflight `done_event`, copies token counts
   onto the request record, and -- when no more generates are
   inflight and `inst._pending_count == 0` -- releases the slot and
   flips state `running -> up`.

The model is left in **slotless `up`** after generate.  Other models
that need a slot can acquire it (and may evict this one back to
`sleep` via Phase 2 of `_step_up`, queued as an `EvictForPeerOp` on
the incumbent's pipeline).

## `pause(model_id)` / `resume(model_id)`

Two control verbs that operate on actively-generating models.  Both
are non-blocking.

- `pause` does **not** queue a regular FIFO successor.  Instead it
  trips the pipeline's `InterruptFlag` synchronously
  (`pipe.interrupt_now("pause")`) and submits `PauseOp` at the
  **head** of the FIFO (`submit_front`).  Any op currently on the
  worker (a long `MoveOp` ladder walk, `GenerateOp` Phase 1, etc.)
  observes the flag at its next cooperative yield-point and bails
  with `Interrupted`; `PauseOp` then runs immediately.  This
  preserves the legacy "pause must reach the engine even when a
  long-running op is in flight" semantics without needing to opt
  out of any future chain.
- `resume` is a successor: it queues `ResumeOp` at the tail of the
  FIFO and runs after any pending `PauseOp` / `MoveOp` for this
  model.

`PauseOp.execute` clears the `InterruptFlag` at the end of its run,
so subsequent ops on the worker see a clean flag.

Each is a thin wrapper over the corresponding `Instance` primitive
(`Instance.pause`/`resume`); the heavy lifting -- snapshot,
re-prefill -- happens in the vLLM child.  See
[`instance_DESIGN.md`](instance_DESIGN.md) "Pause and Resume" for
the full child-side semantics.

### `entry["paused"]` flag

A single boolean per registry entry.  It is **in-memory only** -- not
persisted in `meta.json`.  The child-side `_paused` and
`_saved_requests` ride CRIU + `cuda-checkpoint` for free across
`up <-> sleep <-> checkpoint` (they live as plain Python state inside
the worker_loop closure; see `vllm_child.py`).  The orchestrator-side
flag transitions only at `pause` (False -> True) and `resume`
(True -> False); paused models cannot reach `saved` (see "Walking
down past `up` while paused" below).

### `pause(model_id)`

Pre-condition: `state == "running"` and not already paused; otherwise
no-op.  Effect:

1. Send `inst.pause()`.  The child snapshots every active
   sub-request's `(prompt_token_ids, output_token_ids_so_far,
   sampling_params, t0, first_token_ts)` into `_saved_requests`,
   aborts them in the engine, and sets `_paused = True`.  Pending
   `generate_done` messages are deferred.
2. `Slots.deallocate(entry["slot"])`, `entry["slot"] = None`.
3. `_set_state(model_id, "up")` (slotless), `entry["paused"] = True`.

Effective ladder action: `running -> up` (slotless).  A paused
instance is therefore **never `running`** -- it lives at slotless `up`
or wherever the user moves it on the ladder.

### `resume(model_id)`

Pre-condition: `entry["paused"]` and `state in ("up", "sleep",
"checkpoint")`; otherwise no-op (logs and returns).  `running`
and `saved` cannot legally coexist with `paused=True` -- pause
sets state=up, and `move(saved)` is refused while paused -- so
reaching the no-op branch for those means an invariant
violation; logged as a warning.

`resume` is **generate-shaped**: a no-op when there is nothing
to drive, otherwise reuses `_generate_sync` Phase 1's exact
entry point so it inherits the same self-heal against the
slot-stealing race (see "Resume slot-steal wedge" in Known
Issues).  Effect:

1. Snapshot `_inflight[mid]` under `gen_lock`.  If empty, clear
   `entry["paused"] = False` under `model_lock` and leave the
   model wherever the user parked it -- no walk-up, no slot
   acquisition, no worker `resume` cmd.  Self-heal for any
   stray phantom-paused entry that bypassed the `_pause_sync`
   guard (in healthy code this branch is unreachable).
2. Otherwise, call `_move_sync(model_id, "up",
   announce_state="running")` -- direct call, not the public
   `move()`, so the inner walk doesn't capture
   `prev_gen_events` (the model is paused so they'd never
   fire) and doesn't publish a separate `_futures[mid]`
   future.  This single call covers both the ladder walk
   (when state is `sleep`/`checkpoint`) and the slot
   acquisition + state flip (when state is `up`), all via the
   same code path that `generate` uses.  Critically, it
   inherits `_move_sync`'s post-acquire state re-check: a
   peer's concurrent Phase-2 eviction that steals the
   just-claimed Tier-A slot is detected via `state="sleep"`
   and falls through to `_step_up(sleep, up,
   announce_state="running")` for a heavyweight re-acquire.
   `paused=True` rides through the walk untouched (nothing in
   `_step_up`/`_step_down` mutates `paused`), so saved subreqs
   and queued-during-pause requests stay in `_inflight`
   exactly as `pause` left them.  Typical pattern reaching the
   ladder-walk arm of this branch: `cl.pause();
   cl.move("checkpoint"); cl.resume()`.
3. Send `inst.resume()`.  The child re-adds each saved
   sub-request via `TokensPrompt(prompt_tids + output_tids)`
   with `max_tokens` reduced by pre-pause output length, then
   clears `_paused`.  State is already `running` from step 2.
4. Under `model_lock`: clear `entry["paused"] = False`.  The
   brief window between step 2's state flip and this clear
   (bounded by the `resume` cmd round-trip) is benign:
   `_step_up` Phase 2 eviction targets only `state=="up"`
   slotless squatters (we are `running` with a slot), and
   `pause`/`resume`/`move` all serialise via `_futures[mid]`.

The original per-request futures resolve as the engine finishes the
re-prefilled requests; no new bookkeeping is needed because the
demuxer's generate listener stays installed for the life of the
Instance and `_inflight` remained non-empty across the pause.

**Stochastic-decode caveat.**  Re-prefill at `resume` is bit-exact only
for greedy decoding (`temperature == 0`).  For stochastic sampling the
post-resume token trajectory will diverge from a hypothetical
un-paused run because the per-request RNG state is not captured by
`pause`.  See [`instance_DESIGN.md`](instance_DESIGN.md) "Design
choice: re-prefill, not KV-block snapshot" for the full discussion.

### Generate-while-paused

`Orchestrator.generate(model_id, ...)` while `entry["paused"]` is
True skips `_move_sync` and falls straight through to Phase 2's
batch-drain loop, identically regardless of where on the ladder
the (paused) model sits.  On the child side, `_submit_generate`
applies a single rule: while `_paused`, the vLLM engine sees no
scheduler mutations.  The request is appended to
`_saved_requests` as a synthesised never-stepped record (the
original prompt + sampling params, with empty per-eid state) and
`engine.add_request` is **not** called.  The next `resume()`
re-prefills the deferred entries from `_saved_requests` along
with whatever `pause` itself snapshotted at pause-time, via the
same code path -- both flavours converge on
`engine.add_request(...)` once the engine is back at `up`.

Allowed at every ladder state at which pause itself is allowed
(`up`, `sleep`, `checkpoint` -- see "Walking down past `up` while
paused"); the policy is order-independent w.r.t. any
interleaving of `pause`, `sleep`, `cuda_checkpoint`, and
`generate` on the pipe, because none of those interleavings can
expose the engine to a scheduler mutation while paused.  See
"Generate-while-paused stash" in
[`async_generate_DETAILS.md`](async_generate_DETAILS.md) for the
child-side mechanism and the invariants it preserves.

The per-request future blocks on `done_event` exactly like any
other generate; the orchestrator-side `_request_log`,
`_inflight`, and piggyback classification are independent of
which child-side queue the request transits through.

### Op ordering (formerly "future chaining")

Under the explicit pipeline, ordering between operations is the FIFO
property of `ModelPipeline._q`: every op submitted to the pipeline
runs strictly after every op submitted earlier on the same model,
and the pipeline worker only ever runs one op at a time.  The
legacy `_futures[mid]` / `_last_generate_future[mid]` / manual
`prev_gen_future` snapshotting machinery is gone; the same
guarantees fall out of the pipeline FIFO + the worker-thread-per-
model invariant for free.

Concretely, with the FIFO doing the heavy lifting:

- `move()` queues `MoveOp` behind any in-flight `GenerateOp`.  The
  in-flight `GenerateOp.execute` returned its `PendingRequest` after
  Phase 2 (so the worker is free to dequeue `MoveOp`); Phase 3 runs
  on a daemon thread and is therefore *not* what `MoveOp` waits on
  -- but `MoveOp` does not need to wait, because the demuxer's
  trailing-edge slot release + state flip (`running -> up`) is what
  makes the model evictable, and `MoveOp`'s ladder walk handles
  whatever published state the model is in by the time it dequeues.
  This is the post-migration replacement for the legacy
  `prev_gen_future` snapshot under `gen_lock`.
- `pause` is the one op that **must not** wait for the in-flight
  generate -- a paused-then-resumed generate can take arbitrarily
  long, and the user's intent is "interrupt now".  `pause` jumps
  the FIFO via `submit_front(PauseOp)` AND trips the pipeline's
  `InterruptFlag` synchronously, so any op currently on the worker
  bails at its next yield-point.  `PauseOp.execute` clears the flag
  at the end of its run.
- `resume` is a successor: it queues `ResumeOp` at the tail and
  runs after any pending `PauseOp` / `MoveOp` for the same model.
- `move(target='saved')` and `remove()` are still rejected with a
  warning when `entry["paused"]` is True -- tearing the live process
  down would orphan every per-request `done_event.wait()` parked on
  a daemon thread spawned from `submit_generate`.  The caller must
  `resume()` (and let the in-flight generates drain) first.
- Cross-pipeline ops (Phase-2 eviction) use
  `acquirer_pipe.submit_to_peer_and_wait(incumbent_pipe,
  EvictForPeerOp(...))`, which records the wait edge in
  `_waiting_on` so the cycle detector can refuse a deadlock-shaped
  pair of cross-submits at runtime.

The Known Issues entries "Move-vs-generate Phase-1 sleep race",
"Pause-during-drain ineffective", and "Pause-during-resume future-
chain break" all *remain fixed*: their underlying invariants
(generate must finish Phase 1 before `move` can start a sleep cmd;
pause must reach the worker promptly even mid-resume; pause must
not poison the future chain when it is a no-op) are preserved by
the FIFO + `submit_front` + `InterruptFlag` design, and each
ported `Op.execute` body documents the bug-fix they preserve in
their docstring porting contract.

### Walking down past `up` while paused

Paused models can be walked freely between `up`, `sleep`, and
`checkpoint`: the child-side `_paused` / `_saved_requests` ride CRIU
and `cuda-checkpoint` for free.  But moving all the way to `saved`
would tear the live process down and orphan every per-request
`Future` parked at `done_event.wait()`.  Therefore:

- `move(model_id, "saved")` and `remove(model_id)` are **rejected**
  (logged warning, no-op) when `entry["paused"]` is True.  The
  caller must `resume` and let the in-flight generates drain first.
- `move(model_id, "sleep" | "checkpoint" | "up")` is allowed; the
  paused state survives the ladder walk.

### Result-queue ownership

`_pause_sync` / `_resume_sync` (and every `_step_up` / `_step_down`
cmd) go through `Orchestrator._send_cmd_with_ack` instead of
`inst.wait()` because a paused model's `_pending_count` never drops
to 0 until resume (its generate cmd is deferred in the child).
`_send_cmd_with_ack` installs a fresh `threading.Event` at the tail
of a per-`(model_id, cmd)` FIFO and the demuxer's catch-all listener
pops the head of that FIFO when the cmd's ack lands -- so multiple
concurrent senders for the same cmd (e.g. peer-eviction sleep
racing with a self-evacuation sleep) all get woken in send order.
See [`async_generate_DETAILS.md`](async_generate_DETAILS.md)
"`inst.wait()` vs `_send_cmd_with_ack`" for the full mechanism,
the FIFO ack `Event` protocol, and
"Lock ordering rules" for the consolidated lock inventory and
acquisition order (`gen_lock` -> `model_lock`, with
`_cmd_ack_lock` / `_request_lock` / `Slots._cv` as leaves) that
every `_step_*` and demuxer listener obeys.

### Piggyback classification (dashboard)

The dashboard tags a generate request as a *follower* of an earlier
generate on the same model and renders it as `[wait #L+1]` where
`L` is the lead's request number.  Two flavours exist; both are
permanent (the label survives the lead's completion / model resume /
ladder unwind) because the classifier on `state_server` consults only
immutable timestamps and the immutable `paused_at_submit` flag on the
request record.

The two flavours share one template -- "R piggybacks on L iff L was
submitted earlier on the same model AND L was alive at `R.t_submit`
AND a flavour-specific fact `F` held at `R.t_submit`":

- **Climb-piggyback**: `F` = "L was still climbing the state ladder",
  i.e. `L.t_generate is None or L.t_generate > R.t_submit`.  R landed
  while L's wake-up Phase 1 was still in flight and gets pulled into
  the same batch when the model finally reaches `running`.  This
  flavour predates the async-generate work.
- **Pause-piggyback**: `F` = "R itself was submitted while the model
  was paused", i.e. `R.paused_at_submit is True`.  R is parked on the
  child's request queue behind L (gated by `engine.step()` not running
  under `_paused`) and will be drained alongside L when the user
  resumes.  `paused_at_submit` is set in `submit_generate` from
  `entry["paused"]` and never mutated.

"Alive at `R.t_submit`" means L had not yet completed by then
(`L.t_done is None or L.t_done > R.t_submit`).  Climbing implies
alive, so the climb branch is logically equivalent to its pre-pause
form; stating the alive gate explicitly just makes the parallel
between the two branches visible in `state_server.snapshot_state`.

When multiple leads qualify, R follows the earliest-submitted lead --
the same idiom both flavours used before they were unified.

## `remove(model_id)`

Non-blocking.  `RemoveOp` (queued to the model's pipeline) walks the
model down to `saved` state via an inner `MoveOp`, then deletes the
image directory from disk and removes the model from the registry.
All slot bookkeeping is handled by the ladder walk (specifically,
`sleep -> checkpoint` deallocates).  After `RemoveOp` resolves, a
small daemon thread (`remove-teardown[mid]`) drains and joins the
pipeline worker and pops the model from `Orchestrator._pipelines`,
so a subsequent `register(mid)` for the same id starts fresh.

Per-model only — there is no built-in "remove all" helper.  Fan-out is
the caller's responsibility (use `Orchestrator.models()` to enumerate).

## `wait(model_id)`

Blocks the main thread until everything pending for *model_id*
quiesces:

- `Orchestrator._pipelines[mid].drain()` waits for every queued
  `Op` (including any `MoveOp`, `PauseOp`, `ResumeOp`, `RemoveOp`,
  and the Phase 1+2 of any pending `GenerateOp`s) to finish.
- Each `GenerateOp`'s Phase 3 user-future
  (`Orchestrator._generate_futures`) is then awaited so the call
  also blocks until the engine has actually finished generating
  tokens for those requests.

`wait()` (no argument) drains every registered model's pipeline and
every entry in `_generate_futures`.  Per-model only — for an
unambiguous multi-model barrier scoped to specific ids, fan out via
`Orchestrator.models()`.

## `models()`

Returns the list of currently registered model_ids.  Provided so
callers can write the per-model fan-out loop themselves; the
orchestrator deliberately offers no `*_all` helpers (those concerns
live one layer up, in `client.OrchestratorClient`).

## `status()`

Prints orchestrator config, GPU memory usage, and a compact one-line
summary per model sorted by pinned memory (largest first).  Each line
shows state, GPU assignment, pinned memory (for non-saved), actual
GPU memory usage via NVML per-PID lookup (for up), and image path
(for saved).  A lightweight `_print_states()` one-liner is also
printed automatically after every state change.

## `add(gpu)` / `sub(gpu)` / `wait_gpu(gpu)` -- runtime pool resize

The GPU pool established by `init()` can be grown or shrunk while the
system is live without restarting the orchestrator or any vLLM child.

### `add(gpu)`

Synchronous, fast (pure bookkeeping).  Validates that `gpu` is visible
to NVML (`nvmlDeviceGetCount`); if not, raises `ValueError` rather than
admitting a non-existent device — the slot allocator would eventually
hand it out and `cuda_restore(gpu=N)` would crash inside
`_build_restore_args` on the out-of-range UUID lookup, deadlocking the
dependent `repin` already pipelined into the vLLM child.

On success: appends to `Orchestrator._gpu_ids`, calls `Slots.add(gpu)`
(seeds a single whole-GPU root slot and clears any stale `_draining`
flag), and notifies FIFO waiters that may now be satisfiable.

Idempotent: re-adding a GPU already in the pool logs a warning and
returns.

### `sub(gpu)`

Non-blocking, two-phase drain.

**Phase 1 (synchronous, under `Slots._cv`)**: flag `gpu` in
`Slots._draining`.  This immediately stops `_try_allocate` from handing
out slots on `gpu` (both the explicit-`gpu` and auto-pick branches skip
draining GPUs).  In-flight FIFO waiters get re-evaluated against the
remaining non-draining GPUs.  `move(.., target_gpu=gpu)` is also
rejected with `ValueError` for the duration of the drain.

**Phase 2 (background, on `_pool` via `_sub_sync`)**: snapshot every
model whose registry entry has `gpu == G` *or* `slot.gpu_id == G` (the
double criterion catches a model whose `_step_up` wrote one of the two
fields but not yet the other); for each resident submit
`move(mid, "checkpoint")` and await the move future.  Repeat until no
residents remain *and* no orphan slot is in `Slots._live` for `gpu`,
then call `Slots.pop(gpu)` and remove `gpu` from
`Orchestrator._gpu_ids`.

The orphan-slot branch is the only place this loop polls: a 50 ms
`_cv.wait` covers the microsecond window between `Slots.allocate`
returning and `entry["slot"] = slot` being written under the model's
own lock.  Otherwise the drain is fully event-driven.

`Slots.pop` is called *outside* `_cv` because it re-acquires the
condition internally (the backing lock is non-reentrant).  This is
safe because `_draining` is still set, so no new slot can land on
`gpu` between the cv-held emptiness check and the `pop`.

The drain `Future` is stored in `Orchestrator._gpu_futures[gpu]` and
is awaitable via `wait_gpu`.

### `wait_gpu(gpu)`

Blocks until the pending `sub(gpu)` drain completes.  No-op if no
drain is in flight.  Does **not** participate in `wait(model_id)` —
job futures and GPU futures are separate tracking dicts (`_futures`
and `_last_generate_future` are per-model; `_gpu_futures` is per-GPU).

### Client-side guard

`OrchestratorClient.sub` does one extra check before posting `/sub`:
it refuses to drain the *last* non-draining GPU.  If `gpu_ids -
draining - {gpu}` is empty, residents would have nowhere to migrate
and the drain would hang forever, so we raise `ValueError` on the
client.  The server itself does not enforce this — it's a usability
guard, not an invariant.

## Non-Blocking Design -- Why Threading

The orchestrator uses Python threading -- specifically, an explicit
per-model **pipeline** built from `threading.Thread` workers and
`queue.Queue` FIFOs (see `pipeline.py`, `pipeline_DESIGN.md`).  It
does not use `asyncio`.

- The actual work runs in **forked worker processes** (see
  `instance_DESIGN.md`).  The pipeline worker thread just enqueues
  Instance primitive commands and blocks on `instance.wait()` /
  `_send_cmd_with_ack`.  There is no async I/O to benefit from
  coroutines.
- The existing codebase uses multiprocessing + threading exclusively.
- The caller (demo.py) stays plain synchronous Python.
- `ModelPipeline.submit(op)` returns a `Future` that maps cleanly to
  the existing `wait()` pattern.

### Pipeline-per-model-ID model

Each public method (`register`, `move`, `submit_generate`, `pause`,
`resume`, `remove`) submits an `Op` subclass instance onto the
target model's `ModelPipeline`.  Within a model, ops are serialized
by the pipeline's FIFO -- there is exactly one worker thread per
model_id, so two ops on the same model never run concurrently and
no future-chaining is needed.  Across model IDs, every pipeline runs
concurrently; `Slots` and per-entry locks (`entry["_lock"]`,
`entry["_gen_lock"]`) provide the only cross-model synchronisation
points.  Cross-pipeline dependencies (e.g. Phase-2 HBM eviction --
the acquirer's pipeline blocks on the incumbent's pipeline running an
`EvictForPeerOp`) go through `submit_to_peer_and_wait`, which records
a runtime edge in `_waiting_on` and asserts acyclicity to catch
deadlock cycles loudly.

```
Main thread             Per-model pipeline workers           Worker processes
-----------             --------------------------           ----------------
register("a") -------> Pipeline[a]: RegisterOp              --> Instance A (gpu0, L1 slot)
register("b") -------> Pipeline[b]: RegisterOp              --> Instance B (gpu1, L1 slot)
register("c") -------> Pipeline[c]: RegisterOp              --> (FIFO-blocks on Slots.allocate)
                            |
wait("c")                   |
                       Pipeline[c]: RegisterOp completes    --> Instance C (gpu0)
                            |
submit_generate("b") -> Pipeline[b]: GenerateOp
                          Phase 1: MoveOp("up", announce_state="running")
                            _step_up: saved -> checkpoint  (load)
                            _step_up: checkpoint -> sleep  (Slots.allocate(L=b.level))
                            _step_up: sleep -> up          (tier A/B/C; evict slotless up
                                                            via EvictForPeerOp on the
                                                            incumbent's pipeline; weights
                                                            -> HBM)
                          Phase 2: inst.generate (under entry["_gen_lock"]),
                            append (rid, q_rec, done_event) to _inflight[b],
                            return PendingRequest -- the worker is now free.
                       Daemon thread "generate-wait[b:N]":  Phase 3:
                            done_event.wait()
                            collect outputs, set user-future, drop from
                            _generate_futures eventually via wait().
                       (Demuxer in parallel: per-cmd ack listeners and
                       _on_generate_done -- the latter pops _inflight[b] and,
                       on the trailing edge, releases the slot and flips
                       state running -> up.)
                            |
remove("b") ----------> Pipeline[b]: RemoveOp
                          MoveOp("saved") if needed:
                            _step_down: up -> sleep        (Slots.deallocate if held)
                            _step_down: sleep -> checkpoint (Slots.deallocate)
                            _step_down: checkpoint -> saved (teardown)
                          delete image, pop registry
                       Daemon thread "remove-teardown[b]":  pipe.shutdown(),
                            _pipelines.pop(b).
```

### Why the pipeline replaced `ThreadPoolExecutor`

The legacy design pinned a pool worker for the **entire**
generation duration because Phase 3 (`done_event.wait()`) ran on
the same pool thread that submitted Phase 1+2.  This forced the
pool size up to `max_workers=4096` -- otherwise fan-out workloads
queued inside `concurrent.futures`, hiding the queueing delay
inside the dashboard's `sleep` segment.

The pipeline migration broke that coupling:

- **`GenerateOp.execute` returns after Phase 2** (`PendingRequest`).
  The pipeline worker is then free to dequeue the next op (a
  `PauseOp` jumping the queue, the next `GenerateOp`, an
  `EvictForPeerOp` from a peer's wake-up, etc.).
- **Phase 3 runs on a one-shot daemon thread** spawned by
  `submit_generate`.  Pool occupancy is now proportional to in-flight
  generations, but the threads are not pre-allocated and not bounded
  by a pool size -- they exist only while waiting for `done_event`.

A pipeline worker thread is durable (one per registered model_id)
and lifecycle is explicit: created in `register` /
`init`-image-cache-scan via `_make_pipeline`, joined and popped from
`Orchestrator._pipelines` by `RemoveOp` and `_shutdown_all_pipelines`
(hard-reset).

### Pause-as-interrupt

Pause is **not** a queued successor like move/resume/remove.
`Orchestrator.pause`:

1. Calls `pipe.interrupt_now("pause")` synchronously, which sets the
   pipeline's `InterruptFlag`.
2. Submits `PauseOp` at the **head** of the FIFO (`submit_front`).

Any `Op` currently executing on the worker observes the flag at its
next cooperative yield-point (`InterruptFlag.raise_if_set()` /
`wait_or_interrupt()` -- both throw `Interrupted`).  This replaces
the legacy "0.5 s polling loop" pattern in four places (the loops
have been deleted -- see `pipeline_DESIGN.md`).  `PauseOp.execute()`
clears the flag at the end of its run, so subsequent ops dequeued
after PauseOp see a clean flag.

### Sync between pipeline worker and demuxer

Two sites need explicit synchronization with the demuxer's
`_on_generate_done` listener:

- `_send_cmd_with_ack` -- enqueues a non-`generate` cmd and waits
  for its FIFO ack.  Holds `entry["_gen_lock"]` so the demuxer's
  slot-release decision (`not inflight and inst._pending_count == 0`)
  cannot fire between bumping `_pending_count` and the ack landing.
- `GenerateOp.execute` Phase 2 -- holds `entry["_gen_lock"]` across
  the `inst.generate(...)` send and the `_inflight[mid].append(...)`,
  for the same reason.

`entry["_gen_lock"]` is created on the registry entry in `register`
(and in `init`'s image-cache scan).  It is an `RLock` to tolerate
nested calls from inside an `Op` body that already holds it.

## Known Issues

The 10 entries below are the cumulative race / wedge log for
the orchestrator, all `[resolved]`.  Each one was hit in a
real demo run (dates and victim model IDs are inline).  The
roadmap and pattern table here index them by trigger and by
the cross-cutting concurrency invariant they helped codify.

### Roadmap by cluster

**Eviction / slot-allocation cluster (5)** -- AB / orchestrator-
vs-worker boundary, all rooted in `_step_up` Phase 2 evicting a
slotless `up` incumbent racing the incumbent's own activity:

* [Eviction bypasses future-tracking (save deadlock)](#eviction-bypasses-future-tracking-save-deadlock--resolved)
  -- `demo_10`, 2026-05-04, models 3 & 4.  Phase-2 eviction
  bypassed `_futures` / `_last_generate_future`; combined with
  two threads on `_result_queue.get` and a single-slot
  `_cmd_ack_events` overwrite, left orphan futures that wedged
  every subsequent `move()`.
* [Phase 2 eviction AB-BA deadlock](#phase-2-eviction-ab-ba-deadlock--resolved)
  -- `model 3` / `model 15`.  Held `_lock(incumbent)` across
  `_step_down`, deadlocking against the incumbent's own
  `_acquire_slot_for_running` retreat.
* [Late-sleep slot leak (eviction-vs-rerun race)](#late-sleep-slot-leak-eviction-vs-rerun-race--resolved)
  -- `demo_15`, 2026-05-10, six models.  Incumbent raced back
  to `running` between gate-check and sleep ack; old
  `_step_down(up, sleep)` published `state="sleep"` over the
  re-claimed slot without freeing it, leaking forever.
* [Resume slot-steal wedge](#resume-slot-steal-wedge--resolved)
  -- 2026-05-12, `model 8`.  Peer's eviction stole the slot
  `_resume_sync` had just claimed but not yet published; resume
  blindly sent worker `resume` over a slotless registry entry.
* [Eviction-mid-generate dormant-engine wedge](#eviction-mid-generate-dormant-engine-wedge--resolved)
  -- `model 16` 2026-05-12 + `model 13` 2026-05-13 (4 ms
  residual race).  Generate cmd queued behind sleep on the
  worker pipe → silent hang on dormant engine.  Two-stage
  fix: in-flight drain + sentinel-future gate; worker-side
  `_dormant` fail-fast as defense in depth.

**Pause / resume cluster (4)** -- registry / future-chain
discipline; the recurring failure was either chain bypass or
phantom registry state surviving a no-op body:

* [`sub` busy-spin + silent-pause failure](#sub-busy-spin--silent-pause-failure--resolved)
  -- three layered bugs (chain-only-on-`_last_generate_future`,
  `_sub_sync` re-poisoning `_futures[mid]`, unconditional chain
  await re-raising the prior failure).
* [Deferred-pause / phantom-running hang](#deferred-pause--phantom-running-hang--resolved)
  -- `cl.generate_all(); cl.pause_all()` back-to-back; pause
  queued behind generate, ran on empty engine, but
  `_pause_sync` still committed `paused=True` → phantom
  state propagated to phantom-running on the next resume.
* [Pause-during-drain ineffective](#pause-during-drain-ineffective--resolved)
  -- pause chained on `_futures[mid]` whose move was parked on
  the very `done_event`s pause should have broken; chain
  ordering put pause at the back of its own wait.
* [Pause-during-resume future-chain break](#pause-during-resume-future-chain-break--resolved)
  -- 2026-05-13, `model 7`.  No-op pause unconditionally
  published its (already-done) future, letting the next
  resume slip past `_await_prev` and run in parallel with the
  still-walking earlier resume.

**Move-vs-generate cluster (1)**:

* [Move-vs-generate Phase-1 sleep race](#move-vs-generate-phase-1-sleep-race--resolved)
  -- `model 7` / `model 15`.  `_generate_sync` released
  `gen_lock` between Phase 1 wake-up and Phase 2 enqueue; a
  concurrent `move()` slipped `sleep` into the gap.
  `prev_gen_events` didn't help because the generate hadn't
  reached Phase 2 yet to populate `_inflight`.

### Recurring patterns and the safety rules they codify

A handful of root causes show up across the catalogue.  The
design rules in the right column are what each pattern's
fix ratified:

| Recurring pattern | Where it manifested | Codified safety |
|---|---|---|
| **AB-BA between `gen_lock` and `_lock(mid)`**: peer thread holds one and wants the other. | save deadlock; Phase 2 AB-BA | Lock ordering rule: `gen_lock` outside, `_lock(mid)` inside; `_lock(mid)` released across every `_send_cmd_with_ack`.  `_step_down(up, sleep)` reconciles to the post-sleep worker rather than trusting the caller's pre-sleep snapshot. |
| **TOCTOU on `entry["slot"]`**: Tier A's `Slots.try_allocate` runs outside the model lock; publish into `entry["slot"]` runs inside.  In-between, a peer can sleep the model and free the slot we just took. | late-sleep slot leak; resume slot-steal wedge | `_step_down(up, sleep)` unconditionally reconciles slot + state to the post-sleep worker.  `_resume_sync` routes through `_move_sync(up, announce_state="running")` so it inherits the post-acquire state re-check + heavyweight `sleep -> up` ladder fallback. |
| **Worker-pipe ordering**: `gen_lock` acquisition order = worker-pipe enqueue order; whoever needs to land first must take `gen_lock` first. | move-vs-generate Phase-1 sleep race; eviction-mid-generate wedge | All non-`generate` cmds go through `_send_cmd_with_ack` which sends under `gen_lock`.  `_evict_for_phase2` publishes a sentinel onto `_futures[mid]` under `gen_lock` so the racing generate's `wake_up_weights` enqueues *after* the eviction's `sleep`. |
| **`_futures[mid]` chain hygiene**: publish-without-await, await-without-publish, or chain-trample by a no-op body. | save deadlock; `sub` busy-spin (chain re-poison); pause-during-drain (chain barrier puts pause behind its own wait); pause-during-resume (no-op pause overwrites in-flight resume); move-vs-generate Phase-1 (only-`_last_generate_future` chain missed earlier in-flight) | `_await_prev` log-and-skip helper; `pause` only publishes when the body will commit a state change; `move` snapshots both `_last_generate_future` AND `_inflight[mid]` events under `gen_lock`; `_evict_for_phase2` publishes a sentinel under `gen_lock` so concurrent generates re-check before sending. |
| **Phantom registry state**: `paused=True` with no saved sub-reqs, or `state="running"` with empty `_inflight`. | deferred-pause / phantom-running hang; resume slot-steal wedge (slotless `running`); pause-during-resume break | Pre-act re-checks under `gen_lock`: `_pause_sync` rejects when `_inflight[mid]` is empty; `_resume_sync` self-heals empty `_inflight` as a no-op (clears `paused=False` only) and routes the non-empty branch through `_move_sync(up, announce_state="running")` so registry + slot + worker stay coherent. |
| **Worker accepts cmds against a torn-down engine**: `engine.add_request` / `engine.step()` on an executor `llm.sleep(level=2)` just froze. | move-vs-generate Phase-1 sleep race; eviction-mid-generate dormant-engine wedge (model 16 + the model 13 4 ms residual race) | Orchestrator-side: `_inflight` + `_futures` discipline + Phase-2 sentinel.  Worker-side defense in depth: `vllm_child._dormant` flag (set at the bottom of the `sleep` handler, cleared at the bottom of `wake_up_kv_cache`) gates `_submit_generate` and fails fast with a `RuntimeError` ack so any sentinel regression surfaces as a loud per-request error instead of a silent hang inside `engine.step()`. |

### Eviction bypasses future-tracking (save deadlock)  [resolved]

> **Status:** the underlying race that produced the orphan
> `_last_generate_future` is closed by the demuxer + FIFO ack
> redesign.  The historical analysis below is preserved for
> archaeology; "Resolution" at the end of the section explains how
> each leg of the race is now defused.  Note that
> `prev_gen_future` was later **re-introduced** in `_move_sync`
> for an unrelated bug (Phase-1-vs-sleep race; see
> "Move-vs-generate Phase-1 sleep race" below) but with
> paused-bail-aware `result(timeout=0.5)` polling instead of an
> unconditional `.result()`, so an orphan future would now
> trigger a paused-bail or `Exception`-handler exit rather than
> deadlocking the move.

**Symptom.**  Under concurrent `generate` + Phase-2 eviction, an evicted
model can end up with a permanently-pending entry in
`_last_generate_future`.  Any subsequent `move()` (e.g. `target="saved"`)
submitted for that model chains behind the dead future via
`prev_gen_future.result()` in `_move_sync` and blocks forever.  A later
`generate()` then logs `_generate_sync ... waiting for move future`
and hangs on `prev_move_future.result()`.  The `vllm_child` process
itself is healthy and idle — it just never receives any more commands.

**Reproducer shape.**  Two or more models woken on the same GPU, one
finishes `generate` and parks at slotless `up`, then a peer's
`_step_up` Phase-2 evicts it.  Shortly after (before the caller issues
another `generate`), a batch `move(target="saved")` fan-out is submitted
across all models.  The evicted models' save move never makes progress;
every other model saves normally.

Observed in the demo_10 run on 2026-05-04 for models 3 and 4 (instances
2 and 3): last instance-log line was `sleep OK` from the eviction, then
zero further dispatched commands despite two subsequent
`move target=saved` batches at 22:04:51 and 22:06:16, and a
`generate` batch at 22:06:30.  The orchestrator's "already in 'saved'
state" short-circuit fired for all other models but not these two,
proving their state never advanced past `sleep`.

**Root cause.**  The Phase-2 eviction path in `_step_up` calls
`_step_down(incumbent, "up", "sleep")` **directly**, bypassing the
thread-pool + `_futures` / `_last_generate_future` mechanism that
serialises per-model work:

```python
with Orchestrator._locks_ordered(incumbent):
    if (inc_entry.get("slot") is None
            and inc_entry.get("state") == "up"
            and inc_entry.get("gpu") == home_gpu):
        log.info("%s: evicting %s from GPU %s", ...)
        Orchestrator._step_down(incumbent, "up", "sleep")
```

This synchronously drives `inst.sleep().wait()` on the incumbent under
its `_lock`.  Meanwhile the incumbent's own `_start_generate_waiter`
thread is still alive (its exit condition is "`inflight` empty **and**
`inst._pending_count == 0`", checked after each `_result_queue.get`).
The two threads race:

- If the eviction's `sleep()` is enqueued before the waiter observes
  `_pending_count == 0`, the waiter sees state flipped to `sleep`
  (by `_step_down`) while its own `done_event` for the last request may
  or may not have fired, and its final `_set_state(model_id, "up")`
  block at lines 1013–1018 / 1089–1093 is now inconsistent with reality.
- Whichever way the race resolves, the `Future` returned by
  `submit_generate` for that last request can remain un-set because the
  waiter exits via a branch that skips `done_event.set()`, leaving
  `_last_generate_future[model_id]` pointing at a future that will never
  resolve.

Every subsequent `move()` pops that stale future into its
`prev_gen_future` parameter and deadlocks on `.result()`.  Because
`_futures[model_id]` is updated *before* `_move_sync` runs, the second
and third save attempts chain behind the first, compounding the hang.

**Resolution.**  The race had two legs:

1. *Two threads consuming `_result_queue`*: the per-model
   `_start_generate_waiter` and any `inst.wait()` caller (e.g. the
   evictor's `inst.sleep().wait()`) raced for `get()`.  One could
   grab both replies and exit while the other parked forever.  Now
   the per-instance demuxer is the **sole** consumer of
   `_result_queue`; `inst.wait()` is a condvar wait on
   `_pending_count`, never touches the queue, and is safe under any
   number of concurrent callers.
2. *Single-slot `_cmd_ack_events` overwrite*: two
   `_send_cmd_with_ack(model_id, "sleep")` calls landing on the
   same dict key caused the second Event to overwrite the first,
   so the first caller never woke.  `_cmd_ack_events` is now a
   FIFO of Events per `(model_id, cmd)`; both senders' Events are
   appended and the demuxer's catch-all listener pops them in
   send order, signalling each caller exactly once.

Combined, the orphan `done_event` and orphan
`_last_generate_future` cannot occur: the demuxer's permanent
generate listener fires for every generate ack (no exit branch
that skips `done_event.set()`), and the FIFO discipline means a
peer-eviction sleep racing with a same-model self-evacuation
sleep both complete cleanly.  `_acquire_slot_for_running` also
gained a model-lock-guarded re-check that avoids the redundant
own-side `_step_down` when a peer has already evicted us, so we
don't burn an extra cmd round-trip in the common case.

### `sub` busy-spin + silent-pause failure  [resolved]

> **Status:** fixed.  The combined symptom ("`cl.sub(gpu)` runs at
> 100% CPU forever; subsequent `cl.pause()` on the same model logs
> `pause received` and then nothing") had three layered bugs;
> closing any one of them is enough to break the visible symptom,
> and all three are addressed.

**Symptom.**  After `cl.sub(gpu)` is called while a model on *gpu*
still has an inflight generate (e.g. a deferred-paused generate
that was re-prefilled by a recent `resume`), the orchestrator log
fills with thousands of identical `WARNING sub: GPU N resident move
raised: model 'M' is currently running a generate; ...` lines
within milliseconds.  Subsequent user actions on the same model
(`pause`, `resume`, `move`, …) print only their `<verb> received`
line and silently abort.

**Layered root causes.**

1. *`move()` only chained on `_last_generate_future`.*  When more
   than one generate was inflight, the **last** submitted future
   could resolve while an earlier one was still mid-decode.
   `_move_sync` would see `entry["state"] == "running"` and raise
   `RuntimeError("...currently running a generate...")` despite
   the "drain in-flight generate before moving" semantics that
   `_last_generate_future` was supposed to provide.  Fix:
   `move()` and `remove()` now snapshot every `done_event` in
   `_inflight[mid]` under `gen_lock` and `_move_sync` /
   `_remove_sync` wait on each.  Paused models still skip the
   wait (their inflight events can only complete via a future
   `resume`, so blocking on them would deadlock `paused → sleep`).

2. *`_sub_sync` busy-spin re-poisoned `_futures[mid]` on every
   iteration.*  Each loop iteration called
   `Orchestrator.move(mid, "checkpoint")`, which captures
   `prev = _futures.get(mid)` and then sets
   `_futures[mid] = new_fut`.  The new `_move_sync` chained on
   `prev.result()` and re-raised the prior failure without ever
   re-reading `entry["state"]`, so even after the underlying
   condition resolved (state flipped `running → up`) the spin
   kept logging the original error forever.  Fix: `_sub_sync`
   now tracks per-mid move futures locally, issues at most one
   move per resident per drain, and parks on `Slots._cv` with a
   timeout between passes (waking on `Slots.deallocate` /
   migration).  A model is only re-issued if it has fully drained
   (popped from `submitted`) and a new resident appears (e.g.
   someone migrated back onto the draining GPU).

3. *Chain-prev exceptions silently aborted the next op.*  Every
   per-model orchestrator op (`move`, `pause`, `resume`,
   `submit_generate`, `remove`) chained on
   `_futures.get(mid).result()` unconditionally.  When the chain
   was poisoned by (2)'s spin, every queued op re-raised the
   spin's `RuntimeError` and aborted before its own body ran —
   hence the silent `pause received` followed by nothing.  Fix:
   a single `Orchestrator._await_prev(model_id, label, fut)`
   helper wraps the chain await with `try/except`, logs the
   prior failure once, and lets the successor make its own
   decision based on the *current* world (it already re-checks
   its own preconditions under the model lock).  Successful
   chains are unaffected.

The three fixes are independent and additive.  (1) eliminates
the genuine race; (2) prevents a "spin → poison" topology even
if a future variant of `_move_sync` introduces some new
transient failure mode; (3) is the safety net that ensures any
single failed op cannot mute the next user action.

### Deferred-pause / phantom-running hang  [resolved]

> **Status:** fixed.  Symptom: `cl.move_all("checkpoint")` hangs
> for minutes on models that look "stuck at generate" but show
> no entry in the dashboard's request list.  `cl.pause_all()` on
> those same models flips them out of `running` instantly
> (because there is no actual generate to pause), which lets the
> rest of the system progress.

**Trigger.**  Any user idiom that submits `pause` back-to-back
with `generate` -- typically `cl.generate_all("test", N);
cl.pause_all()` on one line, or two REPL lines in quick
succession -- reliably reproduces it.

**Sequence.**

1. `_generate_sync` flips `state` `up → running`, registers
   `_inflight[mid] = [(req_id, q_rec, done_event)]`, and the
   worker queues `generate`.  The user's `pause_all()` then
   submits `_pause_sync`, which sees `state == "running"` and
   sends `pause` -- the worker queues it **behind** the
   generate.
2. The worker runs the generate to completion.  The demuxer's
   generate listener pops `_inflight[mid]`, sets `done_event`,
   and -- under `gen_lock` -- releases the slot and flips
   `state` `running → up`.
3. The worker then dequeues the `pause`, runs it on the now-empty
   engine, and reports `saved=0 was_paused=False`.  The
   demuxer's catch-all listener wakes the parked `_pause_sync`,
   which marks `entry["paused"] = True`.
4. Registry now holds **phantom-paused**: `paused=True` with
   `_inflight=[]` and no actual saved generates.
5. The next `resume_all()` runs `_resume_sync`, sees
   `paused=True, state=="up"`, sends `resume` to a worker that
   has nothing to restore (`restored=0 synthesized=0`), and
   flips `state` to `running`.  Registry now holds
   **phantom-running**: `state="running"` with `_inflight=[]`.
6. Subsequent `move(checkpoint)` snapshots an empty
   `_inflight`, sees `state=="running"`, and either raises
   `currently running a generate` (which `_await_prev` logs
   harmlessly) or chains behind a stuck `move(up)` whose own
   `_inflight` snapshot is also empty -- in either case the
   system makes no progress on those models until the user
   pauses them by hand.

**Fix.**  Two layers, mirroring the `_await_prev` and inflight
snapshot patterns used elsewhere:

1. *`_pause_sync` (deferred-pause guard).*  After the outer
   `state == "running"` precondition passes, take `gen_lock`
   and re-check `_inflight[mid]` is non-empty.  If empty, log
   `no inflight generates (deferred-pause race); pause is a
   no-op` and return without sending the worker `pause` and
   without marking `paused=True`.  `gen_lock` serialises against
   the generate-done listener, so an empty `_inflight`
   observed under the lock means the listener has already (or
   is about to) flipped the state.

2. *`_resume_sync` (generate-shaped, unified ladder-walk +
   slot-acquire + phantom self-heal).*  After the outer
   `paused` precondition check, gate on `state in ("up",
   "sleep", "checkpoint")` (the only states a paused model can
   legally hold).  Then snapshot `_inflight[mid]` under
   `gen_lock` *before* doing any work.

   * **Empty `_inflight`** -> observable no-op.  Clear
     `paused = False` under `model_lock` and leave the model
     wherever the user parked it -- no walk-up, no slot
     acquisition, no worker `resume` cmd.  The pause-side
     guard already prevents minting `paused=True` without
     sending the worker `pause`, so in healthy code this
     branch is unreachable; it remains as a self-heal for any
     future path that bypasses `_pause_sync`.

   * **Non-empty `_inflight`** -> single unified call followed
     by the worker `resume`:

     a. **Walk + acquire + announce** via the *exact* same
        entry point `_generate_sync` Phase 1 uses:
        `_move_sync(mid, "up", announce_state="running")`.
        Handles `state="sleep"`/`"checkpoint"` (ladder walk to
        `up` with announce on the final `_step_up`) and
        `state="up"` slotless (in-place `_acquire_slot_for_running`
        + post-acquire state re-check + announce) in a single
        code path.  Direct `_move_sync` (not public `move()`)
        so the inner walk doesn't capture `prev_gen_events`
        and doesn't publish a separate `_futures[mid]` future.
        `paused=True` rides through unchanged because nothing
        in `_step_up`/`_step_down` touches it.

        Inheriting `_move_sync`'s post-acquire state re-check
        is what fixes the resume slot-steal wedge (see "Resume
        slot-steal wedge" below).  The earlier two-call
        sequence (`_move_sync(mid, "up")` then bare
        `_acquire_slot_for_running` then unconditional
        `_set_state("running")`) bypassed the re-check on the
        slotless-up branch and could leave the model wedged at
        `state="running", slot=None` over a sleeping engine.

     b. **Worker resume + clear paused.**
        `_send_cmd_with_ack("resume")` re-prefills the saved
        subreqs and unfreezes the engine; then under
        `model_lock`, clear `entry["paused"] = False`.  State
        is already `"running"` from step (a) so we don't flip
        it again.

   `_inflight` is the authoritative orchestrator-side ledger
   for "anything to drive": real `pause` doesn't pop entries
   (only the generate-done listener does), and
   `_generate_sync` Phase 2 appends queued-during-pause new
   generates immediately, so the snapshot covers both classes.

The pause-side guard prevents new phantoms; the resume-side
empty branch self-heals any phantom that already exists, and
the non-empty branch gives `resume` clean generate-shaped
semantics: observably a no-op when there's nothing to drive,
otherwise the same `_move_sync(up, announce_state="running")`
that `generate` uses (typical pattern reaching the ladder-walk
arm: `cl.pause(); cl.move("checkpoint"); cl.resume()`).  See
`async_generate_DETAILS.md` "Deferred-pause / phantom-running
guard" for the full breakdown.

### Phase 2 eviction AB-BA deadlock  [resolved]

`_step_up` Phase 2 picks an incumbent on the home GPU and calls
`_step_down(incumbent, "up", "sleep")` to free HBM.  The
incumbent is a *different* model from the one we're stepping
up, so the eviction has to follow the same lock-ordering rule
that already holds elsewhere: **`gen_lock` outside,
`model_lock` inside, and `model_lock` released across every
`_send_cmd_with_ack`**.

The original eviction block held `_lock(incumbent)` across the
`_step_down(incumbent, "up", "sleep")` call.  `_step_down`
calls `_send_cmd_with_ack("sleep")`, which takes
`gen_lock(incumbent)`.  When the incumbent's *own* thread was
simultaneously self-evacuating (its `_generate_sync` parked in
`_acquire_slot_for_running` already holding
`gen_lock(incumbent)` and waiting for `_lock(incumbent)`
inside its own `_step_down`), the two threads deadlocked
AB-BA.  Symptom: a fresh `generate_all(...); pause_all()`
followed by an `add()` that triggers eviction silently wedges
the incumbent and the new model -- exactly the `model 3` /
`model 15` hang.

The fix validates the incumbent under `_lock(incumbent)` and
releases the lock before invoking `_step_down`:

```python
with Orchestrator._locks_ordered(incumbent):
    evict_now = (inc_entry.get("slot") is None
                 and inc_entry.get("state") == "up"
                 and inc_entry.get("gpu") == home_gpu)
if evict_now:
    Orchestrator._step_down(incumbent, "up", "sleep")
```

Releasing `_lock(incumbent)` between the check and the
`_step_down` opens a window the `_step_down(up, sleep)` side is
designed to absorb: vllm `sleep` is idempotent, the FIFO
`_cmd_ack_events` deque tolerates duplicate sends by
construction, and the trailing block reconciles to the
post-sleep worker by unconditionally freeing any slot the
registry holds and `_set_state(..., "sleep")` -- which is a
no-op same-state stamp when the incumbent self-evacuated to
`sleep` first, and the *only* slot release when the incumbent
instead raced all the way back to `running` while our `sleep`
was queued behind a `generate` on the worker.  See
`async_generate_DETAILS.md` "Eviction lock ordering" and
"Slot release at `up -> sleep`" above for the full timeline.

### Pause-during-drain ineffective  [resolved]

`cl.pause(mid)` issued while `cl.sub(gpu)` is draining `mid`'s
home GPU was a silent no-op: every pause logged `pause
received` and then nothing further until the in-flight generate
naturally completed, by which point the model was already in
`checkpoint` and `_pause_sync` logged `not running, pause is a
no-op`.

The cause was a chain-prev shape: `_sub_sync` schedules
`move(mid, "checkpoint")`, which captures `prev_gen_events`
(the inflight `done_event`s) under `gen_lock` and parks
`_move_sync` on `ev.wait()` for each.  `pause()` reads
`_futures[mid]` (now the move future) and `_pause_sync._await_prev`
blocks on it.  `done_event` only fires on a real
`generate_done`, not on `pause` -- so chaining `pause` on
`_futures[mid]` puts pause at the back of the very wait it
should break.

Two-part fix:

1.  **`_pause_sync` opts out of the `_futures[mid]` chain.**
    Same reasoning the original code already applied to
    `_last_generate_future[mid]`: any future that is itself
    parked on this model's inflight done-events will deadlock
    pause.  Correctness is preserved by the existing
    precondition checks (`state == "running"`, `paused`,
    `_inflight[mid]` under `gen_lock`).  The pause future is
    still published at `_futures[mid]` so downstream ops chain
    on it.

2.  **`_move_sync` polls `entry["paused"]` while waiting on
    `prev_gen_events`** and bails the wait when the model becomes
    paused mid-drain.  Without this, the move would hang
    forever (paused engines never set inflight events).  After
    bailing, `entry["state"]` is `up` and `paused=True`, so the
    move proceeds through the ladder normally and the drain
    completes.

Net effect: `cl.sub(gpu)` followed by `cl.pause(mid)` lands the
model at `state="checkpoint", paused=True` promptly -- saved
sub-requests preserved for a future `resume`, GPU freed for the
drain to finish.  See `async_generate_DETAILS.md`
"Pause-during-drain (chain-vs-interrupt)" for the full
breakdown and the `_move_sync` poll-loop pattern.

### Move-vs-generate Phase-1 sleep race  [resolved]

`cl.sub(gpu)` -> `move(mid, "checkpoint")` racing an in-flight
`generate(mid, ...)` on the same model could hang the vLLM
child.  Symptom: the worker queue accumulated
`wake_up_kv_cache -> sleep -> generate -> unpin ->
cuda_checkpoint` back-to-back; the engine tried to schedule
the request against cumem memory that `sleep` had just
freed, and stuck.

**Cause.**  `_generate_sync` Phase 1 holds `gen_lock` across
`_move_sync(up, announce_state="running")` (which sends
`wake_up_weights`, `restore_weights`, `wake_up_kv_cache`),
releases the lock at Phase 1 exit, then re-acquires it for
Phase 2's `inst.generate(...)`.  A concurrent `move()` from
`_sub_sync` could win `gen_lock` in the released-lock window
and queue `sleep` between the wake-up and the generate.
`prev_gen_events` (the existing inflight-drain wait) didn't
help: the generate hadn't reached Phase 2 yet, so its
`done_event` wasn't in `_inflight` to be captured.  Observed
on `model 7` and `model 15` in mid-2026 when their home GPU
was being drained while a generate was still in wake-up.

**Fix.**  `move()` snapshots `_last_generate_future[mid]`
under the same `gen_lock` that captures `prev_gen_events`,
and passes it as a new `prev_gen_future` parameter to
`_move_sync`.  `_move_sync` polls this future (paused-bail
aware, mirroring the existing `prev_gen_events` loop) before
issuing any cmd of its own.  The future covers the full
lifecycle of the most recent `_generate_sync` -- Phase 1
wake-up, inter-phase window, Phase 2 enqueue, Phase 3
done-wait -- so by the time `_move_sync` proceeds, no
generate can be in a stage that would re-enter `gen_lock`
for this model and queue cmds against the worker.

`prev_gen_events` is retained alongside `prev_gen_future`:
the events catch *earlier* generates already in `_inflight`
that are not tracked by the latest-only `_last_generate_future`
slot; the future catches the *latest* generate in stages
(Phase 1, inter-phase) that `_inflight` does not yet see.
Both skip on paused models (the future / events never
resolve while the engine is frozen) and both use the same
0.5s poll-and-bail loop so a concurrent pause still breaks
the wait.  See `async_generate_DETAILS.md` "Move-vs-generate
Phase-1 race" for the cmd-queue trace and the full poll-loop
listing.  `remove()` was not extended with `prev_gen_future`
because it refuses on paused models and its non-paused use
sites have not exhibited this race; the existing
`prev_gen_events` drain plus the paused-refusal is
sufficient for the observed call patterns.

### Late-sleep slot leak (eviction-vs-rerun race)  [resolved]

A Phase-2 eviction `sleep` racing the incumbent's *own*
return to `running` could leak the incumbent's slot for the
rest of the run and publish `running` over an asleep engine.
Symptom in `outt`: a `running (Xs) -> sleep` transition with
no preceding `Slots.deallocate` log for that slot, even though
the model had been `up`/`running` continuously with a slot
held.  Observed on models 9, 10, 12, 13, 14, 15 in the
demo_15 run on 2026-05-10.

**Cause.**  The window between Phase 2's eviction-gate check
and its `sleep` ack landing is wide enough that a fresh
`generate` can race the incumbent all the way back to
`running`:

```
T0  evictor: gate passes (incumbent is slotless+up on home_gpu)
T0  evictor: release model_lock(incumbent)        # AB-BA fix
T0  evictor: _send_cmd_with_ack("sleep")          # parks on ack
T1  incumbent: new generate arrives, Phase 1 walks ladder up,
               _acquire_slot_for_running Tier A takes the slot
T1  incumbent: Phase 2 enqueues "generate" cmd on worker
               (queued *behind* the still-pending "sleep")
T2  worker: drains "generate" first (FIFO on the worker queue),
            engine runs the request, "generate_done" ack fires
T2  _on_generate_done: inflight empties, but
                       inst._pending_count == 1 (the queued
                       sleep is still in flight) -> skip the
                       slot release; state stays "running"
T3  worker: runs the late "sleep", engine drops weights/KV
T3  evictor: ack lands, _step_down(up, sleep) returns from
             _send_cmd_with_ack
T4  evictor (old code): _set_state(incumbent, "sleep") --
                        overwrites "running" without freeing
                        the slot
```

The slot stays in `Slots._live`; subsequent allocations on
the same GPU undercount free HBM and either OOM or evict
healthy models to make up the shortfall.  Worse, the registry
publishes `running` over an asleep engine -- any later
`move()` or `generate()` against the incumbent observes a
stale `running` state and either refuses (`RuntimeError`) or
re-enters `_step_up` paths that double-allocate.

**Fix.**  `_step_down(up, sleep)` was reshaped from "trust
the caller's pre-sleep snapshot" to "reconcile to the
post-sleep worker".  After `_send_cmd_with_ack("sleep")`
returns, the trailing block now runs under `model_lock`:

```python
with Orchestrator._locks_ordered(model_id):
    if entry.get("slot") is not None:
        Slots.deallocate(entry["slot"])
        entry["slot"] = None
    Orchestrator._set_state(model_id, "sleep")
```

The unconditional `Slots.deallocate` matches the worker's
actual state (it just ran `llm.sleep(level=2)`, so the slot
backing the asleep weights is no longer in use), and the
`_set_state` overwrite is now *correct* rather than racy --
the engine truly is asleep when the ack lands.  Legitimate
callers (`_acquire_slot_for_running` retreat, Phase-2
eviction, `_move_sync` down-walk in the no-race case) all
enter with `slot is None`, so the new branch is a no-op for
them; only the racing-back-to-running case takes the real
release path.

The Phase-2 eviction comment block was updated in lockstep:
the "redundant `sleep` is benign because vllm sleep is
idempotent" rationale still applies, but the comment now
points at the post-sleep reconcile as the recovery point
rather than describing the race as harmless.  See "Slot
release at `up -> sleep`" above for the slot-lifecycle
perspective and `async_generate_DETAILS.md` "Eviction lock
ordering" for the demuxer-side timeline.

### Resume slot-steal wedge  [resolved]

`cl.pause_all()` followed by `cl.resume_all()` could wedge a
model at `state="running", slot=None` over an asleep engine,
leaving its in-flight generates parked on `done_event.wait()`
forever.  Symptom in `outt`: a `model X: resumed
(state=running)` log immediately preceded by a peer's
`evicting model X from GPU N` and `Slots.deallocate Slot(...,
index=I)` for X's slot, with no subsequent `model X: generate
done` ever appearing.  Observed on `model 8` in the demo run
on 2026-05-12 when 18 simultaneous resumes contended for slots
on densely packed GPUs.

**Cause.**  Both `_generate_sync` Phase 1 and `_resume_sync`
called `_acquire_slot_for_running` to claim the running slot
for a slotless `up` model.  Tier A's `Slots.try_allocate`
runs *outside* the model lock; the publish into
`entry["slot"] = slot` runs *inside* the lock.  In between,
a peer's `_step_up` Phase 2 eviction can validate the
incumbent under the same lock (`slot is None and state ==
"up" and gpu == home_gpu`) and proceed to `_step_down(victim,
"up", "sleep")` -- which, after the "Late-sleep slot leak"
fix above, *unconditionally* deallocates whatever slot the
registry holds when the sleep ack lands.  Net effect: the
just-claimed slot is freed and `state` flips to `sleep`.

`_generate_sync` survived this race because `_move_sync`
runs a post-acquire state re-check at line 825:

```python
with Orchestrator._locks_ordered(model_id):
    if entry["state"] == target:                 # still "up"?
        Orchestrator._set_state(model_id, announce_state)
        return
    current = entry["state"]                     # peer evicted us
# fall through to ladder walk for self-heal
```

When the peer steals the slot, the re-check observes
`state="sleep"` and falls through to a heavyweight
`_step_up(sleep, up, announce_state="running")` which
re-acquires a slot via the standard sleep -> up path with
full Phase 1+2+3.  Generate ends up correctly slotted-running
just a few hundred ms late.

`_resume_sync` did **not** have this re-check.  Its body was a
two-call sequence -- `_move_sync(mid, "up")` (no
`announce_state`) followed by a bare
`_acquire_slot_for_running` followed by an unconditional
`_set_state("running")` -- so when the peer stole the slot,
resume blindly sent `resume` to the worker (queued behind the
peer's `sleep` cmd) and flipped `state` to `running` over a
slotless registry entry.  Worker FIFO processed `sleep` first,
asleep-engine ignored the late `resume`, and the model wedged.

**Fix.**  Replace the two-call sequence with a single
`_move_sync(mid, "up", announce_state="running")` -- the
exact same entry point `_generate_sync` Phase 1 uses -- so
`_resume_sync` inherits the post-acquire state re-check for
free.  In the racing case, the re-check falls through to
`_step_up(sleep, up, announce_state="running")` and
re-acquires a slot via the heavyweight path, after which
`_send_cmd_with_ack("resume")` runs against a properly
slotted, awake engine.  The state flip is now done by
`_move_sync` (not by `_resume_sync`), so the resume body
shrinks to:

```python
Orchestrator._move_sync(model_id, "up", announce_state="running")
Orchestrator._send_cmd_with_ack(model_id, "resume")
with Orchestrator._locks_ordered(model_id):
    entry["paused"] = False
```

A side-effect of the new ordering is that `state="running"` is
published *before* the worker `resume` cmd is sent (it used
to be after).  This actually matches `generate`'s ordering --
generate flips to `running` in Phase 1 before
`inst.generate(...)` enqueues in Phase 2 -- so
`_on_generate_done` now correctly sees `state="running"` if
a request happens to complete during the resume cmd
round-trip and takes the proper slot-release + state-flip-to-up
path.

The brief new window where `state="running", paused=True`
(bounded by the `resume` cmd round-trip) is benign:
`_step_up` Phase 2 eviction targets only `state=="up"`
slotless squatters (we are slotted `running`), and
`pause`/`resume`/`move` all serialise via `_futures[mid]` so
no concurrent control op can interleave.  See
`async_generate_DETAILS.md` "Deferred-pause / phantom-running
guard" for the full `_resume_sync` walkthrough.

### Eviction-mid-generate dormant-engine wedge  [resolved]

A `_step_up` Phase-2 eviction landing while the victim has an
in-flight generate could leave the victim's worker pipe in a
`[generate, sleep, generate]` ordering, drain the leading
generate normally, sleep the engine, then drop the trailing
generate onto a dormant engine -- which the worker's saved-
requests deferral path (`vllm_child._submit_generate`'s `if
_paused` / older `_engine_dormant` branch) absorbs *without
sending an ack*.  The orchestrator's `inst._pending_count` for
that generate stayed elevated forever, so the next sync cmd
the orchestrator queued (typically `unpin` from a Tier-C
retreat) blocked the worker on `_drain_pipe_generates` and
the orchestrator on `ev.wait()` for the unpin ack that would
never come.  Symptom in `outt`: an `enqueue unpin
pending=['generate']` line followed by `worker.py:636 >>>
unpin` with no matching `<<< unpin OK`, and no further state
transitions for the model until external intervention.
Observed on `model 16` in the demo run on 2026-05-12 when a
peer's eviction landed between the user's two back-to-back
`cl.generate_all()` calls.

**Cause.**  Phase 2 of `_step_up` (in the *evictor*) called
`_step_down(incumbent, "up", "sleep")` directly, with no
serialisation against the incumbent's in-flight generates.
The incumbent's worker pipe at the moment the eviction
fired:

```
[generate(in-flight)]                       # T0: incumbent generating
[generate(in-flight), sleep]                # T0+e: evictor sends sleep
[generate(in-flight), sleep, generate(new)] # T0+u: user submits next generate
```

The user's new generate slipped in because its
`_generate_sync` Phase 1 saw `state="running"` (the eviction's
`_step_down(up, sleep)` reconcile only flips `state` to
`sleep` *after* the sleep ack lands), so Phase 2 enqueued
`inst.generate(...)` directly.  The worker drained in FIFO
order:

```
generate(in-flight) -> generate done             # engine still up, fine
sleep               -> llm.sleep(level=2) OK     # engine asleep
generate(new)       -> deferred to _saved_requests; engine dormant
                                                 # NO ACK SENT
```

The deferral was originally introduced in commit `253ea9d` to
support `cl.pause(); cl.move("checkpoint"); cl.generate(...)`
-- a paused-then-checkpointed model where the next `resume()`
re-prefills the deferred request -- and ack suppression was
intentional for that case (paused models have their own ack
discipline via `_inflight` done-event suspension across
resume).  But the *eviction*-induced dormancy hits the same
deferral path without `_paused` ever being set: the
orchestrator is still tracking the generate as a normal
pending cmd, expecting a real `generate_done` ack that the
deferral suppresses.

`pending=['generate']` on the next enqueue is the giveaway:
the orchestrator queued `unpin` while the worker still had an
unfulfilled generate cmd from its perspective.  `unpin` (and
every other sync cmd in `vllm_child`) calls
`_drain_pipe_generates` first, which blocks waiting for the
deferred generate's ack that will never come.  Subsequent
chained ops (`move_all(checkpoint)` etc.) inherit the wedge
because their `_futures[mid]` chain runs through the stuck
`_generate_sync`.

**Fix.**  Phase 2 eviction now drains the incumbent's
in-flight generates *before* sending sleep, mirroring the
`prev_gen_future` + `prev_gen_events` snapshot+wait pattern
that `_move_sync` already uses for explicit `move()` calls.
The new helper `Orchestrator._drain_inflight_generates(mid,
label)` snapshots `_last_generate_future[mid]` and
`_inflight[mid]` done-events under the incumbent's
`gen_lock`, then waits OUTSIDE `gen_lock` so a concurrent
`_generate_sync` for the incumbent can still take `gen_lock`
and Phase 2 enqueue a generate -- those enqueues land on the
worker pipe AHEAD of our subsequent `_send_cmd_with_ack(
"sleep")` (since they take `gen_lock` first), so the worker
drains them on the still-up engine and only reaches sleep on
an empty pipe:

```
[generate(in-flight)]                       # T0: incumbent generating
[generate(in-flight), generate(new)]        # T0+u: user submits while we wait
[generate(in-flight), generate(new), sleep] # T1: in-flight done, we send sleep
```

Worker drain order:

```
generate(in-flight) -> generate done             # engine still up, fine
generate(new)       -> generate done             # engine still up, fine
sleep               -> llm.sleep(level=2) OK     # engine asleep on empty pipe
```

Skipped on paused models for the same reason `_move_sync`'s
`prev_gen_events` drain is skipped: a paused generate's
Phase 3 `done_event.wait()` parks until `resume()`, so a
blind wait would deadlock against the pause that the eviction
is trying to evict around.  Polls every 0.5s with a paused-
mid-wait bail to mirror the `_move_sync` polling loop.

**Residual race [resolved].**  The original fix above closed
the eviction-mid-running window where the incumbent had a
generate already in flight at the moment of eviction (the
`model 16` wedge), but acknowledged a residual window: a
fresh `Orchestrator.generate(incumbent, ...)` submitted
*between* the drain returning and the eviction's sleep ack
landing (~one `_send_cmd_with_ack("sleep")` round-trip,
typically 300 ms - 10 s) could still slip behind the sleep
on the worker pipe and re-trigger the same wedge.

This residual race fired in production on the 2026-05-13
demo run as the `model 13` wedge.  Trace from `outt` and
`/tmp/inst12.log` (the worker log for `model 13`):

```
ORCHESTRATOR (outt)                         WORKER (inst12.log)
05:05:03.606 model 12: evicting model 13     05:05:03.606 enqueue sleep pending=[]
                                             05:05:03.609 >>> sleep
05:05:03.607 generate model_id=model 13     ───── 4 ms gap ─────
05:05:03.610 model 13: claimed slot
05:05:03.610 model 13: up -> running         05:05:03.610 enqueue generate pending=['sleep']
                                             05:05:03.611 >>> generate (queued behind sleep)
                                             05:05:03.923 <<< sleep OK (0.314s) ← engine NOW asleep
05:05:03.924 model 13: running -> sleep      05:05:03.924 >>> generate (lands on dormant engine)
                                             05:05:03.924 submitted req_id=inst12-32
                                                          ← LAST LINE, no <<< generate OK
```

The fix is option (b) from the original entry: a sentinel
future on `_futures[incumbent]` for the eviction's lifetime,
gating any racing `_generate_sync(incumbent)`.  Implemented
as `Orchestrator._evict_for_phase2(incumbent)` -- the new
sole caller from the Phase 2 `if evict_now:` branch:

```python
@staticmethod
def _evict_for_phase2(model_id: str) -> None:
    sentinel = Future()
    gen_lock = Orchestrator._get_generate_lock(model_id)
    with gen_lock:
        prev_chain  = Orchestrator._futures.get(model_id)
        prev_gen    = Orchestrator._last_generate_future.get(model_id)
        prev_events = [ev for (_, _, ev)
                       in Orchestrator._inflight.get(model_id, [])]
        Orchestrator._futures[model_id] = sentinel   # publish under gen_lock
    try:
        Orchestrator._await_prev(model_id, "phase-2 evict chain", prev_chain)
        # ... wait for prev_gen + prev_events to drain (model-16 fix) ...
        Orchestrator._step_down(model_id, "up", "sleep")
    finally:
        sentinel.set_result(None)                    # always resolve
```

`_generate_sync(incumbent)` Phase 1 + Phase 2 now share a
single `gen_lock` acquisition, gated by a sentinel re-check
loop:

```python
while True:
    with gen_lock:
        cur = Orchestrator._futures.get(model_id)
        if cur is None or cur is prev_move_future or cur.done():
            ... Phase 1 (move + state flip) + Phase 2 (inst.generate) ...
            break
    Orchestrator._await_prev(model_id, "generate chain (eviction sentinel)", cur)
    prev_move_future = cur
```

Two ordering guarantees this gives:

1. **Phase 1 sees the eviction.**  The publish happens
   under `gen_lock(incumbent)`.  By the time the racing
   `_generate_sync` re-acquires `gen_lock` (after the
   eviction's `_drain_inflight_generates` releases it), the
   sentinel is on `_futures[incumbent]` and the re-check
   forces a release-and-await.  When the await returns,
   `_step_down(up, sleep)` has run -- so `state="sleep"` and
   the model's slot is `None`.  Phase 1 then walks
   `_move_sync(up, announce_state="running")` which calls
   `_step_up(sleep, up, ...)` and queues `wake_up_weights`
   via `_send_cmd_with_ack` (acquires `gen_lock` after the
   eviction's `sleep` cmd send already released it, so
   `wake_up_weights` lands AFTER `sleep` on the worker pipe
   -- correct order).
2. **Phase 1 + Phase 2 are atomic w.r.t. the next eviction.**
   The merge under one `gen_lock` closes the secondary
   window where a sentinel published *between* the old
   separate Phase 1 and Phase 2 `with gen_lock` blocks would
   have left `inst.generate(...)` to fire on a not-yet-
   asleep engine but with the next eviction's `sleep`
   already enqueued behind it -- same wedge, different
   racing thread.

Worker pipe order with the fix (race scenario from
2026-05-13):

```
[sleep]                              # eviction enqueues
[sleep, wake_up_weights, restore,    # racing generate gated;
 wake_up_kv_cache, generate]         # post-await, _step_up + Phase 2 enqueue
```

Worker drain:

```
sleep              -> engine asleep on empty pipe
wake_up_weights    -> engine wakes
restore_weights    -> ...
wake_up_kv_cache   -> ...
generate           -> runs on awake engine, ack returned
```

The slot allocator (`Slots`) handles the HBM contention
naturally: the racing generate's Tier-A allocation in
`_step_up(sleep, up)` Phase 1 may fail if the resumer
(`model 12` in the trace) has already taken the freed
share, in which case Tier B/C eviction kicks in.  This is
already the standard wake-up path -- well-tested across the
existing pause/resume and move surfaces.

`_drain_inflight_generates` is preserved as-is for clarity
(the model-16 fix logic is conceptually distinct from the
sentinel-gating); `_evict_for_phase2` inlines the same
snapshot+wait under its own gen_lock acquisition rather
than calling the helper, so the publish and the snapshot
share one lock window.

**Defense-in-depth: worker-side `_dormant` fail-fast.**  The
sentinel is the load-bearing gate (it's the only thing that
can prevent the orchestrator from enqueueing
`inst.generate(...)` onto a worker whose engine is about to
sleep), but the worker also carries a separate `_dormant`
closure flag that brackets the actual unsafe span:

* `_dormant = True` at the bottom of the `sleep` handler in
  `vllm_child._handle_command` (after `llm.sleep(level=2)` +
  `torch.cuda.synchronize` + `torch.cuda.empty_cache`).
* `_dormant = False` at the bottom of the `wake_up_kv_cache`
  handler (after `llm.wake_up(tags=["kv_cache"])`).
* `_submit_generate` checks `_dormant and not _paused`
  BEFORE the existing `_paused` deferral.  When True, it
  logs at `error` level and sends back
  `("generate_done", 0.0, RuntimeError("...sentinel
  breach..."), {"req_id": rid})` instead of touching the
  engine.  The demuxer's standard `error is not None` path
  latches the error, decrements `_pending_count`, and routes
  the listener call to `Orchestrator._on_generate_done` with
  `error` populated; the orchestrator marks
  `q_rec["state"]="error"`, fires `done_event`, and runs
  the slot-release / state-flip branch -- so a sentinel
  regression surfaces as a loud per-request error in the
  dashboard plus a stack trace in the worker log, instead of
  a silent hang inside `engine.step()` on a torn-down
  executor.

The flag was historically present as `_engine_dormant` in
`253ea9d` and removed in `ad74086` ("unify generate-while-
paused on `_saved_requests`") on the invariant that "while
`_paused` is True, the engine is never mutated".  That
invariant is preserved -- the new `_dormant` is gated on
`_dormant and not _paused`, so paused models still flow
through the deferral path unchanged.  The unification's
removal of `_engine_dormant` did not itself open the
residual race (the historical `_engine_dormant` deferral
also wedged the orchestrator, just with a different
symptom: missing `generate_done` ack vs. silent
`engine.step()` hang -- both blocked the orchestrator on
the next sync cmd's `_drain_pipe_generates`); the residual
race was always orchestrator-level and is closed by the
sentinel.  The worker-side flag is purely a fail-fast
diagnostic so any future regression in the orchestrator-
side gate is impossible to miss.

The Phase 2 eviction comment block in `_step_up` explains
the ordering rationale inline alongside the call site; see
the "Eviction-mid-generate dormant-engine wedge" reference
in `vllm_child.py`'s `_submit_generate` for the worker-side
deferral path that motivates the entire wedge family.

### Pause-during-resume future-chain break  [resolved]

A `cl.pause_all(); cl.resume_all()` issued while an earlier
`resume_all()` was still walking the ladder up could wedge a
model at `state="running"` over an idle engine (no slot held,
`_inflight=[]`, `paused=False`) -- every subsequent
`cl.move(model, ...)` then raised `model 'X' is currently
running a generate; wait for it to finish before calling
move()` and the user could never move the model again.
Symptom in `outt` (2026-05-13 reproduction): `model 7` logged
two consecutive `model 7: resumed (state=running)` lines
~25 seconds apart, the second one with no preceding
`model 7: resume received` between them, and no `model 7:
generate done` ever following.  The orchestrator log later
shows `WARNING model 7: prior move chain raised (model
'model 7' is currently running a generate ...)` from the
user's final `move("checkpoint")`.

**Cause.**  `pause()` deliberately opts out of chaining on
`_futures[mid]` (see "Pause-during-drain" in
`async_generate_DETAILS.md`) so that a pause can break out
of a wait it would otherwise be queued behind.  But the old
`pause()` *also* unconditionally **published** its own
future at `_futures[mid]`, even when the pause's body
would turn out to be a no-op.  In the
`pause / resume_A / pause / resume_B` sequence:

```
P1 (real): _futures[mid] = P1.future, body commits paused=True
R_A:       prev = P1.future; _futures[mid] = R_A.future
            -- R_A walks the ladder up (typically 15-30s)
P2 (mid R_A): _futures[mid] = P2.future
            -- P2 body sees paused=True, logs "already paused,
               skipping", returns immediately
R_B:       prev = P2.future (DONE);  _futures[mid] = R_B.future
            -- R_B's _await_prev returns immediately because
               P2 is already done.  R_B starts walking the
               ladder in parallel with R_A.
```

Both `_resume_sync`s are now alive on the same model.  R_A
completes its walk first, sends `resume`, the engine
restores the saved subreqs, drains them naturally, and the
model goes `running -> up`.  Meanwhile R_B is still in its
own `_move_sync`; by the time it actually executes, the
model has been further evicted to `sleep` and then
checkpointed.  R_B's `_move_sync` then walks
`checkpoint -> ... -> up` and sends `resume` to a worker
whose engine is no longer paused -- the child reports
`resume: restored=0 synthesized=0 was_paused=False`.  R_B
finishes by publishing `state="running"` over an idle
engine with no `_inflight` work and no pending requests on
the child.  `move()`'s `current == "running"` guard then
fires on every subsequent move chain.

The link in the chain that broke was P2's
unconditional `_futures[mid] = P2.future`: P2's body did
nothing observable, but its future-publish replaced the
still-running R_A as the apparent "latest op", letting R_B
slip through `_await_prev` without serialising on R_A.

**Fix.**  `pause()` now only publishes its future to
`_futures[mid]` when the pause is going to commit a state
change.  When the call will be a no-op (model is already
paused, or `state != "running"` so there is nothing to
freeze), `_futures[mid]` is left pointing at whatever
long-running op is in flight -- typically an in-flight
`_resume_sync`'s future:

```python
fut = Orchestrator._pool.submit(
    Orchestrator._pause_sync, model_id,
)
if not entry.get("paused") and entry.get("state") == "running":
    Orchestrator._futures[model_id] = fut
```

Walking the model 7 timeline with the guard in place:

```
P1 (real): _futures[mid] = P1.future, body commits paused=True
R_A:       prev = P1.future;  _futures[mid] = R_A.future
            -- R_A walks the ladder up (still running)
P2 (mid R_A): paused=True -> SKIP publish.
              _futures[mid] STAYS at R_A.future
            -- P2 body still runs and still logs "already
               paused, skipping" (the body is unchanged);
               it just isn't a chain barrier.
R_B:       prev = R_A.future;  _futures[mid] = R_B.future
            -- R_B's _await_prev now waits for R_A to
               finish.  R_A clears paused.  R_B body sees
               paused=False, logs "not paused, resume is a
               no-op", returns.
```

The fix is purely on the "do we publish the future?" line in
`pause()`; the `_pause_sync` body, all its no-op branches,
the deferred-pause guard, and the worker `pause` cmd
semantics are unchanged.  Concurrent real pauses (two pauses
that both think they will commit state) are unaffected --
their bodies still go through the same precondition checks
and the worker's idempotent `pause` cmd tolerates the
duplicate.

The trade-off is one acknowledged narrow window: when a
real pause races with a still-committing earlier pause (both
see `paused=False, state="running"` before either has
flipped the registry), both publish their futures.  This is
no worse than the pre-fix behaviour and is already covered
by `_pause_sync`'s precondition re-checks under `gen_lock`.

## Console Output

The orchestrator separates verbose logs from high-level status messages.

**Verbose logs** go to `stdout` (which `demo.py` redirects to a log
file via `os.dup2`).  These include detailed orchestrator internals,
worker/child process output, and library warnings.

**Console messages** go through the `_console()` helper, which writes
to a separate stream that stays visible to the user.  Console messages
are concise, one-line status updates:

```
qwen3-8b: register received
qwen3-8b: registered (wait=0.0s, cold-start=45.2s, total=45.2s)
  models: qwen3-8b[saved]
qwen3-8b: generate received
qwen3-8b: placed on GPU 0
qwen3-8b: saved -> up (3.1s)
  models: qwen3-8b[up:gpu0]
qwen3-8b: generated (1.2s) -> "Albert Einstein..."
qwen3-8b: remove received
qwen3-8b: up -> saved (2.4s)
  models: qwen3-8b[saved]
qwen3-8b: removed
  models: (none)
```

### `_console()` stream resolution

`_console()` writes to the first available of:

1. `_console_stream` -- module-level override (set by Jupyter notebooks
   to `sys.stderr` so output appears in the active cell).
2. `_terminal` -- a dup of `sys.stderr` taken at import time, before
   any `os.dup2` redirection.  In `demo.py`, this preserves the
   original terminal fd.
3. `sys.stderr` -- fallback.

### Output redirection in `demo.py`

`demo.py` redirects fd 1 (stdout) and fd 2 (stderr) to a timestamped
log file via `os.dup2`.  Forked worker processes inherit these
redirected fds, so all child output (vLLM, CUDA warnings, tqdm) goes
to the log.  `_terminal` (duped before redirection) keeps console
messages on the real terminal.

### Output redirection in `demo.ipynb`

Jupyter's `sys.stderr` is a ZMQ `OutStream` that routes to the active
cell.  The notebook sets `orchestrator._console_stream = sys.stderr`
after importing, so `_console()` output appears in the cell that runs
the code.

## Process Cleanup

Teardown kills the entire process tree (worker + vLLM child +
descendants) using `_kill_process_tree`, which walks descendants
bottom-up via `psutil` and sends SIGKILL.  `Instance._reset()` also
force-kills the worker if it doesn't exit within the join timeout.

## Dashboard

The dashboard (`dashboard.py`) is a standalone curses terminal UI that
polls `GET /state` from the orchestrator's embedded HTTP server
(`state_server.py`).  Each model row shows its memory footprint and,
when applicable, its slot level:

- **Live rows** (`sleep`, `up`, `running`): `"(67.2G, L2)"` when the
  model holds a slot, `"(67.2G)"` when slotless.
- **Image cache rows** (`saved`): `"(40.0G, L2)"` showing the
  model's intrinsic level (the level it *would* allocate at when
  woken).

The slot allocator state is exposed under `/state -> "slots"` as a
per-GPU list of leaves (`{level, index, alloc}`) so the dashboard can
render the buddy tree as a horizontal bar.  `slot_waiters` reports
the size of the FIFO waiter queue.

### Recording and replay

`--record FILE` writes each polled state snapshot to a JSONL file.
Recording starts only after the first generate request appears in
the state.  `--replay FILE` replays a recorded JSONL file in real
time.

### Timestamps

All relative timestamps in the state snapshot (`submit_rel_s`,
`gen_start_rel_s`, `done_rel_s`) are relative to `t0`, which is
anchored to the first `generate()` call (not orchestrator init).

## File Structure

```
semi_persistence/
  orchestrator.py          -- Orchestrator class
  orchestrator_DESIGN.md   -- This file
  pipeline.py              -- ModelPipeline + Op subclasses
  pipeline_DESIGN.md       -- Pipeline design (post-migration)
  slots.py                 -- Buddy allocator (Slots singleton)
  slots_DESIGN.md          -- Buddy allocator design
  instance.py              -- Instance class (see instance_DESIGN.md)
  worker.py                -- Worker process + child thread
  vllm_child.py            -- Spawned vLLM child process
  state_server.py          -- Embedded HTTP server (GET /state)
  dashboard.py             -- Curses dashboard (--record / --replay)
  demo.py                  -- CLI demo script (stdout/stderr -> log file)
  demo.ipynb               -- Jupyter notebook demo
  CRIU_PLUMBING.md         -- CRIU complications and fixes
```
