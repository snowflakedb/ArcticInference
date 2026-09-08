# Async Generate -- Implementation Details

## Architecture Overview

Four processes/threads form the pipeline, connected by three
queue/pipe boundaries:

```
Orchestrator          Instance            Worker _child_thread     vLLM Child
(ThreadPoolExecutor)  (+ Demuxer thread)  (worker process)         (child process)
      |                    |                     |                       |
      |               cmd_queue (mp)        child_queue (thd)       pipe (mp)
      |              ──────────────►       ──────────────►       ──────────────►
      |              result_queue (mp)     (emit_result)         pipe (mp)
      |              ◄──────────────       ◄──────────────       ◄──────────────
                            ▲
                            │ Demuxer is the SOLE consumer of result_queue;
                            │ applies results, decrements _pending_count,
                            │ notifies wait() callers, and dispatches to
                            │ orchestrator-installed cmd / catch-all listeners.
```

## What Is Async

All four layers support fully concurrent generates for the same model:

- **Child engine loop**: drives `LLMEngine` via `add_request()` +
  `step()` instead of blocking `LLM.generate()`.  When a generate
  command arrives, the child drains ALL pending generate commands
  from the pipe before the first `step()`, so they all get
  `add_request`'d to the engine and decode together in every GPU
  forward pass.  New generates arriving mid-decode are picked up
  via `pipe_conn.poll(0)` between steps and added to the batch.
- **Worker fire-and-forget**: sends generate to child pipe without
  waiting for a response; `_get_next_command()` polls both the pipe
  (for `generate_done` results) and the child_queue (for new
  commands) concurrently.
- **Orchestrator per-request Events**: each `_generate_sync` thread
  submits its request and waits only on its own `threading.Event`,
  not on `inst.wait()`.  Each Instance owns a permanent **demuxer
  thread** that reads from `inst._result_queue` and dispatches each
  result to per-cmd listeners; the orchestrator installs a
  `cmd="generate"` listener that resolves the matching per-request
  Event as results arrive.  Multiple generates for the same model
  run truly concurrently -- the second does not wait for the first
  to finish.
- **Orchestrator future separation**: generate futures are stored
  separately from move/register futures.  Generates chain only
  behind moves (to ensure the model is up), not behind other
  generates.

### Concurrency model

```
orch.generate(model, promptA, 5000)  -- submits immediately
orch.generate(model, promptB, 1000)  -- submits immediately, does NOT wait for A

Child engine batches A and B:
  step()  →  decodes 1 token for A AND 1 token for B
  step()  →  decodes 1 token for A AND 1 token for B
  ...
  B finishes at ~2s  →  generate_done(B) sent
  A finishes at ~10s →  generate_done(A) sent

Total wall time ≈ 10s (max), not 12s (sum)
```

### Serialization points (by design)

- **Move-up under gen_lock**: when a model is not yet `"running"`,
  the first `_generate_sync` thread holds `gen_lock` during the
  move-up (load, restore, wake_up).  Other generate threads block
  here until the model is ready.  Once `"running"`, the lock
  acquisition is instant (state check + skip).
- **Move/remove waits for generates**: `move()` and `remove()`
  snapshot every `done_event` in `_inflight[model_id]` under
  `gen_lock` and the corresponding `_move_sync` / `_remove_sync`
  waits on each before stepping the model down.  This drains
  *all* concurrently-inflight generates (not just the last
  submitted one), so a generate that was paused mid-flight and
  re-prefilled by a later `resume` still gates the move
  correctly.  `move()` *also* snapshots
  `_last_generate_future[mid]` under the same `gen_lock` and
  passes it as `prev_gen_future` so `_move_sync` waits on the
  most recent `_generate_sync`'s full lifecycle -- including
  Phase 1 (wake-up) and the brief inter-phase window where
  `_inflight` is not yet populated -- before issuing its own
  `_send_cmd_with_ack("sleep")`.  See "Move-vs-generate
  Phase-1 race" below for the bug shape this closes.  Paused
  models skip both waits (their inflight events / generate
  futures can only complete via a future `resume`, so blocking
  on them would deadlock `paused → sleep`);
  `move(target='saved')` and `remove()` are explicitly
  refused on paused models for the same reason.

## Backward Compatibility with Old CRIU Images

CRIU restores a process from the exact binary/bytecode that was
checkpointed.  Images dumped with the old code run `llm.generate()`
(synchronous) and respond with `("generate", elapsed, error, info)`.
Images dumped with the new code run the engine loop and respond with
`("generate_done", elapsed, error, info)`.

The worker's `_handle_pipe_result()` accepts **both** response types:

```python
if result[0] in ("generate_done", "generate"):
    _pending_generates -= 1
    _emit_result("generate", result[1], result[2], result[3])
```

No re-dump is needed for correctness, but **re-dumping is required to
get the async engine benefit** (concurrent decoding in the child).
Old images serialize generates inside the child via blocking
`llm.generate()`.

## Waiting and Polling Points

### vLLM Child (vllm_child.py)

The child main loop has **two modes**: idle and active.

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| Main loop, idle path | `pipe_conn.recv()` | **Yes** | No engine work, no CPU to spend. Wakes when worker sends a command. |
| Main loop, active path | `pipe_conn.poll(0)` | **No** | Engine is stepping; check if a new command arrived without blocking. |
| `engine.step()` | GPU kernel execution | **Yes** -- GPU-bound | Decodes one batch of tokens. This is the productive work. |
| Generate pipe drain | `pipe_conn.poll(0)` in loop | **No** | After receiving a generate, drains all additional generates from the pipe before the first `step()`. |

When active, the child alternates:
```
engine.step()  →  poll(0)  →  [if data: recv+handle]  →  engine.step()  → ...
```
When idle (no engine requests), it blocks on `recv()` with zero CPU.

### Worker _child_thread (worker.py)

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| `_get_next_command()`, no pending | `child_queue.get()` | **Yes** | No async work, wait for next command. |
| `_get_next_command()`, pending | `pipe.poll(0)` + `child_queue.get(timeout=0.01)` | **Polls** -- 10ms | Must monitor both pipe (for generate_done) and child_queue (for new commands). |
| `_drain_pipe_generates()` | `pipe.recv()` | **Yes** | Called before CRIU ops. Blocks until all in-flight generates complete. |
| `_recv_sync()` | `pipe.recv()` | **Yes** | Waiting for a synchronous command response. Transparently consumes any generate completions that arrive first. |

The 10ms poll loop only runs when `_pending_generates > 0`.

### Instance (instance.py)

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| `_send()` | `cmd_queue.put()` + `Demuxer.notify_send()` | **No** | All primitives are fire-and-forget. |
| `wait()` | `Demuxer.wait_idle()` (condvar on `_pending_count`) | **Yes** | Parks on the demuxer's condition variable; woken whenever `_pending_count` is decremented; safe under any number of concurrent waiters because the demuxer is the sole consumer of `result_queue`. |
| `Demuxer._loop` | `result_queue.get(timeout=0.5)` | **Yes** | Background thread, the **only** reader of `result_queue` for the entire instance lifetime. |

### Orchestrator (orchestrator.py)

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| `_generate_sync` Phase 1 | `gen_lock` + `_move_sync` | **Yes** (first thread) | Ensures model is running. Others skip if already running. |
| `_generate_sync` Phase 2 | `gen_lock` + `inst.generate()` | **Short lock** | Drains queue, submits to engine. The orchestrator's permanent generate listener (installed at `_step_up(saved -> checkpoint)` via `_install_listeners`) handles every ack -- there is no per-batch waiter to spawn. |
| `_generate_sync` Phase 3 | `event.wait()` | **Yes** | Blocks only until THIS request completes. Other requests resolve independently. |
| Generate listener (`_on_generate_done`) | runs on the Instance's demuxer thread | n/a | Per-cmd callback fired by the demuxer for every `cmd="generate"` ack: matches `req_id`, sets the per-request `done_event`, and (when no more inflight + `_pending_count == 0`) releases the slot and flips state `running -> up`.  The `_pending_count` guard is conservative: when a peer's `_send_cmd_with_ack("sleep")` is queued on the worker behind this generate, the guard skips the release; the late `_step_down(up, sleep)` is the secondary release path that reconciles the slot and state.  See `orchestrator_DESIGN.md` "Slot release at `up -> sleep`". |
| `_send_cmd_with_ack` | FIFO `threading.Event` per `(model_id, cmd)` | **Yes** | Sends a non-`generate` command and blocks on the head Event of that cmd's FIFO. The demuxer's catch-all listener pops and signals the head as each non-generate ack arrives.  Used by every step-up/step-down/pause/resume path. |

## Data Flow for Concurrent Generate

```
Time →

Orchestrator         Instance            Worker               Child
    |                    |                   |                    |
    |  generate(A)       |                   |                    |
    |  generate(B)       |                   |                    |
    |──────────────► cmd_queue              |                    |
    |                    | ──────────► child_queue               |
    |                    |                   |──► pipe("generate",A)
    |                    |                   |──► pipe("generate",B)
    |                    |                   |    (fire-and-forget)
    |                    |                   |                    |
    |                    |                   |     recv A
    |                    |                   |     drain pipe → recv B
    |                    |                   |     engine.add_request(A)
    |                    |                   |     engine.add_request(B)
    |                    |                   |                    |
    |                    |                   |     engine.step() ── batched (A+B)
    |                    |                   |     engine.step() ── batched (A+B)
    |  event_A.wait()    |                   |     ...             |
    |  event_B.wait()    |                   |                    |
    |                    |                   |     B finishes:    |
    |                    |                   | ◄── pipe("generate_done",B)
    |                    |   _get_next_cmd   |                    |
    |                    |   polls pipe ──────┘                    |
    |                    | ◄── result_queue("generate",B)         |
    |                    | demuxer drains    |                    |
    |  generate listener |                   |     engine.step()   |
    |  event_B.set()     |                   |     A finishes:    |
    |                    |                   | ◄── pipe("generate_done",A)
    |                    | ◄── result_queue("generate",A)         |
    |                    | demuxer drains    |                    |
    |  generate listener |                   |                    |
    |  event_A.set()     |                   |                    |
    |                    |                   |                    |
    |  both futures      |                   |                    |
    |  resolve           |                   |                    |
```

Generates arriving later (while the engine is active) are picked up
via `poll(0)` between steps and added to the running batch
immediately.

## IPC Protocol

### Worker → Child (pipe)

| Message | Response | Behavior |
|---------|----------|----------|
| `("generate", {"req_id", "prompts", "sampling_params"})` | None (fire-and-forget) | New child: adds to engine, no immediate reply. Old child: responds synchronously with `("generate", ...)`. |
| `("sleep", {})` | `("sleep", elapsed, error, info)` | Synchronous. Child drains engine first. |
| `("exit", {})` | `"exit_ack"` | Synchronous. Child drains engine first. |
| All other commands | `(cmd, elapsed, error, info)` | Synchronous request-response. |

### Child → Worker (pipe, unsolicited)

| Message | When |
|---------|------|
| `("generate_done", elapsed, error, {"req_id", "outputs", ...})` | After all prompts in a req_id finish in the engine |

### Worker → Instance (result_queue)

All results use the same 4-tuple `("generate", elapsed, error, info)`.
The worker maps both `"generate_done"` (new child) and `"generate"`
(old child) to `"generate"` on the result_queue.

## CRIU Safety (Drain Points)

Before any CRIU-sensitive operation, all in-flight generates must
complete.  Drain happens at **both** layers:

| Layer | Operation | Drain mechanism |
|-------|-----------|-----------------|
| **Child** | `sleep` | `_drain_engine()` before `llm.sleep()` |
| **Child** | `prepare_criu_dump` | `_drain_engine()` before FD cleanup |
| **Child** | `exit` | `_drain_engine()` before `exit_ack` |
| **Worker** | `checkpoint` | `_drain_pipe_generates()` before `cuCheckpointProcess` |
| **Worker** | `save` | `_drain_pipe_generates()` before CRIU dump |
| **Worker** | `teardown` | `_drain_pipe_generates()` before kill |
| **Worker** | `exit` | `_drain_pipe_generates()` before sending exit to child |
| **Worker** | Any sync cmd | `_drain_pipe_generates()` before `pipe.send(cmd)` |

## Orchestrator Concurrency Details

### Per-model state

- `_generate_locks[model_id]`: serializes Phase 1 (move-up) and
  Phase 2 (submit).  Short hold when model is already running.
- `_inflight[model_id]`: list of `(req_id, req_record, Event)`
  for in-flight requests.  Modified under `gen_lock`.
- `_cmd_ack_events[(model_id, cmd)]`: per-cmd FIFO `deque` of
  pending `threading.Event`s installed by `_send_cmd_with_ack` and
  popped/signalled by the demuxer's catch-all listener.
- `_last_generate_future[model_id]`: last submitted generate
  `Future`, used by:
  * `wait(mid)` / `wait_all()` for the caller-facing "block
    until the most recent generate finishes" aggregation.
  * `move()` (snapshotted under `gen_lock` alongside
    `prev_gen_events`, passed as `prev_gen_future`) so
    `_move_sync` waits on the most recent generate's full
    lifecycle including Phase 1 and the inter-phase window
    where `_inflight` is not yet populated.  Closes the
    Phase-1-vs-sleep race (see "Move-vs-generate Phase-1
    race").
  * `pause` / `resume` / `remove` do **not** chain on this:
    pause/resume opt out because the future is parked on the
    very generate they would interrupt or unfreeze; remove
    refuses on paused models, so `_inflight` is sufficient
    coverage for non-paused removes.

### Demuxer listener lifecycle

The orchestrator installs **two** demuxer listeners on every
Instance the first time the model is reanimated from `saved`
(in `_step_up(saved -> checkpoint)` via `_install_listeners`):

- `cmd="generate"` -> `_on_generate_done`: matches `req_id`,
  copies token counts onto the request record, sets the
  per-request `done_event`, and -- when `_inflight[model_id]` is
  empty and `inst._pending_count == 0` -- releases the slot and
  flips `running -> up` under `gen_lock` then `model_lock`.
  When the `_pending_count` guard fails (a peer-eviction
  `sleep` was queued on the worker behind this generate), the
  slot release falls through to `_step_down(up, sleep)`'s
  post-sleep reconcile (see `orchestrator_DESIGN.md` "Slot
  release at `up -> sleep`").
- `cmd=None` (catch-all) -> `_on_cmd_ack`: pops the head Event
  from `_cmd_ack_events[(model_id, cmd)]` (if any) and `set()`s
  it.  Skips `cmd == "generate"` so generate acks aren't double-
  handled.

Both listeners persist for the life of the Instance and are
destroyed naturally on teardown when the demuxer stops and the
Instance is GC'd.  No per-batch waiter is spawned.

## Result Attribution

Each `inst.generate()` assigns a unique `req_id` (e.g.
`"inst0-3"`) and stores it in `inst.last_req_id`.  The orchestrator
captures this at submission time.

Results are stored in `inst.generate_results[req_id]` (keyed dict
with `outputs`, `prompt_tokens`, `completion_tokens`).  The
generate listener matches results by `req_id` and sets each
request's Event independently.

## `inst.wait()` vs `_send_cmd_with_ack`

Two ways the orchestrator blocks on a child-side command's ack.
They look interchangeable but differ in **what "done" means** and
**which scope of cmds they synchronise on**.  Both are now safe to
mix freely -- the per-instance demuxer is the single result-queue
consumer for the entire Instance lifetime, so no caller ever
competes with another for `result_queue.get()`.

### The invariant: single demuxer consumer

`inst._result_queue` has exactly one consumer: the per-instance
`Demuxer` thread, alive between `_ensure_queues` and
`_close_queues`.  `inst.wait()` is a thread-safe condvar wait on
the demuxer's `_pending_count`; `_send_cmd_with_ack` parks on a
per-cmd Event signalled by the demuxer's catch-all listener.
Neither path reads from `_result_queue` directly, and neither can
deadlock by racing another caller for `get()`.

### Comparison

| Aspect | `inst.wait()` | `_send_cmd_with_ack(mid, cmd, ...)` |
|--------|---------------|-------------------------------------|
| Queue read | None -- condvar wait on demuxer's `_pending_count` | None -- demuxer's catch-all listener pops the FIFO head Event for this `(model_id, cmd)` and `set()`s it |
| "Done" condition | every pending cmd has been acked (`_pending_count == 0`) | this specific `(model_id, cmd)` ack arrived |
| Safe under concurrent callers | **Yes** -- arbitrary number of waiters all park on the same condvar | **Yes** -- each sender installs a fresh Event in the FIFO; demuxer pops in send order |
| Safe with paused-but-deferred generate | **No** -- `_pending_count` includes the deferred generate, so it never reaches 0 until resume | **Yes** -- only this cmd's ack is awaited |

### Why `_pending_count == 0` is the wrong condition under pause

A paused model has at least one in-flight `generate` whose
`generate_done` is parked in the child until `resume`.  That generate
counts toward `inst._pending_count`.  Any pool-worker
`inst.wait()` issued by `_step_up` / `_step_down` (e.g. for `sleep`,
`cuda_checkpoint`, `cuda_restore`, `wake_up_kv_cache`, ...) would loop
forever waiting for `_pending_count` to drain to 0 -- which won't
happen until resume.

`_send_cmd_with_ack` sidesteps this entirely: it only cares whether
*its* cmd's ack arrived, regardless of how many other generates are
still pending.

### Mechanism

```
pool worker                       demuxer thread (per Instance)
-----------                       -----------------------------
with gen_lock:                     loop:
    with _cmd_ack_lock:                result = result_queue.get(timeout=0.5)
        q = _cmd_ack_events            cmd, elapsed, error, info = result
              .setdefault((mid,cmd),
                          deque())    inst._apply_result(cmd, info)
        q.append(ev)                  with _pending_cv:
    inst.<cmd>(...) # bumps                _pending_count -= 1
                    # _pending_count       _pending_cv.notify_all()
                                       # listener dispatch:
                                       #   generate listener (cmd="generate")
                                       #   catch-all listener:
                                       #     pop head Event of
                                       #     _cmd_ack_events[(mid, cmd)]
                                       #     and ev.set()
ev.wait()
```

`gen_lock` around `install-event + send-cmd` is what makes this
FIFO-ordered: every `_send_cmd_with_ack` for the same `(mid, cmd)`
installs its Event under `_cmd_ack_lock` and then sends under
`gen_lock`, so the install-then-send ordering is preserved across
concurrent senders.  The demuxer pops `_cmd_ack_events` head Events
in arrival order, so the FIFO matches the send order.

`_get_generate_lock` returns an `RLock` so the same thread can hold
`gen_lock` in `_generate_sync` Phase 1, recurse into `_move_sync` ->
`_step_up` / `_step_down` -> `_send_cmd_with_ack`, and re-acquire it
without self-deadlock.

### Where each is used

| Caller | Path | Mechanism | Notes |
|--------|------|-----------|-------|
| `_register_sync` | child boot / image restore | `inst.wait()` | Cold start; no listeners installed yet. The condvar wait is independent of listener state. |
| `_step_up(saved -> checkpoint)` | `inst.criu_restore().plan_restore_weights().wait()` | `inst.wait()` | First time the model is reanimated; listeners are installed on the new Instance just before this chain so subsequent steps can use `_send_cmd_with_ack`. |
| `_step_up`, `_step_down` (every cmd: `sleep`, `unpin`, `cuda_checkpoint`, `cuda_restore`, `repin`, `wake_up_weights`, `restore_weights`, `wake_up_kv_cache`) | move ladder, possibly under a paused model | `_send_cmd_with_ack` | Always synchronises only on its own cmd's ack, so it works whether or not generates are deferred. |
| `_step_down(checkpoint -> saved)` | `inst.teardown().wait().remove()` | `inst.wait()` | Demuxer applies the teardown ack (calls `_reset()`) before notifying the wait; the wait then returns and the inst is GC'd. |
| `_pause_sync`, `_resume_sync` | pause/resume a generating model | `_send_cmd_with_ack` | Issued mid-generate by definition; deferred generate keeps `_pending_count` non-zero so only the FIFO ack path works. |
| `_generate_sync` | submit a new generate | per-request `Event` (not `inst.wait()`) | The orchestrator's permanent generate listener resolves the matching `req_id`. |

### Lock ordering rules

The orchestrator holds five distinct locks across its critical
path.  Concurrency safety hinges on a single global acquisition
order; every code path either follows it or releases the
violating lock before reacquiring lower-rank ones.

#### Lock inventory

| Lock                              | Scope             | Type        | Guards                                                                                                  | Long holds                                           |
|-----------------------------------|-------------------|-------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `gen_lock(mid)`                   | per-model         | `RLock`     | `_inflight[mid]` snapshots, `inst._pending_count` reads, ack-event installation, `inst.<cmd>(...)` send | Held across the worker pipe send; **dropped** before `ev.wait()` |
| `model_lock(mid)` = `entry["_lock"]` | per-model      | `RLock`     | `entry["state"]`, `entry["slot"]`, `entry["gpu"]`, `entry["paused"]`, `entry["instance"]`              | Brief atomic mutations only; never across cmd sends or `ev.wait()` |
| `_cmd_ack_lock`                   | orchestrator-wide | `Lock`      | `_cmd_ack_events: dict[(mid, cmd), deque[Event]]`                                                       | None -- single deque op then released                |
| `_request_lock`                   | orchestrator-wide | `Lock`      | `_request_log`, `_request_counter`                                                                      | Bounded list scan over in-flight requests            |
| `Slots._cv` / `Slots._lock`       | orchestrator-wide | `Condition` | slot pool / waiters / draining                                                                          | Released across `cv.wait()` -- `Slots.allocate` may park here for arbitrarily long |

Two more locks exist but are leaves outside the orchestrator
critical path: `Demuxer._pending_cv` (per-instance, guards
`_pending_count` for `wait_idle`; demuxer drops it before firing
listeners) and `Demuxer._listener_lock` (per-instance, guards the
listener registry; held only for the snapshot copy).  Neither is
ever held while acquiring any orchestrator lock.

#### Global acquisition order

Top is acquired first; arrows point outer -> inner.  Locks at
the same depth are independent (never held simultaneously
across paths).

```
   gen_lock(mid)            <-- outermost orch lock
         |
         +-- _cmd_ack_lock  (brief leaf; only inside gen_lock)
         |
         v
  model_lock(mid)            <-- via _locks_ordered(*ids)
         |
         v
  _request_lock              (brief leaf; may also be taken alone)


  Slots._cv                  (independent; NEVER held while
                              acquiring any orch lock above)
```

Reading top-down: a thread may go from `gen_lock` to
`model_lock` to `_request_lock`, but never the reverse.
`_cmd_ack_lock` is a brief leaf paired only with `gen_lock`.
`_request_lock` is also a leaf and is often taken with no orch
lock held at all (e.g. in `submit_generate`).  `Slots._cv` is
strictly independent: `Slots.allocate` is called while holding
no orchestrator lock, and `Slots.deallocate` -- which is called
under `model_lock` -- never reaches back into orchestrator
locks, so the two domains never form a cycle.

Cross-model locking is permitted only via
`_locks_ordered(*model_ids)`, which sorts the ids and asserts
`len(set) <= 2`.  This bounds the recursion and gives a global
total order over `model_lock` instances, so two threads
acquiring `model_lock(a)` and `model_lock(b)` from opposite
sides cannot deadlock.  `gen_lock` is *never* paired across
models -- if you find yourself wanting `gen_lock(a)` and
`gen_lock(b)` simultaneously, redesign instead.

#### Why this order

Three independent invariants pin the ordering:

1.  **Demuxer generate-done listener takes `gen_lock` then
    `model_lock`.**  The listener pops `_inflight[mid]` (under
    `gen_lock`), and on the trailing edge releases the slot and
    flips state `running -> up` (under `model_lock`).  Anything
    else that touches both must follow the same order or risk
    AB-BA against the listener.

2.  **`_send_cmd_with_ack` takes `gen_lock` (and briefly
    `_cmd_ack_lock`).**  It installs a FIFO ack event under
    `gen_lock`, calls `inst.<cmd>(...)` (which tells the worker
    over a Pipe), and parks on `ev.wait()` -- the wait itself is
    **outside** `gen_lock` so the demuxer can take `gen_lock`
    when the ack arrives and pop the head of the FIFO.  This is
    why `gen_lock` must outrank `model_lock`: the cmd send
    happens while holding `gen_lock` only.

3.  **`Slots.allocate` parks on `Slots._cv.wait()` and is woken
    by `Slots.deallocate` in arbitrary peer threads.**  Holding
    any per-model lock across `Slots.allocate` would deadlock if
    the deallocator needs that lock to make progress
    (`_step_down` -> `Slots.deallocate` is itself called under
    `model_lock`).  So `Slots.allocate` runs lock-free; the
    returned slot is committed under `model_lock` afterwards.

#### Concrete enforcement points

The patterns below are not optional -- each one prevents a
specific deadlock or torn-state bug we've already hit.

-   **`_step_up` / `_step_down` release `model_lock` across
    every `_send_cmd_with_ack`.**  The cmd send takes `gen_lock`
    internally; holding `model_lock` across it would invert the
    listener's `gen_lock -> model_lock` order.  Each step
    re-acquires `model_lock` only for the small atomic
    `slot`/`gpu`/`state` mutation between sends.

-   **`_step_up` Phase 2 eviction (incumbent != self) takes
    `model_lock(incumbent)` only for the validation check, then
    releases it before calling `_step_down(incumbent, ...)`.**
    Holding `model_lock(incumbent)` across `_step_down`
    deadlocks against the incumbent's own thread when it is
    parked in `_send_cmd_with_ack` while still holding
    `gen_lock(incumbent)` (typical pattern: the incumbent is
    self-evacuating from `_acquire_slot_for_running`).  See
    "Eviction lock ordering (Phase 2 of `_step_up`)" below for
    the worked timeline.

-   **The demuxer's generate-done listener (`_on_generate_done`)
    takes `gen_lock` outside `model_lock`.**  All listener-side
    state inspection (`_inflight[mid]`, `_pending_count`)
    happens under `gen_lock`; `_locks_ordered(model_id)` is
    entered only for the slot release / state flip on the
    trailing edge.

-   **`_send_cmd_with_ack` waits on the per-cmd `Event` outside
    `gen_lock`.**  If the wait happened under `gen_lock`, the
    demuxer's catch-all listener (which takes
    `_cmd_ack_lock` -- ranked under `gen_lock`) could still pop
    and set the Event, but a concurrent generate-done listener
    couldn't acquire `gen_lock` to release the slot, freezing
    every running model on this instance.  The implementation
    enters `gen_lock` only for the FIFO append + send; the
    `ev.wait()` is unguarded.

-   **`_pause_sync` re-checks `_inflight[mid]` under `gen_lock`
    before sending `pause`** (deferred-pause / phantom-running
    guard).  The check must be under `gen_lock` so it serialises
    against the generate-done listener: an unguarded check could
    observe a non-empty `_inflight` between the listener's pop
    and its slot release, then mint a phantom-paused entry.

-   **`_resume_sync` snapshots `_inflight[mid]` under
    `gen_lock` before doing any work** (phantom-running
    guard).  Same reasoning -- if empty, clear `paused = False`
    and leave the model wherever the user parked it (no walk-
    up, no slot acquisition, no worker `resume` cmd).  The
    snapshot must happen *before* the optional ladder walk so
    we don't pay the `_step_up` cost when there is nothing to
    drive.

-   **`Slots.allocate` / `Slots.try_allocate` are called outside
    `_locks_ordered(*)`.**  `_acquire_slot_for_running` enters
    `_locks_ordered(model_id)` only for the post-allocate
    commit; the validate-then-retreat dance issues `_step_down`
    *outside* the lock.  Slot release (`Slots.deallocate`) is
    permitted under `model_lock` because it never blocks.

-   **`_request_lock` is acquired only briefly and is always
    leaf.**  `_set_state`, `_pause_sync`, `_resume_sync`, and
    `submit_generate` take it for `_request_log` mutations;
    every call site already holds (or doesn't hold) the
    orchestrator locks above and never reverses.

-   **`_cmd_ack_lock` is acquired only inside `gen_lock`** (in
    `_send_cmd_with_ack` for the FIFO append and in
    `_on_cmd_ack` for the FIFO pop).  No other code touches
    `_cmd_ack_events`, so the order is preserved by
    construction.

-   **`gen_lock` is per-model and never paired across models.**
    No API ever holds `gen_lock(a)` while attempting
    `gen_lock(b)`.  Cross-model coordination always happens
    through `model_lock` (via `_locks_ordered`) or through
    lock-free reads of `_registry`.

### Chain-prev resilience

Per-model orchestrator ops (`move`, `pause`, `resume`,
`submit_generate`, `remove`) chain on `_futures.get(model_id)` so
they serialise behind any in-progress prior op.  The chain await
goes through a single helper:

```python
@staticmethod
def _await_prev(model_id: str, label: str, prev_future) -> None:
    if prev_future is None:
        return
    try:
        prev_future.result()
    except Exception as exc:
        log.warning("%s: prior %s raised (%s); proceeding",
                    model_id, label, exc)
```

This is the safety net for any failure mode that could otherwise
silently mute user actions.  When a prior op raised (e.g. a
transient infrastructure error, or any future variant of
`_move_sync` that raises before completing), the unhandled
re-raise here used to abort the next user-visible op silently --
e.g. a `cl.pause()` would log `pause received` and then nothing,
because `_pause_sync` propagated the prior op's exception through
its own `Future` and never reached its body.

Each call site keeps its own pre-condition check (`_pause_sync`'s
`state != "running"` no-op, `_move_sync`'s `current == "running"`
raise, `_resume_sync`'s `paused` and
`state in ("up", "sleep", "checkpoint")` checks) under the
model lock, so a logged-and-skipped prior failure is safe: the
successor decides what to do based on the *current* world.
Successful chains are unaffected.

`pause` is the one op that opts out of the chain entirely (see
"Pause-during-drain" below) -- chaining `pause` on
`_futures[mid]` has the same deadlock shape as chaining on
`_last_generate_future[mid]` whenever the prior future is itself
parked on this model's in-flight generate.

### Pause-during-drain (chain-vs-interrupt)

`pause` is fundamentally an **interrupt**: its job is to stop
the very generate that other operations are waiting for.  That
makes it incompatible with the chain-prev pattern in two
specific shapes:

1.  **Chaining on `_last_generate_future[mid]`.**  The
    `_generate_sync` future is in Phase 3 `done_event.wait()`
    for the request we're about to interrupt; chaining on it
    deadlocks the pause behind the very wait it should break.
    The original `pause()` already opts out of this for that
    reason.

2.  **Chaining on `_futures[mid]` when it is a `_move_sync`
    parked on `prev_gen_events`.**  This is the
    `cl.sub(gpu)` -> `move(mid, checkpoint)` path: `move()`
    snapshots the inflight `done_event`s under `gen_lock` and
    `_move_sync` then `ev.wait()`s each one before stepping the
    ladder down.  Functionally that wait is identical to
    chaining on the generate future: it can only resolve on a
    real `generate_done`, which is exactly what `pause` would
    suppress.  Chaining `pause` behind that move-future puts
    `pause` at the back of the same wait it should break, so
    `_pause_sync` only runs once the generate has naturally
    finished and the move has walked the model down to
    `checkpoint` -- by which point `state != "running"` and
    pause logs `not running ..., pause is a no-op`.

`pause` therefore opts out of the `_futures[mid]` chain too.
It is submitted directly to the pool and the pause future is
**conditionally** published at `_futures[mid]` so subsequent
ops chain on *it*.  Correctness is preserved by the existing
precondition checks (`state == "running"`, `paused == False`,
`_inflight[mid]` non-empty under `gen_lock`).

The publish is conditional on `pause` actually being about
to commit a state change.  If the call will turn out to be a
no-op (`entry["paused"]` is already true, or
`entry["state"] != "running"`), the publish is skipped and
`_futures[mid]` continues to point at whatever long-running
op is in flight -- in practice, an in-flight
`_resume_sync` that is walking the ladder back up.  Without
this guard, a back-to-back `cl.pause_all(); cl.resume_all()`
issued while an earlier `resume_all()` is still walking up
overwrites `_futures[mid]` with the no-op pause's
immediately-done future; the next resume reads that as
`prev_future` (waits zero time) and runs in parallel with
the first resume, leaving the model wedged at
`state="running"` over an idle engine (see "Pause-during-
resume future-chain break" in `orchestrator_DESIGN.md`
Known Issues for the full walkthrough on `model 7`).

`_pause_sync`'s body is unaffected: even when the publish
is skipped, the body still runs (logs `already paused,
skipping` or `not running, pause is a no-op`) so the user
gets the same observability and the pause-side guard against
phantom-paused entries (see the deferred-pause guard below)
continues to fire.

The other half of the fix is on the `_move_sync` side.  After
pause flips `entry["paused"]` to true, the engine is frozen and
the captured `done_event`s in `prev_gen_events` will never fire
until a future `resume`.  Without intervention, the move would
hang forever waiting on those events.  `_move_sync` therefore
polls `entry["paused"]` while waiting:

```python
entry = Orchestrator._registry.get(model_id, {})
for ev in prev_gen_events:
    while not ev.wait(timeout=0.5):
        if entry.get("paused"):
            log.info("%s: model paused mid-move; abandoning "
                     "inflight-drain wait", model_id)
            break
    else:
        continue
    break
```

When the model is paused mid-wait, the move abandons the
remaining `prev_gen_events.wait()`s and continues with the
ladder walk.  `entry["state"]` is already `up` (set by
`_pause_sync`) and `paused=True`, so the `current == "running"`
guard at the top of `_move_sync` doesn't fire and the move
proceeds normally through `up -> sleep -> checkpoint`.  The
end-state for the user is `state="checkpoint"`, `paused=True`,
which is exactly what `cl.sub(gpu)` followed by `cl.pause()`
should produce: drain completes promptly, model retains its
saved sub-requests for a future `resume`.

The 0.5s poll is cheap relative to the seconds of generate-time
the wait usually spans; in the common (non-paused) case the
first `ev.wait()` returns on the same demuxer-driven `set()`
that resolves the per-request future, so there is no observable
latency penalty when nothing pauses.

### Move-vs-generate Phase-1 race (move-sync vs wake-up ordering)

`_move_sync` issues `_send_cmd_with_ack("sleep")` (and the
rest of the ladder-down cmd sequence) under `gen_lock`.
`_generate_sync` Phase 1 holds `gen_lock` across
`_move_sync(up, announce_state="running")` (which sends
`wake_up_weights`, `restore_weights`, `wake_up_kv_cache`),
*releases* it on Phase 1 exit, then re-acquires it for
Phase 2's `inst.generate(...)` enqueue.  In the released-lock
window between Phase 1's last cmd (`wake_up_kv_cache`) and
Phase 2's `inst.generate`, a concurrent `move()` for the same
model can win `gen_lock` and queue `sleep` ahead of the
generate.  The vLLM child then sees:

```
wake_up_kv_cache       <- last Phase 1 cmd (engine ready)
sleep                  <- move's first cmd (cumem freed)
generate               <- Phase 2's enqueue (request added
                          to a sleeping engine)
unpin / cuda_checkpoint  <- rest of move's ladder-down
```

The engine hangs trying to schedule a request against freed
cumem memory.  This is the race that stuck `model 7` and
`model 15` together when a `cl.sub(gpu)` drain overlapped
with an in-flight `generate` on the same model.

**Fix.**  `move()` snapshots `_last_generate_future[mid]`
under the same `gen_lock` that captures `prev_gen_events`,
and passes it to `_move_sync` as a new `prev_gen_future`
parameter.  `_move_sync` polls this future (paused-bail
aware, mirroring the `prev_gen_events` loop) **before** the
`prev_gen_events` drain:

```python
if prev_gen_future is not None:
    entry = Orchestrator._registry.get(model_id, {})
    if not entry.get("paused"):
        while not prev_gen_future.done():
            if entry.get("paused"):
                log.info("%s: model paused mid-move; "
                         "abandoning generate-chain wait",
                         model_id)
                break
            try:
                prev_gen_future.result(timeout=0.5)
            except FutureTimeoutError:
                continue
            except Exception as exc:
                log.warning("%s: prior generate raised (%s); "
                            "proceeding", model_id, exc)
                break
```

The future covers the full lifecycle of the most recent
`_generate_sync` -- Phase 1 wake-up, the brief inter-phase
window, Phase 2 enqueue, and Phase 3 done-wait -- so by the
time `_move_sync` proceeds, no generate can be in any stage
that would re-enter `gen_lock` for this model.

`prev_gen_events` and `prev_gen_future` are complementary,
and `_move_sync` keeps both:

- `prev_gen_events` covers earlier-but-still-inflight
  generates that have already reached Phase 2 (i.e. have
  entries in `_inflight`).  By the time `prev_gen_future`
  returns, every `done_event` in *that* generate's
  `_inflight` slice has fired and been popped by
  `_on_generate_done`, but generates submitted *before* the
  one tracked by `_last_generate_future` may still be
  in-flight; their events live in earlier `_inflight`
  entries.
- `prev_gen_future` covers only the *latest* generate, but
  catches it in stages (Phase 1, inter-phase) that
  `_inflight` doesn't yet see.

Both are skipped on paused models for the same reason --
the future / events would never resolve while the engine is
frozen.  Both use the same poll-and-bail pattern so a
concurrent pause arriving mid-move still breaks the wait.

### `sub` drain backoff

`Orchestrator.sub(gpu)` schedules `_sub_sync(gpu)` on the pool.
`_sub_sync`:

1. Scans the registry under `Slots._cv` for residents
   (`entry["gpu"] == gpu` or `entry["slot"].gpu_id == gpu`).
2. Issues `move(mid, "checkpoint")` **at most once per resident
   per drain**, tracked in a local `submitted: dict[mid, Future]`.
3. Parks on `Slots._cv` with a 0.5s timeout between passes;
   `Slots.deallocate` / migration `notify_all`s on this cv, so a
   genuine state change wakes the loop promptly.
4. Reaps completed move futures (logs failures), then re-scans.
5. When residents and `submitted` are both empty (and no orphan
   slot is in flight), `Slots.pop(gpu)` and remove the GPU from
   the pool.

The "at most once per resident per drain" discipline is what
prevents the chain-poisoning storm: re-issuing `move()` every
iteration overwrites `_futures[mid]` with a fresh `Future`, and
the new `_move_sync` then chains on the prior (failed) future via
`prev.result()` and re-raises it without re-reading `entry["state"]`.
With `_move_sync` now waiting on the full `_inflight[mid]` event
set (covers earlier-but-still-running generates that the
single-slot `_last_generate_future` would miss) and on a
paused-bail-aware poll of `_last_generate_future` (covers the
latest generate's pre-Phase-2 window; see "Move-vs-generate
Phase-1 race"), the original race that caused this cascade is
gone -- but the at-most-once discipline remains as a structural
guarantee against any future variant re-introducing the same
shape.

### Deferred-pause / phantom-running guard

The worker serialises commands FIFO, and the orchestrator hands
non-`generate` commands off through `_send_cmd_with_ack`, which
parks on a per-`(mid, cmd)` FIFO `Event` until the demuxer
dispatches the matching ack.  When a user submits
`generate_all(...)` followed immediately by `pause_all()` (a
common idiom for "kick off a long generate, then immediately
freeze it"), the orchestrator races two timelines for each model:

1. The worker dequeues `generate`, runs it to completion (~Ks),
   reports `generate_done`.  The demuxer's generate listener pops
   `_inflight[mid]`, sets the per-request `done_event`, and --
   under `gen_lock` -- releases the slot and flips `state`
   `running` -> `up` (because `_inflight` is now empty and
   `_pending_count == 0`).
2. The worker then dequeues the deferred `pause`, runs it on the
   now-empty engine, and reports back `saved=0
   was_paused=False`.  The demuxer's catch-all listener wakes the
   `_pause_sync` parked at `_send_cmd_with_ack`.

If `_pause_sync` blindly marks `entry["paused"] = True` and
re-asserts `state="up"` at this point, the registry now holds a
**phantom-paused** entry: `paused=True` with `_inflight=[]` and
no actual saved generates in the engine.  The next `resume_all()`
runs `_resume_sync`, which sends `resume` to a worker that has
nothing to restore (`restored=0 synthesized=0`), and -- if it
unconditionally flips `state` to `running` -- mints a
**phantom-running** entry: `state="running"` with `_inflight=[]`
and no actual generate in flight.  Every subsequent `move()` then
either hangs (chained behind a stuck `move(up)` whose own
`_inflight` snapshot is empty so it raises immediately, but the
chain-prev poisoning shape is gone after `_await_prev`, so the
next move just sees `state="running"` again) or repeatedly
short-circuits with `currently running a generate`.

Both layers now defend against this:

- **`_pause_sync` (deferred-pause guard)**: after the outer
  `state == "running"` precondition passes, take `gen_lock` and
  re-check `_inflight[mid]` is non-empty.  If empty, log
  `no inflight generates (deferred-pause race); pause is a
  no-op` and return *without* sending the worker `pause` and
  *without* marking `paused=True`.  `gen_lock` serialises against
  the generate-done listener so we observe a consistent
  `inflight` snapshot relative to the listener's pop-and-flip
  sequence.

- **`_resume_sync` (generate-shaped, unified ladder-walk +
  slot-acquire + phantom-running guard)**: after the outer
  `paused` precondition check, gate on `state in ("up",
  "sleep", "checkpoint")` (the only states a paused model can
  legally hold -- pause sets state=up, `move(saved)` is
  refused while paused, and `running` is its own pause-
  incompatible state).  Then snapshot `_inflight[mid]` under
  `gen_lock` *before* doing any work, and split:

  * **Empty `_inflight`** (defensive / nothing to drive): clear
    `entry["paused"]` under `model_lock` and leave the model
    wherever the user parked it (no walk-up cost when there is
    nothing to drive).  No slot acquisition, no worker `resume`
    cmd, no transition to `running`.  The pause-side guard
    already prevents minting `paused=True` without sending the
    worker `pause`, so in healthy code this branch is
    unreachable; keeping it as a self-heal means a future path
    that bypasses `_pause_sync` can't permanently strand the
    orchestrator side.

  * **Non-empty `_inflight`** (saved-from-pause subreqs,
    queued-during-pause new generates, or both): two stages.

    1. **Walk + acquire + announce**, all in a single call to
       `_move_sync(model_id, "up", announce_state="running")`
       -- the *exact* same entry point `_generate_sync` Phase 1
       uses.  Handles the ladder walk (when state is
       `checkpoint`/`sleep`, via `_step_up(..., "up",
       announce_state="running")` on the final step) and the
       in-place slot acquisition + state flip (when state is
       `up`, via `_acquire_slot_for_running` + post-acquire
       state re-check + `_set_state("running")`) in a single
       code path.  Direct `_move_sync` (not the public
       `move()`) so the inner walk doesn't capture
       `prev_gen_events` (the model is paused so they'd never
       fire anyway) and doesn't publish a separate
       `_futures[mid]` future -- the resume future encompasses
       both the walk and the body.  `paused=True` is preserved
       across the walk because nothing in `_step_up` /
       `_step_down` touches `paused`; saved subreqs and queued
       requests in `_inflight` ride the walk untouched.

       Inheriting `_move_sync`'s post-acquire state re-check
       is what fixes the resume slot-steal wedge -- see
       "Resume slot-steal wedge" in
       [`orchestrator_DESIGN.md`](orchestrator_DESIGN.md)
       Known Issues for the cause-and-fix walkthrough.  The
       earlier two-call sequence (`_move_sync(mid, "up")` then
       bare `_acquire_slot_for_running` then unconditional
       `_set_state("running")`) bypassed the re-check on the
       slotless-up branch and could leave the model wedged at
       `state="running", slot=None` over a sleeping engine
       when a peer's concurrent Phase-2 eviction stole the
       just-claimed Tier-A slot.

    2. **Worker resume + clear paused.**
       `_send_cmd_with_ack(model_id, "resume")` re-prefills
       the saved subreqs and unfreezes the engine; then under
       `model_lock`, clear `entry["paused"] = False`.  State
       is already `"running"` from stage 1 so we don't flip it
       again.  The original per-request `done_event`s resolve
       as the engine drives the now-running requests to
       completion.

       The brief window where `state="running", paused=True`
       (bounded by the `resume` cmd round-trip) is benign:
       `_step_up` Phase 2 eviction targets only `state=="up"`
       slotless squatters (we are slotted `running`),
       `pause`/`resume`/`move` all serialise via
       `_futures[mid]`, and `_generate_sync`'s paused fast-
       path tolerates either ordering -- a generate submitted
       in this window lands in the engine and runs immediately
       once the resume ack lands.

The orchestrator-side `_inflight` is the authoritative source
for "anything to drive": a real `pause` does not pop entries
from `_inflight` (only the generate-done listener does), and
`_generate_sync` Phase 2 appends queued-during-pause new
generates immediately, so the snapshot under `gen_lock` covers
both classes unambiguously.

The pause-side guard prevents new phantoms from being minted;
the resume-side empty branch self-heals any phantom that
already exists (e.g. minted by a path that bypasses
`_pause_sync` in the future) and gives `resume` clean
generate-shaped semantics: observably a no-op when there is
nothing to drive, otherwise the same `_move_sync(mid, "up",
announce_state="running")` that `generate` uses (typical
pattern reaching the ladder-walk arm: `cl.pause();
cl.move("checkpoint"); cl.resume()`).

### Generate-while-paused stash

Generate-while-paused is allowed by the orchestrator at any
ladder state at which pause itself is allowed -- `up`, `sleep`,
`checkpoint` (see "Walking down past `up` while paused" in
[`orchestrator_DESIGN.md`](orchestrator_DESIGN.md)).  The
underlying vLLM engine, however, is only safe to call into
while it is at `up`: `llm.sleep(level=2)` discards the cumem
allocator's GPU blocks and tears the executor context down,
and a subsequent `cuda_checkpoint` (via `cuda-checkpoint`)
freezes the entire CUDA context.  Either state makes the engine
APIs hazardous -- `engine.add_request` enqueues into a
scheduler that can no longer `step()`, and `engine.abort_request`
blocks indefinitely on a torn-down executor (the historical
symptom: a `sleep` cmd landing on an already-asleep engine
that had a generate queued from an earlier `pause -> sleep ->
generate` interleaving used to hang in `abort_request`,
backing the entire worker pipe up).

**Invariant.**  The child enforces a single rule that side-steps
the whole hazard class:

> While `_paused` is True, the vLLM engine sees no scheduler
> mutations from this child.  All deferred generates live in
> `_saved_requests`.  `resume` is the only handler that copies
> deferred work back into the engine.

Equivalently, `_active_reqs` and "`_paused` is True" are
mutually exclusive: `pause` empties `_active_reqs` (snapshot
and abort), and `_submit_generate` never populates it while
`_paused` is set.

**Submit-time route.**  When `_paused` is True,
`_submit_generate` short-circuits before touching the engine:
it appends a synthesised entry to `_saved_requests` shaped
exactly like a never-stepped pause snapshot -- per-eid
`prompt_token_ids=[]`, empty `output_token_ids`, empty
`output_text`, the original `prompts` list preserved verbatim,
and the original `sampling_params` / `t0` / `first_token_ts=None`.
This mirrors what `_snapshot_active_into_saved` produces for an
empty per-eid state via its `list(... or [])` clause, so the
resume loop's `len(prompt_tids)` works and its
`if prompt_tids:` test falls through to the
`elif i < len(prompts_orig): prompt_obj = prompts_orig[i]`
branch -- which re-prefills R from the original prompt on the
now-awake engine, identically to how pause-mid-flight requests
whose per-eid state never saw a step are restored.

**Why this is order-independent.**  Because the route is keyed
on `_paused` (set by `pause`, cleared by `resume`) and not on
any engine-state predicate, any pipe interleaving of
`pause`, `sleep`, `unpin`, `cuda_checkpoint`, `cuda_restore`,
`repin`, `wake_up_weights`, `restore_weights`, and `generate`
that the "Walking down past `up` while paused" rule permits is
handled by the same line of code.  Earlier designs needed a
matrix of guards keyed on the dynamic engine state (an
`_engine_dormant` flag plus a `sleep`-handler snapshot fallback
for the awake-paused-then-sleep interleaving); collapsing them
into the `_paused`-keyed submit-time route eliminates both
guards and the flag.

**Why the `sleep` handler no longer needs a snapshot fallback.**
The previous (option-b) design called
`_snapshot_active_into_saved()` inside the `sleep` handler when
`_paused and _active_reqs`, to catch a generate-while-paused
request that had already been `engine.add_request`-ed onto an
awake-paused engine before the sleep landed.  With the
`_paused`-keyed route, no such request can exist: while
`_paused` is True, `_active_reqs` is empty by construction, so
the `sleep` handler can drop the snapshot call entirely and
just `_drain_engine` + `llm.sleep(level=2)`.

**Resume.**  Unchanged.  `resume` iterates `_saved_requests`,
calls `engine.add_request(new_eid, prompt_obj, sp)` for each
entry, clears `_paused`, and lets the main loop's
`engine.step()` drain.  Both pause-mid-flight snapshots and
post-pause submits flow through this same loop; the
classification was relevant only at submit time.

### Eviction lock ordering (Phase 2 of `_step_up`)

`_step_up` runs in three phases: `_acquire_slot_for_running`
(may retreat the same model from `up` to `sleep` if it can't get
a slot), Phase 2 eviction (free a home-GPU peer if HBM pressure
demands it), and Phase 3 wake-up.  Phase 2 picks an eviction
victim from a lock-free scan, then steps that victim down with
`_step_down(incumbent, "up", "sleep")`.

The repeated lock-ordering rule across the codebase is
**`gen_lock` then `model_lock`** -- the demuxer's generate-done
listener takes `gen_lock` outside, then `model_lock` for the
slot/state mutation; `_send_cmd_with_ack` takes `gen_lock`
across the wait so a finishing generate observes a consistent
ack queue.  `_step_up` / `_step_down` deliberately release
`model_lock` before each `_send_cmd_with_ack` to match.

Phase 2 eviction has to follow the same rule when calling
`_step_down` on a *different* model (the incumbent).  Concretely,
the bug shape was:

- T1 (`_generate_sync` for the incumbent) is parked in
  `_acquire_slot_for_running` retreating: it already holds
  `gen_lock(incumbent)` and is waiting for `_lock(incumbent)`
  inside `_step_down(incumbent, "up", "sleep")`.
- T3 (`_step_up` Phase 2 for some other model) holds
  `_lock(incumbent)` from the eviction validation block and is
  inside `_step_down(incumbent, "up", "sleep")` -> first
  `_send_cmd_with_ack("sleep")` -> `gen_lock(incumbent)`.

That is a textbook AB-BA deadlock and is exactly what stuck
`model 3` and `model 15` together: T3 was evicting model 15 on
behalf of model 3's `_step_up`; model 15's own
`_generate_sync` was retreating at the same time.

The fix is to validate the incumbent under `_lock(incumbent)`
and **release `_lock(incumbent)` before calling `_step_down`**:

```python
inc_entry = Orchestrator._registry[incumbent]
with Orchestrator._locks_ordered(incumbent):
    evict_now = (inc_entry.get("slot") is None
                 and inc_entry.get("state") == "up"
                 and inc_entry.get("gpu") == home_gpu)
if evict_now:
    log.info("%s: evicting %s from GPU %s",
             model_id, incumbent, home_gpu)
    Orchestrator._evict_for_phase2(incumbent)
remaining -= share
```

`_evict_for_phase2` is the resolution to the "Eviction-mid-
generate dormant-engine wedge" in `orchestrator_DESIGN.md`
Known Issues.  Without it, a follow-up `inst.generate(
incumbent, ...)` racing the eviction can land on the worker
pipe AFTER our pending `sleep` cmd (its Phase 1 sees
`state="running"` because the eviction hasn't reconciled
yet) and get drained against a dormant engine, which the
`_saved_requests` deferral path absorbs without an ack and
wedges the worker.  The helper combines two mechanisms:

1. **Sentinel publish on `_futures[incumbent]`**, under
   `gen_lock(incumbent)`, *before* sending sleep.  Concurrent
   `_generate_sync(incumbent)` re-checks `_futures[mid]`
   under `gen_lock` (Phase 1 + Phase 2 share one
   acquisition, gated by a re-check loop at the top); on
   mismatch it releases `gen_lock` and awaits the sentinel.
   By the time the await returns, `state="sleep"` and the
   slot is `None`; the racing generate then walks
   `_move_sync(up, announce_state="running")` which queues
   `wake_up_weights` AFTER our `sleep` on the worker pipe
   (correct order, no dormant-engine deferral).
2. **In-flight drain** of `_last_generate_future[incumbent]`
   and per-request `_inflight[incumbent]` done-events,
   *outside* `gen_lock`.  This covers the *original*
   model-16 wedge: an `inst.generate` already on the worker
   pipe when the eviction arrived would otherwise be
   sandwiched as `[generate, sleep]` and the trailing
   generate (from any concurrent submitter that the
   sentinel-gating doesn't yet cover, since the sentinel was
   just published) would land behind sleep.  By waiting for
   the in-flight to complete first, we ensure the worker
   reaches sleep on an empty pipe.

The race window opened by releasing `_lock` early (the
incumbent-validation pre-step) is absorbed on the
`_step_down(up, sleep)` side, which deliberately reconciles
the registry to the *post-sleep worker* rather than trusting
the eviction's pre-sleep snapshot.  Four flavours of the race
show up:

1. *Incumbent self-evacuated first.*  Its own thread (e.g.
   `_acquire_slot_for_running` retreating) flipped state to
   `sleep` before our `_step_down` reached its trailing
   `_set_state`.  Our `_send_cmd_with_ack("sleep")` issues a
   redundant `sleep` to a worker that is already asleep --
   vllm's `sleep` is idempotent, the FIFO ack registry
   tolerates the extra wait by construction (deferred ack
   queue is a per `(mid, cmd)` deque), and the trailing
   `_set_state(..., "sleep")` is the no-op same-state stamp.

2. *Incumbent raced all the way back to `running`.*  Between
   our gate passing (state=`up`, slot=`None`) and our `sleep`
   ack landing, the incumbent's `_generate_sync` walked the
   ladder back up, acquired a slot, queued a `generate` on the
   worker *ahead of* our still-pending `sleep`, and that
   `generate` even completed.  The worker drained the
   `generate` first, then ran the `sleep`.  By the time our
   ack lands, the engine is asleep but the registry reads
   `running` with the slot still held -- because
   `_on_generate_done`'s `_pending_count == 0` guard saw our
   queued `sleep` as still in flight and skipped the slot
   release.  `_step_down(up, sleep)` is the recovery point:
   under `model_lock` it unconditionally `Slots.deallocate`s
   any slot the registry holds and `_set_state`s to `sleep`,
   reconciling the publish view to the asleep engine.  Without
   this branch the slot would leak for the rest of the run.

3. *Incumbent had an in-flight generate when we arrived.*
   The pre-`_step_down` in-flight drain inside
   `_evict_for_phase2` covers this: we snapshot the
   incumbent's `_last_generate_future` and per-request
   done-events under `gen_lock(incumbent)` (where the
   sentinel publish also happens), then wait OUTSIDE
   `gen_lock` for them to drain.  Pending generates that
   *were already submitted* before the snapshot complete on
   the still-up engine; the eviction's `_send_cmd_with_ack(
   "sleep")` then runs against an empty pipe.  Without the
   drain, the trailing generate would land BEHIND the sleep
   and hit the dormant-engine deferral path -- the original
   `model 16` wedge.

4. *Incumbent had a generate submitted DURING the eviction's
   sleep round-trip.*  This is the residual race the
   sentinel resolves.  The racing `_generate_sync(incumbent)`
   re-acquires `gen_lock` after `_evict_for_phase2`'s publish
   has completed, observes the new sentinel on
   `_futures[mid]` (different from its submit-time snapshot),
   releases `gen_lock`, and awaits the sentinel.  When the
   sentinel resolves (after `_step_down` finished), the
   racing generate's Phase 1 finds `state="sleep"` and walks
   the heavyweight wake path, ordering its `wake_up_weights`
   cmd AFTER the eviction's `sleep` on the worker pipe.
   Empirical hit: the `model 13` 4 ms wedge in the
   2026-05-13 demo run.

Holding `_lock` across `_step_down` to *prevent* the redundant
send re-introduces the AB-BA deadlock and is strictly worse
than absorbing the extra command.  See
`orchestrator_DESIGN.md` "Slot release at `up -> sleep`" for
the slot-lifecycle perspective.
