"""Per-model operation pipeline -- explicit serialization of orchestrator ops.

This module provides the infrastructure for the explicit per-model pipeline
described in ``pipeline_DESIGN.md``.  Concrete ``Op`` subclasses
(``RegisterOp``, ``MoveOp``, ``GenerateOp``, ``PauseOp``, ``ResumeOp``,
``RemoveOp``, ``EvictForPeerOp``) live in this file; their bodies are ported
from the legacy ``Orchestrator._*_sync`` methods in ``orchestrator.py``.

Pipeline shape
--------------

* One ``ModelPipeline`` per model_id, owning a single FIFO ``queue.Queue``
  and a single dedicated worker thread.
* All operations on a model are submitted through the pipeline and run in
  FIFO order on its worker thread.  ``submit_front`` is reserved for
  ``PauseOp`` (head-of-queue insertion).
* ``pause`` interrupts the in-flight op via ``InterruptFlag`` *synchronously*
  on the caller's thread, then submits ``PauseOp`` at the head of the queue.
* ``GenerateOp`` is hand-off-and-return: ``execute()`` does Phase 1 + Phase 2
  (walk-up + ``inst.generate(...)`` submit), enqueues a record in
  ``entry["_inflight"]``, and returns a ``PendingRequest`` -- the user's
  thread (in ``Orchestrator.generate``) does the ``done_event.wait()``,
  freeing the pipeline worker for the next op (a ``PauseOp``, an eviction,
  the next generate).

Avoiding circular imports
-------------------------

The Op subclasses defined here need to call ``Orchestrator._step_up`` /
``_step_down`` / ``_set_state`` / ``_send_cmd_with_ack`` / etc. and read
``Orchestrator._registry``.  To avoid ``pipeline.py -> orchestrator.py
-> pipeline.py`` at import time, ``OpContext`` carries an ``orch`` field
(the ``Orchestrator`` *class itself*, injected by ``orchestrator.py`` at
``ModelPipeline`` construction).  Concrete ops call ``ctx.orch._foo(...)``.

Step 0 -- dependency surface inventory
======================================

Inventory of which ``Orchestrator._*`` statics and class-level fields
are read or mutated by ``_step_up`` (orchestrator.py:1102-1391) and
``_step_down`` (orchestrator.py:1394-1469).  Every concrete Op subclass
that does walk-up / walk-down work goes through these two functions, so
this is the canonical surface that ``ctx.orch`` must expose.

``_step_up(model_id, from_state, to_state, *, target_gpu=None,
announce_state=None)``:

* **Reads**: ``Orchestrator._registry[model_id]`` (entry dict).  Iterates
  ``Orchestrator._registry.items()`` for Phase-2 HBM accounting.
* **Mutates** (under ``Orchestrator._locks_ordered(model_id)``):
  ``entry["instance"]``, ``entry["slot"]``, ``entry["gpu"]``,
  ``entry["pinned_cpu_bytes"]``, ``entry["total_gpu_bytes"]``.
* **Calls**: ``Orchestrator._set_state``, ``_locks_ordered``,
  ``_install_listeners`` (saved -> checkpoint),
  ``_send_cmd_with_ack`` (every cmd: ``cuda_restore``, ``repin``,
  ``unpin``, ``cuda_checkpoint``, ``wake_up_weights``,
  ``restore_weights``, ``wake_up_kv_cache``),
  ``_evict_for_phase2`` (Phase-2 HBM eviction; replaced by
  ``EvictForPeerOp`` cross-pipeline submit when ``_use_pipeline``),
  recursive ``_step_down`` / ``_step_up`` (Tier-C slot retreat).
* **Touches**: ``Slots.allocate(level, on_block=...)``,
  ``Slots.try_allocate(level, gpu=...)``, ``Instance(vllm_config)``,
  thread-local ``Orchestrator._timing.gpu_wait_s`` /
  ``_timing.migrate_s``.

``_step_down(model_id, from_state, to_state)``:

* **Reads**: ``Orchestrator._registry[model_id]`` (entry dict).
* **Mutates** (under ``Orchestrator._locks_ordered(model_id)``):
  ``entry["slot"]`` (released and set to None), ``entry["gpu"]``
  (set to None on sleep -> checkpoint), ``entry["instance"]`` (set to
  None on checkpoint -> saved).
* **Calls**: ``Orchestrator._set_state``, ``_locks_ordered``,
  ``_send_cmd_with_ack`` (``sleep``, ``unpin``, ``cuda_checkpoint``).
* **Touches**: ``Slots.deallocate(slot)``,
  ``entry["instance"].teardown().wait().remove()`` (checkpoint -> saved).

Aggregated ``ctx.orch`` surface required by all Op subclasses
=============================================================

* Class-level dicts: ``_registry``.
* Static helpers: ``_set_state``, ``_locks_ordered``,
  ``_install_listeners``, ``_send_cmd_with_ack``, ``_step_up``,
  ``_step_down``, ``_acquire_slot_for_running``, ``_pick_level``
  (module-level helper, but accessed via ``ctx.orch``-adjacent import),
  ``_image_dir_for``.
* Thread-local: ``_timing`` (telemetry; set by ``_step_up`` and read by
  the ``GenerateOp`` queue-record builder).

In addition every Op may touch ``Slots`` (allocate / try_allocate /
deallocate) and ``ctx.entry["instance"]`` (the per-model child-process
proxy from ``instance.py``).
"""
from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from instance import Instance


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interrupt primitives
# ---------------------------------------------------------------------------


class Interrupted(Exception):
    """Raised by ``InterruptFlag.raise_if_set`` / ``wait_or_interrupt``.

    Signals that the surrounding ``Op.execute`` should bail out.  The
    pipeline worker resolves the op's future with this exception and then
    moves on to the next queued op (typically a ``PauseOp`` submitted at
    the head of the queue).  This is *not* an error -- it's the
    cooperative way an op yields control during a pause.
    """


class InterruptFlag:
    """Cooperative interrupt for the in-flight ``Op.execute``.

    Generic threading primitive -- no relation to LLM tokens.  Backed by
    a ``threading.Condition`` so threads parked in ``wait_or_interrupt``
    can be woken by ``set()``.

    Per-pipeline: each ``ModelPipeline`` owns one ``InterruptFlag`` shared
    across the lifetime of the worker.  ``PauseOp`` resets the flag at
    the end of its ``execute`` so subsequent ops (a queued ``move``,
    a deferred ``ResumeOp``) start clean.
    """

    __slots__ = ("_cv", "_set", "_reason")

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._set = False
        self._reason: str | None = None

    def set(self, reason: str = "interrupted") -> None:
        with self._cv:
            self._set = True
            self._reason = reason
            self._cv.notify_all()

    def reset(self) -> None:
        """Clear the flag.  Called by ``PauseOp`` at the end of ``execute``."""
        with self._cv:
            self._set = False
            self._reason = None

    def is_set(self) -> bool:
        return self._set

    def raise_if_set(self) -> None:
        """Polling yield-point.  Op subclasses sprinkle this between
        long-running calls (``_step_up``, slot acquisition, cmd-with-ack
        sequences) so a pause that lands while the worker is between
        I/O sees a quick bail-out.
        """
        if self._set:
            raise Interrupted(self._reason or "interrupted")

    def wait_or_interrupt(
        self,
        ev: threading.Event,
        timeout: float | None = None,
    ) -> bool:
        """Wait for either ``ev`` to be set or the flag to be tripped.

        Returns ``True`` when ``ev`` lands first.  Raises ``Interrupted``
        when the flag wins.  ``ev`` is a vanilla ``threading.Event``
        (typically the per-request ``done_event`` from
        ``entry["_inflight"]``); the flag is the per-pipeline
        ``InterruptFlag``.

        Implementation note: ``ev.set()`` does not notify ``self._cv``,
        so we poll ``ev.is_set()`` on a short interval and rely on the
        cv to wake us promptly when the flag is set.  100 ms is fast
        enough for human-perceptible pause latency without burning CPU.
        """
        import time as _time
        deadline = _time.monotonic() + timeout if timeout is not None else None
        poll = 0.1
        while True:
            if self._set:
                raise Interrupted(self._reason or "interrupted")
            if ev.is_set():
                return True
            if deadline is not None:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    return ev.is_set()
                wait_for = min(poll, remaining)
            else:
                wait_for = poll
            with self._cv:
                if self._set:
                    raise Interrupted(self._reason or "interrupted")
                self._cv.wait(timeout=wait_for)


# ---------------------------------------------------------------------------
# Op + OpContext
# ---------------------------------------------------------------------------


@dataclass
class OpContext:
    """Per-execution context handed to ``Op.execute``.

    Rebuilt by the worker before each op so ``inst`` reflects the current
    ``entry["instance"]`` (which is None during ``saved``/``checkpoint``).

    The ``orch`` field is the ``Orchestrator`` class object, injected at
    ``ModelPipeline`` construction time so Op subclasses can call its
    static helpers (see "Avoiding circular imports" in the module docstring
    and the Step 0 dependency surface inventory above).
    """

    model_id: str
    entry: dict
    interrupt: InterruptFlag
    pipelines: Mapping[str, "ModelPipeline"]
    slots: type
    inst: "Instance | None"
    orch: type


class Op(ABC):
    """Base class for an operation submitted to a ``ModelPipeline``.

    Subclasses implement ``execute(ctx)`` and may return any value (the
    value resolves the op's future).  Subclasses should periodically
    call ``ctx.interrupt.raise_if_set()`` between long-running sub-calls
    so that pause delivery latency is bounded.
    """

    future: Future
    """Set by ``ModelPipeline.submit`` before the op enters the queue."""

    @abstractmethod
    def execute(self, ctx: OpContext) -> Any:  # pragma: no cover - abstract
        ...


# ---------------------------------------------------------------------------
# PendingRequest -- GenerateOp's hand-off-and-return value
# ---------------------------------------------------------------------------


@dataclass
class PendingRequest:
    """Hand-off record returned by ``GenerateOp.execute``.

    The pipeline worker resolves ``GenerateOp.future`` with this value as
    soon as the engine has accepted the request (Phase 2 done).  The
    user's thread inside ``Orchestrator.generate`` then waits on
    ``done_event`` -- this happens off the pipeline worker, so a
    subsequent ``PauseOp`` or peer ``EvictForPeerOp`` can run while the
    generate is still in flight on the engine.
    """

    rid: str
    done_event: threading.Event
    q_rec: dict
    inst: "Instance"


# ---------------------------------------------------------------------------
# ModelPipeline
# ---------------------------------------------------------------------------


# Sentinel pushed onto the queue to signal worker shutdown.
_SHUTDOWN = object()


@dataclass
class _QueueItem:
    op: Op
    front: bool = False


class ModelPipeline:
    """Single FIFO queue + worker thread for one ``model_id``.

    Construction injects the dependencies the worker needs to build an
    ``OpContext`` per op:

    * ``model_id``, ``entry`` -- registry slot for this model.
    * ``slots`` -- the ``Slots`` class (allocate / deallocate access).
    * ``orch`` -- the ``Orchestrator`` class (static helpers + class fields).
    * ``pipelines`` -- the orchestrator's ``_pipelines`` dict (for
      cross-pipeline submissions like ``EvictForPeerOp``).

    The worker rebuilds ``OpContext`` per op so ``inst`` is fresh (it can
    be ``None`` during ``saved``/``checkpoint``).
    """

    def __init__(
        self,
        *,
        model_id: str,
        entry: dict,
        slots: type,
        orch: type,
        pipelines: Mapping[str, "ModelPipeline"],
    ) -> None:
        self.model_id = model_id
        self._entry = entry
        self._slots = slots
        self._orch = orch
        self._pipelines = pipelines
        self.interrupt = InterruptFlag()

        self._q: queue.Queue = queue.Queue()
        # Front-insert items live in a small deque guarded by ``_q``'s
        # lock semantics: we wrap each tail-put with ``submit`` and each
        # head-put with ``submit_front``; the worker drains the front
        # buffer first.  Using two structures keeps ordering simple
        # without re-implementing queue.Queue.
        self._front: list[_QueueItem] = []
        self._front_lock = threading.Lock()

        self._stopping = False
        self._worker = threading.Thread(
            target=self._run,
            name=f"pipeline[{model_id}]",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public submission API
    # ------------------------------------------------------------------

    def submit(self, op: Op) -> Future:
        """Append ``op`` to the tail of the queue.  Returns its Future."""
        self._assert_no_cross_pipeline_cycle()
        op.future = Future()
        self._q.put(_QueueItem(op=op, front=False))
        return op.future

    def submit_front(self, op: Op) -> Future:
        """Insert ``op`` at the head of the queue.  Returns its Future.

        Reserved for ``PauseOp``: pause must run before any queued
        ``MoveOp`` / ``GenerateOp`` / ``ResumeOp`` so that those ops see
        the post-pause registry state.
        """
        self._assert_no_cross_pipeline_cycle()
        op.future = Future()
        with self._front_lock:
            self._front.append(_QueueItem(op=op, front=True))
        # Wake the worker if it's blocked on q.get() with no tail items.
        # An empty no-op item lets us reuse queue.Queue's wakeup; the
        # worker drains the front buffer first regardless.
        self._q.put(_QueueItem(op=_NoOp(), front=False))
        return op.future

    # ------------------------------------------------------------------
    # Cross-pipeline cycle detection
    # ------------------------------------------------------------------
    #
    # The cross-pipeline rule (pipeline_DESIGN.md section 4.3): when a
    # pipeline worker submits an Op onto a peer pipeline and blocks on
    # the result (today: only ``EvictForPeerOp`` from ``_step_up``
    # Phase-2 eviction), the peer must not transitively wait on us.
    # Otherwise we have a cycle: A waits on B which waits on A,
    # silent hang.
    #
    # We catch this at submit time with a simple wait-for-graph walk
    # across the (small, O(num_models)) set of pipeline workers.
    # ``_waiting_on`` is set/cleared by the worker around any
    # ``.result()`` it does on a peer; ``submit`` walks the chain from
    # the destination forward and asserts it doesn't loop back to the
    # caller.

    _waiting_on: "ModelPipeline | None" = None

    def _assert_no_cross_pipeline_cycle(self) -> None:
        caller = threading.current_thread()
        # Find the source pipeline whose worker is the calling thread,
        # if any.  Submissions from non-worker threads (the user's
        # synchronous Orchestrator.move / register / generate / pause)
        # cannot create cross-pipeline cycles since user threads don't
        # have a ``_waiting_on`` chain.
        src: "ModelPipeline | None" = None
        for p in self._pipelines.values():
            if p._worker is caller:
                src = p
                break
        if src is None or src is self:
            return
        # Walk the wait-for chain from us forward; if it loops back to
        # src, we'd deadlock.
        node: "ModelPipeline | None" = self
        seen: set[str] = set()
        while node is not None:
            if node is src:
                raise AssertionError(
                    f"cross-pipeline cycle: pipeline[{src.model_id}] "
                    f"would submit onto pipeline[{self.model_id}] "
                    f"which transitively waits on pipeline[{src.model_id}]"
                )
            if node.model_id in seen:
                # Foreign cycle not involving src; still bad, but not
                # ours to flag here.  Bail to avoid infinite loops.
                return
            seen.add(node.model_id)
            node = node._waiting_on

    def interrupt_now(self, reason: str = "pause") -> None:
        """Trip the interrupt flag.  Synchronous: caller's thread sets it.

        Called by ``Orchestrator.pause`` *before* ``submit_front(PauseOp())``
        so an in-flight ``MoveOp`` / ``GenerateOp`` parked in
        ``wait_or_interrupt`` bails out promptly.
        """
        self.interrupt.set(reason)

    def drain(self, timeout: float | None = None) -> bool:
        """Block until the queue is empty.

        Used by tests and by ``Orchestrator.remove`` to ensure the
        worker has finished with this model before the pipeline is torn
        down.  Returns True on success, False on timeout.
        """
        # queue.Queue.join() requires task_done() pairing, which we do
        # in the worker.  Use it for the tail; for the front buffer
        # poll briefly since front items are short-lived.
        import time as _time
        deadline = _time.monotonic() + timeout if timeout is not None else None
        while True:
            with self._front_lock:
                front_empty = not self._front
            if front_empty and self._q.unfinished_tasks == 0:
                return True
            if deadline is not None and _time.monotonic() >= deadline:
                return False
            _time.sleep(0.01)

    def submit_to_peer_and_wait(self, peer: "ModelPipeline", op: Op) -> Any:
        """Submit ``op`` to ``peer`` and block on its result.

        This is the only sanctioned way for one pipeline's worker to
        wait on another pipeline.  Sets ``self._waiting_on`` so peer
        submits can detect cycles, and clears it on return.

        Used by ``EvictForPeerOp`` orchestration: the acquirer's
        ``_step_up`` Phase-2 path calls
        ``ctx.pipelines[incumbent].submit(EvictForPeerOp(...))`` and
        then needs to wait.  Wrap that wait in this helper so the
        cycle detector sees the edge.

        Ordering note: ``_waiting_on`` is set BEFORE ``peer.submit(op)``
        so a concurrent peer-side ``submit_to_peer_and_wait(self, ...)``
        sees our outgoing edge during its own cycle check.  Without
        this ordering, two workers each calling
        ``submit_to_peer_and_wait`` on each other could both clear
        their cycle checks (each sees the other's ``_waiting_on``
        still as ``None``), both then set their own ``_waiting_on``,
        and both deadlock waiting on each other.  Setting first +
        rolling back on submit failure closes that TOCTOU window.
        """
        self._waiting_on = peer
        try:
            fut = peer.submit(op)
        except BaseException:
            # peer.submit may raise (typically the cycle detector
            # firing on the peer side); roll back the edge so we
            # don't leave a phantom waiter behind.
            self._waiting_on = None
            raise
        try:
            return fut.result()
        finally:
            self._waiting_on = None

    def shutdown(self, *, drain: bool = True, timeout: float | None = None) -> None:
        """Drain the queue, then stop the worker thread.

        Called by ``Orchestrator.remove`` after ``RemoveOp`` resolves and
        by ``Orchestrator.init`` on hard-reset.
        """
        if drain:
            self.drain(timeout=timeout)
        self._stopping = True
        self._q.put(_SHUTDOWN)
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _build_ctx(self) -> OpContext:
        # Re-fetch the entry from the orchestrator's registry per op
        # rather than reusing the construction-time ``self._entry``
        # snapshot.  Rationale: a ``remove(mid)`` schedules a
        # background ``_teardown`` thread that calls
        # ``pipe.shutdown(drain=True)`` BEFORE popping the pipeline
        # from ``Orchestrator._pipelines``.  If a fresh
        # ``register(mid)`` lands during that drain window, the
        # registry's ``mid`` entry is replaced by a new dict but
        # ``self._entry`` still references the old one.  Re-fetching
        # keeps ``ctx.entry`` consistent with ``ctx.orch._registry[mid]``
        # for the very Ops that observe the divergence (the RegisterOp
        # the user just submitted onto the about-to-die pipeline).
        #
        # Fall back to ``self._entry`` when the registry has popped
        # the model entirely (e.g. ops dequeued after RemoveOp during
        # drain) or when ``self._orch`` is a primitive-test stub
        # without a ``_registry`` attribute (the test fixture in
        # ``tests/test_pipeline.py`` passes ``orch=object`` since
        # those tests don't drive a real Orchestrator).
        registry = getattr(self._orch, "_registry", None)
        if registry is None:
            current = self._entry
        else:
            current = registry.get(self.model_id, self._entry)
        return OpContext(
            model_id=self.model_id,
            entry=current,
            interrupt=self.interrupt,
            pipelines=self._pipelines,
            slots=self._slots,
            inst=current.get("instance"),
            orch=self._orch,
        )

    def _next_item(self) -> _QueueItem | object:
        """Return the next item to run.  Front buffer takes precedence."""
        # Drain front buffer first.
        with self._front_lock:
            if self._front:
                item = self._front.pop(0)
                return item
        # Fall back to the tail FIFO; this blocks until a put.
        return self._q.get()

    def _run(self) -> None:
        log.debug("[pipeline %s] worker started", self.model_id)
        try:
            while True:
                item = self._next_item()
                if item is _SHUTDOWN:
                    log.debug("[pipeline %s] shutdown signal", self.model_id)
                    # task_done() not required for sentinels we put ourselves.
                    if isinstance(item, _QueueItem):
                        self._q.task_done()
                    return
                assert isinstance(item, _QueueItem)
                op = item.op
                # The wakeup _NoOp pushed by submit_front lands here as a
                # tail item.  Just account for it and continue.
                if isinstance(op, _NoOp):
                    self._q.task_done()
                    continue
                ctx = self._build_ctx()
                try:
                    result = op.execute(ctx)
                except Interrupted as exc:
                    # Cooperative bail-out: surface to the future, then
                    # carry on with the next op (typically PauseOp at
                    # the head of the queue).
                    op.future.set_exception(exc)
                except BaseException as exc:  # noqa: BLE001
                    log.exception(
                        "[pipeline %s] op %s raised",
                        self.model_id, type(op).__name__,
                    )
                    op.future.set_exception(exc)
                    # When a non-generate worker command failed (e.g.
                    # ``wake_up_kv_cache`` OOM during a resume walk),
                    # the engine is now wedged: re-issued / queued
                    # generates will never produce a ``generate_done``
                    # ack and the user threads parked on their
                    # PendingRequest ``done_event.wait()`` would hang
                    # forever.  Surface the failure to every parked
                    # caller for this model so they see the error and
                    # can release the model.
                    self._fail_inflight_generates_if_worker_failed(exc)
                else:
                    op.future.set_result(result)
                finally:
                    # Tail items always need task_done(); front items
                    # bypassed the queue, so we don't call task_done()
                    # for them.
                    if not item.front:
                        self._q.task_done()
        except BaseException:  # noqa: BLE001
            log.exception("[pipeline %s] worker crashed", self.model_id)
            raise

    def _fail_inflight_generates_if_worker_failed(
            self, exc: BaseException) -> None:
        """If *exc* is a worker-side command failure, fail every
        in-flight generate for this model so parked user threads
        unblock.

        Engine-state corruption from a failed cmd (e.g. CUDA OOM in
        ``wake_up_kv_cache`` mid-resume) leaves the engine unable to
        ever emit ``generate_done`` for the re-injected requests; the
        user's ``PendingRequest.done_event`` would never fire.
        Surfaces a clean exception instead of an infinite hang.

        Lazy-imports ``WorkerCmdFailed`` to avoid the
        ``pipeline.py -> orchestrator.py -> pipeline.py`` import
        cycle described at the top of this module.
        """
        try:
            from orchestrator import WorkerCmdFailed
        except ImportError:
            return
        if not isinstance(exc, WorkerCmdFailed):
            return
        orch_cls = self._orch
        fail = getattr(orch_cls, "_fail_all_inflight", None)
        if fail is None:
            return
        try:
            fail(self.model_id, exc)
        except BaseException:  # noqa: BLE001
            log.exception(
                "[pipeline %s] _fail_all_inflight raised",
                self.model_id,
            )


# ---------------------------------------------------------------------------
# Internal sentinel ops
# ---------------------------------------------------------------------------


class _NoOp(Op):
    """Internal: pushed onto the tail by ``submit_front`` to wake the worker.

    Invisible to callers -- never returned, never has its future awaited.
    """

    def execute(self, ctx: OpContext) -> None:  # pragma: no cover - trivial
        return None


# ---------------------------------------------------------------------------
# Concrete Op subclasses
# ---------------------------------------------------------------------------
#
# Each *Op subclass below carries a structured docstring -- the porting
# contract from the implementation plan.  The "Source" line cites the
# legacy ``Orchestrator._*_sync`` body the Op replaces; "Touches",
# "Preserves fixes", and "Invariants" sections are the audit trail.


class RegisterOp(Op):
    """Cold-start a new model and save its image -- replaces ``_register_sync``.

    Source: orchestrator.py legacy ``_register_sync`` body.

    Touches:
      - ``ctx.orch._registry[mid]`` (created here; populated with the
        registry entry shape from ``_register_sync``).
      - ``ctx.orch._set_state`` / ``_locks_ordered`` /
        ``_pick_level`` / ``_image_dir_for``.
      - ``ctx.slots.allocate`` (cold-start L1 slot) /
        ``ctx.slots.deallocate`` (released inline; never flowed into
        ``entry["slot"]``).
      - ``Instance(vllm_config)`` cold-start sequence:
        ``init(gpu).attach().repin().stage().unpin().sleep()
        .cuda_checkpoint().wait()``, then
        ``criu_dump(image_dir).wait()``, then ``_send("exit")`` +
        ``_reset()``.

    Preserves fixes for: none (cold-start has no Known Issues entry --
        it's serialised by the slot allocator and is the simplest op).

    Invariants (do not weaken):
      - The cold-start slot is local and is deallocated inline before
        ``execute`` returns; never set ``entry["slot"]`` to it.
      - The final ``state`` published is ``saved`` (image on disk, no
        live process).  ``entry["instance"]`` is reset to None and
        ``entry["gpu"]`` is reset to None at the same time.
      - The registry entry must exist *before* the worker starts
        running anything else for this model.  Orchestrator.register
        publishes the entry inline before submitting RegisterOp; the
        body below leaves the entry fields as if ``_register_sync``
        had populated them.
    """

    def __init__(self, model_id: str, vllm_config: dict) -> None:
        # model_id is also carried in ctx.model_id; keep it here too so
        # the Op is self-describing in logs.
        self.model_id = model_id
        self.vllm_config = vllm_config

    def execute(self, ctx: OpContext) -> None:
        # Lazy imports: keep ``pipeline.py`` independent of ``instance``
        # / ``slots`` at module scope.  These modules are leaf deps so
        # importing them here does not introduce a cycle.
        import time as _time
        from instance import Instance

        orch = ctx.orch
        slots = ctx.slots
        mid = ctx.model_id
        vllm_config = self.vllm_config

        t0 = _time.perf_counter()
        image_dir = orch._image_dir_for(mid)
        inst = Instance(vllm_config)

        register_slot = slots.allocate(level=1)
        gpu = register_slot.gpu_id
        t_acquired = _time.perf_counter()

        # The registry entry was created by ``Orchestrator.register``
        # before this Op was submitted, but it may have been minimal
        # (just enough to construct the pipeline).  Reconcile to the
        # full ``_register_sync`` shape: state="init", live instance,
        # cold-start gpu.
        entry = orch._registry[mid]
        entry["state"] = "init"
        entry["instance"] = inst
        entry["gpu"] = gpu
        entry["slot"] = None
        entry["pinned_cpu_bytes"] = 0
        entry["total_gpu_bytes"] = 0
        entry["paused"] = False
        entry["paused_since"] = None
        entry["state_since"] = _time.perf_counter()

        with orch._locks_ordered(mid):
            inst.init(gpu).attach().repin().stage().unpin().sleep().cuda_checkpoint().wait()
            inst.criu_dump(image_dir).wait()
            log.info("%s: image saved to %s", mid, image_dir)

            inst._send("exit")
            inst._reset()

            slots.deallocate(register_slot)
            t_done = _time.perf_counter()
            t_wait = t_acquired - t0
            t_exec = t_done - t_acquired
            log.info(
                "%s: registered on GPU %s  "
                "(wait=%.1fs, cold-start=%.1fs, total=%.1fs)",
                mid, gpu, t_wait, t_exec, t_wait + t_exec,
            )

            entry["instance"] = None
            entry["gpu"] = None
            entry["pinned_cpu_bytes"] = inst.pinned_cpu_bytes
            entry["total_gpu_bytes"] = inst.total_gpu_bytes
            orch._set_state(mid, "saved")


class MoveOp(Op):
    """Walk *model_id* up or down the state ladder -- replaces ``_move_sync``.

    Source: orchestrator.py legacy ``_move_sync`` body.  The
    ``prev_future`` / ``prev_gen_future`` / ``prev_gen_events`` chain
    plumbing drops out: the pipeline FIFO carries op ordering by
    construction, so a ``MoveOp`` queued behind a ``GenerateOp`` waits
    for that generate's hand-off Phase 2 + the user thread's Phase 3
    wait by virtue of being later in the queue.

    The ``running -> X`` race (move() called against a model in the
    transient ``running`` sub-state) is preserved at the entry point
    (``Orchestrator.move`` raises before submitting); inside the Op
    the ladder walk assumes ``state != "running"`` because the queue
    serialises.

    Touches:
      - ``ctx.orch._registry[mid]`` (read state, gpu, slot).
      - ``ctx.orch._step_up`` / ``_step_down`` (the ladder primitives).
      - ``ctx.orch._acquire_slot_for_running`` (sub-state announce).
      - ``ctx.orch._set_state`` / ``_locks_ordered``.
      - ``ctx.slots.deallocate`` (slotless-sleep tail).

    Preserves fixes for:
      - "move-eats-generate" (model 7 / model 15): the
        ``prev_gen_future`` / ``prev_gen_events`` polling loops
        disappear, but the underlying invariant ("a move's sleep cmd
        cannot interleave between a generate's wake_up_kv_cache and
        inst.generate") is preserved structurally: ``GenerateOp``
        submits its ``inst.generate`` synchronously inside its
        ``execute``, so any ``MoveOp`` that lands later in the FIFO
        sees the engine post-generate.
      - "Phantom slot leak" (model X, paused-sub then resume): paused
        check at the entry point still rejects ``move(target='saved')``
        on a paused model.

    Invariants (do not weaken):
      - ``state == "running"`` is rejected at the entry point; the
        body assumes the model is in one of ``saved`` / ``checkpoint``
        / ``sleep`` / ``up``.
      - ``announce_state`` is for sub-state announce only (used by
        ``GenerateOp`` Phase 1 to do ``sleep -> running``); the
        slot-acquisition pass for ``up -> running`` happens INSIDE
        the body, not at the entry point.
      - ``ctx.interrupt.raise_if_set()`` is checked between every
        ladder step so a pause can break the walk cleanly.
    """

    def __init__(
        self,
        target: str,
        *,
        target_gpu: int | None = None,
        announce_state: str | None = None,
    ) -> None:
        self.target = target
        self.target_gpu = target_gpu
        self.announce_state = announce_state

    def execute(self, ctx: OpContext) -> None:
        import time as _time

        orch = ctx.orch
        mid = ctx.model_id
        target = self.target
        target_gpu = self.target_gpu
        announce_state = self.announce_state

        # _STATES is module-level in orchestrator.py; re-derive here to
        # avoid an extra ctx field.  Order must match.
        STATES = ["saved", "checkpoint", "sleep", "up"]

        entry = orch._registry[mid]
        current = entry["state"]
        if current == "running":
            if announce_state == "running":
                return
            raise RuntimeError(
                f"model {mid!r} is currently running a generate; "
                f"wait for it to finish before calling move()"
            )
        if current == target and target_gpu is None and announce_state is None:
            log.info("%s: already in '%s' state", mid, target)
            return
        if current == target and announce_state is not None:
            # Sub-state announce path (typically up -> running).
            # Preserve the invariant that ``running`` always holds a
            # slot: a slotless ``up`` must acquire one first.
            if (target == "up" and announce_state == "running"
                    and entry.get("slot") is None):
                ctx.interrupt.raise_if_set()
                orch._acquire_slot_for_running(mid)
            with orch._locks_ordered(mid):
                if entry["state"] == target:
                    orch._set_state(mid, announce_state)
                    return
            current = entry["state"]

        cur_idx = STATES.index(current)
        tgt_idx = STATES.index(target)

        t0 = _time.perf_counter()

        if (target == "sleep" and target_gpu is not None
                and current in ("up", "sleep")
                and entry.get("gpu") != target_gpu):
            # Migrate to a specific GPU: walk down to checkpoint, then
            # back up to (slotless) sleep with target_gpu pinned.
            chk_idx = STATES.index("checkpoint")
            for step in range(cur_idx, chk_idx, -1):
                ctx.interrupt.raise_if_set()
                orch._step_down(mid, STATES[step], STATES[step - 1])
            ctx.interrupt.raise_if_set()
            orch._step_up(mid, "checkpoint", "sleep", target_gpu=target_gpu)
        elif (target == "sleep" and target_gpu is not None
                and current == "sleep" and entry.get("gpu") == target_gpu):
            # Already on target_gpu in sleep; fall through so the tail
            # below releases any slot still held.
            pass
        elif cur_idx < tgt_idx:
            for step in range(cur_idx, tgt_idx):
                ctx.interrupt.raise_if_set()
                cur = STATES[step]
                nxt = STATES[step + 1]
                kw: dict = {}
                if cur == "checkpoint" and nxt == "sleep":
                    kw["target_gpu"] = target_gpu
                if nxt == "up" and announce_state is not None:
                    kw["announce_state"] = announce_state
                orch._step_up(mid, cur, nxt, **kw)
        else:
            for step in range(cur_idx, tgt_idx, -1):
                ctx.interrupt.raise_if_set()
                orch._step_down(mid, STATES[step], STATES[step - 1])

        # Slotless-sleep flavour tail: when target_gpu was specified for
        # a sleep target, ensure no slot is held.
        if target == "sleep" and target_gpu is not None:
            with orch._locks_ordered(mid):
                if entry.get("slot") is not None:
                    ctx.slots.deallocate(entry["slot"])
                    entry["slot"] = None
                    log.info("%s: released slot on GPU %s (slotless)",
                             mid, target_gpu)

        elapsed = _time.perf_counter() - t0
        log.info("%s: %s -> %s  (%.1fs total)", mid, current, target, elapsed)


class EvictForPeerOp(Op):
    """Phase-2 HBM eviction for a peer's wake-up -- replaces ``_evict_for_phase2``.

    Submitted by the *acquirer*'s pipeline worker onto the *incumbent*'s
    pipeline (the cross-pipeline rule, see pipeline_DESIGN.md section
    4.3 and the cycle-detection assertion in
    ``ModelPipeline._assert_no_cross_pipeline_cycle``).  The acquirer
    blocks on this Op's ``.result()`` so it knows the eviction has
    landed before continuing its own Phase-3 wake-up.

    Source: orchestrator.py legacy ``_evict_for_phase2`` body, minus
    the sentinel-future plumbing and ``_drain_inflight_generates``
    call (the FIFO carries the order: any ``GenerateOp`` already in
    flight on the incumbent's worker has either completed its Phase-2
    submit by the time this Op runs, or is still queued behind it and
    will see the post-sleep state).

    Touches:
      - ``ctx.orch._step_down(mid, "up", "sleep")`` (sends sleep cmd
        + reconciles registry).
      - ``ctx.orch._registry[mid]`` (read state, paused).

    Preserves fixes for:
      - "Eviction-mid-generate dormant-engine wedge" (model 13, model 16):
        the wedge happened because a generate could land on the worker
        pipe AFTER the eviction's ``_drain_inflight_generates`` returned
        but BEFORE the eviction's ``sleep`` cmd reached the worker.
        With the FIFO queue the racing generate either landed BEFORE
        EvictForPeerOp was submitted (in which case EvictForPeerOp waits
        for the existing generate to drain off the worker BEFORE its own
        ``sleep`` is sent -- because this Op's ``execute`` runs to
        completion on the incumbent's worker thread, and any generates
        already enqueued on the worker pipe ack first) or AFTER
        (in which case it lands after the ``sleep`` and walks the
        heavyweight ``_step_up(sleep, up)`` path, NOT a generate-on-
        dormant-engine).
      - "Paused-peer silent OOM" (every L1 wake-up while any L2/L3
        peer on the same GPU was paused -- the six waiting requests
        80/81/82/83/84/86 observed 06:37 on 2026-05-18):
        the old branch returned early on ``entry["paused"]`` but the
        acquirer's Phase-2 loop did ``remaining -= share`` anyway, so
        the acquirer believed the HBM was freed and OOM'd inside
        ``cuda_restore`` (CUresult=2).  Now we walk the paused peer
        down to ``sleep`` like any other incumbent.  This is safe by
        construction: the vllm_child ``sleep`` handler explicitly
        documents the invariant "while ``_paused`` is True,
        ``_active_reqs`` is empty -- ``_drain_engine`` therefore has
        no scheduled work to step through here", so the cmd no-ops
        the drain and proceeds straight to ``llm.sleep(level=2)``.
        Post-sleep the engine is at ``_paused=True, _dormant=True``;
        any further generate submitted while paused still parks into
        ``_saved_requests`` (the ``_dormant and not _paused`` fail-
        fast in ``_submit_generate`` deliberately requires *both*
        flags to fire).  On the user's subsequent ``resume()``,
        ``ResumeOp`` already accepts ``state in ("up", "sleep",
        "checkpoint")`` and composes ``MoveOp(target="up")`` to walk
        the engine back up before re-prefilling ``_saved_requests``,
        so the deferred generates survive the round trip.  Trade-off
        documented on ``PauseOp``: a pause+resume cycle that crosses
        peer contention pays the full sleep<->up cost on resume
        instead of the fast ``inst.resume`` cost.

      - "Stale-incumbent CRIU-frozen sleep segfault" (model 15, model 2
        on 2026-05-18, ~07:27): the acquirer's Phase-2 scan in
        ``_step_up`` is lock-free; by the time ``EvictForPeerOp``
        dequeues on the incumbent's pipeline (potentially seconds
        later, behind a queued ``MoveOp``), the incumbent may have
        been walked from ``up`` to ``checkpoint`` by an unrelated op
        (a client ``move(checkpoint)``, a parallel eviction from a
        different acquirer, etc.).  Sending ``sleep`` to a worker
        whose vllm_child has been ``cuda_checkpoint``'d (CUDA context
        CRIU-frozen) drives vLLM's ``sleep`` handler into
        ``torch.cuda.synchronize(0)`` -> ``cuCtxSynchronize`` ->
        **segfault**, killing the vllm_child and wedging the entire
        pipeline (every subsequent ``_send_cmd_with_ack`` parks on
        the dead worker forever).  We close the window by
        re-validating ``entry["state"] == "up"`` at the top of
        ``execute``: pipeline FIFO ordering guarantees no other op
        is concurrently mutating state, so the check is consistent
        for the whole call.  When the incumbent is already below
        ``up`` the HBM share has already been freed by whoever walked
        it down, so the acquirer's ``remaining -= share`` bookkeeping
        remains correct and we can no-op cleanly.

    Invariants (do not weaken):
      - The acquirer must call ``submit_to_peer_and_wait`` (or set
        ``_waiting_on`` manually) when it blocks on this op's future,
        so the cycle detector sees the cross-pipeline edge.
      - ``_step_down(up, sleep)`` reconciles the registry
        unconditionally (releases any slot, flips state to ``sleep``).
        Safe even if the incumbent self-evacuated between the
        acquirer's gate and this Op running.
      - Paused incumbents are evicted just like un-paused ones; the
        ``entry["paused"]`` flag is preserved across the sleep so a
        later ``ResumeOp`` still triggers the ``_saved_requests``
        replay path.  Do NOT add a ``paused``-skip short-circuit
        here -- see the "Paused-peer silent OOM" entry above for the
        bug it reintroduces.
      - The ``state == "up"`` re-check at the top of ``execute`` is
        load-bearing: removing it reintroduces the "Stale-incumbent
        CRIU-frozen sleep segfault" entry above.  Do NOT replace it
        with "let ``_step_down`` figure it out" -- the segfault
        happens inside the worker's vllm_child, not in any layer the
        orchestrator can recover from.
    """

    def __init__(self, *, acquirer_id: str) -> None:
        self.acquirer_id = acquirer_id

    def execute(self, ctx: OpContext) -> None:
        orch = ctx.orch
        mid = ctx.model_id
        entry = orch._registry.get(mid)
        if entry is None:
            log.info(
                "evict-for-peer(%s): incumbent %s vanished; nothing to do",
                self.acquirer_id, mid,
            )
            return
        # Re-validate state under FIFO ordering.  See the
        # "Stale-incumbent CRIU-frozen sleep segfault" entry in this
        # class's docstring: another op queued ahead of us (client
        # ``move(checkpoint)``, parallel eviction, etc.) may have
        # walked the incumbent below ``up`` since the acquirer's
        # lock-free Phase-2 scan picked it.  If so, the HBM share
        # the acquirer was after has already been freed by that
        # other path; ``remaining -= share`` in ``_step_up`` stays
        # correct and we no-op cleanly.  Calling
        # ``_step_down("up", "sleep")`` here against a
        # checkpoint-CUDA'd engine segfaults vllm_child and wedges
        # the pipeline -- never do it.
        current = entry.get("state")
        if current != "up":
            log.info(
                "evict-for-peer(%s): incumbent %s already at %r "
                "(HBM share freed by another op); nothing to do",
                self.acquirer_id, mid, current,
            )
            return
        paused = bool(entry.get("paused"))
        log.info(
            "evict-for-peer(%s): putting incumbent %s to sleep%s",
            self.acquirer_id, mid,
            " (paused)" if paused else "",
        )
        orch._step_down(mid, "up", "sleep")


class GenerateOp(Op):
    """Submit a generate request -- replaces ``_generate_sync`` Phase 1+2.

    Source: orchestrator.py legacy ``_generate_sync`` Phase 1 (walk-up)
    + Phase 2 (``inst.generate`` + ``_inflight`` append).  Phase 3
    (``done_event.wait()`` + result extraction) moves OUT of the
    pipeline worker and onto the user's calling thread (in
    ``Orchestrator.generate``); see "hand-off and return" below.

    Hand-off-and-return semantics
    -----------------------------

    ``execute(ctx)`` returns a ``PendingRequest`` as soon as Phase 2
    has handed the request to the engine.  The pipeline worker then
    moves on to the next op (a ``PauseOp`` submitted at the head of
    the queue, the next ``GenerateOp``, an ``EvictForPeerOp`` for a
    peer's wake-up).  The user's calling thread inside
    ``Orchestrator.generate`` does ``pending.done_event.wait()`` and
    extracts the result.

    Why this matters: a pause MUST not destroy the user's generate.
    With the user thread doing the wait, the engine can finish
    generating tokens whenever it can (potentially across a pause /
    resume window) and the user's `.result()` call returns the full
    output.  See pipeline_DESIGN.md section 5.0.

    Touches:
      - ``ctx.orch._registry[mid]`` (read state, paused, instance).
      - ``ctx.orch._timing`` (reset before walk-up; read after to
        attribute gpu_wait_s / migrate_s onto the request record).
      - Reuses the ``MoveOp`` ladder-walk implementation by composing
        ``MoveOp(target="up", announce_state="running").execute(ctx)``
        directly (NOT submitting a new op -- that would self-deadlock
        on the same pipeline worker).
      - ``ctx.entry["instance"].generate(prompts, sampling_params)``.
      - ``ctx.orch._inflight[mid]`` (append the per-request record).

    Preserves fixes for:
      - "Eviction-mid-generate dormant-engine wedge" (model 13/16):
        no longer relies on the eviction-sentinel re-check loop.
        ``EvictForPeerOp`` lands on the incumbent's pipeline FIFO
        ahead of any ``GenerateOp`` submitted later, so a generate
        that races a Phase-2 eviction either lands BEFORE the
        eviction (and fires against the still-up engine) or AFTER
        (and walks the heavyweight ``_step_up(sleep, up)`` because
        the eviction's sleep already landed).
      - "Pause kills generate" (the design refinement that motivated
        the rename of CancelToken to InterruptFlag): the engine's
        request id and ``done_event`` are owned by the user's
        thread, not the pipeline worker, so a ``PauseOp`` running on
        the worker after this Op returns does NOT cancel the
        in-flight engine request.  See pipeline_DESIGN.md section 5.0.

    Invariants (do not weaken):
      - ``execute`` returns BEFORE ``done_event.wait()`` so the
        pipeline worker is freed for subsequent ops while the engine
        is still computing tokens.
      - The ``_inflight[mid]`` append happens inside ``execute`` (so
        any peer ``EvictForPeerOp`` that runs later sees the
        in-flight generate); the demuxer's ``_on_generate_done``
        listener pops it and sets ``done_event``, unchanged from the
        legacy path.
      - ``ctx.interrupt.raise_if_set()`` checked at the boundaries:
        before walk-up and between walk-up and Phase 2.  The actual
        ``inst.generate`` send is non-cancellable once it leaves the
        Op (it's now the engine's responsibility); a pause that
        lands AFTER ``inst.generate`` returned simply suspends the
        engine via ``inst.pause``, NOT via ``InterruptFlag``.
    """

    def __init__(
        self,
        prompts: list[str],
        sampling_params: dict,
        q_rec: dict,
    ) -> None:
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.q_rec = q_rec

    def execute(self, ctx: OpContext) -> PendingRequest:
        import time as _time

        orch = ctx.orch
        mid = ctx.model_id
        entry = orch._registry[mid]
        q_rec = self.q_rec

        # Phase 1: walk up to "running", announcing the sub-state
        # atomically (skip the observable "up" window).  Compose
        # ``MoveOp.execute`` directly (don't submit a new op -- that
        # would self-deadlock on this very worker).
        if entry.get("paused"):
            # Generate-while-paused: skip the move (resume is the
            # user's responsibility).  Phase 2 below still enqueues
            # the request via inst.generate(); the child's
            # engine.add_request() is not gated by ``_paused`` --
            # only engine.step() is -- so the request lands in the
            # scheduler and runs after a subsequent resume().
            move_gpu_wait = 0.0
            move_migrate = 0.0
            move_up = 0.0
        elif entry["state"] != "running":
            ctx.interrupt.raise_if_set()
            orch._timing.gpu_wait_s = 0.0
            orch._timing.migrate_s = 0.0
            t_move = _time.perf_counter()
            MoveOp(
                target="up", announce_state="running",
            ).execute(ctx)
            total_move = _time.perf_counter() - t_move
            move_gpu_wait = getattr(orch._timing, "gpu_wait_s", 0.0)
            move_migrate = getattr(orch._timing, "migrate_s", 0.0)
            move_up = total_move - move_gpu_wait - move_migrate
        else:
            move_gpu_wait = 0.0
            move_migrate = 0.0
            move_up = 0.0

        if q_rec.get("state") == "done":
            # Pre-empted (e.g. by a future remove() / cancel that
            # flips ``q_rec["state"] = "done"`` between
            # ``submit_generate`` and us dequeuing).  No external
            # writer hits this state today (only the demuxer's
            # ``_on_generate_done`` sets it, and it runs *after*
            # this Op has already returned a PendingRequest), so
            # this branch is currently dead -- but we keep it as
            # defensive coverage.  The pre-set ``done_event`` is
            # load-bearing: ``Orchestrator.submit_generate``'s
            # ``_wait_and_collect`` daemon parks on
            # ``pending.done_event.wait()`` and would hang forever
            # on an un-set Event.  Setting it here means the wait
            # returns immediately and the user-facing future
            # resolves to an empty result list.
            done_event = threading.Event()
            done_event.set()
            return PendingRequest(
                rid="", done_event=done_event, q_rec=q_rec, inst=None,
            )

        # Phase 2: submit to the engine and append to _inflight.
        ctx.interrupt.raise_if_set()
        inst = entry["instance"]
        if inst is None:
            raise RuntimeError(
                f"generate({mid}): no live instance after walk-up; "
                f"state={entry.get('state')!r}"
            )

        if mid not in orch._inflight:
            orch._inflight[mid] = []

        q_rec["gpu_wait_s"] = move_gpu_wait
        q_rec["migrate_s"] = move_migrate
        q_rec["up_s"] = move_up

        done_event = threading.Event()
        rid: str
        # ``entry["_gen_lock"]`` synchronises this generate submit
        # against the demuxer's ``_on_generate_done`` slot-release
        # decision: appending to ``_inflight`` under the lock and
        # checking ``_inflight`` empty under the lock together
        # guarantee the demuxer never releases a slot between our
        # ``inst.generate(...)`` send and our ``_inflight.append``.
        with entry["_gen_lock"]:
            try:
                q_rec["state"] = "generating"
                q_rec["t_gen_start"] = _time.perf_counter()
                if entry.get("paused"):
                    q_rec["t_pause_started"] = q_rec["t_gen_start"]
                inst.generate(self.prompts, self.sampling_params)
                rid = inst.last_req_id
                orch._inflight[mid].append((rid, q_rec, done_event))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                log.error("%s: generate submit failed: %s", mid, exc)
                q_rec["state"] = "error"
                q_rec["t_done"] = _time.perf_counter()
                rid = ""
                done_event.set()

        return PendingRequest(
            rid=rid, done_event=done_event, q_rec=q_rec, inst=inst,
        )


class PauseOp(Op):
    """Pause an actively-generating model -- replaces ``_pause_sync``.

    Source: orchestrator.py legacy ``_pause_sync`` body.

    Submission shape (set by ``Orchestrator.pause``):

      1. Caller (user thread) calls ``pipe.interrupt_now("pause")``,
         which sets ``InterruptFlag``.  Any in-flight ``MoveOp`` /
         ``GenerateOp`` parked in ``wait_or_interrupt`` /
         ``raise_if_set`` bails out with ``Interrupted``.
      2. Caller submits ``PauseOp`` via ``submit_front`` so it runs
         BEFORE any queued ``MoveOp`` / ``ResumeOp`` (so those see
         post-pause state).
      3. The pipeline worker dequeues the bailed-out op (its future
         resolves with ``Interrupted``), then runs ``PauseOp``.

    At the end of ``execute`` we ``ctx.interrupt.reset()`` so the
    next op (a queued ``MoveOp("checkpoint")`` while paused, a future
    ``ResumeOp``) doesn't see a stale flag and bail spuriously.

    Touches:
      - ``ctx.orch._registry[mid]`` (read state, paused; mutate
        slot, paused, paused_since).
      - ``ctx.orch._inflight[mid]`` (read for deferred-pause guard).
      - ``ctx.orch._send_cmd_with_ack(mid, "pause")``.
      - ``ctx.slots.deallocate``.
      - ``ctx.orch._set_state(mid, "up")``.
      - ``ctx.orch._request_log`` (stamp ``t_pause_started`` on
        in-flight requests for the dashboard).

    Preserves fixes for:
      - "Deferred-pause race / phantom-running" (model 7 / model X):
        re-check ``_inflight[mid]`` non-empty before sending the
        worker pause.  Even though the pipeline FIFO already orders
        operations, the demuxer's ``_on_generate_done`` listener can
        pop ``_inflight`` between our entry and our send (the
        listener fires on the demuxer thread, not on the pipeline
        worker), so the re-check is still load-bearing.
      - "Pause does not destroy generate" (the InterruptFlag rename):
        PauseOp does NOT touch ``done_event`` or
        ``inst.generate_results``; it only sends ``inst.pause`` to
        the engine and lets the engine suspend the in-flight tokens.
        The user's calling thread (in ``Orchestrator.generate``) is
        parked on ``done_event.wait()``, which simply doesn't fire
        until a future ``ResumeOp`` un-suspends the engine.

    HBM contract while paused
    -------------------------
    PauseOp deallocates the slot but keeps the engine warm: weights
    + KV cache stay resident at the model's level share of HBM until
    a later ``ResumeOp`` (cheap path) **or** until a peer evicts us
    via ``EvictForPeerOp`` (cold path).  The cold path is taken when
    a peer wakes up on the same GPU and the Phase-2 HBM accounting
    determines the paused model's share is needed.  In that case our
    ``_active_reqs`` snapshot (in ``_saved_requests``) is preserved
    through ``llm.sleep(level=2)`` and the next ``resume()`` walks
    the engine back up before replaying ``_saved_requests`` via
    prefill.  The ``entry["paused"]`` flag is preserved across the
    eviction so ``ResumeOp``'s ``not entry.get("paused")`` gate
    still routes correctly.

    User-visible consequence: ``pause()`` does *not* guarantee
    fast-resume.  Under peer-contention, a pause+resume bracket pays
    the full sleep<->up walk on resume (weights+KV restore + prefill
    of saved requests) instead of the in-place ``inst.resume`` cost.
    Callers that need fast-resume guarantees should hold the model
    at ``running`` (don't pause until peer contention has cleared)
    or ``move`` it to a less contended GPU before pausing.  See
    ``EvictForPeerOp`` "Paused-peer silent OOM" for the history.

    Invariants (do not weaken):
      - Pre-checks must pass BEFORE sending ``inst.pause`` (paused
        already, state != "running", or empty inflight).
      - ``ctx.interrupt.reset()`` is the LAST thing ``execute`` does
        (or, if we bail early on a no-op pre-check, it is still the
        last thing -- the flag could have been set by a racing
        ``Orchestrator.pause`` call even on a no-op path).
    """

    def execute(self, ctx: OpContext) -> None:
        import time as _time

        orch = ctx.orch
        mid = ctx.model_id
        try:
            entry = orch._registry.get(mid)
            if entry is None:
                return
            if entry.get("paused"):
                log.info("%s: already paused, skipping", mid)
                return
            if entry.get("state") != "running":
                log.info("%s: not running (state=%s), pause is a no-op",
                         mid, entry.get("state"))
                return

            # Deferred-pause race: the demuxer's _on_generate_done
            # listener may have popped ``_inflight[mid]`` between
            # the user's entry to ``Orchestrator.pause`` and us
            # picking up the PauseOp here.  If so, sending a worker
            # ``pause`` would pause an empty engine and mint a
            # phantom paused=True entry that the next resume promotes
            # to phantom-running.  Re-check inflight here.
            if not orch._inflight.get(mid):
                log.info("%s: no inflight generates (deferred-pause "
                         "race); pause is a no-op", mid)
                return

            orch._send_cmd_with_ack(mid, "pause")

            t_pause = _time.perf_counter()
            with orch._locks_ordered(mid):
                slot = entry.get("slot")
                if slot is not None:
                    ctx.slots.deallocate(slot)
                    entry["slot"] = None
                entry["paused"] = True
                entry["paused_since"] = t_pause
                orch._set_state(mid, "up")
            with orch._request_lock:
                for rec in orch._request_log:
                    if (rec.get("model_id") == mid
                            and rec.get("state") == "generating"
                            and rec.get("t_done") is None
                            and rec.get("t_pause_started") is None):
                        rec["t_pause_started"] = t_pause
            log.info("%s: paused (slot released, state=up)", mid)
        finally:
            # Always reset so a subsequent op (typically a queued
            # ``move("checkpoint")`` after a pause-then-park-down user
            # sequence, or a future ``ResumeOp``) starts with a clean
            # flag.  This must run on the no-op pre-check paths too,
            # because the user's ``Orchestrator.pause`` always tripped
            # the flag synchronously regardless of which body path we
            # take.
            ctx.interrupt.reset()


class ResumeOp(Op):
    """Resume a paused model -- replaces ``_resume_sync``.

    Source: orchestrator.py legacy ``_resume_sync`` body.

    Touches:
      - ``ctx.orch._registry[mid]`` (read paused, state, instance;
        clear paused, paused_since).
      - ``ctx.orch._inflight[mid]`` (snapshot for "anything to drive"
        gate).
      - Composes ``MoveOp(target="up", announce_state="running")``
        directly (not a fresh submit) when there's something to
        drive.
      - ``ctx.orch._send_cmd_with_ack(mid, "resume")``.
      - ``ctx.orch._request_log`` (close out the pause window).

    Preserves fixes for:
      - "Pause-during-resume race" (model 7 wedge): under the legacy
        path, ``pause`` while ``_resume_sync`` was mid-walk could
        leave both racing on state publication.  Under the pipeline,
        a pause submitted while ``ResumeOp`` is running interrupts
        ``MoveOp.execute`` (composed inside) at a yield-point;
        ``ResumeOp.execute`` propagates the ``Interrupted`` to its
        future, the queue moves to the head-inserted ``PauseOp``,
        and the engine ends up in a coherent paused state.
      - "Tier-A slot stolen mid-publish" (the ``_acquire_slot_for_running``
        re-check): inherits from ``MoveOp`` -> ``_step_up`` ->
        ``_acquire_slot_for_running`` -> ``_step_up(sleep, up,
        announce_state="running")`` self-heal.

    Invariants (do not weaken):
      - The "nothing to drive" short-circuit fires BEFORE the walk-up
        so we don't pay ladder cost on an empty inflight.
      - ``paused=True`` is preserved across ``MoveOp`` (only this
        Op clears it) so saved sub-reqs ride the walk untouched.
      - ``inst.resume`` is sent AFTER the state is ``running`` from
        the walk; the brief ``running + paused=True`` window is
        bounded by the cmd round-trip and tolerated by readers
        (see ``_resume_sync`` docstring).
    """

    def execute(self, ctx: OpContext) -> None:
        import time as _time

        orch = ctx.orch
        mid = ctx.model_id
        entry = orch._registry.get(mid)
        if entry is None:
            return
        if not entry.get("paused"):
            log.info("%s: not paused, resume is a no-op", mid)
            return
        state = entry.get("state")
        if state not in ("up", "sleep", "checkpoint"):
            log.warning("%s: cannot resume from state=%s; "
                        "resume is a no-op", mid, state)
            return

        # Snapshot inflight: empty -> nothing to drive, just clear
        # paused without walking.  Non-empty -> walk up + worker
        # resume.
        inflight = list(orch._inflight.get(mid) or [])
        if not inflight:
            with orch._locks_ordered(mid):
                entry["paused"] = False
                entry["paused_since"] = None
            log.info("%s: nothing to drive; cleared paused, "
                     "state stays %r", mid, state)
            return

        ctx.interrupt.raise_if_set()
        # Walk to running via MoveOp -- compose its execute, do NOT
        # submit a fresh op (that would self-deadlock on this very
        # worker).
        MoveOp(target="up", announce_state="running").execute(ctx)

        ctx.interrupt.raise_if_set()
        orch._send_cmd_with_ack(mid, "resume")

        with orch._locks_ordered(mid):
            entry["paused"] = False
            entry["paused_since"] = None
        t_resume = _time.perf_counter()
        with orch._request_lock:
            for rec in orch._request_log:
                if (rec.get("model_id") == mid
                        and rec.get("t_pause_started") is not None):
                    rec["paused_s"] = (rec.get("paused_s") or 0.0) + (
                        t_resume - rec["t_pause_started"])
                    rec["t_pause_started"] = None
        log.info("%s: resumed (state=running)", mid)


class RemoveOp(Op):
    """Remove a registered model -- replaces ``_remove_sync``.

    Source: orchestrator.py legacy ``_remove_sync`` body.

    Submission shape (set by ``Orchestrator.remove``):

      1. Caller (user thread) refuses paused models inline (mirrors
         the legacy ``_remove_sync`` paused-block).
      2. Caller submits ``RemoveOp`` on the model's pipeline.
      3. Caller schedules pipeline tear-down (drain + worker join +
         pop from ``_pipelines``) AFTER the RemoveOp future resolves.

    The pipeline FIFO replaces the legacy
    ``Orchestrator._drain_inflight_generates``-equivalent
    ``prev_gen_events`` drain at the entry point: any ``GenerateOp``
    in flight on this model has either already completed (the
    PendingRequest was returned, the user thread has its done_event)
    or is still queued behind RemoveOp (in which case it never runs;
    the worker shuts down after RemoveOp).

    Touches:
      - Composes ``MoveOp(target="saved")`` directly when state !=
        "saved" (walks the ladder all the way down).
      - ``ctx.orch._registry.pop(mid)`` (final removal).
      - ``shutil.rmtree(image_dir)``.

    Preserves fixes for: none (remove has no Known Issues entry --
        the entry-point paused-block is the only safety check).

    Invariants (do not weaken):
      - Paused models are refused at the entry point; this body
        assumes ``not entry["paused"]``.
      - The pipeline tear-down (``pipe.shutdown(...)``) MUST happen
        AFTER this Op resolves.  ``Orchestrator.remove`` schedules
        that tear-down on a background daemon thread so the
        user-visible call stays non-blocking.

    Remove + immediate re-register safety
    -------------------------------------

    A racing ``register(mid)`` between this Op's
    ``_registry.pop(mid)`` and the background ``_teardown`` thread's
    ``_pipelines.pop(mid)`` is handled by the daemon's
    ``shutdown(drain=True)``, NOT by any blocking pattern in
    ``_make_pipeline`` (which simply returns the existing pipeline
    if one is registered).  The sequence:

      1. RemoveOp.execute runs ``_registry.pop(mid)``, returns.
      2. ``_teardown`` daemon wakes on ``_op_fut.result()``.
         Before it calls ``shutdown``, the user lands a fresh
         ``register(mid)``.  That ``register`` (a) sees
         ``mid not in _registry``, (b) creates a new entry,
         (c) calls ``_make_pipeline(mid)`` which returns THIS
         pipeline (because ``_pipelines[mid]`` still references
         it), and (d) submits a new RegisterOp onto our queue.
      3. ``_teardown`` then calls ``shutdown(drain=True)``,
         which waits for every queued op (including that fresh
         RegisterOp) to complete before stopping the worker.
         The RegisterOp populates the new registry entry as
         normal.
      4. ``_teardown`` finally pops the pipeline from
         ``_pipelines``.  Subsequent ``move``/``generate``/etc.
         calls observe ``mid not in _pipelines`` and create a
         brand-new pipeline via ``_make_pipeline``.

    The ``_build_ctx`` re-fetch (see ``ModelPipeline._build_ctx``)
    keeps ``ctx.entry`` pointing at the NEW registry entry for
    the fresh RegisterOp despite the pipeline holding a
    construction-time reference to the OLD entry dict.
    """

    def execute(self, ctx: OpContext) -> None:
        import os as _os
        import shutil as _shutil

        orch = ctx.orch
        mid = ctx.model_id
        entry = orch._registry.get(mid)
        if entry is None:
            return
        if entry["state"] != "saved":
            ctx.interrupt.raise_if_set()
            MoveOp(target="saved").execute(ctx)
        image_dir = entry.get("image_dir")
        if image_dir and _os.path.isdir(image_dir):
            _shutil.rmtree(image_dir)
            log.info("%s: deleted image %s", mid, image_dir)
        orch._registry.pop(mid, None)
        log.info("%s: removed", mid)
