"""Primitive-level tests for ``pipeline.py``.

No orchestrator integration -- everything here exercises the pipeline
infrastructure directly with fake / no-op ``Op`` subclasses.

Run from the package directory::

    python -m pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib

from pipeline import (
    EvictForPeerOp,
    InterruptFlag,
    Interrupted,
    ModelPipeline,
    Op,
    OpContext,
    PauseOp,
    PendingRequest,
    ResumeOp,
)


# ---------------------------------------------------------------------------
# Fake Ops used across tests
# ---------------------------------------------------------------------------


class _RecordOp(Op):
    """Append the op's tag to a shared list when it runs."""

    def __init__(self, tag: str, sink: list[str], *, sleep: float = 0.0) -> None:
        self.tag = tag
        self.sink = sink
        self.sleep = sleep

    def execute(self, ctx: OpContext) -> str:
        if self.sleep:
            time.sleep(self.sleep)
        self.sink.append(self.tag)
        return self.tag


class _RaiseOp(Op):
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def execute(self, ctx: OpContext) -> None:
        raise self.exc


class _WaitOp(Op):
    """Park in ``wait_or_interrupt`` until either ``ev`` fires or pause."""

    def __init__(self, ev: threading.Event) -> None:
        self.ev = ev

    def execute(self, ctx: OpContext) -> str:
        ctx.interrupt.wait_or_interrupt(self.ev)
        return "ok"


class _RaiseIfSetOp(Op):
    """Polling-style op that yields between fake work units."""

    def __init__(
        self,
        steps: int = 5,
        *,
        per_step_sleep: float = 0.05,
        sink: list[str] | None = None,
    ) -> None:
        self.steps = steps
        self.per_step_sleep = per_step_sleep
        self.sink = sink if sink is not None else []

    def execute(self, ctx: OpContext) -> int:
        for i in range(self.steps):
            ctx.interrupt.raise_if_set()
            time.sleep(self.per_step_sleep)
            self.sink.append(f"step{i}")
        return self.steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    model_id: str = "m1",
    *,
    pipelines: dict | None = None,
) -> ModelPipeline:
    """Build a ModelPipeline with stub deps suitable for primitive tests.

    Pass a shared ``pipelines`` dict to register multiple peers under
    the same registry (cross-pipeline tests).
    """
    entry: dict = {"instance": None}
    if pipelines is None:
        pipelines = {}
    pipe = ModelPipeline(
        model_id=model_id,
        entry=entry,
        slots=object,    # tests don't touch Slots
        orch=object,     # tests don't touch Orchestrator
        pipelines=pipelines,
    )
    pipelines[model_id] = pipe
    return pipe


@pytest.fixture
def pipe():
    p = _make_pipeline("m1")
    yield p
    p.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# FIFO + submit_front
# ---------------------------------------------------------------------------


def test_fifo_ordering(pipe: ModelPipeline) -> None:
    sink: list[str] = []
    futs = [pipe.submit(_RecordOp(f"op{i}", sink)) for i in range(5)]
    for f in futs:
        f.result(timeout=2.0)
    assert sink == ["op0", "op1", "op2", "op3", "op4"]


def test_submit_front_runs_ahead_of_queued(pipe: ModelPipeline) -> None:
    sink: list[str] = []
    # Slow op that blocks the worker so we can stack things behind it.
    blocker_release = threading.Event()

    class _BlockOp(Op):
        def execute(self, ctx: OpContext) -> str:
            blocker_release.wait(timeout=2.0)
            sink.append("blocker")
            return "blocker"

    pipe.submit(_BlockOp())
    # Stack two tail ops behind the blocker.
    pipe.submit(_RecordOp("tail-a", sink))
    pipe.submit(_RecordOp("tail-b", sink))
    # Insert a front op while the blocker is still running.
    time.sleep(0.05)
    front_fut = pipe.submit_front(_RecordOp("front", sink))
    blocker_release.set()
    front_fut.result(timeout=2.0)
    pipe.drain(timeout=2.0)
    # Front must run before any of the tail items but after the blocker
    # (which is already executing when submit_front lands).
    assert sink[0] == "blocker"
    assert sink[1] == "front"
    assert sink[2:] == ["tail-a", "tail-b"]


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------


def test_interrupt_raises_in_wait_or_interrupt(pipe: ModelPipeline) -> None:
    ev = threading.Event()
    fut = pipe.submit(_WaitOp(ev))
    # Let the op enter wait_or_interrupt.
    time.sleep(0.05)
    pipe.interrupt_now("test")
    with pytest.raises(Interrupted):
        fut.result(timeout=2.0)


def test_interrupt_when_idle_is_noop(pipe: ModelPipeline) -> None:
    """interrupt() between ops does not crash; the next op sees the flag.

    This documents the rule that ``InterruptFlag`` is sticky until reset.
    Real callers (Orchestrator.pause) reset it inside ``PauseOp``.
    """
    pipe.interrupt_now("between-ops")
    sink: list[str] = []
    # Without a reset, a polling op fires Interrupted immediately.
    op = _RaiseIfSetOp(steps=3, per_step_sleep=0.0, sink=sink)
    fut = pipe.submit(op)
    with pytest.raises(Interrupted):
        fut.result(timeout=2.0)


def test_raise_if_set_yields_promptly(pipe: ModelPipeline) -> None:
    sink: list[str] = []
    op = _RaiseIfSetOp(steps=20, per_step_sleep=0.05, sink=sink)
    fut = pipe.submit(op)
    # Fire interrupt mid-stride.
    time.sleep(0.15)
    pipe.interrupt_now("midstride")
    with pytest.raises(Interrupted):
        fut.result(timeout=2.0)
    # Op took at most ~5 steps before bailing.
    assert len(sink) <= 7, f"raise_if_set was too slow to yield: {sink}"


def test_pipeline_continues_after_interrupted(pipe: ModelPipeline) -> None:
    """An Interrupted op resolves its future; the pipeline keeps running."""
    ev = threading.Event()
    fut1 = pipe.submit(_WaitOp(ev))
    sink: list[str] = []
    fut2 = pipe.submit(_RecordOp("after", sink))
    time.sleep(0.05)
    pipe.interrupt_now("test")
    with pytest.raises(Interrupted):
        fut1.result(timeout=2.0)
    # Reset the flag so fut2 is not interrupted before it starts.
    # This mirrors what PauseOp does at the end of its execute().
    pipe.interrupt.reset()
    # fut2 was queued before the interrupt landed; without reset it would
    # also raise.  After reset, the pipeline keeps draining cleanly.
    fut2.result(timeout=2.0)
    assert sink == ["after"]


# ---------------------------------------------------------------------------
# wait_or_interrupt -- ev-wins / flag-wins / either-or semantics
# ---------------------------------------------------------------------------


def test_wait_or_interrupt_ev_wins(pipe: ModelPipeline) -> None:
    ev = threading.Event()
    fut = pipe.submit(_WaitOp(ev))
    time.sleep(0.05)
    ev.set()
    assert fut.result(timeout=2.0) == "ok"


def test_wait_or_interrupt_flag_wins(pipe: ModelPipeline) -> None:
    ev = threading.Event()
    fut = pipe.submit(_WaitOp(ev))
    time.sleep(0.05)
    pipe.interrupt_now("test")
    with pytest.raises(Interrupted):
        fut.result(timeout=2.0)


def test_wait_or_interrupt_timeout_returns_false() -> None:
    """Direct test of InterruptFlag.wait_or_interrupt timeout semantics."""
    flag = InterruptFlag()
    ev = threading.Event()
    t0 = time.monotonic()
    got = flag.wait_or_interrupt(ev, timeout=0.2)
    elapsed = time.monotonic() - t0
    assert got is False
    assert 0.15 <= elapsed <= 0.6, f"timeout drift: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_op_exception_propagates_to_future(pipe: ModelPipeline) -> None:
    fut = pipe.submit(_RaiseOp(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        fut.result(timeout=2.0)


def test_pipeline_keeps_running_after_op_exception(pipe: ModelPipeline) -> None:
    fut1 = pipe.submit(_RaiseOp(RuntimeError("boom")))
    sink: list[str] = []
    fut2 = pipe.submit(_RecordOp("after", sink))
    with pytest.raises(RuntimeError):
        fut1.result(timeout=2.0)
    fut2.result(timeout=2.0)
    assert sink == ["after"]


# ---------------------------------------------------------------------------
# drain
# ---------------------------------------------------------------------------


def test_drain_waits_for_all_queued(pipe: ModelPipeline) -> None:
    sink: list[str] = []
    for i in range(10):
        pipe.submit(_RecordOp(f"op{i}", sink, sleep=0.01))
    assert pipe.drain(timeout=5.0) is True
    assert sink == [f"op{i}" for i in range(10)]


def test_drain_timeout(pipe: ModelPipeline) -> None:
    blocker_release = threading.Event()

    class _BlockOp(Op):
        def execute(self, ctx: OpContext) -> None:
            blocker_release.wait()

    pipe.submit(_BlockOp())
    assert pipe.drain(timeout=0.1) is False
    blocker_release.set()
    assert pipe.drain(timeout=2.0) is True


# ---------------------------------------------------------------------------
# Multi-pipeline isolation
# ---------------------------------------------------------------------------


def test_multiple_pipelines_run_concurrently() -> None:
    shared: dict = {}
    p_a = _make_pipeline("a", pipelines=shared)
    p_b = _make_pipeline("b", pipelines=shared)
    try:
        sink_a: list[str] = []
        sink_b: list[str] = []
        # Each pipeline's worker is independent; a slow op on A must not
        # delay B.
        ev_release = threading.Event()

        class _BlockedOp(Op):
            def execute(self, ctx: OpContext) -> None:
                ev_release.wait()

        p_a.submit(_BlockedOp())
        p_b.submit(_RecordOp("b1", sink_b))
        p_b.submit(_RecordOp("b2", sink_b))
        p_b.drain(timeout=2.0)
        assert sink_b == ["b1", "b2"]
        assert sink_a == []
        ev_release.set()
        p_a.drain(timeout=2.0)
    finally:
        p_a.shutdown(timeout=2.0)
        p_b.shutdown(timeout=2.0)


def test_one_pipeline_interrupt_does_not_affect_another() -> None:
    shared: dict = {}
    p_a = _make_pipeline("a", pipelines=shared)
    p_b = _make_pipeline("b", pipelines=shared)
    try:
        ev = threading.Event()
        fut_b = p_b.submit(_WaitOp(ev))
        # Interrupt only A.
        p_a.interrupt_now("only-a")
        # B's flag stays clean; ev.set() lets it return normally.
        time.sleep(0.05)
        ev.set()
        assert fut_b.result(timeout=2.0) == "ok"
    finally:
        p_a.shutdown(timeout=2.0)
        p_b.shutdown(timeout=2.0)


# ---------------------------------------------------------------------------
# OpContext surface
# ---------------------------------------------------------------------------


def test_op_context_carries_orch_and_entry(pipe: ModelPipeline) -> None:
    """Op subclasses access ``ctx.orch``, ``ctx.entry``, ``ctx.slots``."""
    seen: dict[str, object] = {}

    class _CaptureOp(Op):
        def execute(self, ctx: OpContext) -> None:
            seen["model_id"] = ctx.model_id
            seen["orch"] = ctx.orch
            seen["slots"] = ctx.slots
            seen["entry"] = ctx.entry
            seen["interrupt"] = ctx.interrupt
            seen["pipelines"] = ctx.pipelines

    pipe.submit(_CaptureOp()).result(timeout=2.0)
    assert seen["model_id"] == "m1"
    assert seen["interrupt"] is pipe.interrupt
    assert pipe.model_id in seen["pipelines"]  # type: ignore[operator]


def test_submit_to_peer_and_wait_acyclic() -> None:
    """A waits on B's result.  No cycle, completes cleanly."""
    shared: dict = {}
    p_a = _make_pipeline("a", pipelines=shared)
    p_b = _make_pipeline("b", pipelines=shared)
    try:
        sink: list[str] = []

        class _CallPeer(Op):
            def execute(self, ctx: OpContext) -> str:
                # B's pipeline registers under ctx.pipelines via the
                # constructor, so we can find p_b through ctx.
                peer = ctx.pipelines["b"]
                # The current pipeline (A) must be the one whose worker
                # is running us; we can't reach "self" from ctx, so
                # fish it out via threading.
                src = ctx.pipelines["a"]
                return src.submit_to_peer_and_wait(
                    peer, _RecordOp("from-a-via-b", sink),
                )

        fut = p_a.submit(_CallPeer())
        assert fut.result(timeout=2.0) == "from-a-via-b"
        assert sink == ["from-a-via-b"]
    finally:
        p_a.shutdown(timeout=2.0)
        p_b.shutdown(timeout=2.0)


def test_submit_to_peer_and_wait_detects_cycle() -> None:
    """A waits on B which submits back to A -> assertion fires."""
    shared: dict = {}
    p_a = _make_pipeline("a", pipelines=shared)
    p_b = _make_pipeline("b", pipelines=shared)
    try:
        # When B's worker tries to submit to A while A's worker is
        # waiting on B (i.e. A._waiting_on is B), that's the cycle the
        # detector catches.  We synthesize this by making B's op submit
        # back to A.

        class _SubmitBackToA(Op):
            def execute(self, ctx: OpContext) -> None:
                # B's worker is currently running us.  A is waiting on
                # us via _waiting_on.  A submit onto A from here MUST
                # raise the cycle assertion.
                a = ctx.pipelines["a"]
                # This call should AssertionError before queuing.
                a.submit(_RecordOp("back-to-a", []))

        class _CallPeerWhichLoops(Op):
            def execute(self, ctx: OpContext) -> None:
                src = ctx.pipelines["a"]
                peer = ctx.pipelines["b"]
                # A._waiting_on = B for the duration of this call.
                src.submit_to_peer_and_wait(peer, _SubmitBackToA())

        fut = p_a.submit(_CallPeerWhichLoops())
        with pytest.raises(AssertionError, match="cross-pipeline cycle"):
            fut.result(timeout=2.0)
    finally:
        p_a.shutdown(timeout=2.0)
        p_b.shutdown(timeout=2.0)


def test_op_context_inst_reflects_current_entry(pipe: ModelPipeline) -> None:
    """``ctx.inst`` is rebuilt per op so it reflects ``entry["instance"]``."""
    seen: list[object] = []

    class _PeekInst(Op):
        def execute(self, ctx: OpContext) -> None:
            seen.append(ctx.inst)

    # First peek: entry["instance"] is None.
    pipe.submit(_PeekInst()).result(timeout=2.0)
    # Mutate the entry as if a register/move had landed.
    pipe._entry["instance"] = object()
    pipe.submit(_PeekInst()).result(timeout=2.0)
    assert seen[0] is None
    assert seen[1] is pipe._entry["instance"]


# ---------------------------------------------------------------------------
# PendingRequest dataclass smoke
# ---------------------------------------------------------------------------


def test_pending_request_dataclass() -> None:
    ev = threading.Event()
    pr = PendingRequest(rid="r1", done_event=ev, q_rec={"foo": 1}, inst=None)
    assert pr.rid == "r1"
    assert pr.done_event is ev
    assert pr.q_rec == {"foo": 1}


# ---------------------------------------------------------------------------
# Regression tests for post-review fixes and the Known Issues catalogue in
# pipeline_DESIGN.md section 8.
#
# These tests cover STRUCTURAL properties of the pipeline that make the
# original bug shapes non-recurrable.  Full end-to-end repros (real vLLM
# subprocesses, multi-GPU eviction) need real hardware, so they live
# outside this directory as the imperative scripts/test_generate.py.
# ---------------------------------------------------------------------------


def test_bidirectional_cross_pipeline_no_deadlock() -> None:
    """Fix #2 regression: simultaneous cross-pipeline submits must not
    deadlock.

    The TOCTOU window the fix closes: A's
    ``submit_to_peer_and_wait(B, opA)`` and B's
    ``submit_to_peer_and_wait(A, opB)`` could both clear their cycle
    checks (each sees the other's ``_waiting_on`` still None), both
    then set their own ``_waiting_on``, and both block waiting on
    each other forever.  With the fix (``_waiting_on`` set BEFORE
    ``peer.submit``), at least one side observes the cycle edge during
    its peer's submit-time check and raises AssertionError, freeing
    the other to complete.

    Without the fix this test deadlocks; the ``fut.result(timeout=...)``
    calls raise TimeoutError.
    """
    shared: dict = {}
    p_a = _make_pipeline("a", pipelines=shared)
    p_b = _make_pipeline("b", pipelines=shared)
    try:
        # Both ops rendezvous on a barrier so the race is forced.
        ready = threading.Barrier(2)

        def _build_cross_op(label: str, src_id: str, peer_id: str) -> Op:
            class _CrossOp(Op):
                def execute(self, ctx: OpContext) -> str:
                    src = ctx.pipelines[src_id]
                    peer = ctx.pipelines[peer_id]
                    ready.wait(timeout=2.0)
                    try:
                        src.submit_to_peer_and_wait(
                            peer, _RecordOp(f"{label}-inner", []),
                        )
                        return f"{label}-completed"
                    except AssertionError:
                        return f"{label}-cycle"
            return _CrossOp()

        fut_a = p_a.submit(_build_cross_op("a", "a", "b"))
        fut_b = p_b.submit(_build_cross_op("b", "b", "a"))

        # Both must return within the timeout.  A deadlock here is the
        # exact regression we're guarding against.
        ra = fut_a.result(timeout=3.0)
        rb = fut_b.result(timeout=3.0)

        # At least one side must have observed the cycle.  Possible
        # outcomes given thread interleaving:
        #   ("a-cycle", "b-completed")  -- B got first, A bailed
        #   ("a-completed", "b-cycle")  -- A got first, B bailed
        #   ("a-cycle", "b-cycle")      -- both saw each other's edge
        outcomes = {ra, rb}
        assert "a-cycle" in outcomes or "b-cycle" in outcomes, (
            f"expected at least one cycle bail, got {outcomes!r}"
        )
    finally:
        p_a.shutdown(timeout=2.0)
        p_b.shutdown(timeout=2.0)


def test_remove_then_register_uses_fresh_entry() -> None:
    """Fix #3 regression: ``ctx.entry`` must not go stale across a
    remove + immediate-register sequence on the same ``model_id``.

    Simulates the orchestrator race:

      1. ``ModelPipeline`` is constructed with ``old_entry`` cached
         on ``self._entry``.
      2. The orchestrator's ``_registry["m1"]`` is replaced by
         ``new_entry`` (i.e., RemoveOp popped + a fresh register
         landed before ``_teardown`` popped the pipeline).
      3. A subsequent op submitted onto THIS pipeline must observe
         ``ctx.entry is new_entry``, not the stale ``self._entry``.
    """
    old_entry: dict = {"instance": None, "tag": "old"}
    new_entry: dict = {"instance": None, "tag": "new"}

    class _StubOrch:
        # Mutable: tests swap the m1 entry between ops.
        _registry: dict = {"m1": old_entry}

    pipe = ModelPipeline(
        model_id="m1",
        entry=old_entry,
        slots=object,
        orch=_StubOrch,
        pipelines={},
    )
    try:
        seen: list[dict] = []

        class _PeekEntryOp(Op):
            def execute(self, ctx: OpContext) -> None:
                seen.append(ctx.entry)

        pipe.submit(_PeekEntryOp()).result(timeout=2.0)
        # Simulate remove + immediate-register: the registry's mid
        # entry is rebound to a brand-new dict.
        _StubOrch._registry["m1"] = new_entry
        pipe.submit(_PeekEntryOp()).result(timeout=2.0)

        assert seen[0] is old_entry
        assert seen[1] is new_entry, (
            "ctx.entry must be re-fetched from _orch._registry per op, "
            "otherwise a remove+register sequence on the same model_id "
            "leaves the pipeline observing a stale construction-time entry."
        )
    finally:
        pipe.shutdown(timeout=2.0)


def test_eviction_then_generate_runs_in_fifo_order() -> None:
    """Known Issues regression: Eviction-mid-generate dormant-engine wedge
    (model 13 / model 16).

    Legacy bug shape: a Phase-2 eviction's ``sleep`` cmd and a racing
    ``generate`` cmd could land out of order on the worker pipe,
    causing the generate to fire against a dormant engine and hang
    forever.

    Under explicit pipelines this is structurally impossible: both
    ops land on the same per-model FIFO queue, ordered by submission
    time, executed serially by the model's worker thread.  Even if
    they're submitted microseconds apart (the original trace showed a
    4ms gap), the second op cannot dequeue until the first has
    returned.

    This test asserts that structural property: an op that mutates
    shared state ("eviction"), submitted just before an op that reads
    it ("generate"), MUST be observed by the reader -- regardless of
    timing or sleep duration.
    """
    pipe = _make_pipeline("m1")
    try:
        state: dict[str, str] = {"engine": "running"}
        observed: list[str] = []

        class _SlowEvictionOp(Op):
            """Mimics EvictForPeerOp Phase 2: sleeps to flush, then
            flips engine state to 'sleep'.  The sleep here stands in
            for the worker cmd round-trip + ack."""
            def execute(self, ctx: OpContext) -> None:
                time.sleep(0.05)
                state["engine"] = "sleep"

        class _FastGenerateOp(Op):
            """Mimics GenerateOp: reads engine state and records it."""
            def execute(self, ctx: OpContext) -> str:
                observed.append(state["engine"])
                return state["engine"]

        # Submit back-to-back, mirroring the model 13 race window.
        fut_e = pipe.submit(_SlowEvictionOp())
        fut_g = pipe.submit(_FastGenerateOp())

        fut_e.result(timeout=2.0)
        assert fut_g.result(timeout=2.0) == "sleep"
        # The generate observed POST-eviction state -- not the
        # pre-eviction "running" that would re-create the dormant-engine
        # wedge.
        assert observed == ["sleep"]
    finally:
        pipe.shutdown(timeout=2.0)


def test_op_exception_does_not_poison_subsequent_ops() -> None:
    """Known Issues regression: ``sub`` busy-spin + silent-pause
    failure (legacy ``Orchestrator.move`` re-raise cascade).

    Legacy bug shape: ``move()`` chained on the shared
    ``_futures[mid]`` via ``prev.result()``.  When an earlier move
    failed (e.g., HBM eviction error), every subsequent move on the
    same model re-raised the original exception without re-reading
    the world, masking the real (recovered) state.

    Under pipelines, each op has its own ``concurrent.futures.Future``
    bound to its own ``Op.execute``.  An exception resolves ONLY that
    one future; the worker dequeues the next op and runs it cleanly.
    """
    pipe = _make_pipeline("m1")
    try:
        sink: list[str] = []
        fut_bad = pipe.submit(_RaiseOp(RuntimeError("simulated move failure")))
        fut_ok1 = pipe.submit(_RecordOp("after-1", sink))
        fut_ok2 = pipe.submit(_RecordOp("after-2", sink))

        with pytest.raises(RuntimeError, match="simulated move failure"):
            fut_bad.result(timeout=2.0)
        # Subsequent ops MUST execute normally; their futures resolve
        # with their own return values, not the prior exception.
        assert fut_ok1.result(timeout=2.0) == "after-1"
        assert fut_ok2.result(timeout=2.0) == "after-2"
        assert sink == ["after-1", "after-2"]
    finally:
        pipe.shutdown(timeout=2.0)


def test_evict_for_peer_walks_paused_incumbent_down() -> None:
    """Bug 'Paused-peer silent OOM' (six waiting requests, 06:37 2026-05-18).

    Legacy bug shape: ``EvictForPeerOp.execute`` short-circuited on
    ``entry.get("paused")`` and returned without sending the worker
    ``sleep`` cmd.  But the acquirer's Phase-2 loop in
    ``_step_up`` ran ``remaining -= share`` unconditionally regardless
    of whether the EvictForPeerOp actually freed anything, so the
    acquirer believed Phase 2 had freed enough HBM.  It then called
    ``wake_up_kv_cache`` and OOM'd inside ``cuda_restore`` with
    ``CUresult=2``.  Every L1 generate that raced any paused L2/L3
    peer on the same GPU hung indefinitely (q_rec stayed at
    ``state="waiting"`` -- that surface is patched in ``orchestrator.py``
    by Fix B; this test guards the underlying eviction trigger).

    Fix (A.3): ``EvictForPeerOp`` walks the paused incumbent down to
    ``sleep`` just like any other.  Safe because the vllm_child
    ``sleep`` handler explicitly documents "while ``_paused`` is True,
    ``_active_reqs`` is empty -- ``_drain_engine`` therefore has no
    scheduled work to step through here", and ``ResumeOp`` already
    accepts ``state in ("up", "sleep", "checkpoint")``.

    This test guards the structural property at the pipeline layer: a
    paused, slotless, state=up incumbent IS walked down via
    ``_step_down(mid, "up", "sleep")``.  The ``entry["paused"]`` flag
    must be preserved across the eviction so a later ResumeOp routes
    through the ``_saved_requests`` replay path.
    """
    step_down_calls: list[tuple[str, str, str]] = []

    class _StubOrch:
        # Incumbent is paused (KV cache + weights still resident at the
        # model's level share), slot already released by PauseOp, state
        # is the post-pause "up" the legacy short-circuit was checking.
        _registry: dict = {
            "incumbent": {
                "state": "up",
                "paused": True,
                "paused_since": 123.456,
                "slot": None,
                "instance": None,
            },
        }

        @staticmethod
        def _step_down(mid: str, from_state: str, to_state: str) -> None:
            step_down_calls.append((mid, from_state, to_state))
            # Mimic the real ``_step_down(up, sleep)`` post-conditions
            # the test cares about: state flips to ``sleep``, the
            # ``paused`` flag is preserved (NOT cleared -- that's
            # ResumeOp's job and the bug we're guarding against).
            ent = _StubOrch._registry[mid]
            ent["state"] = "sleep"

    pipe = ModelPipeline(
        model_id="incumbent",
        entry=_StubOrch._registry["incumbent"],
        slots=object,
        orch=_StubOrch,
        pipelines={},
    )
    try:
        pipe.submit(
            EvictForPeerOp(acquirer_id="acquirer")
        ).result(timeout=2.0)

        assert step_down_calls == [("incumbent", "up", "sleep")], (
            f"EvictForPeerOp must walk paused incumbent down to sleep; "
            f"got step_down_calls={step_down_calls!r}.  If this is "
            f"empty the paused-skip short-circuit regressed and the "
            f"'Paused-peer silent OOM' hang is back."
        )
        entry = _StubOrch._registry["incumbent"]
        assert entry["state"] == "sleep"
        assert entry["paused"] is True, (
            "entry['paused'] must survive the eviction so the next "
            "ResumeOp routes through the _saved_requests replay path "
            "instead of treating the incumbent as a fresh wake-up."
        )
    finally:
        pipe.shutdown(timeout=2.0)


def test_evict_for_peer_handles_vanished_incumbent() -> None:
    """``EvictForPeerOp`` against a model that has been removed must
    no-op cleanly (incumbent vanished between Phase-2 scan and the
    cross-pipeline submission landing on the incumbent's worker).
    """
    class _StubOrch:
        _registry: dict = {}  # incumbent vanished

        @staticmethod
        def _step_down(*_args: object, **_kw: object) -> None:
            raise AssertionError(
                "_step_down must NOT be called for a vanished incumbent"
            )

    pipe = ModelPipeline(
        model_id="ghost",
        entry={"instance": None},  # construction-time stub; _build_ctx
                                   # refetches from _registry per op and
                                   # gets None.
        slots=object,
        orch=_StubOrch,
        pipelines={},
    )
    try:
        # Should resolve without raising.
        pipe.submit(
            EvictForPeerOp(acquirer_id="acquirer")
        ).result(timeout=2.0)
    finally:
        pipe.shutdown(timeout=2.0)


def test_evict_for_peer_noops_on_stale_below_up_incumbent() -> None:
    """Bug 'Stale-incumbent CRIU-frozen sleep segfault' (model 15 and
    model 2, ~07:27 on 2026-05-18).

    Acquirer's Phase-2 scan in ``_step_up`` is lock-free: it picks the
    incumbent at ``state == "up"`` without holding the incumbent's
    pipeline.  By the time the resulting ``EvictForPeerOp`` actually
    dequeues on the incumbent's pipeline (potentially seconds later,
    sitting behind a queued ``MoveOp``), an unrelated op (a client
    ``move(checkpoint)``, a parallel eviction, ...) may have walked
    the incumbent past ``up`` and CRIU-frozen its CUDA context
    (``cuda_checkpoint``).  Sending ``sleep`` to that worker drives
    vLLM's sleep path into ``torch.cuda.synchronize`` ->
    ``cuCtxSynchronize`` -> SEGFAULT, killing vllm_child and
    permanently wedging the pipeline (every later
    ``_send_cmd_with_ack`` parks on the dead worker forever, which
    is exactly what trapped the model 15 / model 2 generates queued
    at ``submit_rel_s=1401.9``).

    Fix: re-check ``entry["state"] == "up"`` at the top of
    ``EvictForPeerOp.execute``; if not, no-op cleanly.  Safe because
    (a) pipeline FIFO ordering guarantees no concurrent state
    mutation while this op runs, and (b) any walker that took the
    incumbent below ``up`` already freed the HBM share the acquirer
    was after, so ``remaining -= share`` in ``_step_up`` Phase 2 is
    still accurate.

    This test guards against regression of the re-check.
    """
    step_down_calls: list[tuple[str, str, str]] = []

    class _StubOrch:
        _registry: dict = {
            "incumbent": {
                "state": "checkpoint",  # walked past "up" by a queued
                                        # MoveOp(checkpoint) that ran
                                        # before us; CUDA context is
                                        # CRIU-frozen.
                "paused": True,         # paused-AND-stale is the exact
                                        # situation that hit production:
                                        # A.3 made paused incumbents
                                        # eligible, which then exposed
                                        # the stale-state TOCTOU.
                "slot": None,
                "instance": None,
            },
        }

        @staticmethod
        def _step_down(mid: str, from_state: str, to_state: str) -> None:
            step_down_calls.append((mid, from_state, to_state))

    pipe = ModelPipeline(
        model_id="incumbent",
        entry=_StubOrch._registry["incumbent"],
        slots=object,
        orch=_StubOrch,
        pipelines={},
    )
    try:
        pipe.submit(
            EvictForPeerOp(acquirer_id="acquirer")
        ).result(timeout=2.0)

        assert step_down_calls == [], (
            "EvictForPeerOp must NOT call _step_down when the incumbent "
            "has already been walked below 'up' -- doing so sends 'sleep' "
            "to a checkpoint-CUDA'd worker and segfaults vllm_child in "
            f"cuCtxSynchronize.  Got step_down_calls={step_down_calls!r}."
        )
        # Stale-state read-only assertions: we must not have mutated
        # anything (no slot reset, no flag flip).
        entry = _StubOrch._registry["incumbent"]
        assert entry["state"] == "checkpoint"
        assert entry["paused"] is True
    finally:
        pipe.shutdown(timeout=2.0)


def test_pauseop_bails_on_empty_inflight() -> None:
    """Known Issues regression: deferred-pause / phantom-running
    (model 7 trace).

    Legacy bug shape: ``generate(mid); pause(mid)`` back-to-back.
    The pause cmd queued behind the generate on the worker pipe.
    If the generate finished BEFORE the pause cmd landed (the
    demuxer's ``_on_generate_done`` popped ``_inflight[mid]`` and the
    engine returned to idle), the worker still processed the pause
    and minted a phantom ``paused=True`` registry entry.  The next
    ``resume`` then promoted that phantom to phantom-running.

    Fix: ``PauseOp.execute`` re-checks ``orch._inflight[mid]``
    non-empty before sending the worker ``pause`` cmd.  This test
    exercises the bail-out path directly: a stub orch with empty
    ``_inflight`` and a ``_send_cmd_with_ack`` that fails loudly if
    called.
    """
    sent: list[tuple] = []

    class _StubOrch:
        _registry: dict = {
            "m1": {
                "state": "running",   # passes the not-running check
                "paused": False,      # passes the already-paused check
            },
        }
        # EMPTY list -> ``orch._inflight.get(mid)`` is falsy -> bail.
        _inflight: dict[str, list] = {"m1": []}
        _request_log: list = []
        _request_lock = threading.Lock()

        @staticmethod
        @contextlib.contextmanager
        def _locks_ordered(*ids: str):
            yield

        @staticmethod
        def _send_cmd_with_ack(mid: str, cmd: str, *args, **kwargs) -> None:
            sent.append((mid, cmd))

        @staticmethod
        def _set_state(mid: str, state: str) -> None:
            pass

    pipe = ModelPipeline(
        model_id="m1",
        entry=_StubOrch._registry["m1"],
        slots=object,
        orch=_StubOrch,
        pipelines={},
    )
    try:
        pipe.submit(PauseOp()).result(timeout=2.0)

        # Bail-out path: no ``pause`` cmd was ever sent to the engine.
        # If it had been sent, the legacy phantom-running bug would
        # have re-emerged.
        assert sent == [], (
            f"PauseOp sent {sent!r} despite empty _inflight; "
            "deferred-pause guard regressed."
        )
        # Registry was not mutated into the phantom paused=True state.
        assert _StubOrch._registry["m1"]["paused"] is False
        # And the InterruptFlag was reset (PauseOp's finally clause)
        # so a subsequent op doesn't bail spuriously.
        assert not pipe.interrupt.is_set()
    finally:
        pipe.shutdown(timeout=2.0)


def test_pause_evict_resume_full_cycle() -> None:
    """End-to-end: paused incumbent gets force-evicted by peer's wake-up,
    then ``ResumeOp`` walks it back up and triggers the worker's
    ``_saved_requests`` replay.

    Covers the full round-trip that Fix A.3 unlocks:

      1. ``PauseOp`` parked incumbent at ``state="up", paused=True,
         slot=None`` and left a deferred ``(rid, q_rec, done_event)``
         record in ``orch._inflight[incumbent]``.  The user's calling
         thread is parked on ``done_event.wait()`` (we don't simulate
         that thread here; we just verify ``done_event`` is NOT
         spuriously set across the eviction).
      2. The peer's Phase-2 walk-up submits ``EvictForPeerOp`` onto
         the incumbent's pipeline.  With Fix A.3 the paused-skip is
         gone, so the eviction walks the incumbent down to ``sleep``.
      3. ``_inflight[incumbent]`` MUST survive the eviction -- the
         deferred record is the only handle to the user's
         ``done_event``.  Popping it here would orphan the user
         thread (the canonical "Paused-peer silent OOM" hang shape
         on the user-thread side).
      4. ``entry["paused"]`` MUST survive the eviction so the next
         ``ResumeOp`` routes through the ``_saved_requests`` replay
         path (in vllm_child) rather than treating the incumbent as
         a fresh wake-up.
      5. ``ResumeOp`` walks the incumbent ``sleep -> up`` via
         composed ``MoveOp(target="up", announce_state="running")``,
         then issues ``_send_cmd_with_ack(mid, "resume")`` which in
         the real worker triggers ``engine.add_request(...)`` for
         every ``_saved_requests`` entry -- the deferred-generate
         replay.
      6. ``entry["paused"]`` is cleared and the engine is at
         ``running``.  The orchestrator's ``_on_generate_done``
         listener (NOT exercised here -- it lives in
         orchestrator.py) will eventually set ``done_event`` once
         the engine emits ``generate_done`` for the re-prefilled
         request, unblocking the user's parked
         ``_wait_and_collect`` daemon.

    This test stubs the orchestrator boundary just enough to drive
    the pipeline ops end-to-end and asserts the observable trace:
    eviction sleep, preserved inflight + paused flag, MoveOp walk-up
    via ``_step_up``, ``resume`` cmd, cleared paused.  The vllm_child
    handlers themselves (``sleep`` while paused, ``resume`` replay)
    have their invariants documented in ``vllm_child.py``'s ``sleep``
    and ``resume`` cmd branches; this test asserts the orchestrator
    sequences them correctly.
    """
    incumbent_id = "incumbent"
    acquirer_id = "acquirer"

    # Recorded interactions with the stub orchestrator, ordered.
    trace: list[tuple] = []

    # The deferred record left by PauseOp before the test starts.
    # ``done_event`` simulates the user's parked PendingRequest.
    done_event = threading.Event()
    q_rec: dict = {
        "req_id": 42,
        "model_id": incumbent_id,
        "state": "generating",
        "t_done": None,
    }
    deferred_record = ("rid-42", q_rec, done_event)

    class _StubOrch:
        # Two-model registry: paused incumbent + idle acquirer
        # (paused incumbent is what EvictForPeerOp targets; acquirer
        # only needs to exist so the cross-pipeline submit_to_peer_
        # and_wait cycle detector can resolve ``_pipelines[acquirer]``
        # if needed).
        _registry: dict = {
            incumbent_id: {
                "state": "up",
                "paused": True,
                "paused_since": 1.0,
                "slot": None,
                "instance": None,
                "level": 2,
                "gpu": 0,
            },
            acquirer_id: {
                "state": "running",
                "paused": False,
                "slot": object(),
                "instance": None,
                "level": 1,
                "gpu": 0,
            },
        }
        _inflight: dict = {incumbent_id: [deferred_record]}
        _request_log: list = []
        _request_lock = threading.Lock()

        @staticmethod
        @contextlib.contextmanager
        def _locks_ordered(*_ids: str):
            yield

        @staticmethod
        def _step_down(mid: str, from_state: str, to_state: str) -> None:
            trace.append(("step_down", mid, from_state, to_state))
            _StubOrch._registry[mid]["state"] = to_state
            # NOTE: paused is intentionally NOT cleared here -- the
            # real ``_step_down(up, sleep)`` only reconciles the
            # state + slot, never the paused flag.  ResumeOp owns
            # the paused-flag lifecycle.

        @staticmethod
        def _step_up(mid: str, from_state: str, to_state: str,
                     **kwargs: object) -> None:
            trace.append(("step_up", mid, from_state, to_state, kwargs))
            entry = _StubOrch._registry[mid]
            entry["state"] = kwargs.get("announce_state") or to_state

        @staticmethod
        def _set_state(mid: str, state: str) -> None:
            trace.append(("set_state", mid, state))
            _StubOrch._registry[mid]["state"] = state

        @staticmethod
        def _send_cmd_with_ack(mid: str, cmd: str, *args, **kwargs) -> None:
            trace.append(("cmd", mid, cmd))

    pipelines: dict[str, ModelPipeline] = {}
    incumbent_pipe = ModelPipeline(
        model_id=incumbent_id,
        entry=_StubOrch._registry[incumbent_id],
        slots=object,
        orch=_StubOrch,
        pipelines=pipelines,
    )
    pipelines[incumbent_id] = incumbent_pipe
    acquirer_pipe = ModelPipeline(
        model_id=acquirer_id,
        entry=_StubOrch._registry[acquirer_id],
        slots=object,
        orch=_StubOrch,
        pipelines=pipelines,
    )
    pipelines[acquirer_id] = acquirer_pipe

    try:
        # Step 1+2: acquirer evicts the paused incumbent.  In the
        # real orchestrator this goes through
        # ``acquirer_pipe.submit_to_peer_and_wait(incumbent_pipe,
        # EvictForPeerOp(...))`` from inside the acquirer's worker;
        # we simulate the same effect from the test thread by
        # submitting directly onto the incumbent's pipeline and
        # waiting on the future.  The cycle-detector aspect of
        # ``submit_to_peer_and_wait`` is exercised separately by
        # ``test_submit_to_peer_and_wait_*``.
        incumbent_pipe.submit(
            EvictForPeerOp(acquirer_id=acquirer_id)
        ).result(timeout=2.0)

        # Eviction landed: _step_down was called, state moved to sleep.
        assert ("step_down", incumbent_id, "up", "sleep") in trace, (
            f"EvictForPeerOp must walk paused incumbent down to sleep; "
            f"trace={trace!r}"
        )
        incumbent_entry = _StubOrch._registry[incumbent_id]
        assert incumbent_entry["state"] == "sleep"

        # Step 3+4: the deferred record + paused flag MUST survive.
        assert _StubOrch._inflight[incumbent_id] == [deferred_record], (
            "EvictForPeerOp must NOT pop _inflight[mid]; the deferred "
            "record is the only handle the user thread has to its "
            "PendingRequest.done_event.  Popping it would orphan the "
            "user thread."
        )
        assert not done_event.is_set(), (
            "EvictForPeerOp must NOT touch done_event; the engine "
            "hasn't actually replied generate_done yet."
        )
        assert incumbent_entry["paused"] is True, (
            "entry['paused'] must survive the eviction so ResumeOp "
            "routes through the _saved_requests replay path."
        )

        # Step 5: user calls resume on the (now sleeping + paused)
        # incumbent.  ResumeOp walks back up via MoveOp.
        trace.clear()
        resume_fut = incumbent_pipe.submit(ResumeOp())
        resume_fut.result(timeout=2.0)

        # MoveOp(target="up", announce_state="running") at state="sleep"
        # composes _step_up(sleep, up, announce_state="running").
        step_up_calls = [t for t in trace if t[0] == "step_up"]
        assert step_up_calls, (
            f"ResumeOp must compose MoveOp to walk back up; trace={trace!r}"
        )
        first_step = step_up_calls[0]
        assert first_step[1] == incumbent_id
        assert first_step[2] == "sleep"
        assert first_step[3] == "up"
        assert first_step[4].get("announce_state") == "running", (
            "MoveOp(target=up, announce_state=running) must propagate "
            "announce_state into _step_up so the published state goes "
            "straight to 'running' (skipping the observable bare 'up' "
            "window where a peer's eviction could steal the slot)."
        )

        # resume cmd was issued AFTER the walk-up (the worker's
        # ``resume`` handler iterates _saved_requests and re-prefills
        # them via engine.add_request -- requires weights + KV cache
        # already loaded by the preceding step_up).
        resume_cmd_calls = [t for t in trace
                            if t[0] == "cmd" and t[2] == "resume"]
        assert resume_cmd_calls == [("cmd", incumbent_id, "resume")], (
            f"Exactly one 'resume' cmd expected; trace={trace!r}"
        )
        # Ordering: resume cmd MUST come after the walk-up step_up.
        cmd_idx = trace.index(resume_cmd_calls[0])
        step_idx = trace.index(first_step)
        assert step_idx < cmd_idx, (
            "resume cmd issued BEFORE walk-up step_up; replay would "
            "hit a dormant engine."
        )

        # Step 6: paused flag cleared, engine at 'running'.
        assert incumbent_entry["paused"] is False, (
            "ResumeOp must clear entry['paused'] after sending the "
            "resume cmd."
        )
        assert incumbent_entry["state"] == "running", (
            "ResumeOp's MoveOp announce_state='running' must leave "
            "the entry at state='running' (NOT bare 'up')."
        )

        # The deferred user thread is STILL parked at this point.
        # In the real system, the engine's eventual generate_done
        # would be popped by ``_on_generate_done`` (orchestrator.py),
        # which sets done_event.  We simulate that final step here
        # to verify the round-trip closes cleanly.
        rid, simulated_q_rec, simulated_done = (
            _StubOrch._inflight[incumbent_id].pop(0))
        simulated_q_rec["state"] = "done"
        simulated_q_rec["t_done"] = time.perf_counter()
        simulated_done.set()
        assert done_event.is_set()
        assert q_rec["state"] == "done"
        assert q_rec["t_done"] is not None
    finally:
        incumbent_pipe.shutdown(timeout=2.0)
        acquirer_pipe.shutdown(timeout=2.0)
