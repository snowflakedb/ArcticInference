"""Per-instance worker-result demuxer.

Owns the read side of an :class:`Instance`'s ``_result_queue``.  A single
thread, alive between :meth:`Demuxer.start` and :meth:`Demuxer.stop`,
drains results, decrements the pending counter, applies results onto
the owning instance, and dispatches each result to per-cmd listener
callbacks.

This is the single replacement for the historical dual-mechanism
(``Instance.wait()`` + ``Orchestrator._start_generate_waiter``) that
both consumed ``_result_queue`` and could deadlock under concurrent
callers.  See ``orchestrator_DESIGN.md`` for the wider rationale.

Listeners are registered per-cmd-type (e.g. ``"generate"``) or as
``cmd=None`` (every cmd).  They fire on the demuxer thread *after*
``_apply_result`` runs, so a listener that needs ordering relative to
``Instance`` state always observes the post-apply view.

Ordering inside :meth:`_handle` is load-bearing:

1. ``_apply_result`` runs first so :class:`Instance` state is updated
   before any waiter or listener can observe it.
2. ``_pending_count -= 1`` plus ``notify_all()`` are inside the same
   ``with self._pending_cv:`` block so the decrement is atomic w.r.t.
   waiters' predicate check.
3. The per-cmd log line and listeners run after the notify, so they
   cannot observe a "still pending" view of ``_pending_count`` but
   still see post-apply state.

Errors from the worker (``error is not None`` on the result tuple) are
latched: the first error encountered is stored on the demuxer.
:meth:`wait_idle` raises and clears the latch, so a subsequent
``wait_idle()`` call after a fresh batch of cmds is unaffected.
Listeners receive both successful and failed results so the orchestrator
can still surface per-request failures from the generate stream.
"""
from __future__ import annotations

import collections
import queue
import threading
from typing import Callable


Listener = Callable[[str, float, object | None, dict], None]


class Demuxer:
    """Background consumer of a worker's result queue."""

    def __init__(
        self,
        *,
        instance_id: int,
        result_queue,
        log,
        apply_result_cb: Callable[[str, dict], None],
        summarise_cb: Callable[[str, dict], object] | None = None,
    ):
        self._instance_id = instance_id
        self._result_queue = result_queue
        self._log = log
        self._apply_result = apply_result_cb
        self._summarise = summarise_cb or (lambda cmd, info: info)

        self._pending_cv = threading.Condition()
        self._pending_count = 0
        self._pending_cmds: collections.deque[str] = collections.deque()
        self._latched_error: tuple[str, object] | None = None  # (cmd, error)

        self._listener_lock = threading.Lock()
        self._cmd_listeners: dict[str, list[Listener]] = {}
        self._all_listeners: list[Listener] = []

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Spawn the consumer thread.  Idempotent: a no-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"inst{self._instance_id}-demuxer",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the consumer thread to exit and join it.

        Safe to call from the demuxer thread itself: a ``teardown`` result
        is applied on this thread (``_apply_result`` -> ``Instance._reset``
        -> ``_close_queues`` -> ``stop``), and joining the current thread
        raises ``RuntimeError``.  In that case we only set the stop flag
        and leave the join to the natural loop exit -- once ``_handle``
        returns, ``_loop`` observes ``_stop`` and the thread ends on its
        own.
        """
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=timeout)
            self._thread = None

    # -- send-side bookkeeping -------------------------------------------------

    def notify_send(self, cmd: str) -> None:
        """Record that *cmd* was just enqueued to the worker.

        Called by ``Instance._send`` immediately after putting on
        ``_cmd_queue``.  Holds ``_pending_cv`` only briefly (no IPC).
        """
        with self._pending_cv:
            self._pending_count += 1
            self._pending_cmds.append(cmd)

    @property
    def pending_count(self) -> int:
        """Number of cmds currently in flight (sent but not acked)."""
        with self._pending_cv:
            return self._pending_count

    @property
    def pending_cmds(self) -> list[str]:
        """Snapshot of currently-pending cmds in FIFO order."""
        with self._pending_cv:
            return list(self._pending_cmds)

    def wait_idle(self) -> None:
        """Block until ``_pending_count`` reaches 0.

        Re-raises and clears the latched first error if any cmd in this
        batch failed, mirroring the legacy ``Instance.wait()``
        ``raise RuntimeError`` behaviour.
        """
        with self._pending_cv:
            while self._pending_count > 0:
                self._pending_cv.wait()
            err = self._latched_error
            self._latched_error = None
        if err is not None:
            cmd, error = err
            raise RuntimeError(f"command '{cmd}' failed: {error}")

    # -- listener registry -----------------------------------------------------

    def add_listener(self, cmd: str | None, callback: Listener) -> None:
        """Register *callback* for one cmd type (e.g. ``"generate"``) or
        for every cmd (*cmd* = None).
        """
        with self._listener_lock:
            if cmd is None:
                self._all_listeners.append(callback)
            else:
                self._cmd_listeners.setdefault(cmd, []).append(callback)

    def remove_listener(self, cmd: str | None, callback: Listener) -> None:
        with self._listener_lock:
            target = (self._all_listeners if cmd is None
                      else self._cmd_listeners.get(cmd, []))
            if callback in target:
                target.remove(callback)

    # -- consumer loop ---------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (OSError, EOFError, ValueError):
                # Queue closed (teardown).  Exit cleanly.
                return
            try:
                self._handle(result)
            except Exception:
                self._log.exception(
                    "demuxer: unhandled error processing %r", result)

    def _handle(self, result) -> None:
        cmd, elapsed, error, info = result

        # Apply BEFORE decrement+notify: the legacy ``Instance.wait()``
        # called ``_apply_result`` on the same thread that decremented
        # ``_pending_count``, so callers of ``wait()`` observed
        # post-apply state when the loop exited.  Preserve that
        # contract here -- if we decremented first, a parked
        # ``wait_idle()`` could return before ``_apply_result`` had
        # updated ``Instance`` state.
        if error is None:
            try:
                self._apply_result(cmd, info)
            except Exception:
                self._log.exception("apply_result(%s) raised", cmd)

        with self._pending_cv:
            self._pending_count -= 1
            if self._pending_cmds:
                self._pending_cmds.popleft()
            if error is not None and self._latched_error is None:
                self._latched_error = (cmd, error)
            self._pending_cv.notify_all()

        status = "OK" if error is None else "FAILED"
        self._log.info("%s %s (%.3fs) %s",
                       cmd, status, elapsed, self._summarise(cmd, info))

        # Listeners fire last, so they too observe post-apply state and
        # a settled ``_pending_count``.  Snapshot under the lock so
        # add/remove during dispatch doesn't trip the iteration.
        with self._listener_lock:
            listeners = list(self._cmd_listeners.get(cmd, ()))
            listeners.extend(self._all_listeners)
        for cb in listeners:
            try:
                cb(cmd, elapsed, error, info)
            except Exception:
                self._log.exception(
                    "listener %r raised on cmd=%s", cb, cmd)
