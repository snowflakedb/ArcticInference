"""Per-replica + per-request metrics collection for the Arctic Inference server.

Two record types:

  * :class:`ReplicaSnapshot` — emitted at fixed intervals from inside vLLM's
    ``AsyncLLM`` step loop via a custom :class:`StatLoggerBase`. One snapshot
    captures the replica's KV-cache utilisation, running/pending request
    counts, the scheduler-side concurrency limit, and the number of tokens
    scheduled in the most recent step.

  * :class:`RequestRecord` — emitted by the :class:`Scheduler` once per
    generation call. Captures the assigned replica, the arrival /
    submission / completion wall-clock timestamps, and the prompt /
    generation / prefix-cache token counts read off the worker's reply.

The collectors are bounded ring buffers; ``drain()`` returns the buffered
items and clears the buffer. Consumers (the API ``/metrics`` route, the
ray_dss zone server, etc.) call ``drain`` periodically so that memory stays
flat regardless of how often the engine produces snapshots.

The vLLM stat logger lives in the engine process. Because each
``InferenceWorker`` Ray actor runs its own engine (and process), the
collector is a process-singleton — the stat logger and the worker share the
same instance via :func:`get_collector`.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, SchedulerStats

# `MultiModalCacheStats` only exists on vLLM >= 0.16; older builds pass `None`
# instead of the typed object.
try:
    from vllm.v1.metrics.stats import MultiModalCacheStats  # noqa: F401
except ImportError:  # pragma: no cover - older vLLM
    MultiModalCacheStats = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """Per-generation-call record produced by the scheduler."""

    request_id: int
    replica_id: int
    arrival_time: float          # scheduler.submit() wall-clock
    submitted_time: float        # routed to a worker and accepted (active_requests++)
    completion_time: float       # worker future resolved
    prompt_len: int
    generation_len: int
    prefix_cache_len: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplicaSnapshot:
    """Per-step snapshot of a replica's vLLM engine state."""

    replica_id: int
    timestamp: float
    kv_cache_usage: float
    num_running_reqs: int
    num_waiting_reqs: int
    max_concurrency: int        # scheduler-side WorkerState.concurrency_limit
    num_tokens_in_step: int     # decode + prefill tokens scheduled this step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Bounded ring with thread-safe drain
# ---------------------------------------------------------------------------


class _BoundedDeque:
    def __init__(self, max_items: int):
        self._buf: deque[Any] = deque(maxlen=max_items)
        self._lock = threading.Lock()

    # threading.Lock isn't picklable, which breaks cloudpickle when a
    # Scheduler (and therefore a ReplicaPool) is returned across a Ray
    # actor boundary in the VERL-integration flow. Snapshot the buffer
    # under the current lock and rebuild a fresh lock on the other side.
    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            return {"_buf": list(self._buf), "_maxlen": self._buf.maxlen}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._buf = deque(state["_buf"], maxlen=state["_maxlen"])
        self._lock = threading.Lock()

    def push(self, item: Any) -> None:
        with self._lock:
            self._buf.append(item)

    def drain(self) -> list[Any]:
        with self._lock:
            out = list(self._buf)
            self._buf.clear()
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ---------------------------------------------------------------------------
# Worker-side collector
# ---------------------------------------------------------------------------


class WorkerMetricsCollector:
    """Holds the per-step replica snapshots produced by the vLLM stat logger.

    Lives in the worker (engine) process. The vLLM stat logger calls
    :meth:`push_snapshot` from inside ``AsyncLLM``'s output handler. The
    worker's RPC method :meth:`drain_snapshots` returns the buffered items
    and clears the buffer.

    ``min_interval`` throttles sampling so the buffer doesn't fill up when
    vLLM is stepping at >1 kHz (e.g. small batches in decode-only mode).
    """

    def __init__(
        self,
        replica_id: int = 0,
        max_snapshots: int = 10_000,
        min_interval_s: float = 1.0,
    ) -> None:
        self.replica_id = replica_id
        self.min_interval_s = max(0.0, min_interval_s)
        self._snapshots = _BoundedDeque(max_items=max_snapshots)
        self._last_record_time: float = 0.0

    def set_replica_id(self, replica_id: int) -> None:
        self.replica_id = replica_id

    def push_snapshot(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
    ) -> None:
        if scheduler_stats is None:
            return
        now = time.time()
        if (
            self.min_interval_s > 0
            and now - self._last_record_time < self.min_interval_s
        ):
            return
        self._last_record_time = now

        tokens_in_step = 0
        if iteration_stats is not None:
            # vLLM >= 0.16 nests prompt token bookkeeping in
            # `prompt_token_stats.computed`; older releases just expose
            # `num_prompt_tokens` directly. Account for both.
            prompt_tokens = 0
            pts = getattr(iteration_stats, "prompt_token_stats", None)
            if pts is not None and hasattr(pts, "computed"):
                prompt_tokens = pts.computed
            else:
                prompt_tokens = getattr(iteration_stats, "num_prompt_tokens", 0)
            tokens_in_step = (
                iteration_stats.num_generation_tokens + prompt_tokens
            )

        self._snapshots.push(ReplicaSnapshot(
            replica_id=self.replica_id,
            timestamp=now,
            kv_cache_usage=float(scheduler_stats.kv_cache_usage),
            num_running_reqs=int(scheduler_stats.num_running_reqs),
            num_waiting_reqs=int(scheduler_stats.num_waiting_reqs),
            # `max_concurrency` is owned by the scheduler (in the parent
            # process). We stamp it later in `Scheduler.drain_metrics` from
            # the per-replica concurrency-limit history.
            max_concurrency=0,
            num_tokens_in_step=int(tokens_in_step),
        ))

    def drain_snapshots(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._snapshots.drain()]

    def latest(self) -> dict[str, Any]:
        """Return the latest scheduler stats without draining the ring.

        Used by the legacy ``InferenceWorker.get_stats`` path (the
        scheduler's existing utilisation poll loop), so that
        ``Scheduler.WorkerState`` keeps getting fresh ``kv_cache_usage`` /
        ``num_requests_running`` / ``num_requests_waiting`` values.
        """
        # Avoid draining: peek at the most recent push. If there's been no
        # snapshot yet (engine just came up), return zeros.
        with self._snapshots._lock:
            buf = self._snapshots._buf
            if not buf:
                return {
                    "gpu_cache_usage": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                }
            last = buf[-1]
        return {
            "gpu_cache_usage": last.kv_cache_usage,
            "num_requests_running": last.num_running_reqs,
            "num_requests_waiting": last.num_waiting_reqs,
        }


# Process-singleton: vLLM's stat logger and the worker actor live in the same
# process; both look up the collector by engine_index.
_COLLECTORS: dict[int, WorkerMetricsCollector] = {}
_COLLECTORS_LOCK = threading.Lock()


def get_collector(engine_index: int = 0) -> WorkerMetricsCollector:
    with _COLLECTORS_LOCK:
        coll = _COLLECTORS.get(engine_index)
        if coll is None:
            coll = WorkerMetricsCollector()
            _COLLECTORS[engine_index] = coll
        return coll


# ---------------------------------------------------------------------------
# vLLM StatLogger plug-in
# ---------------------------------------------------------------------------


class RingStatLogger(StatLoggerBase):
    """A :class:`StatLoggerBase` that pushes each engine step into the
    process-singleton :class:`WorkerMetricsCollector`.

    Pass the *class* (not an instance) as a factory:

        AsyncLLM.from_vllm_config(
            ...,
            stat_loggers=[RingStatLogger],
        )

    vLLM constructs one per engine-index.
    """

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0) -> None:
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self._collector = get_collector(engine_index)

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int = 0,
    ) -> None:
        self._collector.push_snapshot(scheduler_stats, iteration_stats)

    def log_engine_initialized(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Scheduler-side per-replica concurrency-limit history
# ---------------------------------------------------------------------------


class ConcurrencyHistory:
    """Tracks per-replica ``concurrency_limit`` over time.

    Used by the scheduler's metrics drain to back-fill the
    ``max_concurrency`` field on per-step snapshots reported by workers.
    The history is a list of ``(timestamp, limit)`` samples; queries find
    the most recent sample at or before a given timestamp via binary search.

    Entries older than ``retention_s`` are pruned on every push so memory
    stays bounded across long-running jobs.
    """

    def __init__(self, retention_s: float = 3600.0) -> None:
        self._retention_s = retention_s
        self._times: list[float] = []
        self._limits: list[int] = []
        self._lock = threading.Lock()

    # threading.Lock isn't picklable (see _BoundedDeque.__getstate__).
    def __getstate__(self) -> dict[str, Any]:
        with self._lock:
            return {
                "_retention_s": self._retention_s,
                "_times": list(self._times),
                "_limits": list(self._limits),
            }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._retention_s = state["_retention_s"]
        self._times = state["_times"]
        self._limits = state["_limits"]
        self._lock = threading.Lock()

    def record(self, limit: int) -> None:
        now = time.time()
        with self._lock:
            # Skip duplicate-limit pushes that arrive close together — the
            # adjust loop runs every 0.5s and most ticks won't change the
            # limit.
            if self._limits and self._limits[-1] == limit:
                return
            self._times.append(now)
            self._limits.append(limit)
            self._evict_old(now)

    def _evict_old(self, now: float) -> None:
        cutoff = now - self._retention_s
        i = bisect.bisect_left(self._times, cutoff)
        if i > 0:
            del self._times[:i]
            del self._limits[:i]

    def at(self, timestamp: float) -> int:
        with self._lock:
            if not self._times:
                return 0
            i = bisect.bisect_right(self._times, timestamp)
            if i == 0:
                return self._limits[0]
            return self._limits[i - 1]
