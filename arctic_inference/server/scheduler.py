from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import ray

from arctic_inference.server.metrics import (
    ConcurrencyHistory,
    RequestRecord,
    _BoundedDeque,
)

logger = logging.getLogger("arctic_inference.server")

_LOG_AFFINITY = os.environ.get("ARCTIC_LOG_AFFINITY", "") == "1"


@dataclass
class _Request:
    id: int
    prompt: str | list[int]
    sampling_params: dict[str, Any]
    future: asyncio.Future
    worker_idx: int | None = None
    created_at: float = field(default_factory=time.time)
    prefix_hash: int | None = None
    strict: bool = False


@dataclass
class WorkerState:
    handle: ray.actor.ActorHandle
    concurrency_limit: int
    active_requests: int = 0
    available: bool = True
    latest_cache_utilization: float = 0.0
    running_reqs: int = 0
    waiting_reqs: int = 0


RoutingFn = Callable[[_Request, list[WorkerState]], int]
ConcurrencyFn = Callable[[list[WorkerState]], list[int]]


def _compute_prefix_hash(prompt: str | list[int]) -> int:
    """Hash the full prompt for affinity routing."""
    if isinstance(prompt, str):
        return hash(prompt)
    return hash(tuple(prompt))


def least_loaded_routing(request: _Request, workers: list[WorkerState]) -> int:
    """Route to the worker with the lowest load ratio."""
    candidates = [i for i, ws in enumerate(workers) if ws.available and ws.concurrency_limit > 0]
    if not candidates:
        candidates = list(range(len(workers)))
    return min(
        candidates,
        key=lambda i: workers[i].active_requests / max(workers[i].concurrency_limit, 1),
    )


_PREFIX_LOAD_THRESHOLD = 0.85


def prefix_affinity_routing(request: _Request, workers: list[WorkerState]) -> int:
    """Route requests with the same prefix hash to the same worker.

    Falls back through a hash ring on overload and ultimately to
    ``least_loaded_routing`` when all candidates exceed the load threshold.
    """
    candidates = [
        i for i, ws in enumerate(workers)
        if ws.available and ws.concurrency_limit > 0
    ]
    if not candidates:
        return least_loaded_routing(request, workers)

    if request.prefix_hash is None:
        return least_loaded_routing(request, workers)

    preferred = candidates[request.prefix_hash % len(candidates)]
    ws = workers[preferred]
    if ws.active_requests < ws.concurrency_limit * _PREFIX_LOAD_THRESHOLD:
        return preferred

    for offset in range(1, len(candidates)):
        alt = candidates[(request.prefix_hash + offset) % len(candidates)]
        ws_alt = workers[alt]
        if ws_alt.active_requests < ws_alt.concurrency_limit * _PREFIX_LOAD_THRESHOLD:
            return alt

    return least_loaded_routing(request, workers)


def strict_affinity_routing(request: _Request, workers: list[WorkerState]) -> int:
    """Always route to the worker keyed by ``request.prefix_hash``, no ringing.

    Unlike :func:`prefix_affinity_routing`, this never spills to alternate
    workers when the preferred one is overloaded; the scheduler's outer loop
    backs off and retries until capacity opens up on the pinned worker. Used
    for multi-turn rollouts where cache reuse is more valuable than throughput
    smoothing.
    """
    candidates = [
        i for i, ws in enumerate(workers)
        if ws.available and ws.concurrency_limit > 0
    ]
    if not candidates:
        return least_loaded_routing(request, workers)
    if request.prefix_hash is None:
        return least_loaded_routing(request, workers)
    return candidates[request.prefix_hash % len(candidates)]


def utilization_based_concurrency(workers: list[WorkerState]) -> list[int]:
    """Adjust per-worker concurrency limits based on KV cache utilization."""
    TARGET_UTIL = 0.90
    HIGH_UTIL = 0.95
    MAX_QUEUE = 2
    MIN_LIMIT, MAX_LIMIT = 8, 2048
    GROWTH_STEP = 2
    AGGRESSIVE_FACTOR = 10
    BACKOFF_STEP = 1 if random.random() < 0.3 else 0
    PROBE_PROB = 0.1

    new_limits: list[int] = []
    for ws in workers:
        current = max(ws.concurrency_limit, MIN_LIMIT)
        util = ws.latest_cache_utilization

        if util == 0.0 and ws.running_reqs == 0 and ws.waiting_reqs == 0:
            if ws.active_requests > 64:
                new = max(MIN_LIMIT, int(ws.active_requests * 0.95))
            else:
                new = min(max(MIN_LIMIT, int(ws.active_requests * 1.5)), MAX_LIMIT)
        elif util > HIGH_UTIL:
            new = current - BACKOFF_STEP
        elif ws.waiting_reqs > MAX_QUEUE:
            new = current - BACKOFF_STEP
        elif util < TARGET_UTIL:
            if ws.active_requests >= (current - 2):
                gap = max(0.0, TARGET_UTIL - util)
                new = current + GROWTH_STEP + int(gap * AGGRESSIVE_FACTOR)
            else:
                new = current
        else:
            if ws.active_requests >= current and ws.waiting_reqs == 0 and random.random() < PROBE_PROB:
                new = current + 1
            else:
                new = current

        new_limits.append(max(MIN_LIMIT, min(new, MAX_LIMIT)))
    return new_limits


class Scheduler:
    """Routes generation requests across workers with dynamic concurrency control."""

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        initial_concurrency: int = 64,
        routing_fn: RoutingFn = least_loaded_routing,
        concurrency_fn: ConcurrencyFn = utilization_based_concurrency,
        poll_interval: float = 0.5,
        adjust_interval: float = 0.5,
        enable_prefix_hash: bool = False,
        max_request_records: int = 100_000,
    ) -> None:
        if not workers:
            raise ValueError("Scheduler requires at least one worker")

        self._workers = [
            WorkerState(handle=w, concurrency_limit=max(1, initial_concurrency))
            for w in workers
        ]
        self._routing_fn = routing_fn
        self._concurrency_fn = concurrency_fn
        self._enable_prefix_hash = enable_prefix_hash
        self._poll_interval = poll_interval
        self._adjust_interval = adjust_interval
        self._next_id = 0
        self._paused = False
        self._stopped = False
        self._poll_task: asyncio.Task | None = None
        self._adjust_task: asyncio.Task | None = None
        self._inflight_tasks: dict[int, set] = {i: set() for i in range(len(workers))}

        # Group routing state for _equal_affinity_routing (grouped/uid-keyed
        # requests). Keeps every member of a group (same prefix_hash) on one
        # worker while distributing groups evenly. Cleared per rollout via
        # reset_affinity.
        self._group_worker: dict[int, int] = {}  # prefix_hash -> worker_idx
        self._group_counter: int = 0  # dense arrival index for round-robin

        # Per-request metric ring (drained by `drain_metrics`).
        self._request_records: _BoundedDeque = _BoundedDeque(max_request_records)
        # Per-replica `concurrency_limit` history; used to back-fill
        # `max_concurrency` on per-step worker snapshots when draining.
        self._concurrency_history: list[ConcurrencyHistory] = [
            ConcurrencyHistory() for _ in workers
        ]
        # Seed the history so the first drain has something to look up.
        for h, ws in zip(self._concurrency_history, self._workers):
            h.record(ws.concurrency_limit)
        # Best-effort: tell each worker its index so snapshots are
        # tagged with the right replica_id.
        for idx, w in enumerate(workers):
            try:
                w.set_replica_id.remote(idx)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: str | list[int],
        sampling_params: dict[str, Any],
        routing_key: str | None = None,
        strict: bool = False,
    ) -> asyncio.Future:
        """Submit a single prompt for generation.

        Args:
            routing_key: Optional opaque string used as the affinity key for
                routing. When provided, ``hash(routing_key)`` replaces the
                prompt hash as the request's ``prefix_hash``, so any two
                requests sharing the key land on the same worker. Intended
                for multi-turn rollouts where turn N+1 must hit the same
                replica as turn N to reuse its KV cache.
            strict: When True, use :func:`strict_affinity_routing` instead of
                the pool's configured routing function, pinning the request
                to the keyed worker even under load. Ignored if no
                ``routing_key`` (and no prefix hash) is available.
        """
        if self._stopped:
            raise RuntimeError("Scheduler is stopped")
        if self._paused:
            raise RuntimeError("Scheduler is paused")

        self._ensure_background_tasks()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if routing_key is not None:
            prefix_hash: int | None = hash(routing_key)
        elif self._enable_prefix_hash:
            prefix_hash = _compute_prefix_hash(prompt)
        else:
            prefix_hash = None
        req = _Request(
            id=self._next_id,
            prompt=prompt,
            sampling_params=sampling_params,
            future=future,
            prefix_hash=prefix_hash,
            strict=strict,
        )
        self._next_id += 1
        asyncio.create_task(self._process_request(req))
        return future

    async def submit_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: dict[str, Any],
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        if isinstance(routing_key, list):
            if len(routing_key) != len(prompts):
                raise ValueError(
                    f"routing_key list length ({len(routing_key)}) must match "
                    f"prompts length ({len(prompts)})"
                )
            keys: list[str | None] = list(routing_key)
        else:
            keys = [routing_key] * len(prompts)
        futures = [
            self.submit(p, sampling_params, routing_key=k, strict=strict)
            for p, k in zip(prompts, keys)
        ]
        return list(await asyncio.gather(*futures))

    async def drain(self) -> None:
        """Block until all in-flight requests have completed."""
        while any(ws.active_requests > 0 for ws in self._workers):
            await asyncio.sleep(0.1)

    async def drain_worker(self, idx: int, timeout: float = 15) -> None:
        """Block until worker *idx* has zero active requests, or timeout."""
        self._check_idx(idx)
        deadline = time.time() + timeout
        ws = self._workers[idx]
        while ws.active_requests > 0 and time.time() < deadline:
            await asyncio.sleep(0.1)
        if ws.active_requests > 0:
            logger.warning(f"Drain timeout: worker {idx} still has {ws.active_requests} active requests")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def total_concurrency_limit(self) -> int:
        return sum(ws.concurrency_limit for ws in self._workers if ws.available)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def is_worker_available(self, idx: int) -> bool:
        self._check_idx(idx)
        return self._workers[idx].available

    def mark_worker_unavailable(self, idx: int) -> None:
        self._check_idx(idx)
        self._workers[idx].available = False

    def mark_worker_available(self, idx: int) -> None:
        self._check_idx(idx)
        self._workers[idx].available = True

    def update_worker_handle(self, idx: int, new_handle: ray.actor.ActorHandle) -> None:
        self._check_idx(idx)
        self._workers[idx].handle = new_handle
        try:
            new_handle.set_replica_id.remote(idx)
        except Exception:
            pass

    def cancel_worker_inflight(self, idx: int) -> int:
        """Cancel all in-flight tasks for worker *idx*. Returns count cancelled."""
        self._check_idx(idx)
        tasks = self._inflight_tasks.get(idx, set())
        cancelled = 0
        for task in list(tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    def add_worker(self, handle: ray.actor.ActorHandle, concurrency_limit: int = 64) -> None:
        """Add a new worker to the scheduler."""
        idx = len(self._workers)
        new_limit = max(1, concurrency_limit)
        self._workers.append(WorkerState(handle=handle, concurrency_limit=new_limit))
        self._inflight_tasks[idx] = set()
        history = ConcurrencyHistory()
        history.record(new_limit)
        self._concurrency_history.append(history)
        try:
            handle.set_replica_id.remote(idx)
        except Exception:
            pass

    def remove_last_worker(self) -> None:
        """Remove the last worker. Raises if no workers remain."""
        if not self._workers:
            raise RuntimeError("No workers to remove")
        idx = len(self._workers) - 1
        for task in list(self._inflight_tasks.get(idx, set())):
            if not task.done():
                task.cancel()
        self._inflight_tasks.pop(idx, None)
        self._workers.pop()
        if self._concurrency_history:
            self._concurrency_history.pop()

    async def shutdown(self) -> None:
        self._stopped = True
        for task in (self._poll_task, self._adjust_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_idx(self, idx: int) -> None:
        if not (0 <= idx < len(self._workers)):
            raise IndexError(f"Worker index {idx} out of range (have {len(self._workers)} workers)")

    def _ensure_background_tasks(self) -> None:
        loop = asyncio.get_running_loop()
        if self._poll_task is None:
            self._poll_task = loop.create_task(self._utilization_poll_loop())
        if self._adjust_task is None:
            self._adjust_task = loop.create_task(self._concurrency_adjust_loop())

    def _equal_affinity_routing(self, request: _Request, workers: list[WorkerState]) -> int:
        """Strict mode: deterministic equal assignment via round-robin by arrival.

        Keeps every member of a group (same ``prefix_hash``) on one worker so
        groups stay intact across the separate submits that share a uid.
        """
        # Reuse this group's existing assignment if its worker is still usable.
        h = request.prefix_hash
        if h is not None and h in self._group_worker:
            idx = self._group_worker[h]
            if workers[idx].available and workers[idx].concurrency_limit > 0:
                return idx
        candidates = [
            i for i, ws in enumerate(workers)
            if ws.available and ws.concurrency_limit > 0
        ] or list(range(len(workers)))
        idx = candidates[self._group_counter % len(candidates)]
        self._group_counter += 1
        if request.prefix_hash is not None:
            self._group_worker[request.prefix_hash] = idx
        return idx

    def reset_affinity(self) -> None:
        """Clear group->worker assignments and the round-robin counter.

        Called once per rollout so a new generation batch starts fresh.
        """
        self._group_worker.clear()
        self._group_counter = 0

    async def _process_request(self, req: _Request) -> None:
        ws: WorkerState | None = None
        current_task = asyncio.current_task()
        tracked_worker_idx: int | None = None
        submitted_time: float = 0.0
        result: Any = None

        if req.strict and req.prefix_hash is not None:
            routing_fn = self._equal_affinity_routing
        else:
            routing_fn = self._routing_fn

        try:
            while not self._stopped:
                idx = routing_fn(req, self._workers)
                ws = self._workers[idx]
                if ws.active_requests < ws.concurrency_limit:
                    ws.active_requests += 1
                    req.worker_idx = idx
                    submitted_time = time.time()
                    if _LOG_AFFINITY:
                        # Opt-in audit trail: every dispatched request emits
                        # one line so we can grep '(prefix_hash, worker)' and
                        # confirm same-keyed requests land on the same worker.
                        # Emit at WARNING since the env var is the gate and
                        # the default Python logger threshold drops INFO.
                        logger.warning(
                            "AFFINITY req=%d worker=%d prefix_hash=%s "
                            "strict=%s routing_fn=%s",
                            req.id, idx, req.prefix_hash,
                            req.strict, routing_fn.__name__,
                        )
                    break
                await asyncio.sleep(0.005)
            else:
                if not req.future.done():
                    req.future.set_exception(RuntimeError("Scheduler stopped"))
                return

            tracked_worker_idx = req.worker_idx
            if current_task is not None and tracked_worker_idx is not None:
                self._inflight_tasks[tracked_worker_idx].add(current_task)

            try:
                result = await ws.handle.generate.remote(req.prompt, req.sampling_params)
                if not req.future.done():
                    req.future.set_result(result)
            except asyncio.CancelledError:
                if not req.future.done():
                    req.future.set_exception(RuntimeError("Request cancelled for weight update"))

        except asyncio.CancelledError:
            if not req.future.done():
                req.future.set_exception(RuntimeError("Request cancelled for weight update"))
        except Exception as e:
            if not req.future.done():
                req.future.set_exception(e)
        finally:
            if current_task is not None and tracked_worker_idx is not None:
                self._inflight_tasks[tracked_worker_idx].discard(current_task)
            if ws is not None:
                ws.active_requests = max(0, ws.active_requests - 1)
            # Record the per-request metric — only if we successfully
            # submitted to a worker. Skipping cancelled / failed requests
            # keeps the buffer clean for downstream analysis.
            if (
                tracked_worker_idx is not None
                and submitted_time > 0.0
                and isinstance(result, dict)
            ):
                self._request_records.push(RequestRecord(
                    request_id=req.id,
                    replica_id=tracked_worker_idx,
                    arrival_time=req.created_at,
                    submitted_time=submitted_time,
                    completion_time=time.time(),
                    prompt_len=int(result.get("prompt_len", 0) or 0),
                    generation_len=int(result.get("generation_len", 0) or 0),
                    prefix_cache_len=int(result.get("prefix_cache_len", 0) or 0),
                ))

    async def _utilization_poll_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self._poll_interval)
            refs = [ws.handle.get_stats.remote() for ws in self._workers]
            results = await asyncio.gather(*refs, return_exceptions=True)
            for ws, result in zip(self._workers, results):
                if isinstance(result, Exception):
                    ws.latest_cache_utilization = 0.0
                    ws.running_reqs = 0
                    ws.waiting_reqs = 0
                else:
                    ws.latest_cache_utilization = result.get("gpu_cache_usage", 0.0)
                    ws.running_reqs = int(result.get("num_requests_running", 0))
                    ws.waiting_reqs = int(result.get("num_requests_waiting", 0))

    async def _concurrency_adjust_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self._adjust_interval)
            try:
                new_limits = self._concurrency_fn(self._workers)
                for idx, (ws, limit) in enumerate(zip(self._workers, new_limits)):
                    new_limit = max(1, int(limit))
                    ws.concurrency_limit = new_limit
                    if idx < len(self._concurrency_history):
                        self._concurrency_history[idx].record(new_limit)
            except Exception as e:
                logger.error(f"Concurrency adjustment failed: {e}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def drain_metrics(self) -> dict[str, Any]:
        """Drain per-replica snapshots and per-request records.

        Returns a dict with two top-level keys:

          * ``requests``: list of :class:`RequestRecord` dicts (one per
            generation call completed since the last drain).
          * ``replicas``: list of ``{replica_id, snapshots: [...]}`` — each
            snapshot is a :class:`ReplicaSnapshot` dict with
            ``max_concurrency`` back-filled from the scheduler's per-replica
            concurrency-limit history.

        Calling this is idempotent and does not block scheduling.
        """
        # Pull snapshots from each worker. Drains the worker-side ring as a
        # side effect.
        replicas_payload: list[dict[str, Any]] = []
        # Resolve handles to current ones so worker-restart doesn't strand
        # us holding a dead actor.
        worker_handles = [(idx, ws.handle) for idx, ws in enumerate(self._workers)]
        results = await asyncio.gather(
            *[h.drain_metrics.remote() for _, h in worker_handles],
            return_exceptions=True,
        )
        for (idx, _), res in zip(worker_handles, results):
            if isinstance(res, Exception):
                replicas_payload.append({
                    "replica_id": idx,
                    "snapshots": [],
                    "error": str(res),
                })
                continue
            snaps = res.get("snapshots", []) if isinstance(res, dict) else []
            history = (
                self._concurrency_history[idx]
                if idx < len(self._concurrency_history)
                else None
            )
            for s in snaps:
                # Worker doesn't know its own concurrency_limit; back-fill
                # from the scheduler's per-replica history.
                s["replica_id"] = idx
                if history is not None:
                    s["max_concurrency"] = history.at(s.get("timestamp", 0.0))
            replicas_payload.append({
                "replica_id": idx,
                "snapshots": snaps,
            })

        requests_payload = [r.to_dict() for r in self._request_records.drain()]
        return {
            "requests": requests_payload,
            "replicas": replicas_payload,
        }
