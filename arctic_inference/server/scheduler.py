from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import ray

logger = logging.getLogger("arctic_inference.server")


@dataclass
class _Request:
    id: int
    prompt: str | list[int]
    sampling_params: dict[str, Any]
    future: asyncio.Future
    worker_idx: int | None = None
    created_at: float = field(default_factory=time.time)


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


def least_loaded_routing(request: _Request, workers: list[WorkerState]) -> int:
    """Route to the worker with the lowest load ratio."""
    candidates = [i for i, ws in enumerate(workers) if ws.available and ws.concurrency_limit > 0]
    if not candidates:
        candidates = list(range(len(workers)))
    return min(
        candidates,
        key=lambda i: workers[i].active_requests / max(workers[i].concurrency_limit, 1),
    )


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
    ) -> None:
        if not workers:
            raise ValueError("Scheduler requires at least one worker")

        self._workers = [
            WorkerState(handle=w, concurrency_limit=max(1, initial_concurrency))
            for w in workers
        ]
        self._routing_fn = routing_fn
        self._concurrency_fn = concurrency_fn
        self._poll_interval = poll_interval
        self._adjust_interval = adjust_interval
        self._next_id = 0
        self._paused = False
        self._stopped = False
        self._poll_task: asyncio.Task | None = None
        self._adjust_task: asyncio.Task | None = None
        self._inflight_tasks: dict[int, set] = {i: set() for i in range(len(workers))}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, prompt: str | list[int], sampling_params: dict[str, Any]) -> asyncio.Future:
        if self._stopped:
            raise RuntimeError("Scheduler is stopped")
        if self._paused:
            raise RuntimeError("Scheduler is paused")

        self._ensure_background_tasks()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = _Request(
            id=self._next_id,
            prompt=prompt,
            sampling_params=sampling_params,
            future=future,
        )
        self._next_id += 1
        asyncio.create_task(self._process_request(req))
        return future

    async def submit_batch(
        self,
        prompts: list[str | list[int]],
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        futures = [self.submit(p, sampling_params) for p in prompts]
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
        self._workers.append(WorkerState(handle=handle, concurrency_limit=max(1, concurrency_limit)))
        self._inflight_tasks[idx] = set()

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

    async def _process_request(self, req: _Request) -> None:
        ws: WorkerState | None = None
        current_task = asyncio.current_task()
        tracked_worker_idx: int | None = None

        try:
            while not self._stopped:
                idx = self._routing_fn(req, self._workers)
                ws = self._workers[idx]
                if ws.active_requests < ws.concurrency_limit:
                    ws.active_requests += 1
                    req.worker_idx = idx
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
                for ws, limit in zip(self._workers, new_limits):
                    ws.concurrency_limit = max(1, int(limit))
            except Exception as e:
                logger.error(f"Concurrency adjustment failed: {e}")
