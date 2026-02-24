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
class Request:
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


RoutingFn = Callable[[Request, list[WorkerState]], int]
ConcurrencyFn = Callable[[list[WorkerState]], list[int]]


def least_loaded_routing(request: Request, workers: list[WorkerState]) -> int:
    """Route to the worker with the lowest load ratio."""
    candidates = [i for i, ws in enumerate(workers) if ws.available and ws.concurrency_limit > 0]
    if not candidates:
        candidates = list(range(len(workers)))
    return min(
        candidates,
        key=lambda i: workers[i].active_requests / max(workers[i].concurrency_limit, 1),
    )


def utilization_based_concurrency_policy(workers: list[WorkerState]) -> list[int]:
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
        waiting = ws.waiting_reqs
        running = ws.running_reqs

        if util == 0.0 and running == 0 and waiting == 0:
            if ws.active_requests > 64:
                new = max(MIN_LIMIT, int(ws.active_requests * 0.95))
            else:
                new = min(max(MIN_LIMIT, int(ws.active_requests * 1.5)), MAX_LIMIT)
        elif util > HIGH_UTIL:
            new = current - BACKOFF_STEP
        elif waiting > MAX_QUEUE:
            new = current - BACKOFF_STEP
        elif util < TARGET_UTIL:
            if ws.active_requests >= (current - 2):
                gap = max(0.0, TARGET_UTIL - util)
                new = current + GROWTH_STEP + int(gap * AGGRESSIVE_FACTOR)
            else:
                new = current
        else:
            if ws.active_requests >= current and waiting == 0 and random.random() < PROBE_PROB:
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
        concurrency_fn: ConcurrencyFn = utilization_based_concurrency_policy,
        poll_interval: float = 0.5,
        adjust_interval: float = 0.5,
    ) -> None:
        if not workers:
            raise ValueError("Scheduler requires at least one worker")

        self._workers = [WorkerState(handle=w, concurrency_limit=max(1, initial_concurrency)) for w in workers]
        self._routing_fn = routing_fn
        self._concurrency_fn = concurrency_fn
        self._poll_interval = poll_interval
        self._adjust_interval = adjust_interval

        self._next_id = 0
        self._paused = False
        self._stopped = False
        self._poll_task: asyncio.Task | None = None
        self._adjust_task: asyncio.Task | None = None

    def submit(self, prompt: str | list[int], sampling_params: dict[str, Any]) -> asyncio.Future:
        """Submit a generation request. Returns a Future that resolves to a result dict."""
        if self._stopped:
            raise RuntimeError("Scheduler is stopped")
        if self._paused:
            raise RuntimeError("Scheduler is paused")

        self._ensure_background_tasks()

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = Request(
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
        """Submit a batch of prompts and await all results."""
        futures = [self.submit(p, sampling_params) for p in prompts]
        return list(await asyncio.gather(*futures))

    async def drain(self) -> None:
        """Block until all in-flight requests have completed."""
        while any(ws.active_requests > 0 for ws in self._workers):
            await asyncio.sleep(0.1)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def total_concurrency_limit(self) -> int:
        return sum(ws.concurrency_limit for ws in self._workers if ws.available)

    def mark_worker_unavailable(self, idx: int) -> None:
        if 0 <= idx < len(self._workers):
            self._workers[idx].available = False
            self._workers[idx].active_requests = 0

    def mark_worker_available(self, idx: int) -> None:
        if 0 <= idx < len(self._workers):
            self._workers[idx].available = True

    def update_worker_handle(self, idx: int, new_handle: ray.actor.ActorHandle) -> None:
        if 0 <= idx < len(self._workers):
            self._workers[idx].handle = new_handle

    async def shutdown(self) -> None:
        self._stopped = True
        for task in (self._poll_task, self._adjust_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def _ensure_background_tasks(self) -> None:
        loop = asyncio.get_running_loop()
        if self._poll_task is None:
            self._poll_task = loop.create_task(self._utilization_poll_loop())
        if self._adjust_task is None:
            self._adjust_task = loop.create_task(self._concurrency_adjust_loop())

    async def _process_request(self, req: Request) -> None:
        ws: WorkerState | None = None
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

            result = await ws.handle.generate.remote(req.prompt, req.sampling_params)
            if not req.future.done():
                req.future.set_result(result)

        except Exception as e:
            if not req.future.done():
                req.future.set_exception(e)
        finally:
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
