from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import ray
from arctic_inference.server.config import ModelConfig
from arctic_inference.server.scheduler import Scheduler
from arctic_inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_inference.server")


class ReplicaPool:
    """Manages a set of worker replicas for a single model.

    Owns the workers and an internal :class:`Scheduler` that handles
    request routing and concurrency control.
    """

    def __init__(
        self,
        config: ModelConfig,
        worker_cls=InferenceWorker,
        num_replicas: int = 1,
        num_gpus_per_replica: int = 0,
    ) -> None:
        self._config = config
        self._worker_cls = worker_cls
        self._num_gpus_per_replica = num_gpus_per_replica
        self._workers: list[ray.actor.ActorHandle] = []
        self._scheduler: Scheduler | None = None
        self._lock = asyncio.Lock()
        self._stop_monitoring = False
        self._health_task: asyncio.Task | None = None
        self._updating_workers: set[int] = set()
        self._cached_weights_info: list[dict] | None = None
        self._num_replicas_target = num_replicas

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def tp_size(self) -> int:
        return self._config.tensor_parallel_size

    @property
    def num_replicas(self) -> int:
        return len(self._workers)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> int:
        """Create worker actors, initialize vLLM engines, start scheduler.

        Returns the number of workers created.
        """
        ray.init(ignore_reinit_error=True, log_to_driver=False)

        n = self._num_replicas_target
        logger.info(f"Creating {n} workers (TP={self.tp_size}, GPUs/worker={self._num_gpus_per_replica})")

        self._workers = [
            self._worker_cls.options(
                num_gpus=self._num_gpus_per_replica,
                max_concurrency=2048,
            ).remote()
            for _ in range(n)
        ]

        engine_kwargs = self._config.to_engine_kwargs()
        extra_env = self._config.extra_env or None
        await asyncio.gather(*[w.initialize.remote(engine_kwargs, extra_env) for w in self._workers])

        self._scheduler = Scheduler(workers=self._workers, initial_concurrency=64)

        self._stop_monitoring = False
        self._health_task = asyncio.create_task(self._monitor_health())

        logger.info(f"ReplicaPool ready: {n} workers")
        return n

    async def shutdown(self) -> None:
        self._stop_monitoring = True
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._scheduler:
            await self._scheduler.shutdown()
            self._scheduler = None

        for w in self._workers:
            try:
                ray.get(w.shutdown.remote(), timeout=10)
            except Exception:
                pass
        self._workers.clear()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def submit_batch(
        self,
        inputs: list[str | list[int]],
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        return await self._scheduler.submit_batch(inputs, sampling_params)

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    async def scale_down(self, target_count: int) -> None:
        """Remove workers from the end until *target_count* remain."""
        if target_count < 0:
            raise ValueError(f"target_count must be >= 0, got {target_count}")
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            while len(self._workers) > target_count:
                idx = len(self._workers) - 1
                self._scheduler.mark_worker_unavailable(idx)
                await self._scheduler.drain_worker(idx)
                worker = self._workers.pop()
                try:
                    ray.kill(worker)
                except Exception:
                    pass
                self._scheduler.remove_last_worker()
                logger.info(f"Scaled down: removed worker {idx}")

    async def scale_up(self, target_count: int) -> None:
        """Add workers until *target_count* are running."""
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            engine_kwargs = self._config.to_engine_kwargs()
            extra_env = self._config.extra_env or None

            while len(self._workers) < target_count:
                worker = self._worker_cls.options(
                    num_gpus=self._num_gpus_per_replica,
                    max_concurrency=2048,
                ).remote()
                await worker.initialize.remote(engine_kwargs, extra_env)
                self._workers.append(worker)
                self._scheduler.add_worker(worker)
                logger.info(f"Scaled up: added worker {len(self._workers) - 1}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        states = await asyncio.gather(*[w.get_state.remote() for w in self._workers])
        return {
            "num_replicas": len(self._workers),
            "replica_states": list(states),
        }

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_weights_info(self) -> list[dict]:
        if self._cached_weights_info is None:
            from arctic_inference.server.weight_sync import build_weights_info
            infos = build_weights_info(self._config.model)
            self._cached_weights_info = [wi.to_dict() for wi in infos]
        return self._cached_weights_info

    async def sync_weights(
        self,
        groups: list[dict[str, Any]],
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
    ) -> dict[str, Any]:
        """Receive weights from sender(s) and load into all replicas.

        Acquires the pool lock to prevent concurrent scaling or other syncs.
        The *strategy* controls how in-flight requests are handled:
          - **drain**: pause scheduler, wait for in-flight to finish, sync, resume.
          - **skip**: mark workers unavailable, cancel in-flight, sync, re-enable.
          - **hotswap**: sync while serving continues.
        """
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        async with self._lock:
            t0 = time.time()
            n = len(self._workers)

            if strategy == "drain":
                self._scheduler.pause()
                await self._scheduler.drain()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_unavailable(i)
                for i in range(n):
                    self._scheduler.cancel_worker_inflight(i)
            elif strategy != "hotswap":
                raise ValueError(f"Unknown strategy: {strategy!r}. Use: drain, skip, hotswap")

            replica_to_group: dict[int, dict[str, Any]] = {}
            for g in groups:
                for rid in g["replica_ids"]:
                    replica_to_group[rid] = {
                        "master_addr": g["master_addr"],
                        "master_port": g["master_port"],
                        "world_size": 2,
                        "rank_offset": 1,
                    }

            self._updating_workers = set(range(n))

            tasks = []
            for i, worker in enumerate(self._workers):
                gcfg = replica_to_group.get(i)
                if gcfg is None:
                    raise RuntimeError(
                        f"Replica {i} not assigned to any group. "
                        f"Groups cover replicas: {[g['replica_ids'] for g in groups]}"
                    )
                tasks.append(
                    worker.sync_weights.remote(
                        gcfg["master_addr"], gcfg["master_port"],
                        gcfg["rank_offset"], gcfg["world_size"],
                        bucket_size, engine_only, direct_mode,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._updating_workers.clear()

            if strategy == "drain":
                self._scheduler.resume()
            elif strategy == "skip":
                for i in range(n):
                    self._scheduler.mark_worker_available(i)

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            elapsed = time.time() - t0
            logger.info(f"Weight sync: {n} workers, {len(groups)} group(s) in {elapsed:.2f}s")
            return {
                "elapsed": elapsed,
                "num_groups": len(groups),
                "strategy": strategy,
                "strategy_elapsed": elapsed,
                "workers": per_worker,
            }

    async def close_weight_sync(self) -> dict[str, Any]:
        async with self._lock:
            results = await asyncio.gather(*[w.close_weight_sync.remote() for w in self._workers])
            return {"status": "ok", "workers": list(results)}

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def _monitor_health(self) -> None:
        while not self._stop_monitoring:
            await asyncio.sleep(10)
            for i, w in enumerate(self._workers):
                if i in self._updating_workers:
                    continue
                if self._scheduler is not None and not self._scheduler.is_worker_available(i):
                    continue
                try:
                    healthy = await asyncio.wait_for(w.is_healthy.remote(), timeout=30)
                except Exception:
                    healthy = False
                if not healthy:
                    logger.warning(f"Worker {i} unhealthy, attempting restart")
                    await self._restart_worker(i)

    async def _restart_worker(self, idx: int) -> None:
        old = self._workers[idx]
        if self._scheduler is not None:
            self._scheduler.mark_worker_unavailable(idx)

        try:
            ray.kill(old)
        except Exception:
            pass

        new_worker = self._worker_cls.options(
            num_gpus=self._num_gpus_per_replica,
            max_concurrency=2048,
        ).remote()

        engine_kwargs = self._config.to_engine_kwargs()
        extra_env = self._config.extra_env or None
        await new_worker.initialize.remote(engine_kwargs, extra_env)

        self._workers[idx] = new_worker

        if self._scheduler is not None:
            self._scheduler.update_worker_handle(idx, new_worker)
            self._scheduler.mark_worker_available(idx)

        logger.info(f"Worker {idx} restarted successfully")
