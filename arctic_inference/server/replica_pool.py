from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import ray
import torch
from arctic_inference.server.config import ModelConfig
from arctic_inference.server.scheduler import Scheduler
from arctic_inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_inference.server")


def ensure_ray() -> int:
    """Initialize Ray and return the total number of GPUs in the cluster."""
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    nodes = [n for n in ray.nodes() if n["Alive"]]
    if not nodes:
        raise RuntimeError("No alive Ray nodes")
    total = sum(
        int(n["Resources"].get("GPU", torch.cuda.device_count()))
        for n in nodes
    )
    if total == 0:
        raise RuntimeError("No GPUs available in the Ray cluster")
    logger.info(f"{total} GPUs available across {len(nodes)} node(s)")
    return total


class ReplicaPool:
    """Manages a set of worker replicas for a single model.

    Owns the workers and an internal :class:`Scheduler` that handles
    request routing and concurrency control.

    Args:
        worker_cls: Ray actor class for inference workers.
    """

    def __init__(self, worker_cls=InferenceWorker) -> None:
        self._worker_cls = worker_cls
        self._config: ModelConfig | None = None
        self._model_id: str | None = None
        self._workers: list[ray.actor.ActorHandle] = []
        self._scheduler: Scheduler | None = None
        self._lock = asyncio.Lock()
        self._stop_monitoring = False
        self._health_task: asyncio.Task | None = None
        self._updating_workers: set[int] = set()
        self._cached_weights_info: list[dict] | None = None
        self._sleeping = False

    @property
    def config(self) -> ModelConfig:
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        return self._config

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def tp_size(self) -> int:
        return self.config.tensor_parallel_size

    @property
    def num_replicas(self) -> int:
        return len(self._workers)

    def _check_model_id(self, model_id: str | None) -> None:
        if model_id is not None and self._model_id is not None and model_id != self._model_id:
            raise ValueError(
                f"model_id mismatch: got {model_id!r}, "
                f"expected {self._model_id!r}"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: ModelConfig,
        model_id: str | None = None,
        num_replicas: int | None = None,
    ) -> int:
        """Set configuration, create worker actors, start scheduler.

        Args:
            config: Model configuration (defines TP size, model name, etc.).
            model_id: Ignored in single-model mode.
            num_replicas: Number of replicas. If ``None``, uses all
                available GPUs (``total_gpus // tensor_parallel_size``).

        Returns the number of workers created.
        """
        if self._config is not None:
            raise RuntimeError("Already initialized. Call shutdown() first.")

        self._config = config
        self._model_id = model_id

        if num_replicas is None:
            total_gpus = ensure_ray()
            num_replicas = total_gpus // self.tp_size
            if num_replicas == 0:
                raise RuntimeError(
                    f"Not enough GPUs: TP={self.tp_size} needs at least "
                    f"{self.tp_size} GPUs but only {total_gpus} available"
                )

        n = num_replicas
        logger.info(f"Creating {n} workers (TP={self.tp_size})")

        self._workers = [
            self._worker_cls.options(
                num_gpus=self.tp_size,
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

    async def shutdown(self, model_id: str | None = None) -> None:
        self._check_model_id(model_id)
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
        self._config = None
        self._model_id = None
        self._cached_weights_info = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompts: str | list[int] | list[str | list[int]],
        sampling_params: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Generate completions for one or more prompts.

        Args:
            prompts: A single prompt (``str`` or token-id ``list[int]``) or a
                batch of prompts (``list[str | list[int]]``).
            sampling_params: vLLM sampling parameters (dict or SamplingParams).
            model_id: Ignored in single-model mode.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._sleeping:
            raise RuntimeError("Model is sleeping; call /wake_up first")
        params = sampling_params or {}
        if isinstance(prompts, str):
            return [await self._scheduler.submit(prompts, params)]
        if prompts and isinstance(prompts[0], int):
            return [await self._scheduler.submit(prompts, params)]
        return await self._scheduler.submit_batch(prompts, params)

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
                    num_gpus=self.tp_size,
                    max_concurrency=2048,
                ).remote()
                await worker.initialize.remote(engine_kwargs, extra_env)
                self._workers.append(worker)
                self._scheduler.add_worker(worker)
                logger.info(f"Scaled up: added worker {len(self._workers) - 1}")

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    async def sleep(self, level: int = 1) -> dict[str, Any]:
        """Drain requests, then free GPU memory on every worker."""
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._sleeping:
            return {"status": "already_sleeping", "level": level}

        async with self._lock:
            self._scheduler.pause()
            await self._scheduler.drain()

            results = await asyncio.gather(
                *[w.sleep.remote(level=level) for w in self._workers],
                return_exceptions=True,
            )
            self._sleeping = True

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            logger.info("ReplicaPool sleeping (%d workers, level=%d)", len(self._workers), level)
            return {"status": "sleeping", "level": level, "workers": per_worker}

    async def wake_up(self, tags: list[str] | None = None) -> dict[str, Any]:
        """Restore GPU memory on every worker, then resume scheduling."""
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")
        if not self._sleeping:
            return {"status": "already_ready"}

        async with self._lock:
            results = await asyncio.gather(
                *[w.wake_up.remote(tags=tags) for w in self._workers],
                return_exceptions=True,
            )
            self._sleeping = False
            self._scheduler.resume()

            per_worker = [
                {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
                for r in results
            ]
            logger.info("ReplicaPool awake (%d workers)", len(self._workers))
            return {"status": "ready", "workers": per_worker}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        if self._config is None:
            return {"status": "not_initialized"}
        states = await asyncio.gather(*[w.get_state.remote() for w in self._workers])
        return {
            "model": self._config.model,
            "num_replicas": len(self._workers),
            "sleeping": self._sleeping,
            "replica_states": list(states),
        }

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_weights_info(self, model_id: str | None = None) -> list[dict]:
        self._check_model_id(model_id)
        if self._config is None:
            raise RuntimeError("ReplicaPool not initialized")
        if self._cached_weights_info is None:
            from arctic_inference.server.weight_sync import build_weights_info
            infos = build_weights_info(self._config.model)
            self._cached_weights_info = [wi.to_dict() for wi in infos]
        return self._cached_weights_info

    async def sync_weights(
        self,
        groups: list[dict[str, Any]] | None = None,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
        model_id: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        """Receive weights from sender(s) and load into all replicas.

        Accepts either ``groups`` or legacy flat fields (``master_addr``,
        ``master_port``, ``world_size``).  The *strategy* controls how
        in-flight requests are handled:

          - **drain**: pause scheduler, wait for in-flight to finish, sync, resume.
          - **skip**: mark workers unavailable, cancel in-flight, sync, re-enable.
          - **hotswap**: sync while serving continues.
        """
        self._check_model_id(model_id)
        if self._scheduler is None:
            raise RuntimeError("ReplicaPool not initialized")

        if groups is None:
            if master_addr is None or master_port is None:
                raise ValueError(
                    "Provide either 'groups' or legacy flat fields "
                    "(master_addr, master_port, world_size)"
                )
            n = self.num_replicas
            tp = self.tp_size
            groups = [{
                "group_id": 0,
                "master_addr": master_addr,
                "master_port": master_port,
                "world_size": world_size or (1 + n * tp),
                "replica_ids": list(range(n)),
            }]

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

    async def close_weight_sync(self, model_id: str | None = None) -> dict[str, Any]:
        self._check_model_id(model_id)
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
            num_gpus=self.tp_size,
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
