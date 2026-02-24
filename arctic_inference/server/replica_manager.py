from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray
import torch
from arctic_inference.server.config import ModelConfig
from arctic_inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_inference.server")


class ReplicaManager:
    """Manages Ray cluster and worker actor lifecycle."""

    def __init__(
        self,
        worker_cls=InferenceWorker,
        num_workers: int | None = None,
    ) -> None:
        self._worker_cls = worker_cls
        self._num_workers_override = num_workers
        self.workers: list[ray.actor.ActorHandle] = []
        self.config: ModelConfig | None = None
        self._stop_monitoring = False
        self._health_task: asyncio.Task | None = None

    async def initialize(self, config: ModelConfig) -> int:
        """Create worker actors and initialize vLLM engines. Returns worker count."""
        self.config = config
        ray.init(ignore_reinit_error=True, log_to_driver=False)

        if self._num_workers_override is not None:
            num_workers = self._num_workers_override
            num_gpus = 0
        else:
            nodes = [n for n in ray.nodes() if n["Alive"]]
            if not nodes:
                raise RuntimeError("No alive Ray nodes")

            gpus_per_node = int(nodes[0]["Resources"].get("GPU", torch.cuda.device_count()))
            gpus_per_worker = config.tensor_parallel_size
            workers_per_node = gpus_per_node // gpus_per_worker
            num_workers = workers_per_node * len(nodes)
            num_gpus = gpus_per_worker

            if num_workers == 0:
                raise RuntimeError(
                    f"Cannot create workers: {gpus_per_node} GPUs per node, "
                    f"tensor_parallel_size={gpus_per_worker}"
                )

        logger.info(f"Creating {num_workers} workers")

        self.workers = [
            self._worker_cls.options(
                num_gpus=num_gpus,
                max_concurrency=2048,
            ).remote()
            for _ in range(num_workers)
        ]

        engine_kwargs = config.to_engine_kwargs()
        extra_env = config.extra_env or None

        await asyncio.gather(*[w.initialize.remote(engine_kwargs, extra_env) for w in self.workers])

        logger.info(f"All {num_workers} workers initialized")

        self._stop_monitoring = False
        self._health_task = asyncio.create_task(self._monitor_health())

        return num_workers

    async def get_status(self) -> dict[str, Any]:
        states = await asyncio.gather(*[w.get_state.remote() for w in self.workers])
        return {
            "num_replicas": len(self.workers),
            "replica_states": list(states),
        }

    async def shutdown(self) -> None:
        self._stop_monitoring = True
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        for w in self.workers:
            try:
                ray.get(w.shutdown.remote(), timeout=10)
            except Exception:
                pass
        self.workers = []

    async def _monitor_health(self) -> None:
        while not self._stop_monitoring:
            await asyncio.sleep(10)
            for i, w in enumerate(self.workers):
                try:
                    healthy = await asyncio.wait_for(w.is_healthy.remote(), timeout=30)
                except Exception:
                    healthy = False
                if not healthy:
                    logger.warning(f"Worker {i} unhealthy, attempting restart")
                    await self._restart_worker(i)

    async def _restart_worker(self, idx: int) -> None:
        old = self.workers[idx]
        try:
            ray.kill(old)
        except Exception:
            pass

        num_gpus = 0 if self._num_workers_override is not None else self.config.tensor_parallel_size
        new_worker = self._worker_cls.options(
            num_gpus=num_gpus,
            max_concurrency=2048,
        ).remote()

        engine_kwargs = self.config.to_engine_kwargs()
        extra_env = self.config.extra_env or None
        await new_worker.initialize.remote(engine_kwargs, extra_env)

        self.workers[idx] = new_worker
        logger.info(f"Worker {idx} restarted successfully")
