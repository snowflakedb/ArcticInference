from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

import ray
import torch
from arctic_inference.server.config import ModelConfig
from arctic_inference.server.worker import InferenceWorker

if TYPE_CHECKING:
    from arctic_inference.server.scheduler import Scheduler

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
        self._updating_workers: set[int] = set()
        self._scheduler: Scheduler | None = None
        self._cached_weights_info: list[dict] | None = None

    def register_scheduler(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

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
        old = self.workers[idx]

        if self._scheduler is not None:
            self._scheduler.mark_worker_unavailable(idx)

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

        if self._scheduler is not None:
            self._scheduler.update_worker_handle(idx, new_worker)
            self._scheduler.mark_worker_available(idx)

        logger.info(f"Worker {idx} restarted successfully")

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_weights_info(self) -> list[dict]:
        if not self.config:
            raise RuntimeError("Not initialized")
        if self._cached_weights_info is None:
            from arctic_inference.server.weight_sync import build_weights_info
            infos = build_weights_info(self.config.model)
            self._cached_weights_info = [wi.to_dict() for wi in infos]
        return self._cached_weights_info

    async def sync_weights(
        self,
        groups: list[dict[str, Any]],
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        direct_mode: bool = False,
    ) -> dict[str, Any]:
        """Receive + load weights on all replicas via NCCLEngine.

        *groups* maps each NCCL broadcast group to its replicas.  Each
        replica joins the group that contains it, with the correct
        rank offset.  All groups operate in parallel.

        If *direct_mode* is True, per-weight send/recv is used instead of
        bucket packing (BF16 TP=1 zero-copy path).
        """
        if not self.config:
            raise RuntimeError("Not initialized")

        replica_to_group: dict[int, dict[str, Any]] = {}
        for g in groups:
            for rid in g["replica_ids"]:
                replica_to_group[rid] = {
                    "master_addr": g["master_addr"],
                    "master_port": g["master_port"],
                    "world_size": 2,
                    "rank_offset": 1,
                }

        start = time.time()
        self._updating_workers = set(range(len(self.workers)))

        tasks = []
        for i, worker in enumerate(self.workers):
            gcfg = replica_to_group.get(i)
            if gcfg is None:
                raise RuntimeError(
                    f"Replica {i} not assigned to any group. "
                    f"Groups cover replicas: "
                    f"{[g['replica_ids'] for g in groups]}"
                )
            tasks.append(
                worker.sync_weights.remote(
                    gcfg["master_addr"],
                    gcfg["master_port"],
                    gcfg["rank_offset"],
                    gcfg["world_size"],
                    bucket_size,
                    engine_only,
                    direct_mode,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        self._updating_workers.clear()

        per_worker = [
            {"status": "error", "message": str(r)} if isinstance(r, Exception) else r
            for r in results
        ]
        elapsed = time.time() - start
        n_groups = len(groups)
        logger.info(
            f"Weight sync: {len(self.workers)} workers across "
            f"{n_groups} group(s) in {elapsed:.2f}s"
        )
        return {"elapsed": elapsed, "num_groups": n_groups, "workers": per_worker}

    async def close_weight_sync(self) -> dict[str, Any]:
        results = await asyncio.gather(*[
            w.close_weight_sync.remote() for w in self.workers
        ])
        return {"status": "ok", "workers": list(results)}

