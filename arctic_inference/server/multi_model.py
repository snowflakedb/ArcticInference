from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from arctic_inference.server import api
from arctic_inference.server.config import ModelConfig
from arctic_inference.server.replica_pool import ReplicaPool, ensure_ray
from arctic_inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_inference.server")


class Driver:
    """Manages multiple models across a GPU cluster.

    Drop-in replacement for :class:`ReplicaPool` as the ``api.backend``.
    Each :meth:`initialize` call loads a model and returns a ``model_id``.
    All subsequent operations route to the correct pool via ``model_id``.
    """

    def __init__(self, worker_cls=InferenceWorker) -> None:
        self._worker_cls = worker_cls
        self._total_gpus: int = 0
        self._ray_initialized = False
        self._pools: dict[str, ReplicaPool] = {}

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    def _ensure_ray(self) -> None:
        if self._ray_initialized:
            return
        self._total_gpus = ensure_ray()
        self._ray_initialized = True

    @property
    def _allocated_gpus(self) -> int:
        return sum(p.num_replicas * p.tp_size for p in self._pools.values())

    @property
    def _available_gpus(self) -> int:
        return self._total_gpus - self._allocated_gpus

    def _compute_even_share(self, tp_sizes: dict[str, int]) -> dict[str, int]:
        n = len(tp_sizes)
        if n == 0:
            return {}
        gpus_per_model = self._total_gpus // n
        result: dict[str, int] = {}
        for model_id, tp in tp_sizes.items():
            replicas = gpus_per_model // tp
            if replicas == 0:
                raise RuntimeError(
                    f"Not enough GPUs for {model_id!r}: TP={tp} needs at least "
                    f"{tp} GPUs but only {gpus_per_model} available per model "
                    f"({self._total_gpus} total / {n} models)"
                )
            result[model_id] = replicas
        return result

    def _rebalance_up(self) -> None:
        tp_sizes = {mid: p.tp_size for mid, p in self._pools.items()}
        plan = self._compute_even_share(tp_sizes)
        for mid, p in self._pools.items():
            target = plan[mid]
            if target > p.num_replicas:
                logger.info(f"Scaling up {mid!r}: {p.num_replicas} -> {target} replicas (background)")
                # Cancel any earlier in-flight scale before starting a new one,
                # and register the new task on the pool so pool.shutdown() can
                # cancel it cleanly if the model is torn down before scaling
                # finishes.
                if p._scale_task and not p._scale_task.done():
                    p._scale_task.cancel()
                p._scale_task = asyncio.ensure_future(p.scale_up(target))

    def _get_pool(self, model_id: str | None) -> ReplicaPool:
        if model_id is None:
            raise ValueError("model_id is required in multi-model mode")
        try:
            return self._pools[model_id]
        except KeyError:
            raise KeyError(f"Unknown model_id {model_id!r}. Loaded: {list(self._pools)}")

    # ------------------------------------------------------------------
    # Lifecycle  (same interface as ReplicaPool)
    # ------------------------------------------------------------------

    async def initialize(
        self,
        config: ModelConfig,
        model_id: str | None = None,
        num_replicas: int | None = None,
    ) -> int:
        self._ensure_ray()

        if model_id is None:
            model_id = uuid.uuid4().hex[:8]
        if model_id in self._pools:
            raise ValueError(f"model_id {model_id!r} already loaded")

        if num_replicas is None:
            tp_sizes = {mid: p.tp_size for mid, p in self._pools.items()}
            tp_sizes[model_id] = config.tensor_parallel_size
            plan = self._compute_even_share(tp_sizes)

            for mid, pool in self._pools.items():
                target = plan[mid]
                if target < pool.num_replicas:
                    logger.info(f"Rebalancing {mid!r}: {pool.num_replicas} -> {target} replicas")
                    await pool.scale_down(target)

            num_replicas = plan[model_id]

        pool = ReplicaPool(worker_cls=self._worker_cls)
        await pool.initialize(config, num_replicas=num_replicas)
        self._pools[model_id] = pool
        return num_replicas

    async def shutdown(self, model_id: str | None = None) -> None:
        if model_id is not None:
            pool = self._get_pool(model_id)
            await pool.shutdown()
            del self._pools[model_id]
            if self._pools:
                self._rebalance_up()
        else:
            for pool in self._pools.values():
                await pool.shutdown()
            self._pools.clear()

    # ------------------------------------------------------------------
    # Inference  (route by model_id, delegate to pool)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompts: str | list[int] | list[str | list[int]],
        sampling_params: dict[str, Any] | None = None,
        model_id: str | None = None,
        routing_key: str | list[str | None] | None = None,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Route a /generate to the pool for *model_id*.

        ``routing_key`` and ``strict`` are forwarded to
        :meth:`ReplicaPool.generate` so multi-turn rollouts can pin
        same-keyed prompts to one replica for KV cache reuse. See the
        scheduler module for the full affinity contract.
        """
        return await self._get_pool(model_id).generate(
            prompts, sampling_params,
            routing_key=routing_key, strict=strict,
        )

    # ------------------------------------------------------------------
    # Weight sync  (route by model_id, delegate to pool)
    # ------------------------------------------------------------------

    def get_weights_info(self, model_id: str | None = None) -> list[dict]:
        return self._get_pool(model_id).get_weights_info()

    async def sync_weights(
        self,
        groups: list[dict[str, Any]] | None = None,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
        reverse: bool = False,
        model_id: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).sync_weights(
            groups, bucket_size, strategy=strategy,
            engine_only=engine_only, direct_mode=direct_mode,
            reverse=reverse,
            master_addr=master_addr, master_port=master_port,
            world_size=world_size,
        )

    def get_spec_weights_info(self, model_id: str | None = None) -> list[dict]:
        return self._get_pool(model_id).get_spec_weights_info()

    async def sync_spec_weights(
        self,
        groups: list[dict[str, Any]] | None = None,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        reverse: bool = False,
        model_id: str | None = None,
        master_addr: str | None = None,
        master_port: int | None = None,
        world_size: int | None = None,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).sync_spec_weights(
            groups, bucket_size, strategy=strategy,
            engine_only=engine_only, reverse=reverse,
            master_addr=master_addr, master_port=master_port,
            world_size=world_size,
        )

    async def close_weight_sync(self, model_id: str | None = None) -> dict[str, Any]:
        return await self._get_pool(model_id).close_weight_sync()

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    async def sleep(self, model_id: str, level: int = 1) -> dict[str, Any]:
        """Free GPU memory for *model_id* (drain requests first)."""
        pool = self._get_pool(model_id)
        return await pool.sleep(level=level)

    async def wake_up(
        self, model_id: str, tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Restore GPU memory for *model_id* and resume serving."""
        pool = self._get_pool(model_id)
        return await pool.wake_up(tags=tags)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for mid, pool in self._pools.items():
            models[mid] = await pool.get_status()
        return {
            "total_gpus": self._total_gpus,
            "allocated_gpus": self._allocated_gpus,
            "available_gpus": self._available_gpus,
            "models": models,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def drain_metrics(
        self, model_id: str | None = None,
    ) -> dict[str, Any]:
        """Drain metrics for one model (when ``model_id`` is set) or all."""
        if model_id is not None:
            payload = await self._get_pool(model_id).drain_metrics()
            payload["model_id"] = model_id
            return payload
        models: dict[str, Any] = {}
        for mid, pool in self._pools.items():
            models[mid] = await pool.drain_metrics()
        return {"models": models}


# Swap backend and reuse the app — no endpoint duplication.
api.backend = Driver()
app = api.app
