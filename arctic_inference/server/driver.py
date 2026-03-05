from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import ray
import torch

from arctic_inference.server.config import ModelConfig
from arctic_inference.server.replica_pool import ReplicaPool
from arctic_inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_inference.server")


class Driver:
    """Main server class. Manages multiple models across a GPU cluster.

    Each :meth:`init` call loads a model, allocates GPUs via even sharing,
    and returns a ``model_id``.  All subsequent operations require that
    ``model_id``.

    Connects to an existing Ray cluster or starts one using all visible GPUs.
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
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        nodes = [n for n in ray.nodes() if n["Alive"]]
        if not nodes:
            raise RuntimeError("No alive Ray nodes")
        self._total_gpus = sum(
            int(n["Resources"].get("GPU", torch.cuda.device_count()))
            for n in nodes
        )
        if self._total_gpus == 0:
            raise RuntimeError("No GPUs available in the Ray cluster")
        self._ray_initialized = True
        logger.info(f"Driver: {self._total_gpus} GPUs available across {len(nodes)} node(s)")

    @property
    def _allocated_gpus(self) -> int:
        return sum(p.num_replicas * p.tp_size for p in self._pools.values())

    @property
    def _available_gpus(self) -> int:
        return self._total_gpus - self._allocated_gpus

    def _compute_even_share(self, tp_sizes: dict[str, int]) -> dict[str, int]:
        """Return model_id -> num_replicas under even GPU sharing."""
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

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    async def init(
        self,
        config: ModelConfig,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Load a model and allocate GPU resources.

        GPUs are split evenly across all loaded models; existing pools are
        automatically scaled down to make room.
        """
        self._ensure_ray()

        if model_id is None:
            model_id = uuid.uuid4().hex[:8]
        if model_id in self._pools:
            raise ValueError(f"model_id {model_id!r} already loaded")

        tp_sizes = {mid: p.tp_size for mid, p in self._pools.items()}
        tp_sizes[model_id] = config.tensor_parallel_size
        plan = self._compute_even_share(tp_sizes)

        for mid, pool in self._pools.items():
            target = plan[mid]
            if target < pool.num_replicas:
                logger.info(f"Rebalancing {mid!r}: {pool.num_replicas} -> {target} replicas")
                await pool.scale_down(target)

        new_replicas = plan[model_id]
        pool = ReplicaPool(
            config=config,
            worker_cls=self._worker_cls,
            num_replicas=new_replicas,
            num_gpus_per_replica=config.tensor_parallel_size,
        )
        await pool.initialize()
        self._pools[model_id] = pool

        return {
            "status": "ready",
            "model_id": model_id,
            "num_replicas": new_replicas,
        }

    async def shutdown_model(self, model_id: str) -> dict[str, Any]:
        pool = self._get_pool(model_id)
        await pool.shutdown()
        del self._pools[model_id]

        # Rebalance remaining pools in the background so we return immediately.
        if self._pools:
            self._rebalance_up()

        return {"status": "shutdown", "model_id": model_id}

    def _rebalance_up(self) -> None:
        """Compute even share and scale up any under-allocated pools (fire-and-forget)."""
        tp_sizes = {mid: p.tp_size for mid, p in self._pools.items()}
        plan = self._compute_even_share(tp_sizes)
        for mid, p in self._pools.items():
            target = plan[mid]
            if target > p.num_replicas:
                logger.info(f"Scaling up {mid!r}: {p.num_replicas} -> {target} replicas (background)")
                asyncio.ensure_future(p.scale_up(target))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def sample(
        self,
        model_id: str,
        prompts: list[str] | None = None,
        prompt_token_ids: list[list[int]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pool = self._get_pool(model_id)
        if prompts is not None:
            inputs: list[str | list[int]] = prompts
        elif prompt_token_ids is not None:
            inputs = prompt_token_ids
        else:
            raise ValueError("Provide either prompts or prompt_token_ids")
        return await pool.submit_batch(inputs, sampling_params or {})

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def get_weights_info(self, model_id: str) -> list[dict]:
        return self._get_pool(model_id).get_weights_info()

    async def sync_weights(
        self,
        model_id: str,
        groups: list[dict[str, Any]],
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
    ) -> dict[str, Any]:
        return await self._get_pool(model_id).sync_weights(
            groups, bucket_size, strategy=strategy,
            engine_only=engine_only, direct_mode=direct_mode,
        )

    async def close_weight_sync(self, model_id: str) -> dict[str, Any]:
        return await self._get_pool(model_id).close_weight_sync()

    # ------------------------------------------------------------------
    # Status / shutdown
    # ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        models: dict[str, Any] = {}
        for mid, pool in self._pools.items():
            info = await pool.get_status()
            models[mid] = {"model": pool.config.model, **info}
        return {
            "total_gpus": self._total_gpus,
            "allocated_gpus": self._allocated_gpus,
            "available_gpus": self._available_gpus,
            "models": models,
        }

    async def shutdown(self) -> dict[str, Any]:
        for pool in self._pools.values():
            await pool.shutdown()
        self._pools.clear()
        return {"status": "shutdown"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pool(self, model_id: str) -> ReplicaPool:
        try:
            return self._pools[model_id]
        except KeyError:
            raise KeyError(f"Unknown model_id {model_id!r}. Loaded: {list(self._pools)}")
