from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any

from arctic_inference.server.config import ModelConfig
from arctic_inference.server.replica_manager import ReplicaManager
from arctic_inference.server.scheduler import Scheduler
from arctic_inference.server.worker import InferenceWorker

logger = logging.getLogger("arctic_inference.server")


class DriverState(str, Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"


class Driver:
    """Top-level orchestrator. Manages ReplicaManager + Scheduler lifecycle
    and exposes the operations that the API layer calls."""

    def __init__(self, worker_cls=InferenceWorker, num_workers: int | None = None) -> None:
        self.state = DriverState.UNINITIALIZED
        self.replica_manager = ReplicaManager(worker_cls=worker_cls, num_workers=num_workers)
        self.scheduler: Scheduler | None = None
        self._op_lock = asyncio.Lock()

    async def init(self, config: ModelConfig) -> dict[str, Any]:
        async with self._op_lock:
            if self.state not in (DriverState.UNINITIALIZED,):
                raise RuntimeError(f"Cannot init from state={self.state.value}")

            self.state = DriverState.INITIALIZING
            num_workers = await self.replica_manager.initialize(config)

            self.scheduler = Scheduler(
                workers=self.replica_manager.workers,
                initial_concurrency=64,
            )

            self.replica_manager.register_scheduler(self.scheduler)

            self.state = DriverState.READY
            return {"status": "ready", "num_replicas": num_workers}

    async def sample(
        self,
        prompts: list[str] | None = None,
        prompt_token_ids: list[list[int]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.state != DriverState.READY:
            raise RuntimeError(f"Cannot sample in state={self.state.value}")
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized")

        if prompts is not None:
            inputs: list[str | list[int]] = prompts
        elif prompt_token_ids is not None:
            inputs = prompt_token_ids
        else:
            raise ValueError("Provide either prompts or prompt_token_ids")

        return await self.scheduler.submit_batch(inputs, sampling_params or {})

    def get_weights_info(self) -> list[dict]:
        if self.state != DriverState.READY:
            raise RuntimeError(f"Cannot get weights info in state={self.state.value}")
        return self.replica_manager.get_weights_info()

    async def sync_weights(
        self,
        groups: list[dict[str, Any]],
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
    ) -> dict[str, Any]:
        """Receive weights from sender(s) and load into all replicas.

        *groups* is a list of dicts, each describing one independent NCCL
        broadcast group::

            {"group_id": 0, "master_addr": "...", "master_port": 29500,
             "world_size": 3, "replica_ids": [0]}

        Applies the requested *strategy* before/after the transfer:
          - **drain**: pause scheduler, wait for in-flight to finish, sync, resume.
          - **skip**: mark workers unavailable, cancel in-flight, sync, re-enable.
          - **hotswap**: sync while serving continues.
        """
        async with self._op_lock:
            if self.state != DriverState.READY:
                raise RuntimeError(f"Cannot sync weights in state={self.state.value}")
            if self.scheduler is None:
                raise RuntimeError("Scheduler not initialized")

            t0 = time.time()
            n = len(self.replica_manager.workers)

            if strategy == "drain":
                self.scheduler.pause()
                await self.scheduler.drain()
            elif strategy == "skip":
                for i in range(n):
                    self.scheduler.mark_worker_unavailable(i)
                for i in range(n):
                    self.scheduler.cancel_worker_inflight(i)
            elif strategy != "hotswap":
                raise ValueError(f"Unknown strategy: {strategy!r}. Use: drain, skip, hotswap")

            result = await self.replica_manager.sync_weights(
                groups, bucket_size, engine_only=engine_only, direct_mode=direct_mode)

            if strategy == "drain":
                self.scheduler.resume()
            elif strategy == "skip":
                for i in range(n):
                    self.scheduler.mark_worker_available(i)

            result["strategy"] = strategy
            result["strategy_elapsed"] = time.time() - t0
            return result

    async def close_weight_sync(self) -> dict[str, Any]:
        async with self._op_lock:
            if self.state != DriverState.READY:
                raise RuntimeError(f"Cannot close weight sync in state={self.state.value}")
            return await self.replica_manager.close_weight_sync()

    async def status(self) -> dict[str, Any]:
        info = await self.replica_manager.get_status()
        return {"state": self.state.value, **info}

    async def shutdown(self) -> dict[str, Any]:
        async with self._op_lock:
            if self.scheduler:
                await self.scheduler.shutdown()
                self.scheduler = None
            await self.replica_manager.shutdown()
            self.state = DriverState.UNINITIALIZED
            return {"status": "shutdown"}
