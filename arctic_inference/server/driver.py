from __future__ import annotations

import logging
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

    async def init(self, config: ModelConfig) -> dict[str, Any]:
        if self.state not in (DriverState.UNINITIALIZED,):
            raise RuntimeError(f"Cannot init from state={self.state.value}")

        self.state = DriverState.INITIALIZING
        num_workers = await self.replica_manager.initialize(config)

        self.scheduler = Scheduler(
            workers=self.replica_manager.workers,
            initial_concurrency=64,
        )

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

    async def status(self) -> dict[str, Any]:
        info = await self.replica_manager.get_status()
        return {"state": self.state.value, **info}

    async def shutdown(self) -> dict[str, Any]:
        if self.scheduler:
            await self.scheduler.shutdown()
            self.scheduler = None
        await self.replica_manager.shutdown()
        self.state = DriverState.UNINITIALIZED
        return {"status": "shutdown"}
