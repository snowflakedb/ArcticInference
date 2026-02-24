"""Dummy worker for CPU-only testing. Drop-in replacement for InferenceWorker."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import ray
from arctic_inference.server.worker import WorkerLifecycleState


@ray.remote
class DummyWorker:
    """Mimics InferenceWorker without vLLM. Generates deterministic dummy outputs."""

    def __init__(self) -> None:
        self.state = WorkerLifecycleState.UNINITIALIZED

    async def initialize(self, engine_kwargs: dict[str, Any], extra_env: dict[str, str] | None = None) -> None:
        self.state = WorkerLifecycleState.READY
        self._model = engine_kwargs.get("model", "dummy")

    async def generate(self, prompt: str | list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        if self.state != WorkerLifecycleState.READY:
            raise RuntimeError(f"Worker not ready: state={self.state.value}")

        await asyncio.sleep(0.01)

        if isinstance(prompt, list):
            text = f"dummy({len(prompt)} tokens)"
            token_ids = list(range(len(prompt), len(prompt) + 5))
        else:
            text = f"dummy({prompt})"
            token_ids = list(range(5))

        return {
            "text": text,
            "token_ids": token_ids,
            "finish_reason": "stop",
        }

    def is_healthy(self) -> bool:
        return self.state == WorkerLifecycleState.READY

    def get_state(self) -> str:
        return self.state.value

    def get_stats(self) -> dict[str, Any]:
        return {"state": self.state.value, "pid": os.getpid()}

    def pid(self) -> int:
        return os.getpid()

    def shutdown(self) -> None:
        self.state = WorkerLifecycleState.UNINITIALIZED
