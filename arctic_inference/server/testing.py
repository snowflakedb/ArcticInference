"""Dummy worker for CPU-only testing. Drop-in replacement for InferenceWorker."""
from __future__ import annotations

import asyncio
import math
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
            num_prompt_tokens = len(prompt)
        else:
            text = f"dummy({prompt})"
            token_ids = list(range(5))
            num_prompt_tokens = max(1, len(prompt.split()))

        result: dict[str, Any] = {
            "text": text,
            "token_ids": token_ids,
            "finish_reason": "stop",
        }

        top_k = sampling_params.get("prompt_logprobs")
        if top_k is not None:
            result["prompt_logprobs"] = _dummy_prompt_logprobs(
                num_prompt_tokens, int(top_k),
            )

        top_k_sample = sampling_params.get("logprobs")
        if top_k_sample is not None:
            result["logprobs"] = _dummy_sample_logprobs(
                len(token_ids), int(top_k_sample),
            )

        return result

    def is_healthy(self) -> bool:
        return self.state == WorkerLifecycleState.READY

    def get_state(self) -> str:
        return self.state.value

    def get_stats(self) -> dict[str, Any]:
        return {"state": self.state.value, "pid": os.getpid()}

    def pid(self) -> int:
        return os.getpid()

    async def sync_weights(
        self, master_addr: str, master_port: int, rank_offset: int,
        world_size: int, bucket_size: int = 256 * 1024 * 1024,
    ) -> dict[str, Any]:
        return {"status": "done", "params_loaded": 0, "elapsed": 0.0}

    async def close_weight_sync(self) -> dict[str, Any]:
        return {"status": "ok"}

    def shutdown(self) -> None:
        self.state = WorkerLifecycleState.UNINITIALIZED


def _dummy_prompt_logprobs(
    num_tokens: int, top_k: int,
) -> list[dict[int, dict] | None]:
    """Produce deterministic dummy prompt_logprobs matching vLLM's format."""
    out: list[dict[int, dict] | None] = [None]  # first position is always None
    for pos in range(1, num_tokens):
        out.append({
            pos + k: {"logprob": -0.1 * (k + 1), "rank": k + 1}
            for k in range(top_k)
        })
    return out


def _dummy_sample_logprobs(
    num_tokens: int, top_k: int,
) -> list[dict[int, dict]]:
    """Produce deterministic dummy sample logprobs."""
    return [
        {
            pos + k: {"logprob": -0.2 * (k + 1), "rank": k + 1}
            for k in range(top_k)
        }
        for pos in range(num_tokens)
    ]
