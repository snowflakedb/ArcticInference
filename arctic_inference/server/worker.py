from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any
from uuid import uuid4

import ray
import torch

logger = logging.getLogger("arctic_inference.server")


def _serialize_logprobs_position(pos_data: dict | None) -> dict[int, dict] | None:
    """Convert a single position's {token_id: Logprob} to {token_id: dict}.

    Works for both prompt_logprobs and sample_logprobs positions.
    vLLM's Logprob is a dataclass with .logprob and .rank attributes.
    """
    if pos_data is None:
        return None
    out = {}
    for tok_id, lp in pos_data.items():
        if hasattr(lp, "logprob"):
            out[tok_id] = {"logprob": lp.logprob, "rank": lp.rank}
        else:
            out[tok_id] = {"logprob": float(lp), "rank": None}
    return out


class WorkerLifecycleState(str, Enum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    SLEEPING = "sleeping"


@ray.remote
class InferenceWorker:
    """Ray actor that hosts an in-process vLLM AsyncLLM engine."""

    def __init__(self) -> None:
        self.llm = None
        self.state = WorkerLifecycleState.UNINITIALIZED

    async def initialize(self, engine_kwargs: dict[str, Any], extra_env: dict[str, str] | None = None) -> None:
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs

        if extra_env:
            os.environ.update(extra_env)

        engine_kwargs.setdefault(
            "worker_extension_cls",
            "arctic_inference.server.weight_sync.WeightSyncExtension",
        )
        engine_args = AsyncEngineArgs(**engine_kwargs)
        logger.info("Worker %d engine_args.enable_sleep_mode=%s",
                     os.getpid(), getattr(engine_args, 'enable_sleep_mode', 'N/A'))
        vllm_config = engine_args.create_engine_config()
        self.llm = AsyncLLM.from_vllm_config(vllm_config)
        self.state = WorkerLifecycleState.READY
        logger.info("Worker %d initialized: model=%s", os.getpid(), engine_kwargs.get("model"))

    async def generate(self, prompt: str | list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        if self.state != WorkerLifecycleState.READY:
            raise RuntimeError(f"Worker not ready: state={self.state.value}")

        from vllm import SamplingParams

        params = SamplingParams(**sampling_params)
        request_id = str(uuid4())

        if isinstance(prompt, list):
            prompt_input: Any = {"prompt_token_ids": prompt}
        else:
            prompt_input = prompt

        final_output = None
        async for output in self.llm.generate(prompt_input, params, request_id=request_id):
            final_output = output

        if not final_output or not final_output.outputs:
            raise RuntimeError("Empty generation output")

        choice = final_output.outputs[0]
        result: dict[str, Any] = {
            "text": choice.text,
            "token_ids": list(choice.token_ids),
            "finish_reason": choice.finish_reason,
        }

        if final_output.prompt_logprobs is not None:
            result["prompt_logprobs"] = [
                _serialize_logprobs_position(pos) for pos in final_output.prompt_logprobs
            ]

        if choice.logprobs is not None:
            result["logprobs"] = [
                _serialize_logprobs_position(pos) for pos in choice.logprobs
            ]

        return result

    def is_healthy(self) -> bool:
        return self.llm is not None and self.state in (
            WorkerLifecycleState.READY, WorkerLifecycleState.SLEEPING,
        )

    def get_state(self) -> str:
        return self.state.value

    def get_stats(self) -> dict[str, Any]:
        return {"state": self.state.value, "pid": os.getpid()}

    def pid(self) -> int:
        return os.getpid()

    # ------------------------------------------------------------------
    # Sleep / Wake
    # ------------------------------------------------------------------

    async def sleep(self, level: int = 1, offload_weights: bool = False) -> dict[str, Any]:
        """Free GPU memory by offloading weights and/or KV cache.

        Args:
            level: 1 = free KV cache only, 2 = free KV cache + weights.
            offload_weights: If True, also manually move model weights to CPU
                after vLLM sleep (for colocated mode where CuMemAllocator
                doesn't manage model weight allocations).
        """
        if self.state != WorkerLifecycleState.READY:
            raise RuntimeError(f"Cannot sleep: worker state is {self.state.value}")
        await self.llm.collective_rpc("sleep", kwargs={"level": level})
        if offload_weights:
            await self.llm.collective_rpc("offload_model_weights")
        self.state = WorkerLifecycleState.SLEEPING
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Worker %d sleeping (level=%d, offload_weights=%s, %.0f MB remaining)",
                     os.getpid(), level, offload_weights, mem_mb)
        return {"status": "sleeping", "level": level, "gpu_mb": mem_mb}

    async def wake_up(self, tags: list[str] | None = None, restore_weights: bool = False) -> dict[str, Any]:
        """Restore GPU memory (reverse of :meth:`sleep`).

        Args:
            tags: What to restore, e.g. ``["weights"]`` or
                  ``["weights", "kv_cache"]``.  ``None`` restores everything.
            restore_weights: If True, also restore manually offloaded model
                weights before the vLLM wake_up.
        """
        if self.state != WorkerLifecycleState.SLEEPING:
            if self.state == WorkerLifecycleState.READY:
                return {"status": "already_ready"}
            raise RuntimeError(f"Cannot wake up: worker state is {self.state.value}")
        if restore_weights:
            await self.llm.collective_rpc("backload_model_weights")
        kwargs: dict[str, Any] = {}
        if tags is not None:
            kwargs["tags"] = tags
        await self.llm.collective_rpc("wake_up", kwargs=kwargs)
        self.state = WorkerLifecycleState.READY
        logger.info("Worker %d awake", os.getpid())
        return {"status": "ready"}

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    async def sync_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        direct_mode: bool = False,
        reverse: bool = False,
    ) -> dict[str, Any]:
        """Receive + load weights on all TP workers via a single call."""
        results = await self.llm.collective_rpc(
            "sync_weights",
            args=(master_addr, master_port, rank_offset, world_size,
                  bucket_size, engine_only, direct_mode, reverse),
        )
        return results[0] if results else {}

    async def sync_weights_ipc(
        self,
        group_id: int,
        timeout: float = 300,
    ) -> dict[str, Any]:
        """Receive + load weights via shared memory (colocated mode)."""
        results = await self.llm.collective_rpc(
            "sync_weights_ipc",
            args=(group_id, timeout),
        )
        return results[0] if results else {}

    async def load_weights_cuda_ipc(self, ipc_payload: dict) -> dict[str, Any]:
        """Load weights from CUDA IPC handles (zero-copy, same GPU)."""
        results = await self.llm.collective_rpc(
            "load_weights_cuda_ipc",
            args=(ipc_payload,),
        )
        return results[0] if results else {}

    async def load_weights_from_cpu(self, weights: list) -> dict[str, Any]:
        """Load weights from CPU tensors (via Ray object store)."""
        results = await self.llm.collective_rpc(
            "load_weights_from_cpu",
            args=(weights,),
        )
        return results[0] if results else {}

    async def load_weights_from_shm_path(self, path: str) -> dict[str, Any]:
        """Load weights from a shared-memory file path."""
        results = await self.llm.collective_rpc(
            "load_weights_from_shm_path",
            args=(path,),
        )
        return results[0] if results else {}

    async def sync_spec_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        reverse: bool = False,
    ) -> dict[str, Any]:
        """Receive + load spec (drafter) model weights on all TP workers."""
        results = await self.llm.collective_rpc(
            "sync_spec_weights",
            args=(master_addr, master_port, rank_offset, world_size,
                  bucket_size, engine_only, reverse),
        )
        return results[0] if results else {}

    async def close_weight_sync(self) -> dict[str, Any]:
        """Destroy persistent NCCLEngine on all TP workers."""
        results = await self.llm.collective_rpc("close_weight_sync")
        return results[0] if results else {}

    def shutdown(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.state = WorkerLifecycleState.UNINITIALIZED
