from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any
from uuid import uuid4

import ray

logger = logging.getLogger("arctic_inference.server")


class WorkerLifecycleState(str, Enum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"


@ray.remote
class InferenceWorker:
    """Ray actor that hosts an in-process vLLM AsyncLLM engine."""

    def __init__(self) -> None:
        self.llm = None  # AsyncLLM, set in initialize()
        self.state = WorkerLifecycleState.UNINITIALIZED

    async def initialize(self, engine_kwargs: dict[str, Any], extra_env: dict[str, str] | None = None) -> None:
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs

        if extra_env:
            os.environ.update(extra_env)

        engine_args = AsyncEngineArgs(**engine_kwargs)
        vllm_config = engine_args.create_engine_config()
        self.llm = AsyncLLM.from_vllm_config(vllm_config)
        self.state = WorkerLifecycleState.READY
        logger.info(f"Worker {os.getpid()} initialized: model={engine_kwargs.get('model')}")

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
        return {
            "text": choice.text,
            "token_ids": list(choice.token_ids),
            "finish_reason": choice.finish_reason,
        }

    def is_healthy(self) -> bool:
        return self.llm is not None and self.state == WorkerLifecycleState.READY

    def get_state(self) -> str:
        return self.state.value

    def get_stats(self) -> dict[str, Any]:
        return {"state": self.state.value, "pid": os.getpid()}

    def pid(self) -> int:
        return os.getpid()

    def shutdown(self) -> None:
        if self.llm is not None:
            del self.llm
            self.llm = None
        self.state = WorkerLifecycleState.UNINITIALIZED
