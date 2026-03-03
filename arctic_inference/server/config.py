from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

_QUANT_SUFFIXES = re.compile(r"-(FP8|NVFP4|FP4)$", re.IGNORECASE)


def normalize_model_name(name: str) -> str:
    basename = name.rstrip("/").split("/")[-1]
    return _QUANT_SUFFIXES.sub("", basename)


class GPUType(str, Enum):
    H200 = "H200"
    B200 = "B200"


class ModelConfig(BaseModel):
    """Configuration for vLLM engine initialization.

    Field names match AsyncEngineArgs where possible so to_engine_kwargs()
    can pass them through directly.
    """

    model: str = Field(description="Model name, HuggingFace ID, or local path")
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    max_num_seqs: int = 1024
    gpu_memory_utilization: float = 0.95
    quantization: str | None = None
    enable_chunked_prefill: bool = True
    kv_cache_dtype: str | None = None
    enable_expert_parallel: bool = False
    trust_remote_code: bool = True

    extra_engine_kwargs: dict[str, Any] = Field(default_factory=dict, exclude=True)
    extra_env: dict[str, str] = Field(default_factory=dict, exclude=True)

    def to_engine_kwargs(self) -> dict[str, Any]:
        kwargs = {k: v for k, v in self.model_dump().items() if v is not None}
        kwargs.update(self.extra_engine_kwargs)
        return kwargs

    @classmethod
    def from_registry(
        cls,
        name: str,
        gpu_type: GPUType,
        max_model_len: int = 32768,
    ) -> ModelConfig:
        key = normalize_model_name(name)
        if gpu_type not in MODEL_CONFIGS or key not in MODEL_CONFIGS[gpu_type]:
            raise ValueError(f"No config for model={key!r} on gpu={gpu_type.value}")
        return cls(max_model_len=max_model_len, **MODEL_CONFIGS[gpu_type][key])


MODEL_CONFIGS: dict[GPUType, dict[str, dict[str, Any]]] = {
    GPUType.H200: {
        "Qwen3-30B-A3B": dict(
            model="Qwen/Qwen3-30B-A3B",
            tensor_parallel_size=1,
            quantization="fp8",
        ),
        "Qwen3-30B-A3B-Instruct-2507": dict(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            tensor_parallel_size=1,
            quantization="fp8",
        ),
        "Qwen3-235B-A22B-Instruct-2507": dict(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            tensor_parallel_size=2,
        ),
        "Qwen3-Coder-480B-A35B-Instruct": dict(
            model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            tensor_parallel_size=4,
        ),
    },
    GPUType.B200: {
        "Qwen3-30B-A3B": dict(
            model="Qwen/Qwen3-30B-A3B",
            quantization="fp8",
            kv_cache_dtype="fp8",
            enable_expert_parallel=True,
            extra_engine_kwargs={
                "compilation_config": {"pass_config": {"enable_fi_allreduce_fusion": True, "enable_noop": True}}
            },
        ),
        "Qwen3-235B-A22B-Instruct-2507": dict(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            tensor_parallel_size=2,
            kv_cache_dtype="fp8",
            enable_expert_parallel=True,
            extra_env={"VLLM_USE_FLASHINFER_MOE_FP8": "1"},
        ),
    },
}
