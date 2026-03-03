from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from arctic_inference.server.config import ModelConfig
from arctic_inference.server.driver import Driver
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


class SampleRequest(BaseModel):
    prompts: list[str] | None = None
    prompt_token_ids: list[list[int]] | None = None
    sampling_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_inputs(self) -> "SampleRequest":
        if self.prompts is None and self.prompt_token_ids is None:
            raise ValueError("Provide either 'prompts' or 'prompt_token_ids'")
        if self.prompts is not None and self.prompt_token_ids is not None:
            raise ValueError("Provide only one of 'prompts' or 'prompt_token_ids'")
        return self


class GroupConfig(BaseModel):
    group_id: int
    master_addr: str
    master_port: int
    world_size: int
    replica_ids: list[int]


class SyncWeightsRequest(BaseModel):
    groups: list[GroupConfig] | None = None
    bucket_size: int = 256 * 1024 * 1024
    strategy: str = "hotswap"
    engine_only: bool = False
    direct_mode: bool = False

    # Legacy flat fields — auto-wrapped into a single group covering all replicas
    master_addr: str | None = None
    master_port: int | None = None
    world_size: int | None = None


driver = Driver()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await driver.shutdown()


app = FastAPI(title="Arctic Inference", lifespan=lifespan)


@app.post("/init")
async def init_endpoint(config: ModelConfig):
    try:
        return await driver.init(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sample")
async def sample_endpoint(request: SampleRequest):
    """Generate text and/or compute log probabilities.

    Log-prob support is built into vLLM's SamplingParams:

    - ``prompt_logprobs``: set to top-k (int) to get per-prompt-token logprobs.
      Returned under the ``prompt_logprobs`` key in each result.
      For GRPO-style reference log-probs, concatenate prompt + completion as the
      prompt text and pass ``{"prompt_logprobs": k, "max_tokens": 1}``.

    - ``logprobs``: set to top-k (int) to get per-generated-token logprobs.
      Returned under the ``logprobs`` key in each result.
    """
    try:
        results = await driver.sample(
            prompts=request.prompts,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
        )
        return {"results": results}
    except RuntimeError as e:
        msg = str(e).lower()
        if "paused" in msg or "cancelled" in msg:
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weights_info")
async def weights_info_endpoint():
    try:
        infos = driver.get_weights_info()
        return {"weights_info": infos, "count": len(infos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync_weights")
async def sync_weights_endpoint(request: SyncWeightsRequest):
    try:
        if request.groups is not None:
            groups = [g.model_dump() for g in request.groups]
        elif request.master_addr is not None and request.master_port is not None:
            num_replicas = (await driver.status()).get("num_replicas", 1)
            groups = [{
                "group_id": 0,
                "master_addr": request.master_addr,
                "master_port": request.master_port,
                "world_size": request.world_size
                    or (1 + num_replicas * (driver.replica_manager.config.tensor_parallel_size if driver.replica_manager.config else 1)),
                "replica_ids": list(range(num_replicas)),
            }]
        else:
            raise ValueError("Provide either 'groups' or legacy flat fields (master_addr, master_port, world_size)")

        return await driver.sync_weights(
            groups,
            bucket_size=request.bucket_size,
            strategy=request.strategy,
            engine_only=request.engine_only,
            direct_mode=request.direct_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close_weight_sync")
async def close_weight_sync_endpoint():
    try:
        return await driver.close_weight_sync()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status_endpoint():
    try:
        return await driver.status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shutdown")
async def shutdown_endpoint():
    try:
        await driver.shutdown()
        return {"status": "shutdown"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
