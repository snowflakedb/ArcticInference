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
    try:
        results = await driver.sample(
            prompts=request.prompts,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
        )
        return {"results": results}
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
