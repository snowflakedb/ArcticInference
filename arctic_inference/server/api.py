from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from arctic_inference.server.replica_pool import ReplicaPool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from arctic_inference.server.config import ModelConfig


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class InitRequest(BaseModel):
    config: ModelConfig
    model_id: str | None = None


class GenerateRequest(BaseModel):
    model_id: str | None = None
    prompts: list[str | list[int]]
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class GroupConfig(BaseModel):
    group_id: int
    master_addr: str
    master_port: int
    world_size: int
    replica_ids: list[int]


class SleepRequest(BaseModel):
    model_id: str
    level: int = 1


class WakeUpRequest(BaseModel):
    model_id: str
    tags: list[str] | None = None


class SyncWeightsRequest(BaseModel):
    model_id: str | None = None
    groups: list[GroupConfig] | None = None
    bucket_size: int = 256 * 1024 * 1024
    strategy: str = "hotswap"
    engine_only: bool = False
    direct_mode: bool = False

    # Legacy flat fields
    master_addr: str | None = None
    master_port: int | None = None
    world_size: int | None = None


# ---------------------------------------------------------------------------
# Backend – a bare ReplicaPool by default, swapped to Driver by multi_model.py
# ---------------------------------------------------------------------------

backend: Any = ReplicaPool()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await backend.shutdown()


app = FastAPI(title="Arctic Inference", lifespan=lifespan)


@app.post("/init")
async def init_endpoint(request: InitRequest):
    try:
        n = await backend.initialize(request.config, model_id=request.model_id)
        return {"status": "ready", "model_id": request.model_id, "num_replicas": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        results = await backend.generate(**request.model_dump())
        return {"results": results}
    except RuntimeError as e:
        msg = str(e).lower()
        if "paused" in msg or "cancelled" in msg:
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sleep")
async def sleep_endpoint(request: SleepRequest):
    """Free GPU memory for a model (drain in-flight requests first)."""
    try:
        return await driver.sleep(request.model_id, level=request.level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wake_up")
async def wake_up_endpoint(request: WakeUpRequest):
    """Restore GPU memory for a model and resume serving."""
    try:
        return await driver.wake_up(request.model_id, tags=request.tags)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weights_info")
async def weights_info_endpoint(model_id: str | None = None):
    try:
        infos = backend.get_weights_info(model_id=model_id)
        return {"weights_info": infos, "count": len(infos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync_weights")
async def sync_weights_endpoint(request: SyncWeightsRequest):
    try:
        return await backend.sync_weights(**request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close_weight_sync")
async def close_weight_sync_endpoint(model_id: str | None = None):
    try:
        return await backend.close_weight_sync(model_id=model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status_endpoint():
    try:
        return await backend.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shutdown")
async def shutdown_endpoint(model_id: str | None = None):
    try:
        await backend.shutdown(model_id=model_id)
        return {"status": "shutdown", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
