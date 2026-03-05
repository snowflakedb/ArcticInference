## Arctic Inference Server

Multi-model inference server built on Ray and vLLM. Serves multiple models simultaneously with automatic GPU sharing, dynamic rebalancing, and per-pool concurrency control.

### Architecture

```
Client ──> FastAPI ──> Driver ──> ReplicaPool (model-a) ──> Scheduler ──> Worker 0
                         │                                            └──> Worker 1
                         └──────> ReplicaPool (model-b) ──> Scheduler ──> Worker 0
```

**Components:**

- **Driver** — Main server class. Tracks replica pools, manages GPU allocation with even sharing, and routes requests by `model_id`.
- **ReplicaPool** — Manages workers for a single model. Owns a Scheduler, handles scaling, health monitoring, weight sync, and request submission.
- **Scheduler** — Routes requests to the least-loaded worker with utilization-based concurrency adjustment.
- **Worker** — Light wrapper around a vLLM `AsyncLLM` engine.
- **Pipeline** — High-level dataset processing supporting single or multi-model workflows.

### Quick Start

```console
$ pip install -e ".[server]"
```

### Usage

#### HTTP API

Start the server, then use curl to load models and send requests:

```console
$ arctic_inference_server --port 8000
$ bash projects/server/api_example.sh
```

#### Python API (Driver)

```console
$ python projects/server/driver_example.py
```

#### Python API (Pipeline)

```console
$ python projects/server/pipeline_example.py
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/init` | Load a model. Body: `{config: ModelConfig, model_id?: str}`. Returns `model_id`. |
| `POST` | `/sample` | Generate from prompts or token IDs. Requires `model_id`. |
| `GET` | `/weights_info` | Get weight metadata for a model. Requires `model_id` query param. |
| `POST` | `/sync_weights` | Receive weights via NCCL. Requires `model_id`. |
| `POST` | `/close_weight_sync` | Close NCCL engines. Requires `model_id` query param. |
| `GET` | `/status` | GPU allocation and per-model replica status. |
| `GET` | `/models` | Alias for `/status`. |
| `POST` | `/shutdown_model` | Shut down a single model. Requires `model_id` query param. |
| `POST` | `/shutdown` | Shut down all models. |

### Multi-Model GPU Sharing

When multiple models are loaded, GPUs are split evenly. Loading a new model automatically scales down existing pools:

```
Init model-a (4 GPUs available) → model-a gets 4 replicas
Init model-b                     → model-a scales to 2, model-b gets 2
Shutdown model-a                 → model-b keeps 2 (can re-init to reclaim)
```

### Model Registry

Pre-tuned configs via `ModelConfig.from_registry()`:

| GPU | Model |
|-----|-------|
| H200 | Qwen3-30B-A3B, Qwen3-30B-A3B-Instruct-2507, Qwen3-235B-A22B-Instruct-2507, Qwen3-Coder-480B-A35B-Instruct |
| B200 | Qwen3-30B-A3B, Qwen3-235B-A22B-Instruct-2507 |
