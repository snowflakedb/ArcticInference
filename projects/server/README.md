## Arctic Inference Server

Multi-replica inference server built on Ray and vLLM. Automatically distributes workers across available GPUs, routes requests with dynamic concurrency control, and monitors worker health with auto-restart.

### Architecture

```
Client ──> FastAPI ──> Driver ──> Scheduler ──> Worker 0 (vLLM AsyncLLM)
                         │                 ├──> Worker 1
                         │                 └──> Worker N
                         └── ReplicaManager (lifecycle + health monitoring)
```

**Components:**

- **Driver** — Top-level orchestrator managing the full init/sample/shutdown lifecycle.
- **ReplicaManager** — Creates Ray worker actors, auto-detects GPU topology, and runs background health checks with auto-restart.
- **Scheduler** — Routes requests across workers using least-loaded routing with utilization-based concurrency adjustment.
- **Pipeline** — High-level dataset processing with the `@task` decorator, managed concurrency, progress tracking, and retry logic.

### Quick Start

Install with the server extra:

```console
$ pip install -e ".[server]"
```

### Usage

There are two ways to use the server: via the **HTTP API** or the **Python API**.

#### HTTP API

Start the server:

```console
$ arctic_inference_server --port 8000
```

Then interact with it via curl (see `api_example.sh`):

```console
$ bash projects/server/api_example.sh
```

#### Python API (Driver)

Use the `Driver` directly for programmatic control (see `driver_example.py`):

```console
$ python projects/server/driver_example.py
```

#### Python API (Pipeline)

Use the `Pipeline` for batch dataset processing (see `pipeline_example.py`):

```console
$ python projects/server/pipeline_example.py
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/init` | Initialize workers with a `ModelConfig` |
| `POST` | `/sample` | Generate from text prompts or token IDs |
| `GET` | `/status` | Get driver state and per-replica status |
| `POST` | `/shutdown` | Shut down all workers and the scheduler |

### Model Registry

Pre-tuned configs are available for common GPU + model combinations via `ModelConfig.from_registry()`:

| GPU | Model |
|-----|-------|
| H200 | Qwen3-30B-A3B, Qwen3-30B-A3B-Instruct-2507, Qwen3-235B-A22B-Instruct-2507, Qwen3-Coder-480B-A35B-Instruct |
| B200 | Qwen3-30B-A3B, Qwen3-235B-A22B-Instruct-2507 |
