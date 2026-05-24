"""Multi-model kvcached example: 5 models (0.6B–32B) sharing GPUs.

kvcached uses CUDA VMM to elastically share GPU memory across models.
Models can be loaded, slept (freeing physical GPU pages), and woken
without reloading weights from disk.

Two modes:
  driver  — calls the Driver Python API directly (in-process)
  http    — starts a FastAPI server and drives it via HTTP

Phases (identical in both modes):
  1. Init each model one at a time (sleep previous before loading next)
  2. Wake all models, fire requests to all simultaneously
  3. Prefix cache reuse: shared prefix across active models
  4. Full lifecycle: sleep all → wake all → generate → shutdown

Requires:
  - kvcached daemon: nvidia-cuda-mps-control -d
  - CUDA MPS active
  - KVCACHED_AUTOPATCH=true ENABLE_KVCACHED=1

Usage:
  # Driver mode (default):
  python projects/server/kvcached_multimodel_example.py

  # HTTP mode:
  python projects/server/kvcached_multimodel_example.py --mode http

  # HTTP mode against an already-running server:
  python projects/server/kvcached_multimodel_example.py --mode http --no-server

  # Fewer / smaller models:
  python projects/server/kvcached_multimodel_example.py \\
      --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import concurrent.futures
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kvcached_multimodel")

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    ("Qwen/Qwen3-0.6B",  1),
    ("Qwen/Qwen3-1.7B",  1),
    ("Qwen/Qwen3-4B",    1),
    ("Qwen/Qwen3-8B",    1),
    ("Qwen/Qwen3-32B",   1),
]

PROMPTS = [
    "Explain how photosynthesis works step by step.",
    "What are the key differences between TCP and UDP?",
    "Write a short story about a robot that learns to paint.",
    "Describe the process of nuclear fusion in stars.",
    "What is the significance of the Turing test?",
]

SHARED_PREFIX = (
    "You are a helpful AI assistant. Please provide a clear, concise, "
    "and accurate answer to the following question. "
    "Think step by step before answering.\n\nQuestion: "
)
PREFIX_SUFFIXES = [
    "What is gradient descent?",
    "How does a hash table work?",
    "Explain the CAP theorem.",
    "What is a Fourier transform?",
    "How does TLS encryption work?",
]

SAMPLING_PARAMS: dict[str, Any] = {"max_tokens": 128, "temperature": 0.7}
SHORT_PARAMS: dict[str, Any] = {"max_tokens": 32, "temperature": 0.0}

KVCACHED_ENV = {
    "KVCACHED_AUTOPATCH": "true",
    "ENABLE_KVCACHED": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "ARCTIC_INFERENCE_ENABLED": "1",
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
}

EXTRA_ENGINE_KWARGS = {
    "enable_sleep_mode": True,
    "enable_prefix_caching": True,
    "dtype": "bfloat16",
}

SERVER_PORT = 8321
LOG_DIR = "/tmp/kvcached_multimodel_test_logs"
W = 92


# ── Helpers ───────────────────────────────────────────────────────────

def hr(char="─"):
    print(char * W)


def banner(msg: str):
    hr("═")
    print(f"  {msg}")
    hr("═")


def model_id_for(idx: int, model_name: str) -> str:
    short = model_name.rsplit("/", 1)[-1].lower().replace("-", "_")
    return f"m{idx}_{short}"


# ── Backend abstraction ───────────────────────────────────────────────

class Backend(abc.ABC):
    """Thin interface over Driver (in-process) or HTTP."""

    @abc.abstractmethod
    def init_model(self, model: str, model_id: str, tp: int) -> dict: ...

    @abc.abstractmethod
    def generate(self, model_id: str, prompts: list[str],
                 params: dict[str, Any]) -> list[dict]: ...

    @abc.abstractmethod
    def sleep(self, model_id: str) -> None: ...

    @abc.abstractmethod
    def wake_up(self, model_id: str) -> None: ...

    @abc.abstractmethod
    def get_status(self) -> dict: ...

    @abc.abstractmethod
    def shutdown(self) -> None: ...

    def generate_timed(self, model_id: str, prompts: list[str],
                       params: dict[str, Any] | None = None) -> list[dict]:
        params = params or SAMPLING_PARAMS
        t0 = time.perf_counter()
        results = self.generate(model_id, prompts, params)
        elapsed = time.perf_counter() - t0
        total_toks = sum(len(r["token_ids"]) for r in results)
        print(f"  {model_id}: {len(prompts)} prompts → {total_toks} tokens "
              f"in {elapsed:.2f}s ({total_toks / elapsed:.0f} tok/s)")
        return results


class DriverBackend(Backend):
    """Calls the Driver Python API directly (in-process, async).

    The event loop runs in a dedicated background thread so that
    multiple caller threads can submit coroutines concurrently via
    run_coroutine_threadsafe.
    """

    def __init__(self):
        import threading
        from arctic_inference.server import ModelConfig
        from arctic_inference.server.multi_model import Driver
        self._ModelConfig = ModelConfig
        self._driver = Driver()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _make_config(self, model: str, tp: int):
        return self._ModelConfig(
            model=model,
            tensor_parallel_size=tp,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            extra_engine_kwargs=EXTRA_ENGINE_KWARGS,
            extra_env=KVCACHED_ENV,
            ray_num_gpus=0,
        )

    def init_model(self, model, model_id, tp):
        print(f"  Initializing {model_id} ({model}, TP={tp}) ...")
        config = self._make_config(model, tp)
        n = self._run(self._driver.initialize(config, model_id=model_id, num_replicas=1))
        print(f"  {model_id} ready ({n} replicas)")
        return {"status": "ready", "model_id": model_id, "num_replicas": n}

    def generate(self, model_id, prompts, params):
        return self._run(self._driver.generate(prompts, params, model_id=model_id))

    def sleep(self, model_id):
        print(f"  Sleeping {model_id} ...")
        self._run(self._driver.sleep(model_id))
        print(f"  {model_id} sleeping")

    def wake_up(self, model_id):
        print(f"  Waking {model_id} ...")
        self._run(self._driver.wake_up(model_id))
        print(f"  {model_id} awake")

    def get_status(self):
        return self._run(self._driver.get_status())

    def shutdown(self):
        self._run(self._driver.shutdown())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class HttpBackend(Backend):
    """Drives the FastAPI server via HTTP."""

    def __init__(self):
        self._base = f"http://localhost:{SERVER_PORT}"

    def _post(self, endpoint, body, label, timeout=600):
        r = requests.post(f"{self._base}{endpoint}", json=body, timeout=timeout)
        if r.status_code != 200:
            print(f"  ERROR [{label}] {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        return r.json()

    def _get(self, endpoint, label, timeout=30):
        r = requests.get(f"{self._base}{endpoint}", timeout=timeout)
        if r.status_code != 200:
            print(f"  ERROR [{label}] {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
        return r.json()

    def init_model(self, model, model_id, tp):
        print(f"  Initializing {model_id} ({model}, TP={tp}) ...")
        config = {
            "model": model,
            "tensor_parallel_size": tp,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "extra_engine_kwargs": EXTRA_ENGINE_KWARGS,
            "extra_env": KVCACHED_ENV,
            "ray_num_gpus": 0,
        }
        data = self._post("/init",
                          {"config": config, "model_id": model_id, "num_replicas": 1},
                          label=f"init {model_id}")
        print(f"  {model_id} ready: {data}")
        return data

    def generate(self, model_id, prompts, params):
        data = self._post("/generate",
                          {"model_id": model_id, "prompts": prompts,
                           "sampling_params": params},
                          label=f"gen {model_id}", timeout=300)
        return data["results"]

    def sleep(self, model_id):
        print(f"  Sleeping {model_id} ...")
        self._post("/sleep", {"model_id": model_id, "level": 1},
                   label=f"sleep {model_id}", timeout=120)
        print(f"  {model_id} sleeping")

    def wake_up(self, model_id):
        print(f"  Waking {model_id} ...")
        self._post("/wake_up", {"model_id": model_id},
                   label=f"wake {model_id}", timeout=300)
        print(f"  {model_id} awake")

    def get_status(self):
        return self._get("/status", "status")

    def shutdown(self):
        pass


# ── Phases (backend-agnostic) ─────────────────────────────────────────

def print_status(backend: Backend):
    status = backend.get_status()
    hr()
    print(f"  GPUs: {status.get('allocated_gpus', '?')}/{status.get('total_gpus', '?')}")
    for mid, info in status.get("models", {}).items():
        states = info.get("replica_states", [])
        ready = sum(1 for s in states if s == "ready")
        sleeping = sum(1 for s in states if s == "sleeping")
        print(f"  {mid:30s}  replicas={info['num_replicas']}  "
              f"ready={ready}  sleeping={sleeping}")
    hr()


def phase_init(backend: Backend, models: list[tuple[str, int]]) -> list[str]:
    """Phase 1: Load every model, sleeping the previous before loading next."""
    banner("Phase 1: Initialize all models on shared GPU(s)")
    model_ids: list[str] = []
    for idx, (model, tp) in enumerate(models):
        mid = model_id_for(idx, model)
        model_ids.append(mid)

        if idx > 0:
            prev = model_ids[idx - 1]
            print(f"  Sleeping {prev} before loading next model ...")
            backend.sleep(prev)

        backend.init_model(model, mid, tp)

    backend.sleep(model_ids[-1])
    print_status(backend)
    return model_ids


def _fire_batch(backend: Backend, model_id: str, prompts: list[str],
                params: dict[str, Any]) -> dict:
    """Send a batch and return stats."""
    t0 = time.perf_counter()
    results = backend.generate(model_id, prompts, params)
    elapsed = time.perf_counter() - t0
    toks = sum(len(r["token_ids"]) for r in results)
    return {"model_id": model_id, "n": len(results), "tokens": toks,
            "elapsed": elapsed}


def phase_all_active(backend: Backend, model_ids: list[str],
                     requests_per_model: int = 1000):
    """Phase 2: Wake ALL models, fire thousands of requests concurrently."""
    banner(f"Phase 2: All models active — {requests_per_model} requests/model")
    for mid in model_ids:
        backend.wake_up(mid)
    print_status(backend)

    batch_size = len(PROMPTS)
    waves = requests_per_model // batch_size
    n_models = len(model_ids)
    total_requests = waves * batch_size * n_models
    print(f"  {n_models} models × {waves} waves × {batch_size} prompts = "
          f"{total_requests} total requests")
    print(f"  max_tokens={SAMPLING_PARAMS['max_tokens']}, "
          f"concurrency={n_models} (one thread per model)\n")

    t0_all = time.perf_counter()
    stats: dict[str, dict] = {mid: {"tokens": 0, "requests": 0, "elapsed": 0.0}
                              for mid in model_ids}
    failed = 0

    for wave_idx in range(waves):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_models) as pool:
            futures = {
                pool.submit(_fire_batch, backend, mid, PROMPTS, SAMPLING_PARAMS): mid
                for mid in model_ids
            }
            for f in concurrent.futures.as_completed(futures):
                mid = futures[f]
                try:
                    r = f.result()
                    stats[mid]["tokens"] += r["tokens"]
                    stats[mid]["requests"] += r["n"]
                    stats[mid]["elapsed"] += r["elapsed"]
                except Exception as e:
                    failed += 1
                    logger.error("%s wave %d FAILED: %s", mid, wave_idx + 1, e)

        if (wave_idx + 1) % 20 == 0 or wave_idx == waves - 1:
            done = (wave_idx + 1) * batch_size * n_models
            elapsed_so_far = time.perf_counter() - t0_all
            toks_so_far = sum(s["tokens"] for s in stats.values())
            print(f"  [{wave_idx + 1}/{waves}] {done} requests, "
                  f"{toks_so_far} tokens, "
                  f"{toks_so_far / elapsed_so_far:.0f} tok/s aggregate, "
                  f"{elapsed_so_far:.1f}s elapsed")

    elapsed_all = time.perf_counter() - t0_all
    total_toks = sum(s["tokens"] for s in stats.values())
    total_reqs = sum(s["requests"] for s in stats.values())

    hr()
    print(f"  LOAD TEST RESULTS")
    print(f"  Total: {total_reqs} requests, {total_toks} tokens "
          f"in {elapsed_all:.1f}s")
    print(f"  Aggregate throughput: {total_toks / elapsed_all:.0f} tok/s")
    print(f"  Failed batches: {failed}")
    print()
    for mid in model_ids:
        s = stats[mid]
        avg_batch = s["elapsed"] / max(s["requests"] / batch_size, 1)
        tps = s["tokens"] / s["elapsed"] if s["elapsed"] > 0 else 0
        print(f"  {mid:30s}  {s['requests']:5d} reqs  {s['tokens']:7d} toks  "
              f"{tps:6.0f} tok/s  avg_batch={avg_batch:.2f}s")
    hr()


def phase_prefix_cache(backend: Backend, model_ids: list[str]):
    """Phase 5: Prefix cache reuse — same prefix, different suffixes."""
    banner("Phase 3: Prefix cache reuse (shared prefix)")
    active = model_ids[:3]
    for mid in active:
        prompts = [SHARED_PREFIX + s for s in PREFIX_SUFFIXES]
        print(f"  {mid}: {len(prompts)} prompts with shared prefix "
              f"({len(SHARED_PREFIX)} chars)")
        results = backend.generate_timed(mid, prompts, SHORT_PARAMS)
        for i, r in enumerate(results):
            print(f"    Q: {PREFIX_SUFFIXES[i][:40]:40s}  "
                  f"→ {r['text'][:50]}")


def phase_lifecycle(backend: Backend, model_ids: list[str]):
    """Phase 6: Full sleep-all → wake-all → generate cycle."""
    banner("Phase 4: Full lifecycle (sleep all → wake all → generate)")
    print("  Sleeping all models ...")
    for mid in model_ids:
        try:
            backend.sleep(mid)
        except Exception:
            pass
    print_status(backend)

    print("  Waking all models ...")
    for mid in model_ids:
        backend.wake_up(mid)
    print_status(backend)

    print("  Final generation round ...")
    for mid in model_ids:
        results = backend.generate_timed(mid, [PROMPTS[0]], SHORT_PARAMS)
        print(f"    [{mid}] {results[0]['text'][:80]}...")


# ── Server management (HTTP mode only) ────────────────────────────────

def start_server():
    env = {**os.environ, **KVCACHED_ENV}
    os.makedirs(LOG_DIR, exist_ok=True)
    log_fh = open(f"{LOG_DIR}/server.log", "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "arctic_inference.server.multi_model:app",
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
        ],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    return proc, log_fh


def wait_server_ready(proc, timeout=90):
    base = f"http://localhost:{SERVER_PORT}"
    for i in range(timeout):
        if proc.poll() is not None:
            print(f"  Server died (exit={proc.returncode})")
            with open(f"{LOG_DIR}/server.log") as fh:
                print(fh.read()[-3000:])
            sys.exit(1)
        try:
            r = requests.get(f"{base}/status", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready after {i + 1}s")
                return
        except Exception:
            pass
        time.sleep(1)
    print("  Server did not start in time")
    sys.exit(1)


def kill_proc(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model kvcached test — all models share GPU(s)")
    parser.add_argument(
        "--mode", choices=["driver", "http"], default="driver",
        help="Backend mode: 'driver' (in-process) or 'http' (FastAPI server)")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="HF model names (default: Qwen 0.6B, 1.7B, 4B, 8B, 32B)")
    parser.add_argument(
        "--tp", nargs="+", type=int, default=None,
        help="TP size per model (must match --models length)")
    parser.add_argument(
        "--skip-big", action="store_true",
        help="Drop the 32B model from the default list")
    parser.add_argument(
        "--no-server", action="store_true",
        help="(HTTP mode) Don't start server; assume it's already running")
    parser.add_argument(
        "--requests-per-model", type=int, default=1000,
        help="Number of requests per model in the load test phase (default: 1000)")
    args = parser.parse_args()

    if args.models:
        tp_list = args.tp or [1] * len(args.models)
        if len(tp_list) != len(args.models):
            parser.error("--tp must have same length as --models")
        models = list(zip(args.models, tp_list))
    else:
        models = list(DEFAULT_MODELS)
        if args.skip_big:
            models = models[:-1]

    os.environ.update(KVCACHED_ENV)

    banner(f"kvcached Multi-Model Test ({args.mode} mode): {len(models)} models")
    for model, tp in models:
        print(f"  {model} (TP={tp})")
    print(f"  All models share GPU via kvcached CUDA VMM + MPS")
    hr()

    server_proc = log_fh = None

    if args.mode == "http" and not args.no_server:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True)
        time.sleep(1)
        print("Starting server ...")
        server_proc, log_fh = start_server()
        wait_server_ready(server_proc)

    backend: Backend
    if args.mode == "driver":
        backend = DriverBackend()
    else:
        backend = HttpBackend()

    try:
        model_ids = phase_init(backend, models)
        phase_all_active(backend, model_ids, args.requests_per_model)
        phase_prefix_cache(backend, model_ids)
        phase_lifecycle(backend, model_ids)

        banner("ALL PHASES PASSED")
        if args.mode == "http":
            print(f"  Server logs: {LOG_DIR}/server.log")

    except Exception:
        logger.exception("Test failed")
        if args.mode == "http":
            print(f"  Server logs: {LOG_DIR}/server.log")
        raise

    finally:
        backend.shutdown()
        if server_proc is not None:
            kill_proc(server_proc)
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    main()
