#!/usr/bin/env python3
"""Two-model lifecycle test for kvcached prefix caching.

Phases:
  1. Model A alone -- full GPU, prefix caching warm
  2. Wake Model B -- A and B co-exist, pressure-driven eviction active
  3. Sleep A -- Model B alone, prefix caching at full capacity

Uses the ArcticInference multi-model server (Driver + Ray workers)
with kvcached for elastic GPU memory sharing.

Usage:
  python test_kvcached_prefix_caching.py                    # both TP=1
  python test_kvcached_prefix_caching.py --tp 2             # both TP=2
  python test_kvcached_prefix_caching.py --tp-a 2 --tp-b 1  # mixed TP
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import requests

MODEL_A = "Qwen/Qwen3-32B"
MODEL_B = "Qwen/Qwen3-1.7B"
MODEL_ID_A = "model_a"
MODEL_ID_B = "model_b"
SERVER_PORT = 8000
BASE = f"http://localhost:{SERVER_PORT}"
LOG_DIR = "/tmp/test_kvcached_prefix_caching_logs"

PREFIX = (
    "Once upon a time in a land far away there lived a great king "
    "who ruled over a vast kingdom with wisdom and courage."
)
SUFFIXES = [
    "castle", "army", "sword", "kingdom", "horses", "enemies", "legacy",
]

SERVER_ENV = {
    "KVCACHED_AUTOPATCH": "true",
    "ENABLE_KVCACHED": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "ARCTIC_INFERENCE_ENABLED": "1",
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
}

WORKER_EXTRA_ENV = {
    "KVCACHED_AUTOPATCH": "true",
    "ENABLE_KVCACHED": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "ARCTIC_INFERENCE_ENABLED": "1",
}

EXTRA_ENGINE_KWARGS = {
    "enable_sleep_mode": True,
    "enable_prefix_caching": True,
    "dtype": "bfloat16",
}


def make_model_config(model, tp):
    return {
        "model": model,
        "tensor_parallel_size": tp,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "extra_engine_kwargs": EXTRA_ENGINE_KWARGS,
        "extra_env": WORKER_EXTRA_ENV,
        "ray_num_gpus": 0,
    }


def start_server():
    env = {**os.environ, **SERVER_ENV}
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


def wait_server_ready(proc, timeout=60):
    for i in range(timeout):
        if proc.poll() is not None:
            print(f"Server died (exit={proc.returncode}).")
            with open(f"{LOG_DIR}/server.log") as fh:
                print(fh.read()[-3000:])
            sys.exit(1)
        try:
            r = requests.get(f"{BASE}/status", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready after {i + 1}s")
                return
        except Exception:
            pass
        time.sleep(1)
    print("  Server did not start in time")
    sys.exit(1)


def api_post(endpoint, body, label, timeout=600):
    r = requests.post(f"{BASE}{endpoint}", json=body, timeout=timeout)
    if r.status_code != 200:
        print(f"  ERROR [{label}] {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
    return r.json()


def init_model(model, model_id, tp):
    print(f"  Initializing {model_id} ({model}, TP={tp}) ...")
    data = api_post(
        "/init",
        {"config": make_model_config(model, tp), "model_id": model_id,
         "num_replicas": 1},
        label=f"init {model_id}",
    )
    print(f"  {model_id} ready: {data}")
    return data


def sleep_model(model_id, level=1):
    print(f"  Sleeping {model_id} (level={level}) ...")
    api_post("/sleep", {"model_id": model_id, "level": level},
             label=f"sleep {model_id}", timeout=120)
    print(f"  {model_id} sleeping")


def wake_model(model_id):
    print(f"  Waking {model_id} ...")
    api_post("/wake_up", {"model_id": model_id},
             label=f"wake {model_id}", timeout=120)
    print(f"  {model_id} awake")


def gen(model_id, suffix, max_tokens=10):
    prompt = f"{PREFIX} {suffix}"
    data = api_post(
        "/generate",
        {"model_id": model_id, "prompts": [prompt],
         "sampling_params": {"max_tokens": max_tokens, "temperature": 0}},
        label=f"gen {model_id}", timeout=120,
    )
    text = data["results"][0]["text"].strip()
    print(f"    [{model_id}:{suffix:>10s}] {text[:60]}")
    return data


def kill_proc(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=None,
                        help="Tensor parallel size for both models")
    parser.add_argument("--tp-a", type=int, default=None,
                        help="Tensor parallel size for Model A")
    parser.add_argument("--tp-b", type=int, default=None,
                        help="Tensor parallel size for Model B")
    args = parser.parse_args()

    tp_a = args.tp_a or args.tp or 1
    tp_b = args.tp_b or args.tp or 1

    os.makedirs(LOG_DIR, exist_ok=True)

    subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True)
    time.sleep(1)

    print(f"=== Starting ArcticInference multi-model server ===")
    print(f"  Model A: {MODEL_A} (TP={tp_a})")
    print(f"  Model B: {MODEL_B} (TP={tp_b})")
    proc, log_fh = start_server()

    try:
        wait_server_ready(proc)

        init_model(MODEL_A, MODEL_ID_A, tp_a)
        init_model(MODEL_B, MODEL_ID_B, tp_b)

        print("\n  Sleeping Model B so Model A has full GPU ...")
        sleep_model(MODEL_ID_B)

        print("\n=== Phase 1: Model A alone ===")
        for s in SUFFIXES[:4]:
            gen(MODEL_ID_A, s)

        print("\n=== Phase 2: Wake B, both models co-exist ===")
        wake_model(MODEL_ID_B)
        print("  Interleaved requests:")
        for s in SUFFIXES:
            gen(MODEL_ID_A, s)
            gen(MODEL_ID_B, s)

        print("\n=== Phase 3: Sleep A, Model B alone ===")
        sleep_model(MODEL_ID_A)
        for s in SUFFIXES[:4]:
            gen(MODEL_ID_B, s)

        print("\n=== All phases passed ===")
        print(f"Logs: {LOG_DIR}/server.log")

    finally:
        kill_proc(proc)
        log_fh.close()


if __name__ == "__main__":
    main()
