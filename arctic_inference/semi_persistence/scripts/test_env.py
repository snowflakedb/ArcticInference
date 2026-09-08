"""Smoke test for the ``vllm_config["_env"]`` reserved key.

Exercises the per-model env-var path end to end:

1. Two Instances with the same model but different ``_env`` see
   different ``os.environ`` in their child processes (read via
   ``/proc/<pid>/environ``).
2. Reserved keys in ``_env`` (``CUDA_VISIBLE_DEVICES``,
   ``VLLM_ENABLE_V1_MULTIPROCESSING``, ``USE_LIBUV``) do *not* override
   the values the child sets at the top of ``vllm_child_loop``.
3. The on-disk ``meta.json`` from ``criu_dump`` preserves ``_env`` in
   the persisted ``vllm_config`` so the orchestrator can rediscover it
   on reboot and client-side dedup stays honest.

Imperative script requiring two real GPUs: run with
``python scripts/test_env.py [gpu_a] [gpu_b]``.  Defaults to GPUs 0 and 1.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

from arctic_inference.semi_persistence import Instance


CONFIG = {"model": "Qwen/Qwen3.5-2B", "gpu_memory_utilization": 0.4}
IMAGE_ROOT = "/data-fast/image-cache/test/_env_smoke"


def _read_env(pid: int) -> dict[str, str]:
    """Parse ``/proc/<pid>/environ`` into a dict."""
    with open(f"/proc/{pid}/environ", "rb") as f:
        raw = f.read()
    out: dict[str, str] = {}
    for kv in raw.split(b"\x00"):
        if not kv:
            continue
        k, _, v = kv.partition(b"=")
        out[k.decode(errors="replace")] = v.decode(errors="replace")
    return out


def main() -> None:
    gpu_a = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gpu_b = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if os.path.isdir(IMAGE_ROOT):
        shutil.rmtree(IMAGE_ROOT)

    cfg_a = dict(
        CONFIG,
        _env={
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "SEMIP_PROBE_VAR": "value-A",
            "CUDA_VISIBLE_DEVICES": "99",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "USE_LIBUV": "1",
        },
    )
    cfg_b = dict(
        CONFIG,
        _env={
            "VLLM_LOGGING_LEVEL": "WARNING",
            "SEMIP_PROBE_VAR": "value-B",
        },
    )

    inst_a = Instance(cfg_a)
    inst_a.init(gpu=gpu_a)
    inst_b = Instance(cfg_b)
    inst_b.init(gpu=gpu_b)
    inst_a.wait()
    inst_b.wait()

    env_a = _read_env(inst_a.pid)
    env_b = _read_env(inst_b.pid)

    assert env_a.get("SEMIP_PROBE_VAR") == "value-A", (
        f"A SEMIP_PROBE_VAR expected 'value-A', "
        f"got {env_a.get('SEMIP_PROBE_VAR')!r}")
    assert env_b.get("SEMIP_PROBE_VAR") == "value-B", (
        f"B SEMIP_PROBE_VAR expected 'value-B', "
        f"got {env_b.get('SEMIP_PROBE_VAR')!r}")
    assert env_a.get("VLLM_LOGGING_LEVEL") == "DEBUG", env_a.get("VLLM_LOGGING_LEVEL")
    assert env_b.get("VLLM_LOGGING_LEVEL") == "WARNING", env_b.get("VLLM_LOGGING_LEVEL")

    assert env_a.get("CUDA_VISIBLE_DEVICES") == str(gpu_a), (
        f"A CUDA_VISIBLE_DEVICES expected '{gpu_a}', "
        f"got {env_a.get('CUDA_VISIBLE_DEVICES')!r} -- reserved-key drop failed")
    assert env_a.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", (
        f"A VLLM_ENABLE_V1_MULTIPROCESSING expected '0', "
        f"got {env_a.get('VLLM_ENABLE_V1_MULTIPROCESSING')!r}")
    assert env_a.get("USE_LIBUV") == "0", (
        f"A USE_LIBUV expected '0', got {env_a.get('USE_LIBUV')!r}")

    assert env_b.get("SEMIP_PROBE_VAR") == "value-B"
    assert "SEMIP_PROBE_VAR" in env_a and env_a["SEMIP_PROBE_VAR"] != env_b["SEMIP_PROBE_VAR"]

    image_dir = os.path.join(IMAGE_ROOT, "model_a")
    inst_a.attach().repin().stage().unpin().sleep().cuda_checkpoint()
    inst_a.criu_dump(image_dir).wait()

    with open(os.path.join(image_dir, "meta.json")) as f:
        meta = json.load(f)
    persisted_env = meta["vllm_config"].get("_env")
    assert persisted_env == cfg_a["_env"], (
        f"meta.json _env mismatch: persisted={persisted_env!r} "
        f"expected={cfg_a['_env']!r}")
    assert "CUDA_VISIBLE_DEVICES" in persisted_env, (
        "reserved key should remain on disk -- only dropped at apply "
        "time in the child")

    inst_b.teardown().wait()

    print("OK: per-model _env applied, reserved keys dropped, meta.json preserved")


if __name__ == "__main__":
    main()
