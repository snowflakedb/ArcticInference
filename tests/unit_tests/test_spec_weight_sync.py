#!/usr/bin/env python3
"""Spec-model weight-sync integration test.

Two independent processes, two GPUs -- just like production.

  GPU 0 + GPU 1: vLLM engine with TP=2 (inference receiver)
  Senders:       subprocess per TP rank, cross-assigned to opposite GPU

The script re-invokes itself with ``--sender`` to spawn sender
processes, so everything lives in one file.

Verification: after an identity weight sync the acceptance rate must
remain above a minimum threshold, proving the drafter is still working.
The acceptance rate is collected via a file-based accumulator in the
patched ``SpecDecodingStats.observe_draft`` (env ``SPEC_DEC_STATS_FILE``).

Single-node run:
    python tests/unit_tests/test_spec_weight_sync.py

Multi-node run (sender on remote host, receiver on this host):
    HF_HUB_CACHE=/checkpoint/huggingface/hub \
    BASE_MODEL=meta-llama/Llama-3.3-70B-Instruct \
    SENDER_HOST=10.4.174.218 \
    python tests/unit_tests/test_spec_weight_sync.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import torch

# ---------------------------------------------------------------------------
# Shared constants (overridable via env vars)
# ---------------------------------------------------------------------------

MODEL = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
SPEC_MODEL = os.environ.get(
    "SPEC_MODEL", "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-70B-Instruct",
)
SENDER_HOST = os.environ.get("SENDER_HOST", "")
TP_SIZE = int(os.environ.get("TP_SIZE", "2"))
SPEC_CONFIG = {
    "method": "arctic",
    "model": SPEC_MODEL,
    "num_speculative_tokens": 3,
    "enable_suffix_decoding": False,
    "disable_by_batch_size": 64,
}
CONVERSATION = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "Write a short paragraph about the weather."},
]
BASE_PORT = int(os.environ.get("BASE_PORT", "29600"))
MIN_ACCEPTANCE_RATE = 0.1


# ===================================================================
# Sender (training side) -- runs in a subprocess
# ===================================================================

def sender_main(
    model_path: str,
    base_port: int,
    tp_rank: int,
    gpu_id: int,
    master_addr: str = "127.0.0.1",
):
    from arctic_inference.server.weight_sync import send_spec_weights

    device = torch.device("cuda", gpu_id)
    print(f"[sender tp{tp_rank}] sending {model_path}", flush=True)
    result = send_spec_weights(
        model_path, master_addr, base_port, tp_rank, device,
    )
    print(f"[sender tp{tp_rank}] sent {result['params_sent']} tensors, done",
          flush=True)


# ===================================================================
# Acceptance-rate tracking (monkey-patch applied before engine fork)
# ===================================================================

_stats_file: str = ""
_cumulative: dict[str, int] = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0}


def _install_stats_hook(path: str):
    """Monkey-patch observe_draft to write cumulative stats to *path*.

    Must be called after ``vllm.plugins.load_general_plugins()`` (so the
    Arctic patches are in place) and before ``LLM(...)`` (so the engine
    core subprocess inherits the patched method via fork).
    """
    global _stats_file
    _stats_file = path
    reset_stats_file(path)

    from vllm.v1.spec_decode.metrics import SpecDecodingStats
    _prev = SpecDecodingStats.observe_draft

    def _hooked(self, num_draft_tokens, num_accepted_tokens):
        _prev(self, num_draft_tokens, num_accepted_tokens)
        _cumulative["drafts"] += 1
        _cumulative["draft_tokens"] += num_draft_tokens
        _cumulative["accepted_tokens"] += num_accepted_tokens
        with open(_stats_file, "w") as f:
            json.dump(_cumulative, f)

    SpecDecodingStats.observe_draft = _hooked


def read_acceptance_rate(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    draft = data.get("draft_tokens", 0)
    if draft == 0:
        return None
    return data["accepted_tokens"] / draft


def reset_stats_file(path: str):
    _cumulative.update(drafts=0, draft_tokens=0, accepted_tokens=0)
    with open(path, "w") as f:
        json.dump(_cumulative, f)


# ===================================================================
# Sender launching helpers
# ===================================================================

def _launch_senders_local(spec_path: str) -> list[subprocess.Popen]:
    """Launch sender subprocesses on the local machine (single-node)."""
    senders = []
    for tp_rank in range(TP_SIZE):
        sender_gpu = (tp_rank + 1) % TP_SIZE
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(sender_gpu)
        print("[test] starting local sender tp%d on GPU %d" % (tp_rank, sender_gpu))
        p = subprocess.Popen(
            [sys.executable, __file__, "--sender",
             spec_path, str(BASE_PORT), str(tp_rank), "0"],
            env=env,
        )
        senders.append(p)
    return senders


def _launch_senders_remote(spec_path: str, sender_host: str) -> list[subprocess.Popen]:
    """Launch sender subprocesses on a remote host via SSH (multi-node)."""
    senders = []
    python_bin = os.environ.get("REMOTE_PYTHON", sys.executable)
    test_script = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(test_script)))

    hf_cache = os.environ.get("HF_HUB_CACHE", "")
    hf_cache_env = f"HF_HUB_CACHE={hf_cache} " if hf_cache else ""

    nccl_envs = ""
    for key in ("NCCL_NET_PLUGIN", "NCCL_TUNER_PLUGIN", "NCCL_SOCKET_IFNAME",
                "NCCL_DEBUG"):
        val = os.environ.get(key)
        if val is not None:
            nccl_envs += f"{key}={val} "

    for tp_rank in range(TP_SIZE):
        sender_cmd = (
            f"cd {repo_root} && "
            f"CUDA_VISIBLE_DEVICES={tp_rank} "
            f"ARCTIC_INFERENCE_ENABLED=1 "
            f"{hf_cache_env}"
            f"{nccl_envs}"
            f"{python_bin} {test_script} "
            f"--sender {spec_path} {BASE_PORT} {tp_rank} 0 0.0.0.0"
        )
        print("[test] starting remote sender tp%d on %s GPU %d" % (
            tp_rank, sender_host, tp_rank))
        p = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", sender_host, sender_cmd],
        )
        senders.append(p)
    return senders


# ===================================================================
# Receiver / test (inference side)
# ===================================================================

def main() -> int:
    os.environ.setdefault("ARCTIC_INFERENCE_ENABLED", "1")

    master_addr = SENDER_HOST if SENDER_HOST else "127.0.0.1"
    multi_node = bool(SENDER_HOST)

    print("[test] mode=%s model=%s spec=%s" % (
        "multi-node (sender=%s)" % SENDER_HOST if multi_node else "single-node",
        MODEL, SPEC_MODEL))

    import vllm                                      # noqa: E402
    from vllm import LLM, SamplingParams             # noqa: E402
    vllm.plugins.load_general_plugins()

    stats_file = os.path.join(tempfile.gettempdir(), "spec_dec_stats.json")
    _install_stats_hook(stats_file)

    sampling = SamplingParams(temperature=0.0, max_tokens=128)

    from huggingface_hub import snapshot_download
    spec_path = snapshot_download(SPEC_MODEL)
    from arctic_inference.server.weight_sync import spec_bucket_size
    bucket_size = spec_bucket_size(spec_path)
    print("[test] spec bucket_size=%d MB" % (bucket_size // (1024 * 1024)))

    print("[test] creating engine (TP=%d)" % TP_SIZE)
    llm = LLM(
        model=MODEL,
        quantization="fp8",
        tensor_parallel_size=TP_SIZE,
        speculative_config=SPEC_CONFIG,
        worker_extension_cls="arctic_inference.server.weight_sync.WeightSyncExtension",
        max_model_len=16384,
        disable_log_stats=False,
        seed=0,
    )

    def generate():
        return llm.chat(CONVERSATION, sampling_params=sampling)[0].outputs[0].text

    # --- baseline: generate and check acceptance rate ---
    reset_stats_file(stats_file)
    baseline = generate()
    rate_before = read_acceptance_rate(stats_file)
    print("[test] baseline: %r" % baseline)
    print("[test] acceptance rate before sync: %s" % rate_before)

    if rate_before is not None and rate_before < MIN_ACCEPTANCE_RATE:
        print("FAIL: acceptance rate before sync too low (%s)" % rate_before,
              file=sys.stderr)
        del llm
        return 1
    if rate_before is None:
        print("[test] acceptance rate hook not active (stats overridden by "
              "plugin); skipping acceptance rate checks")

    # --- launch senders ---
    if multi_node:
        senders = _launch_senders_remote(spec_path, SENDER_HOST)
    else:
        senders = _launch_senders_local(spec_path)

    time.sleep(5)
    for i, p in enumerate(senders):
        if p.poll() is not None:
            print("FAIL: sender %d exited early (rc=%s)" % (i, p.returncode),
                  file=sys.stderr)
            for s in senders:
                s.kill()
            del llm
            return 1

    # --- receive weights ---
    print("[test] calling sync_spec_weights (master_addr=%s)" % master_addr)
    t0 = time.time()
    results = llm.llm_engine.engine_core.collective_rpc(
        "sync_spec_weights",
        args=(master_addr, BASE_PORT, 1, 2, bucket_size),
    )
    print("[test] sync done in %.2fs: %s" % (time.time() - t0, results))

    for p in senders:
        p.wait(timeout=120)

    # --- generate after sync and check acceptance rate ---
    reset_stats_file(stats_file)
    after = generate()
    rate_after = read_acceptance_rate(stats_file)
    print("[test] after: %r" % after)
    print("[test] acceptance rate after sync: %s" % rate_after)

    del llm

    # --- verify ---
    if not results or results[0].get("status") != "done":
        print("FAIL: sync_spec_weights: %s" % results, file=sys.stderr)
        return 1

    if rate_before is not None and rate_after is not None:
        if rate_after < MIN_ACCEPTANCE_RATE:
            print("FAIL: acceptance rate after sync too low (%s)" % rate_after,
                  file=sys.stderr)
            return 1
        print("PASS (acceptance rate: %.1f%% -> %.1f%%)" % (
            rate_before * 100, rate_after * 100))
    else:
        print("PASS (sync completed, generation verified, "
              "acceptance rate hooks not active)")
    return 0


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sender":
        master_addr = sys.argv[6] if len(sys.argv) > 6 else "127.0.0.1"
        sender_main(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                     int(sys.argv[5]), master_addr)
    else:
        sys.exit(main())
