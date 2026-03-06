#!/usr/bin/env python3
"""Two-node weight sync benchmark with concurrent serving load.

Strategies: drain (pause+wait), skip (cancel in-flight), hotswap (zero downtime).

Uses the new NCCLEngine with bucket-packed, double-buffered NCCL broadcast.
Sender provides full (name, tensor) pairs — no TP assumption on the sender.

Usage (small model, single-node test):
    # Node B (receiver): CUDA_VISIBLE_DEVICES=0 python -m arctic_inference.server.cli --port 8100
    # Node A (sender):   python benchmark_weight_sync.py \
    #     --server-url http://<B>:8100 --sender-gpu 0 --master-addr <A_IP>

Usage (Qwen3-32B, two-node):
    # Node B (receiver): python -m arctic_inference.server.cli --port 8100
    # Node A (sender):   python benchmark_weight_sync.py --preset qwen3-32b \
    #     --server-url http://<B>:8100 --sender-gpu 0 --master-addr <A_IP>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import glob as _glob
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import requests
import torch


# ---------------------------------------------------------------------------
# Model presets — tuned defaults for known large models
# ---------------------------------------------------------------------------

MODEL_PRESETS: dict[str, dict] = {
    "qwen3-32b": dict(
        model="Qwen/Qwen3-32B",
        quantization="fp8",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        load_concurrency=8,
        load_max_tokens=32,
        n_iters=3,
        warmup_iters=1,
        load_warmup_secs=15.0,
        load_cooldown_secs=10.0,
        timeout=1200,
    ),
    "qwen3-32b-bf16-tp1": dict(
        model="Qwen/Qwen3-32B",
        quantization=None,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        load_concurrency=8,
        load_max_tokens=32,
        n_iters=3,
        warmup_iters=1,
        load_warmup_secs=15.0,
        load_cooldown_secs=10.0,
        timeout=1200,
    ),
    "qwen3-32b-bf16-tp2": dict(
        model="Qwen/Qwen3-32B",
        quantization=None,
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        load_concurrency=8,
        load_max_tokens=32,
        n_iters=3,
        warmup_iters=1,
        load_warmup_secs=15.0,
        load_cooldown_secs=10.0,
        timeout=1200,
    ),
    "qwen3-32b-fp8-tp2": dict(
        model="Qwen/Qwen3-32B",
        quantization="fp8",
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        load_concurrency=8,
        load_max_tokens=32,
        n_iters=3,
        warmup_iters=1,
        load_warmup_secs=15.0,
        load_cooldown_secs=10.0,
        timeout=1200,
    ),
    "opt-125m": dict(
        model="facebook/opt-125m",
        max_model_len=512,
        gpu_memory_utilization=0.3,
    ),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    server_url: str = "http://localhost:8100"
    model: str = "facebook/opt-125m"
    server_gpu: int | None = None
    sender_gpu: int = 1
    sender_gpus: list[int] | None = None
    training_sharding: str = "dp"
    master_addr: str = "127.0.0.1"
    nccl_port: int | None = None
    n_iters: int = 5
    warmup_iters: int = 1
    load_concurrency: int = 16
    load_prompt: str = "The quick brown fox jumps over the lazy dog. Once upon a time"
    load_max_tokens: int = 64
    strategies: list[str] = field(default_factory=lambda: ["drain", "skip", "hotswap"])
    load_warmup_secs: float = 10.0
    load_cooldown_secs: float = 5.0
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.3
    quantization: str | None = None
    tensor_parallel_size: int = 1
    timeout: int = 600

    @property
    def effective_sender_gpus(self) -> list[int]:
        """Resolved list of sender GPU indices."""
        if self.sender_gpus is not None:
            return self.sender_gpus
        return [self.sender_gpu]

    @property
    def use_direct_mode(self) -> bool:
        """True when BF16 TP=1 — enables per-weight zero-copy transfer."""
        return not self.quantization and self.tensor_parallel_size == 1


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/status", timeout=5).status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def load_model_weights(model_path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    sf_files = sorted(_glob.glob(os.path.join(model_path, "*.safetensors")))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")
    params: dict[str, torch.Tensor] = {}
    for idx, sf in enumerate(sf_files):
        shard_params = load_file(sf, device="cpu")
        params.update(shard_params)
        shard_bytes = sum(t.numel() * t.element_size() for t in shard_params.values())
        print(f"    shard {idx + 1}/{len(sf_files)}: "
              f"{len(shard_params)} tensors, {shard_bytes / 1e9:.2f} GB  "
              f"({os.path.basename(sf)})")
    return params


def _check_gpu_memory(device_idx: int, required_bytes: int) -> None:
    try:
        free, total = torch.cuda.mem_get_info(device_idx)
        required_gb = required_bytes / 1e9
        free_gb = free / 1e9
        total_gb = total / 1e9
        print(f"[gpu] Device {device_idx}: {free_gb:.1f} GB free / {total_gb:.1f} GB total, "
              f"need ~{required_gb:.1f} GB for sender weights")
        if required_bytes > free * 0.95:
            print(f"  WARNING: sender weights ({required_gb:.1f} GB) may not fit in "
                  f"available GPU memory ({free_gb:.1f} GB)")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Load generator — hammers /sample with concurrent requests
# ---------------------------------------------------------------------------

class LoadGenerator:
    """Generates continuous serving load via threaded HTTP requests."""

    def __init__(self, base_url: str, concurrency: int,
                 prompt: str, max_tokens: int) -> None:
        self.base_url = base_url
        self.concurrency = concurrency
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._stop = threading.Event()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._futures: list[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._latencies: list[float] = []

    def start(self) -> None:
        self._stop.clear()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency)
        self._futures = [
            self._executor.submit(self._worker_loop)
            for _ in range(self.concurrency)
        ]

    def stop(self) -> None:
        self._stop.set()
        for f in self._futures:
            try:
                f.result(timeout=120)
            except Exception:
                pass
        self._futures.clear()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def snapshot_and_reset(self) -> dict:
        with self._lock:
            snap = {
                "requests": self._request_count,
                "errors": self._error_count,
                "latencies": list(self._latencies),
            }
            self._request_count = 0
            self._error_count = 0
            self._latencies.clear()
            return snap

    def _worker_loop(self) -> None:
        session = requests.Session()
        while not self._stop.is_set():
            t0 = time.time()
            try:
                resp = session.post(
                    f"{self.base_url}/generate",
                    json={
                        "prompts": [self.prompt],
                        "sampling_params": {
                            "max_tokens": self.max_tokens,
                            "temperature": 0.7,
                        },
                    },
                    timeout=120,
                )
                elapsed = time.time() - t0
                with self._lock:
                    self._request_count += 1
                    if resp.status_code == 200:
                        self._latencies.append(elapsed)
                    elif resp.status_code == 503:
                        pass
                    else:
                        self._error_count += 1
            except Exception:
                with self._lock:
                    self._request_count += 1
                    self._error_count += 1
        session.close()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

class WeightSyncBenchmark:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.cfg = config
        self.server_proc: subprocess.Popen | None = None
        self._senders: dict[int, object] = {}  # group_id → WeightSender
        self._schedule = None
        self._nccl_base_port: int = 0

    @staticmethod
    def _weight_iter(weights_info, gpu_params):
        """Yield (name, tensor) pairs for all known weights on a GPU."""
        return (
            (wi.name, gpu_params[wi.name])
            for wi in weights_info
            if wi.name in gpu_params
        )

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg
        sender_gpus = cfg.effective_sender_gpus
        print(f"\n{'=' * 64}")
        print(f"  Weight Sync Benchmark")
        print(f"  Server:      {cfg.server_url}")
        print(f"  Model:       {cfg.model}")
        print(f"  Quant:       {cfg.quantization or 'none'}")
        print(f"  TP:          {cfg.tensor_parallel_size}")
        print(f"  Sender GPUs: {sender_gpus}")
        print(f"  Sharding:    {cfg.training_sharding}")
        print(f"  Master addr: {cfg.master_addr}")
        print(f"  Strategies:  {cfg.strategies}")
        print(f"  Direct mode: {cfg.use_direct_mode}")
        print(f"  Iterations:  {cfg.n_iters} (+{cfg.warmup_iters} warmup)")
        print(f"  Load conc.:  {cfg.load_concurrency} workers")
        print(f"{'=' * 64}\n")

        params = None
        try:
            self._maybe_start_server()
            self._wait_for_server()
            self._init_model()
            model_path, params, weights_info, total_bytes = self._load_weights()

            status = requests.get(f"{cfg.server_url}/status", timeout=10).json()
            if "models" in status:
                models = status["models"]
                num_replicas = next(iter(models.values()))["num_replicas"] if models else 1
            else:
                num_replicas = status.get("num_replicas", 1)
            from arctic_inference.server.weight_sync import TransferSchedule
            self._schedule = TransferSchedule.build(
                training_sharding=cfg.training_sharding,
                training_gpus=len(sender_gpus),
                inference_replicas=num_replicas,
                inference_tp=cfg.tensor_parallel_size,
            )
            print(f"[schedule] {self._schedule.num_groups} group(s), "
                  f"{num_replicas} replica(s), tp={cfg.tensor_parallel_size}")
            for g in self._schedule.groups:
                gpu = sender_gpus[g.sender_train_rank]
                tgt = ", ".join(f"R{r}T{t}" for r, t in g.targets)
                print(f"  Group {g.group_id}: GPU {gpu} → [{tgt}]")

            active_gpu_set = {
                sender_gpus[g.sender_train_rank]
                for g in self._schedule.groups
            }
            for gpu_idx in active_gpu_set:
                _check_gpu_memory(gpu_idx, total_bytes)

            gpu_params = self._move_to_gpus(params, sorted(active_gpu_set))

            max_weight_bytes = max(wi.nbytes for wi in weights_info)
            self._bucket_size = max(max_weight_bytes, 256 * 1024 * 1024)
            print(f"[engine] bucket_size = {self._bucket_size / 1e6:.1f} MB "
                  f"(max_weight = {max_weight_bytes / 1e6:.1f} MB)")

            baseline_text, baseline_tokens = self._capture_baseline()

            all_results: dict[str, dict] = {}
            for strategy in cfg.strategies:
                print(f"\n{'=' * 64}")
                print(f"  STRATEGY: {strategy.upper()}")
                print(f"{'=' * 64}")
                result = self._benchmark_strategy(
                    strategy, gpu_params, weights_info, total_bytes)
                all_results[strategy] = result

                ok = self._verify_weights(params, weights_info, strategy,
                                          baseline_text, baseline_tokens)
                result["verify_pass"] = ok

            self._print_summary(all_results, total_bytes)

        finally:
            if params is not None:
                del params
            gc.collect()
            self._cleanup()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _maybe_start_server(self) -> None:
        if self.cfg.server_gpu is None:
            print("[setup] Using external server (--server-gpu not set)")
            return
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.cfg.server_gpu)
        port = self.cfg.server_url.rsplit(":", 1)[-1]
        cmd = [sys.executable, "-m", "arctic_inference.server.cli",
               "--port", port]
        print(f"[setup] Starting server on GPU {self.cfg.server_gpu}...")
        self.server_proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def _wait_for_server(self) -> None:
        print("[setup] Waiting for server...")
        if not wait_for_server(self.cfg.server_url, self.cfg.timeout):
            if self.server_proc:
                out = self.server_proc.stdout.read().decode(errors="replace")[-3000:]
                print(f"Server output:\n{out}")
            raise RuntimeError("Server did not become ready in time")
        print("[setup] Server is up!")

    def _init_model(self) -> None:
        cfg = self.cfg
        for _retry in range(5):
            try:
                status = requests.get(
                    f"{cfg.server_url}/status", timeout=120).json()
                break
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError):
                print(f"  status check timeout, retrying ({_retry + 1}/5)...")
                time.sleep(10)
        else:
            raise RuntimeError("Server status endpoint unreachable")
        if status.get("model") or status.get("models"):
            print("[setup] Model already initialized")
            return
        config: dict = {
            "model": cfg.model,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "max_model_len": cfg.max_model_len,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
        }
        if cfg.quantization:
            config["quantization"] = cfg.quantization
        print(f"[setup] Initializing model: {cfg.model}")
        resp = requests.post(f"{cfg.server_url}/init",
                             json={"config": config}, timeout=cfg.timeout)
        if resp.status_code == 500 and "already" in resp.text.lower() or "state=ready" in resp.text:
            print("[setup] Model already initialized (race with status check)")
            return
        if resp.status_code != 200:
            print(f"  INIT FAILED ({resp.status_code}): {resp.text[:2000]}")
        resp.raise_for_status()
        print(f"  -> {resp.json()}")

    def _load_weights(self):
        cfg = self.cfg
        from huggingface_hub import snapshot_download
        from arctic_inference.server.weight_sync import WeightInfo

        print(f"[setup] Resolving model checkpoint: {cfg.model}")
        model_path = snapshot_download(cfg.model)

        wi_data = requests.get(
            f"{cfg.server_url}/weights_info", timeout=300).json()
        weights_info = [WeightInfo.from_dict(d) for d in wi_data["weights_info"]]
        print(f"  -> {wi_data['count']} parameters")

        print(f"[setup] Loading safetensors from: {model_path}")
        params = load_model_weights(model_path)
        total_bytes = sum(t.numel() * t.element_size() for t in params.values())
        print(f"  -> {len(params)} tensors total ({total_bytes / 1e9:.2f} GB)")
        return model_path, params, weights_info, total_bytes

    # ------------------------------------------------------------------
    # Weight correctness verification
    # ------------------------------------------------------------------

    _VERIFY_PROMPT = "The capital of France is"
    _VERIFY_SAMPLING = {"temperature": 0.0, "max_tokens": 32}

    def _sample_inference(self, retries: int = 5) -> tuple[str, list[int]]:
        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.cfg.server_url}/generate",
                    json={"prompts": [self._VERIFY_PROMPT],
                          "sampling_params": self._VERIFY_SAMPLING},
                    timeout=180)
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as e:
                if attempt < retries - 1:
                    print(f"  [inference] attempt {attempt+1}/{retries} "
                          f"failed ({type(e).__name__}), retrying in 5s...")
                    time.sleep(5)
                    continue
                raise
            if resp.status_code in (503, 500) and attempt < retries - 1:
                print(f"  [inference] attempt {attempt+1}/{retries} "
                      f"got {resp.status_code}, retrying in 5s...")
                time.sleep(5)
                continue
            resp.raise_for_status()
            result = resp.json()["results"][0]
            return result["text"], result["token_ids"]
        resp.raise_for_status()
        raise RuntimeError("unreachable")

    def _capture_baseline(self) -> tuple[str, list[int]]:
        print("[verify] Capturing baseline inference output...")
        text, token_ids = self._sample_inference()
        preview = text.strip().replace("\n", " ")[:120]
        print(f"  Prompt:   {self._VERIFY_PROMPT!r}")
        print(f"  Output:   {preview!r}")
        print(f"  Tokens:   {len(token_ids)}")
        return text, token_ids

    def _verify_weights(self, params: dict[str, torch.Tensor],
                        weights_info, strategy: str,
                        baseline_text: str,
                        baseline_token_ids: list[int]) -> bool:
        print(f"\n[verify] Checking weight correctness after {strategy}...")

        infer_ok = True
        try:
            text, token_ids = self._sample_inference()
            preview = text.strip().replace("\n", " ")[:120]
            if token_ids == baseline_token_ids:
                print(f"  [inference] PASS — output matches baseline "
                      f"({len(token_ids)} tokens)")
            else:
                infer_ok = False
                print(f"  [inference] FAIL — output differs from baseline")
                print(f"    baseline: {baseline_text.strip()[:100]!r}")
                print(f"    got:      {preview!r}")
            print(f"  Output: {preview!r}")
        except Exception as e:
            infer_ok = False
            print(f"  [inference] ERROR: {e}")

        print(f"  [verify] {'PASS' if infer_ok else 'FAIL'}")
        return infer_ok

    def _move_to_gpus(
        self,
        params: dict[str, torch.Tensor],
        gpu_indices: list[int],
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Copy model weights to each active sender GPU.

        Returns ``{gpu_idx: {name: tensor_on_gpu}}``.
        """
        total = len(params)
        total_bytes = sum(t.numel() * t.element_size() for t in params.values())

        all_gpu_params: dict[int, dict[str, torch.Tensor]] = {}

        for gpu_idx in gpu_indices:
            device = torch.device("cuda", gpu_idx)
            torch.cuda.set_device(gpu_idx)
            print(f"[setup] Moving {total_bytes / 1e9:.2f} GB "
                  f"({total} tensors) to sender GPU {gpu_idx}...")

            gpu_params: dict[str, torch.Tensor] = {}
            moved_bytes = 0
            last_report = time.time()
            for i, (k, v) in enumerate(params.items()):
                gpu_params[k] = v.to(device)
                moved_bytes += v.numel() * v.element_size()
                now = time.time()
                if now - last_report >= 5.0 or i == total - 1:
                    pct = moved_bytes / total_bytes * 100
                    print(f"  -> GPU {gpu_idx}: {i + 1}/{total} tensors, "
                          f"{moved_bytes / 1e9:.1f}/{total_bytes / 1e9:.1f} GB ({pct:.0f}%)")
                    last_report = now

            torch.cuda.synchronize(device)
            alloc_gb = torch.cuda.memory_allocated(device) / 1e9
            print(f"  -> GPU {gpu_idx}: {alloc_gb:.1f} GB allocated")
            all_gpu_params[gpu_idx] = gpu_params

        return all_gpu_params

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------

    def _build_groups_payload(self) -> list[dict]:
        """Build the ``groups`` JSON array for the /sync_weights endpoint.

        Emits one entry per unique replica (de-duplicated across sender
        groups).  The receiver only needs to know the base port for each
        replica; individual TP workers derive their ports via
        ``master_port + tp_rank``.
        """
        schedule = self._schedule
        tp = schedule.inference_tp
        base_port = self._nccl_base_port
        seen: set[int] = set()
        payload: list[dict] = []
        for g in schedule.groups:
            for r, _t in g.targets:
                if r not in seen:
                    seen.add(r)
                    payload.append({
                        "group_id": r,
                        "master_addr": self.cfg.master_addr,
                        "master_port": base_port + r * tp,
                        "world_size": g.world_size,
                        "replica_ids": [r],
                    })
        return payload

    def _create_senders(self, gpu_params, weights_info):
        """Create WeightSenders, connect, and do the first weight sync."""
        from arctic_inference.server.weight_sync import WeightSender

        cfg = self.cfg
        schedule = self._schedule
        sender_gpus = cfg.effective_sender_gpus

        base_port = cfg.nccl_port or find_free_port()
        self._nccl_base_port = base_port

        n_groups = schedule.num_groups
        total_conns = sum(len(g.targets) for g in schedule.groups)
        print(f"[engine] {n_groups} sender(s), {total_conns} NCCL conn(s) "
              f"(base_port={base_port}, bucket={self._bucket_size / 1e6:.0f}MB)")
        for g in schedule.groups:
            gpu = sender_gpus[g.sender_train_rank]
            sender = WeightSender(
                group=g, schedule=schedule,
                master_addr=cfg.master_addr, base_port=base_port,
                device=torch.device("cuda", gpu),
                bucket_size=self._bucket_size,
            )
            tgt = ", ".join(f"R{r}T{t}" for r, t in g.targets)
            print(f"  Group {g.group_id}: GPU {gpu} → [{tgt}]  ports {sender.ports}")
            self._senders[g.group_id] = sender

        senders = self._senders
        errors: dict[int, str] = {}

        def _connect_and_send(gid):
            s = senders[gid]
            gpu = sender_gpus[schedule.groups[gid].sender_train_rank]
            wp = gpu_params[gpu]
            try:
                s.send(
                    list(self._weight_iter(weights_info, wp)),
                    direct=cfg.use_direct_mode,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                errors[gid] = str(e)

        t0 = time.time()
        threads = [
            threading.Thread(target=_connect_and_send, args=(gid,))
            for gid in senders
        ]
        for t in threads:
            t.start()
        time.sleep(0.3)

        groups_payload = self._build_groups_payload()
        resp = requests.post(f"{cfg.server_url}/sync_weights", json={
            "groups": groups_payload,
            "bucket_size": self._bucket_size,
            "strategy": "hotswap",
            "direct_mode": cfg.use_direct_mode,
        }, timeout=cfg.timeout)

        for t in threads:
            t.join(cfg.timeout)
        setup_time = time.time() - t0

        if errors:
            raise RuntimeError(f"Sender setup failed: {errors}")
        resp.raise_for_status()
        print(f"  -> {n_groups} sender(s), {total_conns} conn(s), "
              f"first sync in {setup_time:.3f}s")
        return setup_time

    def _destroy_senders(self) -> None:
        for s in self._senders.values():
            try:
                s.destroy()
            except Exception:
                pass
        self._senders.clear()
        try:
            requests.post(
                f"{self.cfg.server_url}/close_weight_sync", timeout=30)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Single weight sync iteration
    # ------------------------------------------------------------------

    def _do_weight_sync(self, gpu_params, weights_info, strategy):
        """Run one weight sync across all groups.

        Returns ``(total_time, max_sender_elapsed, receiver_response)``.
        """
        cfg = self.cfg
        schedule = self._schedule
        sender_gpus = cfg.effective_sender_gpus
        result: dict = {}
        error: dict = {}

        def _send_group(group):
            gid = group.group_id
            gpu = sender_gpus[group.sender_train_rank]
            wp = gpu_params[gpu]
            try:
                r = self._senders[gid].send(
                    self._weight_iter(weights_info, wp),
                    direct=cfg.use_direct_mode,
                )
                result[f"sender_{gid}"] = {
                    "elapsed": r["max_target_elapsed"], "status": "done",
                }
            except Exception as e:
                error[f"sender_{gid}"] = str(e)

        def _recv():
            try:
                groups_payload = self._build_groups_payload()
                resp = requests.post(
                    f"{cfg.server_url}/sync_weights",
                    json={
                        "groups": groups_payload,
                        "bucket_size": self._bucket_size,
                        "strategy": strategy,
                        "direct_mode": cfg.use_direct_mode,
                    },
                    timeout=cfg.timeout)
                resp.raise_for_status()
                result["receiver"] = resp.json()
            except Exception as e:
                error["receiver"] = str(e)

        t0 = time.time()
        send_threads = [
            threading.Thread(target=_send_group, args=(g,))
            for g in schedule.groups
        ]
        recv_thread = threading.Thread(target=_recv)
        for t in send_threads:
            t.start()
        recv_thread.start()
        for t in send_threads:
            t.join(cfg.timeout)
        recv_thread.join(cfg.timeout)
        total_time = time.time() - t0

        if error:
            raise RuntimeError(f"Weight sync failed: {error}")

        sender_times = [
            result.get(f"sender_{g.group_id}", {}).get("elapsed", -1)
            for g in schedule.groups
        ]
        max_sender_elapsed = max(sender_times) if sender_times else -1
        return total_time, max_sender_elapsed, result.get("receiver", {})

    # ------------------------------------------------------------------
    # Per-strategy benchmark
    # ------------------------------------------------------------------

    def _benchmark_strategy(self, strategy, gpu_params, weights_info, total_bytes):
        cfg = self.cfg

        if not self._senders:
            setup_time = self._create_senders(gpu_params, weights_info)
        else:
            setup_time = 0.0

        try:
            print(f"\n[load] Starting load generator ({cfg.load_concurrency} workers)...")
            load_gen = LoadGenerator(
                cfg.server_url, cfg.load_concurrency,
                cfg.load_prompt, cfg.load_max_tokens)
            load_gen.start()

            print(f"[load] Warming up serving for {cfg.load_warmup_secs}s...")
            time.sleep(cfg.load_warmup_secs)
            warmup_snap = load_gen.snapshot_and_reset()
            w_reqs = warmup_snap["requests"]
            w_errs = warmup_snap["errors"]
            w_rps = w_reqs / cfg.load_warmup_secs if cfg.load_warmup_secs > 0 else 0
            w_lats = warmup_snap["latencies"]
            w_p50 = statistics.median(w_lats) if w_lats else 0
            print(f"  -> Warmup: {w_rps:.1f} req/s, p50={w_p50:.3f}s, "
                  f"{w_errs}/{w_reqs} errors")

            if cfg.warmup_iters > 0:
                print(f"\n[sync] Warmup syncs ({cfg.warmup_iters})...")
                for i in range(cfg.warmup_iters):
                    _ = load_gen.snapshot_and_reset()
                    tt, st_t, _ = self._do_weight_sync(
                        gpu_params, weights_info, strategy)
                    print(f"  warmup {i}: {tt:.3f}s")

            print(f"\n[sync] Benchmarking {strategy.upper()} "
                  f"({cfg.n_iters} iterations)...")
            sync_times: list[float] = []
            sender_times: list[float] = []
            load_during: list[dict] = []
            mem_extras: list[int] = []

            for i in range(cfg.n_iters):
                _ = load_gen.snapshot_and_reset()

                total_t, sender_t, recv_resp = self._do_weight_sync(
                    gpu_params, weights_info, strategy)

                sync_snap = load_gen.snapshot_and_reset()
                sync_times.append(total_t)
                sender_times.append(sender_t)
                load_during.append(sync_snap)

                recv_worker_t = "?"
                update_path = "?"
                mem_extra_mb = 0.0
                workers = recv_resp.get("workers", [{}])
                if workers and isinstance(workers[0], dict):
                    w0 = workers[0]
                    if isinstance(w0.get("elapsed"), (int, float)):
                        recv_worker_t = f"{w0['elapsed']:.3f}s"
                    if w0.get("update_path"):
                        update_path = w0["update_path"]
                    if isinstance(w0.get("mem_extra_bytes"), (int, float)):
                        mem_extra_mb = w0["mem_extra_bytes"] / 1e6
                        mem_extras.append(w0["mem_extra_bytes"])

                s_reqs = sync_snap["requests"]
                s_errs = sync_snap["errors"]
                s_rps = s_reqs / total_t if total_t > 0 else 0
                bw_iter = total_bytes / total_t / 1e9 if total_t > 0 else 0
                print(f"  iter {i}: total={total_t:.3f}s  sender={sender_t:.3f}s  "
                      f"worker={recv_worker_t}  bw={bw_iter:.2f}GB/s  "
                      f"path={update_path}  mem_extra={mem_extra_mb:.1f}MB  "
                      f"load={s_rps:.1f}req/s  errors={s_errs}/{s_reqs}")

            print(f"\n[load] Cooldown ({cfg.load_cooldown_secs}s)...")
            _ = load_gen.snapshot_and_reset()
            time.sleep(cfg.load_cooldown_secs)
            cool_snap = load_gen.snapshot_and_reset()
            cool_rps = cool_snap["requests"] / cfg.load_cooldown_secs if cfg.load_cooldown_secs > 0 else 0
            cool_lats = cool_snap["latencies"]
            cool_p50 = statistics.median(cool_lats) if cool_lats else 0
            print(f"  -> Post-sync: {cool_rps:.1f} req/s, p50={cool_p50:.3f}s, "
                  f"{cool_snap['errors']}/{cool_snap['requests']} errors")

            load_gen.stop()

            avg_sync = statistics.mean(sync_times)
            std_sync = statistics.stdev(sync_times) if len(sync_times) > 1 else 0
            avg_sender = statistics.mean(sender_times)
            bw = total_bytes / avg_sync / 1e9 if avg_sync > 0 else 0
            tot_reqs = sum(s["requests"] for s in load_during)
            tot_errs = sum(s["errors"] for s in load_during)
            tot_t = sum(sync_times)
            rps = tot_reqs / tot_t if tot_t else 0
            all_lats = [l for s in load_during for l in s["latencies"]]
            p50 = statistics.median(all_lats) if all_lats else 0
            p99 = sorted(all_lats)[int(len(all_lats) * 0.99)] if len(all_lats) > 1 else p50

            avg_mem_extra = statistics.mean(mem_extras) if mem_extras else 0
            max_mem_extra = max(mem_extras) if mem_extras else 0

            result = {
                "strategy": strategy, "setup_time": setup_time,
                "sync_times": sync_times,
                "avg_sync": avg_sync, "std_sync": std_sync,
                "min_sync": min(sync_times), "max_sync": max(sync_times),
                "avg_sender": avg_sender, "bandwidth_gbps": bw,
                "load_rps_during_sync": rps, "load_errors_during_sync": tot_errs,
                "load_total_during_sync": tot_reqs,
                "load_p50_during": p50, "load_p99_during": p99,
                "warmup_rps": w_rps, "cooldown_rps": cool_rps,
                "avg_mem_extra_mb": avg_mem_extra / 1e6,
                "max_mem_extra_mb": max_mem_extra / 1e6,
            }

            print(f"\n  --- {strategy.upper()} Results ---")
            print(f"  Sync time:  {avg_sync:.3f}s +/- {std_sync:.3f}s  "
                  f"(min={min(sync_times):.3f}, max={max(sync_times):.3f})")
            print(f"  Sender:     {avg_sender:.3f}s")
            print(f"  Bandwidth:  {bw:.2f} GB/s")
            print(f"  Memory:     avg_extra={avg_mem_extra / 1e6:.1f}MB  "
                  f"max_extra={max_mem_extra / 1e6:.1f}MB")
            print(f"  Load:       {rps:.1f} req/s during sync  (p50={p50:.3f}s, p99={p99:.3f}s)")
            print(f"  Errors:     {tot_errs}/{tot_reqs}")
            return result

        except Exception:
            self._destroy_senders()
            raise

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self, all_results: dict[str, dict], total_bytes: int) -> None:
        cfg = self.cfg
        gb = total_bytes / 1e9
        print(f"\n{'=' * 80}")
        n_groups = self._schedule.num_groups if self._schedule else 1
        print(f"  SUMMARY  (model={cfg.model}, {gb:.1f} GB, "
              f"quant={cfg.quantization or 'none'}, tp={cfg.tensor_parallel_size}, "
              f"{n_groups} sender group(s), "
              f"{cfg.n_iters} iters, {cfg.load_concurrency} load workers)")
        print(f"{'=' * 80}")
        header = (f"  {'Strategy':<10} {'Sync (avg)':>11} {'Sync (std)':>11} "
                  f"{'BW GB/s':>9} {'MemExtra':>10} {'Load RPS':>10} "
                  f"{'p50':>8} {'p99':>8} {'Errors':>8} {'Verify':>8}")
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        for strategy, r in all_results.items():
            mem_str = f"{r.get('max_mem_extra_mb', 0):.0f}MB"
            verify_str = "PASS" if r.get("verify_pass") else "FAIL"
            print(
                f"  {strategy:<10} {r['avg_sync']:>10.3f}s {r['std_sync']:>10.3f}s "
                f"{r['bandwidth_gbps']:>9.2f} {mem_str:>10} "
                f"{r['load_rps_during_sync']:>10.1f} "
                f"{r['load_p50_during']:>7.3f}s {r['load_p99_during']:>7.3f}s "
                f"{r['load_errors_during_sync']:>8d} {verify_str:>8}"
            )

        print(f"\n  Baseline serving (warmup):  "
              f"{list(all_results.values())[0].get('warmup_rps', 0):.1f} req/s")
        print(f"  Post-sync serving (cooldown): "
              f"{list(all_results.values())[-1].get('cooldown_rps', 0):.1f} req/s")
        print(f"{'=' * 80}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        self._destroy_senders()
        if self.cfg.server_gpu is not None:
            try:
                requests.post(f"{self.cfg.server_url}/shutdown", timeout=10)
            except Exception:
                pass
        if self.server_proc:
            print("\n[cleanup] Shutting down server...")
            self.server_proc.send_signal(signal.SIGINT)
            try:
                self.server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark weight sync under serving load",
                                epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(MODEL_PRESETS.keys()), default=None,
                   help="Apply tuned defaults for a known model "
                        f"({', '.join(MODEL_PRESETS.keys())}). "
                        "Explicit flags override preset values.")
    p.add_argument("--server-url", default=None)
    p.add_argument("--server-gpu", type=int, default=None)
    p.add_argument("--sender-gpu", type=int, default=None,
                   help="Single sender GPU (legacy). Overridden by --sender-gpus.")
    p.add_argument("--sender-gpus", type=int, nargs="+", default=None,
                   help="List of sender GPU indices for multi-sender parallel transfer.")
    p.add_argument("--training-sharding", default=None, choices=["dp", "fsdp"],
                   help="Training sharding mode (default: dp).")
    p.add_argument("--master-addr", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--quantization", default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--n-iters", type=int, default=None)
    p.add_argument("--warmup-iters", type=int, default=None)
    p.add_argument("--load-concurrency", type=int, default=None)
    p.add_argument("--load-max-tokens", type=int, default=None)
    p.add_argument("--load-warmup-secs", type=float, default=None)
    p.add_argument("--load-cooldown-secs", type=float, default=None)
    p.add_argument("--strategies", nargs="+", default=None,
                   choices=["drain", "skip", "hotswap"])
    p.add_argument("--nccl-port", type=int, default=None)
    p.add_argument("--timeout", type=int, default=None)

    args = p.parse_args()

    defaults = {}
    if args.preset:
        defaults.update(MODEL_PRESETS[args.preset])
        print(f"[config] Using preset: {args.preset}")

    cli_overrides = {
        k: v for k, v in vars(args).items()
        if k != "preset" and v is not None
    }
    merged = {**defaults, **cli_overrides}

    config = BenchmarkConfig(**{
        f.name: merged[f.name]
        for f in BenchmarkConfig.__dataclass_fields__.values()
        if f.name in merged
    })

    WeightSyncBenchmark(config).run()


if __name__ == "__main__":
    main()
