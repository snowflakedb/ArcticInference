"""WeightSyncExtension — vLLM worker extension for receiving weights.

Registered via ``--worker-extension-cls arctic_inference.server.weight_sync.WeightSyncExtension``

Provides two entry points:
  1. ``sync_weights()``      — for the main (base) model.
  2. ``sync_spec_weights()`` — for the spec (drafter) model.

Each lazily creates / reuses a persistent ``NCCLEngine`` and receives
weight tensors via NCCL.  Loading strategies for the base model:
  - **direct_zero_copy** (BF16, TP=1): receive straight into parameter views
  - **direct** (BF16, TP=1, fallback): bucket receive + copy into param views
  - **fp8**: bucket receive + FP8 in-place quantization
  - **batched**: bucket receive + model.load_weights for TP slicing

Spec (drafter) weight sync defaults to the **hotswap** strategy (continue
serving while syncing) because the drafter model is small enough that sync
completes quickly, and updating spec weights never affects the base model's
output correctness — at worst, speculative acceptance rate degrades
transiently.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


class WeightSyncExtension:
    """vLLM worker extension for NCCL weight sync.

    Usage:
        --worker-extension-cls arctic_inference.server.weight_sync.WeightSyncExtension
    """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_or_create_engine(
        self,
        master_addr: str,
        master_port: int,
        bucket_size: int,
        reverse: bool,
        *,
        engine_attr: str,
        key_attr: str,
        label: str = "",
    ):
        """Return a (possibly cached) NCCLEngine for the given config.

        *engine_attr* / *key_attr* are the instance attribute names used
        to cache the engine and its key, allowing separate engines for
        base and spec models.
        """
        from vllm.distributed.parallel_state import get_world_group
        from arctic_inference.server.weight_sync.engine import NCCLEngine

        tp_rank = get_world_group().rank
        my_port = master_port + tp_rank
        engine_key = (master_addr, my_port)

        engine = getattr(self, engine_attr, None)
        if engine is None or getattr(self, key_attr, None) != engine_key:
            if engine is not None:
                engine.destroy()
            logger.info(
                "Creating %sNCCLEngine rank=1 ws=2 port=%d tp_rank=%d bucket=%dMB",
                label, my_port, tp_rank, bucket_size // (1024 * 1024),
            )
            engine = NCCLEngine(
                master_addr=master_addr,
                master_port=my_port,
                rank=1,
                world_size=2,
                device=self.device,
                bucket_size=bucket_size,
                reverse=reverse,
            )
            setattr(self, engine_attr, engine)
            setattr(self, key_attr, engine_key)

        return engine

    def _build_result(self, start: float, mem_before: int, path: str, loaded: int) -> dict:
        """Synchronize CUDA and return timing / memory metrics."""
        torch.cuda.synchronize(self.device)
        peak_mem = torch.cuda.max_memory_allocated(self.device)
        mem_after = torch.cuda.memory_allocated(self.device)
        elapsed = time.time() - start

        return {
            "status": "done",
            "params_loaded": loaded,
            "elapsed": elapsed,
            "update_path": path,
            "mem_before_bytes": mem_before,
            "mem_peak_bytes": peak_mem,
            "mem_after_bytes": mem_after,
            "mem_extra_bytes": peak_mem - mem_before,
        }

    # ------------------------------------------------------------------
    # Param-name validation (catches sender/receiver model mismatch)
    # ------------------------------------------------------------------

    def _validate_weight_sync_names(
        self,
        model,
        sender_names: Iterable[str],
        *,
        context: str = "",
    ) -> None:
        """Raise if the sender's param names do not match the model's expected
        HF-style param name set.

        Catches the common failure mode where the training side ships a
        different architecture than the inference side (e.g. Qwen3 vs.
        Qwen2.5), or where weights are silently dropped because of a name
        mismatch.  Names that are legitimately unsynced (rotary inv_freq
        buffers, FP8/GPTQ quantization metadata) are filtered on both
        sides before comparison.

        Set ``ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0`` to demote the mismatch
        from an error to a warning (default: strict).
        """
        from arctic_inference.server.weight_sync.utils import (
            _name_is_non_synced,
            compute_expected_hf_param_names,
        )

        sender_set = {n for n in sender_names if not _name_is_non_synced(n)}

        try:
            expected_set = compute_expected_hf_param_names(model)
        except Exception as exc:
            logger.warning(
                "Weight-sync name check skipped: failed to derive expected "
                "param names for the receiver model (%s).", exc,
            )
            return

        unexpected = sender_set - expected_set
        missing = expected_set - sender_set

        if not unexpected and not missing:
            print(
                f"[weight-sync names validated] context={context or '?'} "
                f"sender={len(sender_set)} expected={len(expected_set)}",
                flush=True,
            )
            return

        ctx = f" [{context}]" if context else ""
        msg_parts = [f"Weight-sync param name mismatch{ctx}:"]
        if unexpected:
            sample = sorted(unexpected)[:10]
            tail = ("" if len(unexpected) <= len(sample)
                    else f" ... (+{len(unexpected) - len(sample)} more)")
            msg_parts.append(
                f"  Sender shipped {len(unexpected)} names the model does "
                f"NOT expect: {', '.join(sample)}{tail}"
            )
        if missing:
            sample = sorted(missing)[:10]
            tail = ("" if len(missing) <= len(sample)
                    else f" ... (+{len(missing) - len(sample)} more)")
            msg_parts.append(
                f"  Model expects {len(missing)} names the sender did NOT "
                f"ship: {', '.join(sample)}{tail}"
            )
        msg_parts.append(
            f"  (sender_count={len(sender_set)}, "
            f"expected_count={len(expected_set)})"
        )
        msg = "\n".join(msg_parts)

        if os.environ.get("ARCTIC_WEIGHT_SYNC_STRICT_NAMES", "1") == "0":
            logger.warning("%s\n(non-strict mode: continuing despite mismatch)",
                           msg)
            return

        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sync_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        direct_mode: bool = False,
        reverse: bool = False,
    ) -> dict:
        """Receive weights via NCCL and load them into the model.

        On the first call, a persistent ``NCCLEngine`` is created (NCCL
        rendezvous blocks until the sender also creates its engine).
        Subsequent calls with the same config reuse the existing engine.

        If *engine_only* is True, only the NCCL rendezvous is performed.
        If *direct_mode* is True, per-weight send/recv is used (BF16 TP=1).
        """
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        engine = self._get_or_create_engine(
            master_addr, master_port, bucket_size, reverse,
            engine_attr="_ws_engine", key_attr="_ws_engine_key",
        )

        if engine_only:
            return {"status": "engine_ready", "rank": 1, "world_size": 2}

        start = time.time()
        model = self.model_runner.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        tp = get_tensor_model_parallel_world_size()

        if self.model_config.quantization:
            path = "fp8"
            loaded = self._load_fp8(model, engine)
        elif tp == 1 and direct_mode:
            path = "direct_zero_copy"
            loaded = self._load_direct_zero_copy(model, engine)
        elif tp == 1:
            path = "direct"
            loaded = self._load_direct(model, engine)
        else:
            path = "batched"
            loaded = self._load_batched(model, engine)

        return self._build_result(start, mem_before, path, loaded)

    # ------------------------------------------------------------------
    # Loading strategies
    # ------------------------------------------------------------------

    def _load_fp8(self, model, engine) -> int:
        from arctic_inference.server.weight_sync.utils import _FP8InplaceUpdater

        updater = _FP8InplaceUpdater(
            model, self.model_config.dtype, self.device,
        )
        loaded = 0
        for name, tensor in engine.receive_weights():
            updater.feed(name, tensor)
            loaded += 1
        if updater.pending:
            logger.warning(
                "FP8 updater has %d incomplete modules", updater.pending,
            )
        return loaded

    def _load_direct_zero_copy(self, model, engine) -> int:
        """True zero-copy: receive each weight directly into its parameter view.

        Uses ``engine.receive_weights_direct()`` so that NCCL writes bytes
        straight into the model's parameter storage — no intermediate buffers
        or copies.  Requires BF16, TP=1.
        """
        from arctic_inference.server.weight_sync.utils import _DirectParamWriter

        writer = _DirectParamWriter(model, self.device)
        param_views: dict[str, torch.Tensor] = {}
        for name in writer.all_keys():
            v = writer.get_view(name)
            if v is not None:
                param_views[name] = v.contiguous()

        result = engine.receive_weights_direct(param_views)
        return result.get("params_loaded", 0)

    def _load_direct(self, model, engine) -> int:
        """Bucket path for TP=1 non-quantized models.

        Copies from the engine's bucket buffer into pre-computed parameter views.
        """
        from arctic_inference.server.weight_sync.utils import _DirectParamWriter

        writer = _DirectParamWriter(model, self.device)
        loaded = 0
        for name, tensor in engine.receive_weights():
            view = writer.get_view(name)
            if view is not None:
                view.copy_(tensor)
            else:
                logger.debug("Orphan weight %s — discarded", name)
            loaded += 1
        return loaded

    def _load_batched(self, model, engine) -> int:
        """Batched path for TP>1 models (including FP8+TP>1).

        Each received tensor is a full (un-sharded) weight.
        ``model.load_weights`` handles TP slicing and quantization.
        """
        loaded = 0
        for name, tensor in engine.receive_weights():
            model.load_weights([(name, tensor)])
            loaded += 1
        return loaded

    # ------------------------------------------------------------------
    # Spec (drafter) model weight sync
    # ------------------------------------------------------------------

    def sync_spec_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        reverse: bool = False,
    ) -> dict:
        """Receive weights via NCCL and load them into the drafter model.

        Mirrors :meth:`sync_weights` but targets the spec (drafter) model.
        Uses a separate NCCL engine instance so it can operate on a
        different port/connection.

        Spec weight sync defaults to the **hotswap** strategy: weights are
        received and loaded while inference continues uninterrupted.  This
        is safe because the drafter model is small (sync completes quickly)
        and updating it never affects the base model's output correctness —
        at worst, speculative acceptance rate degrades transiently.
        """
        drafter = getattr(self.model_runner, "drafter", None)
        if drafter is None or getattr(drafter, "model", None) is None:
            raise RuntimeError(
                "sync_spec_weights called but no drafter model is loaded"
            )

        engine = self._get_or_create_engine(
            master_addr, master_port, bucket_size, reverse,
            engine_attr="_ws_spec_engine", key_attr="_ws_spec_engine_key",
            label="spec ",
        )

        if engine_only:
            return {"status": "engine_ready", "rank": 1, "world_size": 2}

        start = time.time()
        spec_model = drafter.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        all_weights = [(n, t.cpu()) for n, t in engine.receive_weights()]
        loaded = len(all_weights)
        spec_model.load_weights(all_weights)

        return self._build_result(start, mem_before, "batched", loaded)

    # ------------------------------------------------------------------
    # Shared-memory path (colocated mode — no NCCL)
    # ------------------------------------------------------------------

    def sync_weights_ipc(
        self,
        group_id: int,
        timeout: float = 300,
    ) -> dict:
        """Receive weights via shared memory (colocated mode).

        Used when training and inference share a physical GPU, making
        NCCL communicator creation impossible.  The training worker writes
        weights to ``/dev/shm`` and this method reads them back.
        """
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )
        from arctic_inference.server.weight_sync.ipc_engine import (
            load_weights_from_shm,
        )

        start = time.time()
        model = self.model_runner.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        weights = load_weights_from_shm(group_id, timeout=timeout)
        self._validate_weight_sync_names(
            model, (name for name, _ in weights), context="ipc_shm",
        )
        tp = get_tensor_model_parallel_world_size()

        if self.model_config.quantization:
            from arctic_inference.server.weight_sync.utils import _FP8InplaceUpdater
            updater = _FP8InplaceUpdater(
                model, self.model_config.dtype, self.device,
            )
            for name, tensor in weights:
                updater.feed(name, tensor.to(self.device))
            loaded = len(weights)
        elif tp == 1:
            from arctic_inference.server.weight_sync.utils import _DirectParamWriter
            writer = _DirectParamWriter(model, self.device)
            loaded = 0
            for name, tensor in weights:
                view = writer.get_view(name)
                if view is not None:
                    view.copy_(tensor.to(self.device))
                loaded += 1
        else:
            loaded = 0
            for name, tensor in weights:
                model.load_weights([(name, tensor.to(self.device))])
                loaded += 1

        return self._build_result(start, mem_before, "ipc_shm", loaded)

    def load_weights_cuda_ipc(self, ipc_payload: dict) -> dict:
        """Load weights from CUDA IPC handles (zero-copy, same GPU).

        The sender (DeepSpeed worker) creates IPC handles via
        ``torch.multiprocessing.reductions.reduce_tensor`` and sends
        them as a pickled dict.  We open each handle to get a direct
        view of the sender's GPU memory and copy into our model params.

        Uses model.load_weights() which handles TP sharding natively
        via AutoWeightsLoader (column/row parallel slicing).

        NOTE: TP>1 has a known issue with the embedding layer weight_loader
        assertion in some vLLM model implementations.  Verified working
        with TP=1.  TP>1 colocated mode also requires placement group
        changes (see PR comment).

        Works even when model weights were offloaded (params are empty) —
        model.load_weights re-allocates as needed.
        """
        import base64
        import pickle

        start = time.time()
        model = self.model_runner.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        names = ipc_payload["names"]
        self._validate_weight_sync_names(model, names, context="cuda_ipc")

        handles_list = pickle.loads(
            base64.b64decode(ipc_payload["ipc_handles_pickled"]))

        device_index = torch.cuda.current_device()
        physical_gpu_id = str(
            torch.cuda.get_device_properties(device_index).uuid)

        self._offloaded_state = {}

        loaded = 0
        for name, handle_dict in zip(names, handles_list):
            if physical_gpu_id not in handle_dict:
                continue
            func, args = handle_dict[physical_gpu_id]
            args_list = list(args)
            args_list[6] = device_index
            src_tensor = func(*args_list)
            model.load_weights([(name, src_tensor)])
            loaded += 1

        torch.cuda.synchronize()
        return self._build_result(start, mem_before, "cuda_ipc", loaded)

    def load_weights_from_shm_path(self, path: str) -> dict:
        """Load weights from a shared memory file path."""
        weights = torch.load(path, map_location="cpu", weights_only=True)
        return self._load_cpu_weights_into_model(weights)

    def _load_cpu_weights_into_model(self, weights: list) -> dict:
        """Common logic for loading CPU weight list into vLLM model.

        Uses model.load_weights() which handles TP-aware sharding via
        vLLM's weight_loader. Params must be properly shaped (not offloaded)
        before calling this — use backload_model_weights first if needed.
        """
        start = time.time()
        model = self.model_runner.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        self._validate_weight_sync_names(
            model, (name for name, _ in weights), context="cpu_to_gpu",
        )

        self._offloaded_state = {}

        loaded = 0
        for name, cpu_tensor in weights:
            model.load_weights([(name, cpu_tensor.to(self.device))])
            loaded += 1

        torch.cuda.synchronize()
        return self._build_result(start, mem_before, "cpu_to_gpu", loaded)

    def load_weights_from_cpu(self, weights_bytes: bytes) -> dict:
        """Load weights from serialized CPU tensors into the vLLM model.

        First ensures all parameters have proper GPU storage allocated,
        then uses _DirectParamWriter for efficient in-place copy.
        """
        import io as _io

        start = time.time()
        model = self.model_runner.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        weights = torch.load(
            _io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)

        weights_offloaded = any(
            p.untyped_storage().size() == 0 for p in model.parameters())

        if weights_offloaded and hasattr(self, '_offloaded_state') and self._offloaded_state:
            for name, cpu_tensor in self._offloaded_state.items():
                parts = name.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                param = getattr(mod, parts[-1])
                param.data = torch.empty_like(cpu_tensor, device=self.device)
            self._offloaded_state = {}

        from arctic_inference.server.weight_sync.utils import _DirectParamWriter
        writer = _DirectParamWriter(model, self.device)
        loaded = 0
        for name, cpu_tensor in weights:
            view = writer.get_view(name)
            if view is not None:
                view.copy_(cpu_tensor.to(self.device))
                loaded += 1

        torch.cuda.synchronize()
        return self._build_result(start, mem_before, "cpu_to_gpu", loaded)

    # ------------------------------------------------------------------
    # Model weight offload/backload for colocated mode
    # ------------------------------------------------------------------

    def offload_model_weights(self) -> dict:
        """Move vLLM model weights to CPU to free GPU for training.

        Saves CPU copies, then replaces param data with tiny CPU placeholders
        to release the GPU memory. The original shapes are preserved in
        _offloaded_state for re-allocation during load.
        """
        model = self.model_runner.model
        self._offloaded_state = {}
        for name, p in model.named_parameters():
            self._offloaded_state[name] = p.data.to("cpu")
            p.data = torch.zeros(1, dtype=p.dtype, device="cpu")

        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        mem_mb = torch.cuda.memory_allocated(self.device) / 1e6
        logger.info("vLLM model weights offloaded to CPU (%.0f MB GPU remaining)", mem_mb)
        return {"status": "offloaded", "gpu_mb": mem_mb}

    def backload_model_weights(self) -> dict:
        """Restore vLLM model weights from CPU.

        Restores param.data directly from _offloaded_state which has the
        correct TP-sharded shapes saved before offload.
        """
        if not hasattr(self, '_offloaded_state') or not self._offloaded_state:
            return {"status": "nothing_to_restore"}
        model = self.model_runner.model
        param_dict = dict(model.named_parameters())
        loaded = 0
        for name, cpu_tensor in self._offloaded_state.items():
            if name in param_dict:
                param_dict[name].data = cpu_tensor.to(self.device)
                loaded += 1
        self._offloaded_state = {}
        torch.cuda.synchronize()
        logger.info("vLLM model weights restored to GPU (%d params)", loaded)
        return {"status": "restored", "loaded": loaded}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def probe_engine_mem(self) -> dict:
        """Report GPU memory state from inside the vLLM engine worker.

        Called via ``collective_rpc("probe_engine_mem")`` from the host-side
        ``InferenceWorker``. The host-side Ray actor itself never allocates
        CUDA tensors, so its local ``torch.cuda.memory_allocated()`` is
        always ``0``; this method is the only way to see what the actual
        engine worker is holding.

        Also reports two diagnostic fields that are specific to the
        colocate / sleep story:
          * ``sleep_mode_on``  — whether ``ModelConfig.enable_sleep_mode``
            actually reached the engine. If this is False, the pool path
            is a no-op even with patches applied.
          * ``pool_mb``        — total bytes currently tracked by
            ``CuMemAllocator.get_current_usage()``. If this stays near
            zero while ``alloc_mb`` is ~the full reservation, weights
            and KV cache are not being routed through the pluggable
            allocator and ``sleep`` will never free them.
          * ``worker_patched`` — whether the running ``Worker.load_model``
            is the ArcticInference patched version (i.e. defined in
            ``arctic_inference.vllm.patches``). The engine runs in a
            ``spawn`` child process, so patches applied in the parent
            don't necessarily carry over.
        """
        import os
        free_b, total_b = torch.cuda.mem_get_info()
        out: dict = {
            "pid": os.getpid(),
            "alloc_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "free_mb": free_b / 1e6,
            "total_mb": total_b / 1e6,
            "sleep_mode_on": None,
            "pool_mb": None,
            "worker_patched": None,
        }
        try:
            vllm_config = getattr(self, "vllm_config", None)
            if vllm_config is not None:
                out["sleep_mode_on"] = bool(getattr(
                    vllm_config.model_config, "enable_sleep_mode", False))
        except Exception as exc:  # noqa: BLE001
            out["sleep_mode_on"] = f"err:{exc}"

        try:
            from vllm.device_allocator.cumem import CuMemAllocator
            out["pool_mb"] = CuMemAllocator.get_instance().get_current_usage() / 1e6
        except Exception as exc:  # noqa: BLE001
            out["pool_mb"] = f"err:{exc}"

        try:
            from vllm.v1.worker.gpu_worker import Worker as _BaseWorker
            mod = getattr(_BaseWorker.load_model, "__module__", "?")
            out["worker_patched"] = mod
        except Exception as exc:  # noqa: BLE001
            out["worker_patched"] = f"err:{exc}"

        return out

    def close_weight_sync(self) -> dict:
        """Destroy persistent NCCLEngine instances."""
        for attr, key_attr in [
            ("_ws_engine", "_ws_engine_key"),
            ("_ws_spec_engine", "_ws_spec_engine_key"),
        ]:
            engine = getattr(self, attr, None)
            if engine is not None:
                engine.destroy()
                setattr(self, attr, None)
                setattr(self, key_attr, None)
        return {"status": "ok"}
