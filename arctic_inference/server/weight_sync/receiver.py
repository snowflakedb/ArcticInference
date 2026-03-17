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
import time

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
    # Cleanup
    # ------------------------------------------------------------------

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
