"""WeightSyncExtension — vLLM worker extension for receiving weights.

Registered via ``--worker-extension-cls arctic_inference.server.weight_sync.WeightSyncExtension``

Provides a single ``sync_weights()`` entry point that:
  1. Lazily creates / reuses a persistent ``NCCLEngine``.
  2. Receives weight tensors via NCCL send/recv (pipelined buckets or direct).
  3. Loads them into the model using the appropriate strategy:
     - **direct_zero_copy** (BF16, TP=1): receive straight into parameter views
     - **direct** (BF16, TP=1, fallback): bucket receive + copy into param views
     - **fp8**: bucket receive + FP8 in-place quantization
     - **batched**: bucket receive + model.load_weights for TP slicing
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
    ) -> dict:
        """Receive weights via NCCL and load them into the model.

        On the first call, a persistent ``NCCLEngine`` is created (NCCL
        rendezvous blocks until the sender also creates its engine).
        Subsequent calls with the same config reuse the existing engine.

        If *engine_only* is True, only the NCCL rendezvous is performed.
        If *direct_mode* is True, per-weight send/recv is used (BF16 TP=1).
        """
        from vllm.distributed.parallel_state import (
            get_world_group,
            get_tensor_model_parallel_world_size,
        )
        from arctic_inference.server.weight_sync.engine import NCCLEngine

        tp_rank = get_world_group().rank
        my_port = master_port + tp_rank
        engine_key = (master_addr, my_port)

        engine = getattr(self, "_ws_engine", None)
        if engine is None or getattr(self, "_ws_engine_key", None) != engine_key:
            if engine is not None:
                engine.destroy()
            logger.info(
                "Creating NCCLEngine rank=1 ws=2 port=%d tp_rank=%d bucket=%dMB",
                my_port, tp_rank, bucket_size // (1024 * 1024),
            )
            engine = NCCLEngine(
                master_addr=master_addr,
                master_port=my_port,
                rank=1,
                world_size=2,
                device=self.device,
                bucket_size=bucket_size,
            )
            self._ws_engine = engine
            self._ws_engine_key = engine_key

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
    # Cleanup
    # ------------------------------------------------------------------

    def close_weight_sync(self) -> dict:
        """Destroy the persistent NCCLEngine."""
        engine = getattr(self, "_ws_engine", None)
        if engine is not None:
            engine.destroy()
            self._ws_engine = None
            self._ws_engine_key = None
        return {"status": "ok"}
