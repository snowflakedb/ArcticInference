# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
    GroupShape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights,
    pack_rows,
)


class W4A8LmHeadMethod:
    """Quantization method for ParallelLMHead using CUTLASS W4A8.

    Usage:
      - Assign an instance to `lm_head.quant_method`.
      - It will lazily quantize `lm_head.weight` to int4 (group_size=128)
        the first time `apply()` is called, and then call
        `ops.cutlass_w4a8_mm` with FP8 activations.

    This deliberately does NOT integrate with Fp8Config/Fp8ConfigWithEmbedding.
    It is entirely opt-in and only affects lm_heads you explicitly patch.
    """

    def __init__(self, group_size: int = 128) -> None:
        self.group_size = group_size
        # Same dynamic per-token FP8 activation quant as CutlassW4A8LinearKernel.
        self.quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    # ---------------- weight preparation (INT4 + scales) ----------------

    @torch.inference_mode()
    def _prepare_once(self, lm_head: torch.nn.Module) -> None:
        """Quantize and pack lm_head.weight only once, after weights are loaded."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A8LmHeadMethod._prepare_once() called during CUDA graph capture. "
                "Call quant_method.prepare(lm_head) before enabling graph capture."
            )
        if hasattr(lm_head, "w4a8_w_q"):
            return  # already prepared

        if not current_platform.is_cuda():
            raise RuntimeError("W4A8 LM head requires CUDA")

        if not current_platform.has_device_capability(90):
            raise RuntimeError("W4A8 LM head requires Hopper (sm90)")

        if not hasattr(ops, "cutlass_w4a8_mm"):
            raise RuntimeError(
                "ops.cutlass_w4a8_mm not found. Rebuild vLLM with W4A8 support."
            )

        # ParallelLMHead stores its local shard as [vocab_shard, hidden_dim].
        weight: torch.Tensor = lm_head.weight
        vocab_shard, hidden_dim = weight.shape

        if hidden_dim % self.group_size != 0 or vocab_shard % self.group_size != 0:
            raise ValueError(
                "W4A8 kernel requires both hidden_dim and local vocab shard "
                f"to be multiples of {self.group_size}, got weight.shape="
                f"{tuple(weight.shape)}"
            )

        # CUTLASS kernel expects W as [K, N] = [hidden_dim, vocab_shard].
        w = weight.detach().to(torch.float16).t().contiguous()  # [K, N]

        # Group-wise int4 quantization along K with group_size=128.
        _, w_q, w_s, _ = quantize_weights(
            w,
            scalar_types.int4,
            group_size=self.group_size,
            zero_points=False,
        )

        # Pack 4-bit values into int32 and reorder into CUTLASS layout.
        w_q = pack_rows(w_q & 0x0F, scalar_types.int4.size_bits, *w_q.shape)
        # Column-major before the custom op's internal reordering.
        w_q = w_q.t().contiguous().t()
        w_q_packed = ops.cutlass_encode_and_reorder_int4b(w_q)

        # Pack group scales to FP8.
        w_s_packed = ops.cutlass_pack_scale_fp8(
            w_s.to(torch.float8_e4m3fn)
        )

        # Per-output-channel scales: keep them at 1.0 (same as vLLM tests).
        b_channel_scales = torch.ones(
            vocab_shard, device=weight.device, dtype=torch.float32
        )

        # Register as buffers so they follow the module across devices/DDP.
        lm_head.register_buffer("w4a8_w_q", w_q_packed)
        lm_head.register_buffer("w4a8_w_group_scales", w_s_packed)
        lm_head.register_buffer("w4a8_w_channel_scales", b_channel_scales)

    @torch.inference_mode()
    def prepare(self, lm_head: torch.nn.Module) -> None:
        """Optional explicit hook if you want to pre-quantize."""
        self._prepare_once(lm_head)

    # ---------------- forward path (FP8 act × INT4 weight) ----------------

    @torch.inference_mode()
    def apply(
        self,
        lm_head: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Matches LogitsProcessorOpt expectation:

            logits = lm_head.quant_method.apply(lm_head, hidden_states, bias)

        Args:
            lm_head: ParallelLMHead instance.
            x: [..., hidden_dim] tensor (usually bf16).
            bias: Optional bias (usually None for ParallelLMHead).
        Returns:
            [..., local_vocab_shard] logits in bf16.
        """
        # Lazily quantize weights the first time after loading.
        self._prepare_once(lm_head)

        hidden_dim = x.shape[-1]
        if hasattr(lm_head, "embedding_dim"):
            assert hidden_dim == lm_head.embedding_dim, (
                f"hidden_dim={hidden_dim}, expected {lm_head.embedding_dim}"
            )

        # Flatten to [M, K].
        x_2d = x.reshape(-1, hidden_dim)

        # Dynamic per-token FP8 activation quantization.
        x_fp8, token_scales = self.quant_fp8(x_2d)
        token_scales = token_scales.reshape(-1)  # [M]

        # CUTLASS GEMM: FP8 activations × INT4 weights → BF16 logits.
        out_2d = ops.cutlass_w4a8_mm(
            a=x_fp8,
            b_q=lm_head.w4a8_w_q,
            b_group_scales=lm_head.w4a8_w_group_scales,
            b_group_size=self.group_size,
            a_token_scales=token_scales,
            b_channel_scales=lm_head.w4a8_w_channel_scales,
        )

        if bias is not None:
            out_2d = out_2d + bias

        return out_2d

