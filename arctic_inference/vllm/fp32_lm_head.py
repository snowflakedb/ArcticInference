# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP32 LM head: run the lm_head matmul in fp32 without changing weights.

For RL workloads we need precise logits / log-probs at the LM head:
- the lm_head matmul must be computed in fp32 (not bf16/fp16), and
- the softmax must be in fp32.

vLLM's V1 sampler already runs softmax in fp32 (it casts logits to
fp32 before softmax — see ``vllm/v1/sample/sampler.py`` lines 90 and
207, and ``vllm/v1/sample/ops/topk_topp_sampler.py`` which uses
``softmax(..., dtype=torch.float32)``). The remaining bf16 op is the
LM-head matmul itself.

This module fixes that with a single patch on
``LogitsProcessor._get_logits`` (and the parallel ``get_top_tokens``
greedy fast path):

    logits = F.linear(hidden_states.to(fp32), lm_head.weight.to(fp32),
                      bias.to(fp32))

Notes:
- The lm_head **weight is not modified**: it stays in the model's
  native dtype (typically bf16) in GPU memory. Only the matmul
  operands are upcast on the fly.
- This bypasses ``lm_head.quant_method.apply`` (which would error on
  dtype mismatch). For unquantized lm_heads the original ``apply`` is
  just ``F.linear`` anyway, so we lose no functionality.
- For quantized lm_heads (rare) the patch falls back to the original
  ``_get_logits`` so quant kernels still run; the result is the
  upstream behavior, not fp32 logits.
- Per-step cost: a single ``vocab_size * hidden * 2`` byte read of the
  weight tensor through HBM (sub-ms on H100/MI300 for 128k-vocab x
  4k-hidden). No extra VRAM is used for an fp32 weight copy.

Enable via the env var ``ARCTIC_FP32_LM_HEAD=1`` or the CLI flag
``--fp32-lm-head``.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from arctic_inference.patching import ArcticPatch

logger = init_logger(__name__)

# Module-level toggle. Set to True (e.g. via the env var or CLI flag)
# *before* ``apply_fp32_lm_head_patches`` runs and *before* the model
# is constructed. The patches are always installed but are no-ops when
# this is False.
_FP32_LM_HEAD_ENABLED = False


def set_fp32_lm_head_enabled(enabled: bool) -> None:
    """Enable or disable fp32 lm_head globally."""
    global _FP32_LM_HEAD_ENABLED
    if enabled and not _FP32_LM_HEAD_ENABLED:
        logger.info("FP32 LM head enabled: lm_head matmul will run in "
                    "fp32 (weights stay bf16; on-the-fly upcast).")
    _FP32_LM_HEAD_ENABLED = enabled


def is_fp32_lm_head_enabled() -> bool:
    return _FP32_LM_HEAD_ENABLED


def _fp32_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """``F.linear`` with all operands promoted to fp32 (no-op if already fp32).

    The weight is upcast on the fly; the underlying Parameter is not
    modified.
    """
    if weight.dtype != torch.float32:
        weight = weight.to(torch.float32)
    if hidden_states.dtype != torch.float32:
        hidden_states = hidden_states.to(torch.float32)
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    return torch.nn.functional.linear(hidden_states, weight, bias)


class LogitsProcessorFp32Patch(ArcticPatch[LogitsProcessor]):
    """Run the lm_head matmul in fp32 when the toggle is on."""

    _orig_get_logits = LogitsProcessor._get_logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        embedding_bias,
    ):
        if not _FP32_LM_HEAD_ENABLED:
            return self._orig_get_logits(
                hidden_states, lm_head, embedding_bias)

        # Quantized lm_heads use a quant-specific ``apply`` that we
        # cannot replace with a plain matmul; fall back to the
        # original (no fp32 upcast for them).
        if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
            return self._orig_get_logits(
                hidden_states, lm_head, embedding_bias)

        logits = _fp32_linear(hidden_states, lm_head.weight, embedding_bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    _orig_get_top_tokens = LogitsProcessor.get_top_tokens

    def get_top_tokens(
        self,
        lm_head,
        hidden_states: torch.Tensor,
        embedding_bias=None,
    ):
        # ``get_top_tokens`` is the local-argmax fast path used for
        # greedy decoding. We replace just the matmul step with an
        # fp32 one and keep the rest (soft_cap, scale, padding mask,
        # argmax, TP all-gather) unchanged.
        if (not _FP32_LM_HEAD_ENABLED
                or not isinstance(lm_head.quant_method,
                                  UnquantizedEmbeddingMethod)):
            return self._orig_get_top_tokens(
                lm_head, hidden_states, embedding_bias)

        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather,
        )

        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not "
                "supported for non-positive logit scaling factors.")
        tp_size = get_tensor_model_parallel_world_size()

        logits = _fp32_linear(hidden_states, lm_head.weight, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)


def apply_fp32_lm_head_patches() -> None:
    """Install the fp32 lm_head patch.

    The patch is always installed (so the toggle can be flipped at
    runtime via ``set_fp32_lm_head_enabled(True)`` before model
    construction) but is a no-op when the toggle is False.
    """
    LogitsProcessorFp32Patch.apply_patch()
