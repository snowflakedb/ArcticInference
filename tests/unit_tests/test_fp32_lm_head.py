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

"""Unit tests for the FP32 LM head patch.

The patch promotes the LM-head matmul to fp32 by upcasting both
``hidden_states`` and ``lm_head.weight`` on the fly inside
``LogitsProcessor._get_logits``. The lm_head weight is **not**
modified (it stays in the model's native dtype).
"""

from __future__ import annotations

import os

import pytest
import torch


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _init_distributed():
    """Set up a 1-rank distributed/model-parallel environment."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", world_size=1, rank=0)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        ensure_model_parallel_initialized,
    )
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="env://",
    )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )


@pytest.fixture(scope="module")
def distributed():
    _skip_if_no_cuda()
    torch.cuda.set_device(0)
    _init_distributed()


@pytest.fixture(scope="module")
def patches_installed(distributed):
    """Install the patch exactly once for the test module.

    ``apply_patch()`` raises ``ValueError`` if called twice on the
    same target attribute, so we tolerate that to allow multiple test
    modules to coexist in the same Python process.
    """
    from arctic_inference.vllm.fp32_lm_head import (
        apply_fp32_lm_head_patches)
    try:
        apply_fp32_lm_head_patches()
    except ValueError as e:
        if "is already" not in str(e):
            raise


@pytest.fixture
def fp32_enabled(patches_installed):
    """Flip the toggle on for the duration of one test."""
    from arctic_inference.vllm.fp32_lm_head import set_fp32_lm_head_enabled
    set_fp32_lm_head_enabled(True)
    try:
        yield
    finally:
        set_fp32_lm_head_enabled(False)


@pytest.fixture
def vllm_cfg():
    """A current-vllm-config context that lasts for the whole test.

    Both ``ParallelLMHead`` (CustomOp via VocabParallelEmbedding) and
    ``LogitsProcessor`` (CustomOp) call ``get_current_vllm_config()``
    in ``__init__``, so the context must wrap their construction.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()) as cfg:
        yield cfg


def _build_bf16_parallel_lm_head(vocab_size: int, hidden_size: int):
    """Build a bf16 ``ParallelLMHead``. Caller must hold ``vllm_cfg``."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        lm_head = ParallelLMHead(vocab_size, hidden_size).cuda()
    finally:
        torch.set_default_dtype(prev)
    assert lm_head.weight.dtype == torch.bfloat16
    return lm_head


def test_fp32_logits_with_bf16_weight(fp32_enabled, vllm_cfg):
    """Bf16 lm_head weight + bf16 hidden_states → fp32 logits.

    Verifies the on-the-fly upcast: the result must equal
    ``F.linear(hidden.float(), weight_bf16.float())`` exactly, and
    the underlying ``lm_head.weight`` Parameter must remain bf16
    (we never modify it in place).
    """
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    vocab_size, hidden_size = 256, 64
    lm_head = _build_bf16_parallel_lm_head(vocab_size, hidden_size)
    logits_processor = LogitsProcessor(vocab_size).cuda()

    torch.manual_seed(0)
    weight_bf16 = torch.randn(
        lm_head.weight.shape, dtype=torch.bfloat16, device="cuda")
    lm_head.weight.data.copy_(weight_bf16)
    hidden_bf16 = torch.randn(
        4, hidden_size, dtype=torch.bfloat16, device="cuda")

    weight_ptr_before = lm_head.weight.data_ptr()
    weight_dtype_before = lm_head.weight.dtype

    logits = logits_processor(lm_head, hidden_bf16)

    assert logits.dtype == torch.float32

    # Underlying weight Parameter is unchanged (still bf16, same storage).
    assert lm_head.weight.dtype == weight_dtype_before == torch.bfloat16
    assert lm_head.weight.data_ptr() == weight_ptr_before

    ref = torch.nn.functional.linear(
        hidden_bf16.float(), weight_bf16.float())
    ref = ref[..., :vocab_size]
    torch.testing.assert_close(logits, ref, rtol=0, atol=0)


def test_fp32_logits_handles_already_fp32_inputs(fp32_enabled, vllm_cfg):
    """If the weight happens to be fp32, the upcast is a no-op."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead)

    vocab_size, hidden_size = 256, 64
    lm_head = ParallelLMHead(
        vocab_size, hidden_size, params_dtype=torch.float32).cuda()
    logits_processor = LogitsProcessor(vocab_size).cuda()

    assert lm_head.weight.dtype == torch.float32

    torch.manual_seed(0)
    weight_fp32 = torch.randn(
        lm_head.weight.shape, dtype=torch.float32, device="cuda")
    lm_head.weight.data.copy_(weight_fp32)
    hidden_fp32 = torch.randn(
        4, hidden_size, dtype=torch.float32, device="cuda")

    logits = logits_processor(lm_head, hidden_fp32)
    assert logits.dtype == torch.float32

    ref = torch.nn.functional.linear(hidden_fp32, weight_fp32)
    ref = ref[..., :vocab_size]
    torch.testing.assert_close(logits, ref, rtol=0, atol=0)


def test_no_op_when_disabled(patches_installed, vllm_cfg):
    """With the toggle off, ``_get_logits`` runs the original bf16 path."""
    from arctic_inference.vllm.fp32_lm_head import set_fp32_lm_head_enabled
    set_fp32_lm_head_enabled(False)

    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    vocab_size, hidden_size = 256, 64
    lm_head = _build_bf16_parallel_lm_head(vocab_size, hidden_size)
    logits_processor = LogitsProcessor(vocab_size).cuda()

    torch.manual_seed(0)
    lm_head.weight.data.copy_(torch.randn_like(lm_head.weight.data))
    hidden_bf16 = torch.randn(
        4, hidden_size, dtype=torch.bfloat16, device="cuda")

    logits = logits_processor(lm_head, hidden_bf16)
    # Original behavior: bf16 in, bf16 out.
    assert logits.dtype == torch.bfloat16


def test_tied_lm_head_qwen3_style(fp32_enabled, vllm_cfg):
    """Qwen3-style tying: ``self.lm_head = self.model.embed_tokens``.

    The lm_head IS the bf16 ``VocabParallelEmbedding`` Parameter. The
    patch must still produce fp32 logits via on-the-fly upcast and
    must not touch the embed_tokens weight.
    """
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding)

    vocab_size, hidden_size = 256, 64
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        embed_tokens = VocabParallelEmbedding(
            vocab_size, hidden_size).cuda()
    finally:
        torch.set_default_dtype(prev)
    lm_head = embed_tokens  # the qwen3 pattern: lm_head IS embed_tokens
    logits_processor = LogitsProcessor(vocab_size).cuda()

    assert lm_head.weight.dtype == torch.bfloat16

    torch.manual_seed(0)
    lm_head.weight.data.copy_(torch.randn_like(lm_head.weight.data))
    hidden_bf16 = torch.randn(
        4, hidden_size, dtype=torch.bfloat16, device="cuda")

    weight_dtype_before = lm_head.weight.dtype
    logits = logits_processor(lm_head, hidden_bf16)

    assert logits.dtype == torch.float32
    # Embed_tokens weight is unchanged (still bf16, no in-place mutation).
    assert lm_head.weight.dtype == weight_dtype_before == torch.bfloat16
    assert embed_tokens.weight is lm_head.weight

    ref = torch.nn.functional.linear(
        hidden_bf16.float(), lm_head.weight.float())
    ref = ref[..., :vocab_size]
    torch.testing.assert_close(logits, ref, rtol=0, atol=0)
