# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-cache dynamic NTK RoPE support for ArcticInference.

This subpackage adds a new ``rope_type="multi_cache_ntk"`` that
implements a *bucketed* dynamic-NTK scheme: one static-rope cache per
configured factor, concatenated into a single ``cos_sin_cache``.  At
forward time, each token is routed (per its request's seq_len) to the
factor whose cache best covers that length, and the existing CUDA rope
kernel is re-used unchanged -- we simply add a per-token offset tensor
to ``positions`` before the kernel indexes the unified cache.

Why this shape
--------------
* Within any given bucket, the behavior is bit-identical to vLLM's
  static :class:`~vllm.model_executor.layers.rotary_embedding.DynamicNTKScalingRotaryEmbedding`
  at that bucket's factor.  This matches the training distribution for
  models that were fine-tuned with static rope at some factor.
* No new CUDA kernel.  The forward reuses ``ops.rotary_embedding`` with
  ``positions + per_token_offset``.  Cost is a single elementwise add.
* CUDA-graph safe.  The per-token offset tensor lives in a registered
  buffer; the model runner writes it in place each forward.  Captured
  graphs see fresh values on replay with no re-capture.

Design summary
--------------
* :class:`MultiCacheDynamicNTKRotaryEmbedding` is a drop-in
  :class:`~vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding`
  subclass.  It owns a registered GPU buffer ``runtime_bucket_offsets``
  sized for ``max_num_batched_tokens`` and a concatenated
  ``cos_sin_cache`` covering every factor.
* The model runner
  (:class:`arctic_inference.vllm.model_runner.GPUModelRunnerPatch`)
  computes per-token seq-lens and calls
  :meth:`MultiCacheDynamicNTKRotaryEmbedding.update_runtime_seq_lens`
  before invoking the model; that method does the seq_len -> offset
  translation in-place on the GPU buffer.

Public API
----------
* :class:`MultiCacheDynamicNTKRotaryEmbedding`
* :func:`apply_rope_runtime_patches`
"""

from arctic_inference.vllm.rope.multi_cache_ntk import (
    DEFAULT_FACTORS,
    MultiCacheDynamicNTKRotaryEmbedding,
)

__all__ = [
    "DEFAULT_FACTORS",
    "MultiCacheDynamicNTKRotaryEmbedding",
    "apply_rope_runtime_patches",
]


def apply_rope_runtime_patches() -> None:
    """Install the multi-cache RoPE patches.

    This must be called *before* any model loads so that
    :func:`vllm.model_executor.layers.rotary_embedding.get_rope` knows
    how to dispatch the new ``multi_cache_ntk`` rope type.  The
    companion :class:`GPUModelRunnerPatch` hook is installed elsewhere
    (in :mod:`arctic_inference.vllm.model_runner`) because it needs to
    run after CUDA is safely importable.
    """
    from arctic_inference.vllm.rope.patches import (
        _install_apply_dict_overrides_patch,
        _install_get_rope_patch,
    )

    _install_apply_dict_overrides_patch()
    _install_get_rope_patch()
