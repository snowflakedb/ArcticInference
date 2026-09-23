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

"""Model-specific training→vLLM weight conversions.

``plan_sync`` / ``convert_weights`` no-op for architectures that already match
vLLM storage names (Qwen3, Llama, ...).
"""

from __future__ import annotations

from typing import Sequence

import torch

from arctic_inference.server.weight_sync.adapters.qwen35 import SyncOp
from arctic_inference.server.weight_sync.adapters.qwen35 import apply_qwen35_sync_op
from arctic_inference.server.weight_sync.adapters.qwen35 import dest_shape_for_op
from arctic_inference.server.weight_sync.adapters.qwen35 import dest_sync_descriptors
from arctic_inference.server.weight_sync.adapters.qwen35 import expected_hf_names_for_text_sync
from arctic_inference.server.weight_sync.adapters.qwen35 import pack_qwen35_gdn_layer
from arctic_inference.server.weight_sync.adapters.qwen35 import plan_qwen35_vllm_sync
from arctic_inference.server.weight_sync.adapters.qwen35 import to_vllm_sync_weights


def plan_sync(
    names: Sequence[str],
    *,
    tie_word_embeddings: bool | None = None,
) -> list[SyncOp] | None:
    """Return conversion ops, or ``None`` when names should pass through."""
    return plan_qwen35_vllm_sync(names, tie_word_embeddings=tie_word_embeddings)


def apply_sync_op(op: SyncOp, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return apply_qwen35_sync_op(op, tensors)


def convert_weights(
    weights: Sequence[tuple[str, torch.Tensor]],
    *,
    tie_word_embeddings: bool | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Convert trainer named weights to the vLLM storage layout when needed."""
    return to_vllm_sync_weights(weights, tie_word_embeddings=tie_word_embeddings)


__all__ = [
    "SyncOp",
    "apply_sync_op",
    "convert_weights",
    "dest_shape_for_op",
    "dest_sync_descriptors",
    "expected_hf_names_for_text_sync",
    "pack_qwen35_gdn_layer",
    "plan_sync",
]
