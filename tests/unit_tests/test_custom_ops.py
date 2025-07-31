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

import pytest
import torch
from typing import List

CUDA_DEVICES = [f"cuda:{0}"]


def reshape_and_cache_flash_bulk_ref(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scales: List[torch.Tensor],
    v_scales: List[torch.Tensor],
    num_heads: int,
    head_size: int,
) -> None:
    from vllm import _custom_ops as ops

    num_layers = len(key_caches)
    key_list = torch.chunk(keys, num_layers, dim=-1)
    value_list = torch.chunk(values, num_layers, dim=-1)

    for i in range(num_layers):
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key_list[i].reshape(-1, num_heads, head_size),
            value_list[i].reshape(-1, num_heads, head_size),
            key_caches[i],
            value_caches[i],
            slot_mapping,
            kv_cache_dtype,
            k_scales[i],
            v_scales[i],
        )

    return None


@pytest.mark.parametrize("num_layers", [4])
@pytest.mark.parametrize("num_tokens", [2])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("head_size", [16])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_reshape_and_cache_flash_bulk(
    device: str,
    num_layers: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
) -> None:
    from arctic_inference.py_custom_ops import (try_load_torch_library,
                                                reshape_and_cache_flash_bulk)
    if not try_load_torch_library():
        pytest.skip("Custom ops not available, skipping test.")

    torch.set_default_device(device)

    hidden_size = num_heads * head_size

    keys = torch.randn(num_layers * num_tokens, hidden_size, device=device)
    values = torch.randn(num_layers * num_tokens, hidden_size, device=device)
    key_caches = [
        torch.randn(num_tokens, hidden_size, device=device)
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.randn(num_tokens, hidden_size, device=device)
        for _ in range(num_layers)
    ]
    key_caches_ref = key_caches.copy()
    value_caches_ref = value_caches.copy()
    slot_mapping = torch.randint(0, num_tokens, (num_tokens, ), device=device)
    kv_cache_dtype = "auto"
    k_scales = [
        torch.tensor(0.1, dtype=torch.float32, device=device)
        for _ in range(num_layers)
    ]
    v_scales = [
        torch.tensor(0.1, dtype=torch.float32, device=device)
        for _ in range(num_layers)
    ]

    reshape_and_cache_flash_bulk_ref(keys, values, key_caches_ref,
                                     value_caches_ref, slot_mapping,
                                     kv_cache_dtype, k_scales, v_scales,
                                     num_heads, head_size)

    reshape_and_cache_flash_bulk(keys, values, key_caches, value_caches,
                                 slot_mapping, kv_cache_dtype, k_scales,
                                 v_scales, num_heads, head_size)

    for i in range(num_layers):
        assert torch.allclose(
            key_caches[i],
            key_caches_ref[i]), f"Key caches do not match for layer {i}"
        assert torch.allclose(
            value_caches[i],
            value_caches_ref[i]), f"Value caches do not match for layer {i}"

    return None
