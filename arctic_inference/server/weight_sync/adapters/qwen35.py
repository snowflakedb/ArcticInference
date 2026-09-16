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

"""HF Qwen3.5 → vLLM storage layout for training→sampling weight sync.

HuggingFace ``AutoModelForCausalLM`` ships unpacked GDN projections under
``model.*``. vLLM's ``Qwen3_5ForConditionalGeneration`` stores a fused GDN
layout under ``language_model.model.*`` plus a frozen vision tower. NCCL
TP=1 copies by name into vLLM param views, so the sender must emit that
storage layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import Sequence

import torch

_UNPACKED_GDN_SUFFIX = "linear_attn.in_proj_qkv.weight"
_QKV_SUFFIX = ".linear_attn.in_proj_qkv.weight"
_CONV1D_SUFFIX = ".linear_attn.conv1d.weight"
_QKV_BIAS_SUFFIX = ".self_attn.qkv_proj.bias"


def has_unpacked_qwen35_gdn(names: Iterable[str]) -> bool:
    """True when the name set is HF Qwen3.5 GDN (not Qwen3 / Qwen3-Next)."""
    return any(name.endswith(_UNPACKED_GDN_SUFFIX) for name in names)


def is_optional_frozen_vllm_param(name: str) -> bool:
    """Vision / MTP params that text-only training does not train or ship."""
    if name.startswith("visual.") or name.startswith("model.visual."):
        return True
    if name.startswith("mtp.") or name.startswith("model.mtp."):
        return True
    return ".mtp." in name


def _unpacked_qkv_biases(qkv_bias_name: str) -> tuple[str, str, str]:
    prefix = qkv_bias_name[: -len("qkv_proj.bias")]
    return (f"{prefix}q_proj.bias", f"{prefix}k_proj.bias", f"{prefix}v_proj.bias")


def expected_hf_names_for_text_sync(expected: set[str], sender_names: Iterable[str]) -> set[str]:
    """Drop frozen vision/MTP names from *expected* unless the sender shipped them.

    vLLM fuses Qwen2 ``q/k/v_proj.bias`` into ``qkv_proj.bias``. The HF trainer
    still ships the three unpacked biases; accept those in place of the fused
    name so name-check matches the loader that already cats q/k/v.
    """
    sender_set = set(sender_names)
    kept = {n for n in expected if n in sender_set or not is_optional_frozen_vllm_param(n)}
    rewritten: set[str] = set()
    for name in kept:
        if name.endswith(_QKV_BIAS_SUFFIX):
            unpacked = _unpacked_qkv_biases(name)
            if all(part in sender_set for part in unpacked):
                rewritten.update(unpacked)
                continue
        rewritten.add(name)
    return rewritten


def pack_qwen35_gdn_layer(layer_sd: dict, prefix: str) -> dict:
    """Pack unpacked HF GDN projections into vLLM fused ``in_proj_qkvz`` / ``in_proj_ba``.

    Row order matches vLLM ``Qwen3_5GatedDeltaNet``:
    ``in_proj_qkvz = cat([qkv, z], dim=0)``, ``in_proj_ba = cat([b, a], dim=0)``.
    Mutates ``layer_sd`` in place (same contract as the MoE layer converter).
    """
    qkv_key = f"{prefix}.linear_attn.in_proj_qkv.weight"
    z_key = f"{prefix}.linear_attn.in_proj_z.weight"
    if qkv_key in layer_sd and z_key in layer_sd:
        qkv = layer_sd.pop(qkv_key)
        z = layer_sd.pop(z_key)
        layer_sd[f"{prefix}.linear_attn.in_proj_qkvz.weight"] = torch.cat([qkv, z], dim=0)

    b_key = f"{prefix}.linear_attn.in_proj_b.weight"
    a_key = f"{prefix}.linear_attn.in_proj_a.weight"
    if b_key in layer_sd and a_key in layer_sd:
        b = layer_sd.pop(b_key)
        a = layer_sd.pop(a_key)
        layer_sd[f"{prefix}.linear_attn.in_proj_ba.weight"] = torch.cat([b, a], dim=0)
    return layer_sd


def to_vllm_param_name(name: str) -> str | None:
    """Map an HF / Prime-RL Qwen3.5 name onto vLLM's VLM internal name.

    Returns ``None`` for frozen vision / MTP params that should not be synced.
    """
    if is_optional_frozen_vllm_param(name):
        return None
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    if name.startswith("lm_head."):
        return "language_model.lm_head." + name[len("lm_head.") :]
    if name.startswith("model."):
        return "language_model.model." + name[len("model.") :]
    return name


@dataclass(frozen=True)
class SyncOp:
    dest: str
    sources: tuple[str, ...]
    kind: str  # copy | cat0 | unsqueeze1


def _gdn_layer_prefix(qkv_name: str) -> str:
    if not qkv_name.endswith(_QKV_SUFFIX):
        raise ValueError(f"not an unpacked GDN qkv name: {qkv_name!r}")
    return qkv_name[: -len(_QKV_SUFFIX)]


def plan_qwen35_vllm_sync(names: Sequence[str]) -> list[SyncOp] | None:
    """Build copy/pack ops from HF names. ``None`` means leave the list unchanged."""
    if not has_unpacked_qwen35_gdn(names):
        return None

    name_set = set(names)
    consumed: set[str] = set()
    ops: list[SyncOp] = []

    for name in names:
        if not name.endswith(_UNPACKED_GDN_SUFFIX):
            continue
        prefix = _gdn_layer_prefix(name)
        qkv_key = f"{prefix}.linear_attn.in_proj_qkv.weight"
        z_key = f"{prefix}.linear_attn.in_proj_z.weight"
        b_key = f"{prefix}.linear_attn.in_proj_b.weight"
        a_key = f"{prefix}.linear_attn.in_proj_a.weight"
        if qkv_key in name_set and z_key in name_set:
            dest = to_vllm_param_name(f"{prefix}.linear_attn.in_proj_qkvz.weight")
            if dest is not None:
                ops.append(SyncOp(dest, (qkv_key, z_key), "cat0"))
            consumed.update((qkv_key, z_key))
        if b_key in name_set and a_key in name_set:
            dest = to_vllm_param_name(f"{prefix}.linear_attn.in_proj_ba.weight")
            if dest is not None:
                ops.append(SyncOp(dest, (b_key, a_key), "cat0"))
            consumed.update((b_key, a_key))

    has_embed = any(name.endswith("embed_tokens.weight") for name in names)
    for name in names:
        if name in consumed or is_optional_frozen_vllm_param(name):
            continue
        if has_embed and name.endswith("lm_head.weight"):
            continue
        dest = to_vllm_param_name(name)
        if dest is None:
            continue
        kind = "unsqueeze1" if name.endswith(_CONV1D_SUFFIX) else "copy"
        ops.append(SyncOp(dest, (name,), kind))
    return ops


def apply_qwen35_sync_op(op: SyncOp, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if op.kind == "cat0":
        return torch.cat(list(tensors), dim=0)
    if op.kind == "unsqueeze1":
        tensor = tensors[0]
        return tensor.unsqueeze(1) if tensor.ndim == 2 else tensor
    return tensors[0]


def to_vllm_sync_weights(
    weights: Sequence[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Convert HF named weights to the vLLM storage layout, or return ``weights`` unchanged."""
    names = [name for name, _ in weights]
    ops = plan_qwen35_vllm_sync(names)
    if ops is None:
        return list(weights)
    by_name = dict(weights)
    out: list[tuple[str, torch.Tensor]] = []
    for op in ops:
        tensors = [by_name[src] for src in op.sources]
        out.append((op.dest, apply_qwen35_sync_op(op, tensors)))
    return out
