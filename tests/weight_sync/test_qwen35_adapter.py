# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for HF Qwen3.5 → vLLM weight-sync conversion."""

from __future__ import annotations

import pytest
import torch

from arctic_inference.server.weight_sync.adapters import convert_weights
from arctic_inference.server.weight_sync.adapters import expected_hf_names_for_text_sync
from arctic_inference.server.weight_sync.adapters import pack_qwen35_gdn_layer
from arctic_inference.server.weight_sync.receiver import TextOnlyWeightSyncExtension


def _qwen35_2b_like_layer(layer: int) -> list[tuple[str, torch.Tensor]]:
    prefix = f"model.layers.{layer}"
    return [
        (f"{prefix}.input_layernorm.weight", torch.ones(2048)),
        (f"{prefix}.linear_attn.in_proj_qkv.weight", torch.arange(6144 * 2048).reshape(6144, 2048).float()),
        (f"{prefix}.linear_attn.in_proj_z.weight", torch.arange(2048 * 2048).reshape(2048, 2048).float() + 1),
        (f"{prefix}.linear_attn.in_proj_b.weight", torch.arange(16 * 2048).reshape(16, 2048).float() + 2),
        (f"{prefix}.linear_attn.in_proj_a.weight", torch.arange(16 * 2048).reshape(16, 2048).float() + 3),
        (f"{prefix}.linear_attn.conv1d.weight", torch.arange(2048 * 4).reshape(2048, 4).float()),
        (f"{prefix}.linear_attn.out_proj.weight", torch.ones(2048, 2048)),
        (f"{prefix}.mlp.down_proj.weight", torch.ones(2048, 2048)),
    ]


def test_qwen35_packs_gdn_prefix_and_conv1d():
    embed = torch.ones(100, 2048)
    layer_weights = _qwen35_2b_like_layer(0)
    by_hf = dict(layer_weights)
    weights = [
        ("model.embed_tokens.weight", embed),
        *layer_weights,
        ("model.visual.patch_embed.weight", torch.ones(8)),
        ("mtp.layers.0.weight", torch.ones(4)),
    ]
    converted = dict(convert_weights(weights))
    qkv = by_hf["model.layers.0.linear_attn.in_proj_qkv.weight"]
    z = by_hf["model.layers.0.linear_attn.in_proj_z.weight"]
    assert converted["language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"].shape == (8192, 2048)
    assert torch.equal(converted["language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"][:6144], qkv)
    assert torch.equal(converted["language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"][6144:], z)
    assert "visual.patch_embed.weight" not in converted


def test_qwen3_passthrough_without_unpacked_gdn():
    weights = [
        ("model.embed_tokens.weight", torch.ones(4, 8)),
        ("model.layers.0.self_attn.q_proj.weight", torch.ones(8, 8)),
        ("lm_head.weight", torch.ones(4, 8)),
    ]
    converted = convert_weights(weights)
    assert [name for name, _ in converted] == [name for name, _ in weights]


def test_pack_qwen35_gdn_layer_matches_vllm_cat_order():
    prefix = "model.layers.1"
    qkv = torch.arange(12).reshape(6, 2).float()
    z = torch.arange(4).reshape(2, 2).float() + 100
    layer_sd = {
        f"{prefix}.linear_attn.in_proj_qkv.weight": qkv,
        f"{prefix}.linear_attn.in_proj_z.weight": z,
        f"{prefix}.linear_attn.in_proj_b.weight": torch.ones(1, 2),
        f"{prefix}.linear_attn.in_proj_a.weight": torch.ones(1, 2),
    }
    pack_qwen35_gdn_layer(layer_sd, prefix)
    assert torch.equal(layer_sd[f"{prefix}.linear_attn.in_proj_qkvz.weight"], torch.cat([qkv, z], dim=0))


def test_expected_hf_names_drop_missing_visual():
    expected = {"language_model.model.embed_tokens.weight", "visual.patch_embed.weight"}
    filtered = expected_hf_names_for_text_sync(expected, {"language_model.model.embed_tokens.weight"})
    assert filtered == {"language_model.model.embed_tokens.weight"}


def test_text_only_extension_allows_missing_visual():
    ext = TextOnlyWeightSyncExtension()
    expected = {"language_model.model.embed_tokens.weight", "visual.patch_embed.weight"}
    import arctic_inference.server.weight_sync.utils as ws_utils

    orig = ws_utils.compute_expected_hf_param_names
    ws_utils.compute_expected_hf_param_names = lambda model: set(expected)
    try:
        ext._validate_weight_sync_names(object(), ["language_model.model.embed_tokens.weight"], context="test")
    finally:
        ws_utils.compute_expected_hf_param_names = orig


def test_text_only_extension_still_rejects_unexpected_lm_name():
    ext = TextOnlyWeightSyncExtension()
    import arctic_inference.server.weight_sync.utils as ws_utils

    orig = ws_utils.compute_expected_hf_param_names
    ws_utils.compute_expected_hf_param_names = lambda model: {
        "language_model.model.embed_tokens.weight",
        "visual.patch_embed.weight",
    }
    try:
        with pytest.raises(RuntimeError, match="does NOT expect"):
            ext._validate_weight_sync_names(
                object(),
                ["language_model.model.embed_tokens.weight", "language_model.lm_head.weight"],
                context="test",
            )
    finally:
        ws_utils.compute_expected_hf_param_names = orig
