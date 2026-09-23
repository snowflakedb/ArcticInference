# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for HF Qwen3.5 → vLLM weight-sync conversion."""

from __future__ import annotations

import pytest
import torch

from arctic_inference.server.weight_sync.adapters import convert_weights
from arctic_inference.server.weight_sync.adapters import dest_shape_for_op
from arctic_inference.server.weight_sync.adapters import dest_sync_descriptors
from arctic_inference.server.weight_sync.adapters import expected_hf_names_for_text_sync
from arctic_inference.server.weight_sync.adapters import pack_qwen35_gdn_layer
from arctic_inference.server.weight_sync.adapters import plan_sync
from arctic_inference.server.weight_sync.adapters import SyncOp
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


def test_passthrough_drops_visual_from_payload_and_descriptors():
    weights = [
        ("model.embed_tokens.weight", torch.ones(4, 8)),
        ("visual.patch_embed.weight", torch.ones(8)),
        ("mtp.layers.0.weight", torch.ones(4)),
    ]
    converted = convert_weights(weights)
    assert [name for name, _ in converted] == ["model.embed_tokens.weight"]
    descriptors = dest_sync_descriptors(
        [name for name, _ in weights],
        {"model.embed_tokens.weight": (4, 8), "visual.patch_embed.weight": (8,)},
        {
            "model.embed_tokens.weight": "bfloat16",
            "visual.patch_embed.weight": "bfloat16",
        },
    )
    assert [item["name"] for item in descriptors] == ["model.embed_tokens.weight"]


def test_dest_sync_descriptors_passthrough_requires_shapes():
    with pytest.raises(KeyError, match="missing from shapes"):
        dest_sync_descriptors(
            ["model.embed_tokens.weight"],
            {},
            {"model.embed_tokens.weight": "bfloat16"},
        )


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


def test_plan_sync_packs_qkv_bias_to_fused_dest():
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
    ]
    ops = plan_sync(names)
    dests = [op.dest for op in ops]
    assert "language_model.model.layers.0.self_attn.qkv_proj.bias" in dests
    assert "language_model.model.layers.0.self_attn.q_proj.bias" not in dests
    bias_op = next(op for op in ops if op.dest.endswith("qkv_proj.bias"))
    assert bias_op.kind == "cat0"
    assert len(bias_op.sources) == 3


def test_plan_sync_keeps_untied_lm_head():
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
        "lm_head.weight",
    ]
    dests = [op.dest for op in plan_sync(names, tie_word_embeddings=False)]
    assert "language_model.lm_head.weight" in dests
    dests_unknown = [op.dest for op in plan_sync(names)]
    assert "language_model.lm_head.weight" in dests_unknown


def test_plan_sync_drops_tied_lm_head():
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
        "lm_head.weight",
    ]
    dests = [op.dest for op in plan_sync(names, tie_word_embeddings=True)]
    assert "language_model.lm_head.weight" not in dests


def test_dest_shape_for_cat0_and_unsqueeze1():
    cat = dest_shape_for_op(SyncOp("d", ("a", "b"), "cat0"), [(6144, 2048), (2048, 2048)])
    assert cat == (8192, 2048)
    unsqueezed = dest_shape_for_op(SyncOp("d", ("c",), "unsqueeze1"), [(2048, 4)])
    assert unsqueezed == (2048, 1, 4)


def test_dest_sync_descriptors_include_packed_qkv_bias():
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.visual.patch_embed.weight",
    ]
    shapes = {
        "model.embed_tokens.weight": (100, 2048),
        "model.layers.0.linear_attn.in_proj_qkv.weight": (6144, 2048),
        "model.layers.0.linear_attn.in_proj_z.weight": (2048, 2048),
        "model.layers.0.linear_attn.in_proj_b.weight": (16, 2048),
        "model.layers.0.linear_attn.in_proj_a.weight": (16, 2048),
        "model.layers.0.self_attn.q_proj.bias": (2048,),
        "model.layers.0.self_attn.k_proj.bias": (256,),
        "model.layers.0.self_attn.v_proj.bias": (2048,),
        "model.visual.patch_embed.weight": (8,),
    }
    dtypes = {name: "bfloat16" for name in names}
    descriptors = dest_sync_descriptors(names, shapes, dtypes)
    by_name = {item["name"]: item for item in descriptors}
    assert "visual.patch_embed.weight" not in by_name
    assert by_name["language_model.model.layers.0.self_attn.qkv_proj.bias"]["shape"] == [4352]
    assert by_name["language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"]["shape"] == [8192, 2048]


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
