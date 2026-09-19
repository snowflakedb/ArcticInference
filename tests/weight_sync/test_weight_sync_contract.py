# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the init-time weight-sync name contract."""

from __future__ import annotations

import pytest

from arctic_inference.server.weight_sync.receiver import WeightSyncExtension


class _FakeExt(WeightSyncExtension):
    def __init__(self, writable, quantized=False):
        self._writable = set(writable)
        self.model_config = type("C", (), {"quantization": quantized})()
        self.device = "cpu"
        self.model_runner = type("R", (), {"model": object()})()

    def _writable_dest_names(self, model) -> set[str]:
        return set(self._writable)

    def _check_contract_shapes(self, model, descriptors) -> None:
        return None


def _desc(*names: str) -> list[dict]:
    return [{"name": name, "shape": [1], "dtype": "bfloat16"} for name in names]


def test_bind_accepts_trainer_dest_subset_with_text_only_visual():
    ext = _FakeExt(
        {
            "language_model.model.embed_tokens.weight",
            "visual.patch_embed.weight",
            "mtp.layers.0.weight",
        }
    )
    result = ext.bind_weight_sync_contract(
        _desc("language_model.model.embed_tokens.weight"),
        policy="text_only",
    )
    assert result["status"] == "bound"
    assert result["count"] == 1


def test_bind_fails_when_trainer_dest_is_not_writable():
    ext = _FakeExt({"language_model.model.embed_tokens.weight"})
    with pytest.raises(RuntimeError, match="not sampler-writable"):
        ext.bind_weight_sync_contract(
            _desc(
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
            ),
            policy="text_only",
        )


def test_bind_fails_on_extra_writable_without_text_only_policy():
    ext = _FakeExt(
        {
            "language_model.model.embed_tokens.weight",
            "visual.patch_embed.weight",
        }
    )
    with pytest.raises(RuntimeError, match="extra writable"):
        ext.bind_weight_sync_contract(
            _desc("language_model.model.embed_tokens.weight"),
            policy="default",
        )


def test_first_payload_asserts_ordered_names_later_uses_hash():
    dest = "language_model.model.embed_tokens.weight"
    ext = _FakeExt({dest})
    ext.bind_weight_sync_contract(_desc(dest), policy="text_only")
    ext._validate_weight_sync_names(object(), [dest], context="nccl:direct")
    assert ext._ws_contracts["base"]["first_ok"] is True
    ext._validate_weight_sync_names(object(), [dest], context="nccl:direct")
    with pytest.raises(RuntimeError, match="drifted from init contract"):
        ext._validate_weight_sync_names(
            object(),
            [dest, "language_model.lm_head.weight"],
            context="nccl:direct",
        )


def test_rebind_same_contract_keeps_first_ok():
    dest = "language_model.model.embed_tokens.weight"
    ext = _FakeExt({dest})
    ext.bind_weight_sync_contract(_desc(dest), policy="text_only")
    ext._validate_weight_sync_names(object(), [dest], context="first")
    reused = ext.bind_weight_sync_contract(_desc(dest), policy="text_only")
    assert reused.get("reused") is True
    assert ext._ws_contracts["base"]["first_ok"] is True


def test_quantized_receiver_skips_bind():
    ext = _FakeExt({"language_model.model.embed_tokens.weight"}, quantized=True)
    result = ext.bind_weight_sync_contract(
        _desc("language_model.model.embed_tokens.weight"),
        policy="text_only",
    )
    assert result["status"] == "skipped"
    assert ext._weight_sync_contract() is None
