# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the init-time weight-sync name contract."""

from __future__ import annotations

import asyncio

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
    assert result == {"status": "skipped", "reason": "quantized"}
    assert ext._weight_sync_contract() is None


def test_quantized_skip_does_not_require_writable_dests():
    """Bind must return before the writable ⊆ check, or FP8 receivers fail init."""
    ext = _FakeExt(writable=set(), quantized=True)
    result = ext.bind_weight_sync_contract(
        _desc("language_model.lm_head.weight"),
        policy="text_only",
    )
    assert result["status"] == "skipped"
    assert ext._weight_sync_contract() is None


def test_quantized_skip_keeps_legacy_name_check():
    dest = "language_model.model.embed_tokens.weight"
    ext = _FakeExt({dest}, quantized=True)
    ext.bind_weight_sync_contract(_desc(dest), policy="text_only")
    assert ext._weight_sync_contract() is None

    import arctic_inference.server.weight_sync.utils as ws_utils

    orig = ws_utils.compute_expected_hf_param_names
    ws_utils.compute_expected_hf_param_names = lambda model: {dest}
    try:
        ext._validate_weight_sync_names(object(), [dest], context="nccl:fp8")
        with pytest.raises(RuntimeError, match="does NOT expect"):
            ext._validate_weight_sync_names(
                object(), [dest, "visual.patch_embed.weight"], context="nccl:fp8",
            )
    finally:
        ws_utils.compute_expected_hf_param_names = orig


def test_fp8_string_counts_as_quantized():
    ext = WeightSyncExtension()
    ext.model_config = type("C", (), {"quantization": "fp8"})()
    assert ext._is_quantized() is True
    ext.model_config = type("C", (), {"quantization": None})()
    assert ext._is_quantized() is False


class _Remote:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _RecordingWorker:
    def __init__(self):
        self.binds: list[tuple] = []
        self.initialize = _Remote(self._initialize)
        self.bind_weight_sync_contract = _Remote(self._bind)

    async def _initialize(self, *args, **kwargs):
        return None

    async def _bind(self, descriptors, policy="default", model_key="base"):
        self.binds.append((list(descriptors), policy, model_key))
        return {"status": "bound", "count": len(descriptors)}


def test_rebind_is_noop_without_stored_contract():
    from arctic_inference.server.replica_pool import ReplicaPool

    pool = ReplicaPool.__new__(ReplicaPool)
    pool._weight_sync_contract = None
    worker = _RecordingWorker()
    asyncio.run(pool._rebind_weight_sync_contract(worker))
    assert worker.binds == []


def test_restart_worker_rebinds_stored_contract(monkeypatch):
    from arctic_inference.server import replica_pool as replica_pool_mod
    from arctic_inference.server.replica_pool import ReplicaPool

    replacement = _RecordingWorker()
    descriptors = _desc("language_model.model.embed_tokens.weight")

    class _Options:
        def remote(self):
            return replacement

    pool = ReplicaPool.__new__(ReplicaPool)
    pool._workers = [_RecordingWorker()]
    pool._scheduler = None
    pool._weight_sync_contract = {
        "descriptors": descriptors,
        "policy": "text_only",
        "model_key": "base",
    }
    pool._worker_cls = type("Cls", (), {"options": staticmethod(lambda **kwargs: _Options())})()
    pool._worker_ray_options = lambda replica_idx: {}
    pool._config = type(
        "Cfg",
        (),
        {"to_engine_kwargs": lambda self: {"model": "dummy"}, "extra_env": None},
    )()
    monkeypatch.setattr(replica_pool_mod.ray, "kill", lambda actor: None)

    asyncio.run(pool._restart_worker(0))
    assert pool._workers[0] is replacement
    assert replacement.binds == [(descriptors, "text_only", "base")]
