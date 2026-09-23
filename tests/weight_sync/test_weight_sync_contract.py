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

    def _check_contract_shapes(self, model, descriptors) -> bool:
        return True


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


def test_first_payload_rejects_reordered_names():
    names = [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    ]
    ext = _FakeExt(set(names))
    ext.bind_weight_sync_contract(_desc(*names), policy="text_only")
    with pytest.raises(RuntimeError, match="first payload != init contract"):
        ext._validate_weight_sync_names(
            object(), list(reversed(names)), context="nccl:direct",
        )


def test_bind_exempts_tied_lm_head_only():
    ext = _FakeExt(
        {
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
        }
    )
    result = ext.bind_weight_sync_contract(
        _desc("language_model.model.embed_tokens.weight"),
        policy="text_only",
        tie_word_embeddings=True,
    )
    assert result["status"] == "bound"
    with pytest.raises(RuntimeError, match="extra writable"):
        ext.bind_weight_sync_contract(
            _desc("language_model.model.embed_tokens.weight"),
            policy="text_only",
            tie_word_embeddings=False,
        )


def test_non_strict_mismatch_does_not_store(monkeypatch):
    monkeypatch.setenv("ARCTIC_WEIGHT_SYNC_STRICT_NAMES", "0")
    ext = _FakeExt(
        {
            "language_model.model.embed_tokens.weight",
            "visual.patch_embed.weight",
        }
    )
    result = ext.bind_weight_sync_contract(
        _desc("language_model.model.embed_tokens.weight"),
        policy="default",
    )
    assert result["status"] == "mismatch"
    assert result["bound"] is False
    assert ext._weight_sync_contract() is None


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
    assert result == {
        "status": "skipped",
        "reason": "quantized",
        "policy": "text_only",
    }
    assert ext._weight_sync_contract() is None
    assert ext._weight_sync_policy() == "text_only"


def test_quantized_skip_does_not_require_writable_dests():
    """Bind must return before the writable ⊆ check, or FP8 receivers fail init."""
    ext = _FakeExt(writable=set(), quantized=True)
    result = ext.bind_weight_sync_contract(
        _desc("language_model.lm_head.weight"),
        policy="text_only",
    )
    assert result["status"] == "skipped"
    assert ext._weight_sync_contract() is None


def test_quantized_skip_keeps_text_only_filter():
    dest = "language_model.model.embed_tokens.weight"
    ext = _FakeExt({dest}, quantized=True)
    result = ext.bind_weight_sync_contract(_desc(dest), policy="text_only")
    assert result["status"] == "skipped"
    assert ext._weight_sync_contract() is None
    assert ext._weight_sync_policy() == "text_only"

    import arctic_inference.server.weight_sync.utils as ws_utils

    orig = ws_utils.compute_expected_hf_param_names
    ws_utils.compute_expected_hf_param_names = lambda model: {
        dest, "visual.patch_embed.weight",
    }
    try:
        ext._validate_weight_sync_names(
            object(), [dest], context="nccl:fp8",
        )
        with pytest.raises(RuntimeError, match="does NOT expect"):
            ext._validate_weight_sync_names(
                object(), [dest, "language_model.lm_head.weight"], context="nccl:fp8",
            )
    finally:
        ws_utils.compute_expected_hf_param_names = orig


def test_quantized_skip_default_policy_still_rejects_visual():
    dest = "language_model.model.embed_tokens.weight"
    ext = _FakeExt({dest}, quantized=True)
    ext.bind_weight_sync_contract(_desc(dest), policy="default")
    assert ext._weight_sync_policy() == "default"

    import arctic_inference.server.weight_sync.utils as ws_utils

    orig = ws_utils.compute_expected_hf_param_names
    ws_utils.compute_expected_hf_param_names = lambda model: {
        dest, "visual.patch_embed.weight",
    }
    try:
        with pytest.raises(RuntimeError, match="names the sender did NOT"):
            ext._validate_weight_sync_names(object(), [dest], context="nccl:fp8")
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
        self.cleared = 0
        self.initialize = _Remote(self._initialize)
        self.bind_weight_sync_contract = _Remote(self._bind)
        self.clear_weight_sync_contract = _Remote(self._clear)

    async def _initialize(self, *args, **kwargs):
        return None

    async def _bind(self, descriptors, policy="default", model_key="base",
                    tie_word_embeddings=None):
        self.binds.append((list(descriptors), policy, model_key, tie_word_embeddings))
        return {"status": "bound", "count": len(descriptors)}

    def _clear(self):
        self.cleared += 1
        return {"status": "cleared"}


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
    assert replacement.binds == [(descriptors, "text_only", "base", None)]


def test_pool_caches_contract_only_after_workers_accept():
    from arctic_inference.server.replica_pool import ReplicaPool

    pool = ReplicaPool.__new__(ReplicaPool)
    pool._workers = [_RecordingWorker(), _RecordingWorker()]
    pool._weight_sync_contract = None
    pool._check_model_id = lambda model_id: None
    descriptors = _desc("language_model.model.embed_tokens.weight")
    result = asyncio.run(
        pool.bind_weight_sync_contract(descriptors, "text_only", "base")
    )
    assert result["status"] == "bound"
    assert pool._weight_sync_contract["policy"] == "text_only"
    assert pool._workers[0].binds
    assert pool._workers[1].binds


def test_pool_does_not_cache_mismatch():
    from arctic_inference.server.replica_pool import ReplicaPool

    class _MismatchWorker(_RecordingWorker):
        async def _bind(self, descriptors, policy="default", model_key="base",
                        tie_word_embeddings=None):
            await super()._bind(descriptors, policy, model_key, tie_word_embeddings)
            return {"status": "mismatch", "bound": False}

    pool = ReplicaPool.__new__(ReplicaPool)
    pool._workers = [_MismatchWorker()]
    pool._weight_sync_contract = None
    pool._check_model_id = lambda model_id: None
    asyncio.run(
        pool.bind_weight_sync_contract(
            _desc("language_model.model.embed_tokens.weight"),
            "text_only",
            "base",
        )
    )
    assert pool._weight_sync_contract is None


def test_restart_worker_rebind_failure_leaves_old_handle(monkeypatch):
    from arctic_inference.server import replica_pool as replica_pool_mod
    from arctic_inference.server.replica_pool import ReplicaPool

    class _FailingWorker(_RecordingWorker):
        async def _bind(self, descriptors, policy="default", model_key="base",
                        tie_word_embeddings=None):
            await super()._bind(descriptors, policy, model_key, tie_word_embeddings)
            raise RuntimeError("not sampler-writable")

    replacement = _FailingWorker()
    old = _RecordingWorker()

    class _Options:
        def remote(self):
            return replacement

    pool = ReplicaPool.__new__(ReplicaPool)
    pool._workers = [old]
    pool._scheduler = None
    pool._weight_sync_contract = {
        "descriptors": _desc("language_model.model.embed_tokens.weight"),
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

    with pytest.raises(RuntimeError, match="not sampler-writable"):
        asyncio.run(pool._restart_worker(0))
    assert pool._workers[0] is old
    assert replacement.binds


def test_clear_weight_sync_contract_clears_workers():
    from arctic_inference.server.replica_pool import ReplicaPool

    pool = ReplicaPool.__new__(ReplicaPool)
    workers = [_RecordingWorker(), _RecordingWorker()]
    pool._workers = workers
    pool._weight_sync_contract = {"policy": "text_only"}
    pool.clear_weight_sync_contract()
    assert pool._weight_sync_contract is None
    assert workers[0].cleared == 1
    assert workers[1].cleared == 1


class _ViewWriter:
    def __init__(self, views):
        self._views = views

    def get_view(self, name):
        return self._views.get(name)

    def all_keys(self):
        return list(self._views)


def _patch_tp_and_writer(monkeypatch, views, tp=1):
    import vllm.distributed.parallel_state as ps
    import arctic_inference.server.weight_sync.utils as ws_utils

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(
        ws_utils, "_DirectParamWriter", lambda model, device: _ViewWriter(views),
    )


def test_writable_dest_names_drops_fused_qkv_alias(monkeypatch):
    import torch

    views = {
        "layers.0.self_attn.qkv_proj.weight": torch.ones(6, 2),
        "layers.0.self_attn.q_proj.weight": torch.ones(2, 2),
        "layers.0.self_attn.k_proj.weight": torch.ones(2, 2),
        "layers.0.self_attn.v_proj.weight": torch.ones(2, 2),
    }

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = torch.nn.Parameter(torch.ones(6, 2))

        def named_parameters(self, *args, **kwargs):
            yield "layers.0.self_attn.qkv_proj.weight", self.fused

    _patch_tp_and_writer(monkeypatch, views)
    ext = WeightSyncExtension()
    ext.device = "cpu"
    keys = ext._writable_dest_names(_Model())
    assert "layers.0.self_attn.qkv_proj.weight" not in keys
    assert "layers.0.self_attn.q_proj.weight" in keys


def test_check_contract_shapes_rejects_numel_mismatch(monkeypatch):
    import torch

    views = {"a": torch.ones(2, 4)}
    _patch_tp_and_writer(monkeypatch, views)
    ext = WeightSyncExtension()
    ext.device = "cpu"
    with pytest.raises(RuntimeError, match="dest_numel=16 view_numel=8"):
        ext._check_contract_shapes(
            object(),
            [{"name": "a", "shape": [4, 4], "dtype": "float32"}],
        )


def test_check_contract_shapes_skips_offload_placeholder(monkeypatch):
    import torch

    views = {"a": torch.ones(1)}
    _patch_tp_and_writer(monkeypatch, views)
    ext = WeightSyncExtension()
    ext.device = "cpu"
    assert ext._check_contract_shapes(
        object(),
        [{"name": "a", "shape": [4, 4], "dtype": "float32"}],
    ) is True


def test_check_contract_shapes_rejects_missing_view(monkeypatch):
    _patch_tp_and_writer(monkeypatch, {})
    ext = WeightSyncExtension()
    ext.device = "cpu"
    with pytest.raises(RuntimeError, match="no sampler view"):
        ext._check_contract_shapes(
            object(),
            [{"name": "a", "shape": [2], "dtype": "float32"}],
        )


def test_check_contract_shapes_accepts_matching_shape_and_dtype(monkeypatch):
    import torch

    views = {"a": torch.ones(2, 2, dtype=torch.float32)}
    _patch_tp_and_writer(monkeypatch, views)
    ext = WeightSyncExtension()
    ext.device = "cpu"
    assert ext._check_contract_shapes(
        object(),
        [{"name": "a", "shape": [2, 2], "dtype": "float32"}],
    ) is True


def test_check_contract_shapes_rejects_dtype_mismatch(monkeypatch):
    import torch

    views = {"a": torch.ones(2, 2, dtype=torch.float32)}
    _patch_tp_and_writer(monkeypatch, views)
    ext = WeightSyncExtension()
    ext.device = "cpu"
    with pytest.raises(RuntimeError, match="dest_dtype"):
        ext._check_contract_shapes(
            object(),
            [{"name": "a", "shape": [2, 2], "dtype": "bfloat16"}],
        )


def test_zero_copy_collects_received_names(monkeypatch):
    import arctic_inference.server.weight_sync.utils as ws_utils

    class _Engine:
        def receive_weights_direct(self, views):
            return {"params_loaded": 2, "names": ["a", "b"]}

    monkeypatch.setattr(
        ws_utils, "_DirectParamWriter", lambda model, device: _ViewWriter({}),
    )
    ext = WeightSyncExtension()
    ext.device = "cpu"
    received: list[str] = []
    loaded = ext._load_direct_zero_copy(object(), _Engine(), received)
    assert loaded == 2
    assert received == ["a", "b"]
