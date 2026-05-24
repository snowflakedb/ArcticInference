# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""ArcticInference patches for kvcached prefix caching with automatic
memory-pressure-driven eviction."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_kvcached_prefix_cache_patches() -> None:
    _patch_coordinator_setup()
    logger.info("kvcached prefix-cache patches applied")


def _patch_coordinator_setup() -> None:
    kvcoord_mod = importlib.import_module("vllm.v1.core.kv_cache_coordinator")
    KVCacheCoordinator = getattr(kvcoord_mod, "KVCacheCoordinator")
    if not hasattr(KVCacheCoordinator, "_setup_kvcached_coordinator"):
        logger.debug("_setup_kvcached_coordinator not found")
        return

    from arctic_inference.vllm.kvcached.prefix_block_pool import (
        PrefixCacheableElasticBlockPool,
    )

    def _setup_with_prefix_cache(self: Any) -> None:
        kv_cache_config = getattr(self, "kv_cache_config")
        kv_groups = kv_cache_config.kv_cache_groups
        if len(kv_groups) != 1:
            raise ValueError(
                "kvcached prefix-cache requires exactly one kv cache group"
            )
        kv_cache_group = kv_groups[0]
        kv_cache_spec = kv_cache_group.kv_cache_spec
        block_size = kv_cache_spec.block_size
        cell_size = kv_cache_spec.page_size_bytes // block_size // 2
        num_layers = len(kv_cache_config.kv_cache_tensors)
        enable_caching = getattr(self, "enable_caching", False)
        hash_block_size = getattr(
            self.block_pool, "hash_block_size", block_size
        )
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_world_size,
            )
            tp_size = int(get_tensor_model_parallel_world_size())
        except Exception:
            tp_size = 1
        from kvcached.integration.vllm import interfaces as kvi
        kvi.init_kvcached(tp_rank=0, tp_size=tp_size, is_worker=False)
        self.block_pool = PrefixCacheableElasticBlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            block_size=block_size,
            cell_size=cell_size,
            num_layers=num_layers,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
        )
        for manager in self.single_type_managers:
            manager.block_pool = self.block_pool
            manager._null_block = self.block_pool.null_block

    KVCacheCoordinator._setup_kvcached_coordinator = _setup_with_prefix_cache
