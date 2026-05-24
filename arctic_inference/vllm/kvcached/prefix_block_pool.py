# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PrefixCacheableElasticBlockPool -- a drop-in replacement for kvcached's
``ElasticBlockPool`` that adds prefix caching with automatic memory-
pressure-driven eviction.

Physical block management is delegated to kvcached's ``KVCacheManager``
(CUDA VMM page alloc/free).  Prefix caching state (hash table, LRU
eviction queue, ref counts) is maintained on top.

Freed blocks stay in the eviction queue with their physical pages
mapped, enabling fast prefix reuse.  When kvcached reports memory
pressure (``available_size()`` below a configurable threshold), the
eviction queue is automatically flushed, releasing physical pages for
other processes sharing the GPU.  Prefix caching can also be toggled
explicitly at runtime via ``set_caching_enabled()``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

from vllm.v1.core.block_pool import BlockHashToBlockMap
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from kvcached.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


class PrefixCacheableElasticBlockPool:
    """BlockPool backed by kvcached with optional prefix caching.

    This class exposes the same interface as ``vllm.v1.core.block_pool.BlockPool``
    so that ``KVCacheCoordinator`` and ``SingleTypeKVCacheManager`` can use it
    transparently.

    Args:
        num_gpu_blocks: Maximum number of logical blocks (for metrics).
        block_size: Number of tokens per block.
        cell_size: Per-block cell size for kvcached.
        num_layers: Number of KV cache layers.
        enable_caching: Initial state of prefix caching.
        hash_block_size: Block size used for hash computation.
        enable_kv_cache_events: Whether to emit KV cache events.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        block_size: int,
        cell_size: int,
        num_layers: int,
        enable_caching: bool,
        hash_block_size: int = 0,
        enable_kv_cache_events: bool = False,
        pressure_threshold_pct: float = 0.10,
    ) -> None:
        from kvcached.integration.vllm.interfaces import get_kv_cache_manager

        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size or block_size
        self.enable_kv_cache_events = False
        self.kv_event_queue: list = []
        self._pressure_threshold = max(
            int(num_gpu_blocks * pressure_threshold_pct), 32,
        )

        self.kv_cache_manager: KVCacheManager = get_kv_cache_manager(
            num_gpu_blocks, block_size, cell_size, num_layers,
        )

        # block_id -> KVCacheBlock for all blocks we have ever allocated.
        self._block_cache: dict[int, KVCacheBlock] = {}

        # Eviction queue (doubly linked list) for blocks with ref_cnt == 0
        # that still have their physical pages mapped.
        self._eviction_blocks: list[KVCacheBlock] = []
        self._eviction_queue = FreeKVCacheBlockQueue(self._eviction_blocks)

        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Allocate a null block.
        null_ids = self.kv_cache_manager.alloc(1)
        assert null_ids is not None and len(null_ids) == 1
        self.null_block = KVCacheBlock(null_ids[0])
        self.null_block.is_null = True
        self._block_cache[null_ids[0]] = self.null_block

        self.metrics_collector = None

    # ------------------------------------------------------------------
    # BlockPool interface
    # ------------------------------------------------------------------

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int],
    ) -> Optional[list[KVCacheBlock]]:
        if not self.enable_caching:
            return None
        cached_blocks: list[KVCacheBlock] = []
        for group_id in kv_cache_group_ids:
            key = make_block_hash_with_group_id(block_hash, group_id)
            block = self.cached_block_hash_to_block.get_one_block(key)
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        if not self.enable_caching:
            return
        if num_cached_blocks >= num_full_blocks:
            return

        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks

        if block_size == self.hash_block_size:
            block_hashes: BlockHashList = request.block_hashes
        else:
            assert block_size % self.hash_block_size == 0
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size,
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        for i, blk in enumerate(new_full_blocks):
            if blk.is_null:
                continue
            if blk.block_hash is not None:
                continue
            bh = new_block_hashes[i]
            key = make_block_hash_with_group_id(bh, kv_cache_group_id)
            blk.block_hash = key
            self.cached_block_hash_to_block.insert(key, blk)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool"
            )

        ret: list[KVCacheBlock] = []
        remaining = num_blocks

        if self.enable_caching:
            while remaining > 0 and self._eviction_queue.num_free_blocks > 0:
                block = self._eviction_queue.popleft()
                self._maybe_evict_cached_block(block)
                block.ref_cnt = 1
                ret.append(block)
                remaining -= 1

        if remaining > 0:
            new_ids = self.kv_cache_manager.alloc(remaining)
            if new_ids is None or len(new_ids) < remaining:
                raise ValueError(
                    f"kvcached: failed to allocate {remaining} blocks"
                )
            for bid in new_ids:
                block = self._block_cache.get(bid)
                if block is None:
                    block = KVCacheBlock(bid)
                    self._block_cache[bid] = block
                else:
                    block.reset_hash()
                block.ref_cnt = 1
                ret.append(block)

        return ret

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1

        freed = [
            b for b in blocks_list if b.ref_cnt == 0 and not b.is_null
        ]
        if not freed:
            return

        if self.enable_caching:
            self._eviction_queue.append_n(freed)
            self._maybe_relieve_pressure()
        else:
            ids = [b.block_id for b in freed]
            for b in freed:
                self._evict_hash(b)
            self.kv_cache_manager.free(ids)

    def touch(self, blocks: Any) -> None:
        if not self.enable_caching:
            return
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self._eviction_queue.remove(block)
            block.ref_cnt += 1

    def reset_prefix_cache(self) -> bool:
        self._flush_eviction_queue()
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        for block in self._block_cache.values():
            block.reset_hash()
        logger.info("kvcached: prefix cache reset")
        return True

    def get_num_free_blocks(self) -> int:
        eviction_count = (
            self._eviction_queue.num_free_blocks if self.enable_caching else 0
        )
        return eviction_count + self.kv_cache_manager.available_size()

    def get_usage(self) -> float:
        total = self.num_gpu_blocks - 1
        if total <= 0:
            return 0.0
        return 1.0 - (self.get_num_free_blocks() / total)

    def take_events(self) -> list:
        return []

    def evict_blocks(self, block_ids: set[int]) -> None:
        for bid in block_ids:
            block = self._block_cache.get(bid)
            if block is not None:
                self._maybe_evict_cached_block(block)

    # ------------------------------------------------------------------
    # Runtime toggle
    # ------------------------------------------------------------------

    def set_caching_enabled(self, enabled: bool) -> None:
        """Toggle prefix caching at runtime.

        When turning OFF, all eviction-candidate blocks are freed back to
        kvcached (physical pages unmapped, memory shared with other
        processes).  When turning ON, the cache starts empty.
        """
        if self.enable_caching and not enabled:
            self._flush_eviction_queue()
            self.cached_block_hash_to_block = BlockHashToBlockMap()
            for block in self._block_cache.values():
                if not block.is_null:
                    block.reset_hash()
            logger.info("kvcached: prefix caching disabled, eviction queue flushed")
        elif not self.enable_caching and enabled:
            logger.info("kvcached: prefix caching enabled")
        self.enable_caching = enabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_relieve_pressure(self) -> None:
        """Flush the eviction queue if kvcached is under memory pressure.

        When another model on the same GPU is consuming physical pages,
        available_size() drops.  If it falls below the threshold we
        release all cached blocks so kvcached can hand those pages to the
        other process.
        """
        if self._eviction_queue.num_free_blocks == 0:
            return
        available = self.kv_cache_manager.available_size()
        if available < self._pressure_threshold:
            n = self._eviction_queue.num_free_blocks
            self._flush_eviction_queue()
            logger.info(
                "kvcached: flushed %d eviction blocks due to memory "
                "pressure (available=%d < threshold=%d)",
                n, available, self._pressure_threshold,
            )

    def _flush_eviction_queue(self) -> None:
        """Free all blocks in the eviction queue back to kvcached."""
        ids_to_free: list[int] = []
        while self._eviction_queue.num_free_blocks > 0:
            block = self._eviction_queue.popleft()
            self._evict_hash(block)
            ids_to_free.append(block.block_id)
        if ids_to_free:
            self.kv_cache_manager.free(ids_to_free)

    def _evict_hash(self, block: KVCacheBlock) -> None:
        """Remove a block from the hash table if present."""
        bh = block.block_hash
        if bh is not None:
            self.cached_block_hash_to_block.pop(bh, block.block_id)
            block.reset_hash()

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        bh = block.block_hash
        if bh is None:
            return False
        if self.cached_block_hash_to_block.pop(bh, block.block_id) is None:
            return False
        block.reset_hash()
        return True
