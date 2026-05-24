"""Unit tests for PrefixCacheableElasticBlockPool.

Tests the prefix caching, eviction queue, and memory-pressure logic
using a mock KVCacheManager -- no GPU, kvcached daemon, or model files
required.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _install_kvcached_mock():
    """Install a fake kvcached package so the import succeeds without
    the real kvcached library installed."""
    kvcached_mod = types.ModuleType("kvcached")
    kv_cache_manager_mod = types.ModuleType("kvcached.kv_cache_manager")
    integration_mod = types.ModuleType("kvcached.integration")
    vllm_mod = types.ModuleType("kvcached.integration.vllm")
    interfaces_mod = types.ModuleType("kvcached.integration.vllm.interfaces")

    class FakeKVCacheManager:
        """Simulates kvcached's KVCacheManager with a simple block pool."""

        def __init__(self, capacity: int):
            self._capacity = capacity
            self._allocated: set[int] = set()
            self._next_id = 0

        def alloc(self, n: int) -> list[int] | None:
            if n > self.available_size():
                return None
            ids = []
            for _ in range(n):
                while self._next_id in self._allocated:
                    self._next_id += 1
                self._allocated.add(self._next_id)
                ids.append(self._next_id)
                self._next_id += 1
            return ids

        def free(self, ids: list[int]) -> None:
            for bid in ids:
                self._allocated.discard(bid)

        def available_size(self) -> int:
            return self._capacity - len(self._allocated)

    kv_cache_manager_mod.KVCacheManager = FakeKVCacheManager

    _manager_store: dict[str, FakeKVCacheManager] = {}

    def get_kv_cache_manager(num_gpu_blocks, block_size, cell_size, num_layers):
        key = f"{num_gpu_blocks}_{block_size}_{cell_size}_{num_layers}"
        if key not in _manager_store:
            _manager_store[key] = FakeKVCacheManager(num_gpu_blocks)
        return _manager_store[key]

    def init_kvcached(**kwargs):
        pass

    interfaces_mod.get_kv_cache_manager = get_kv_cache_manager
    interfaces_mod.init_kvcached = init_kvcached

    vllm_mod.interfaces = interfaces_mod
    integration_mod.vllm = vllm_mod
    kvcached_mod.integration = integration_mod
    kvcached_mod.kv_cache_manager = kv_cache_manager_mod

    sys.modules["kvcached"] = kvcached_mod
    sys.modules["kvcached.kv_cache_manager"] = kv_cache_manager_mod
    sys.modules["kvcached.integration"] = integration_mod
    sys.modules["kvcached.integration.vllm"] = vllm_mod
    sys.modules["kvcached.integration.vllm.interfaces"] = interfaces_mod

    return _manager_store


_manager_store = _install_kvcached_mock()

from arctic_inference.vllm.kvcached.prefix_block_pool import (
    PrefixCacheableElasticBlockPool,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)

_hash_counter = 0

def _make_pool(
    num_blocks: int = 100,
    enable_caching: bool = True,
    pressure_threshold_pct: float = 0.10,
) -> PrefixCacheableElasticBlockPool:
    _manager_store.clear()
    return PrefixCacheableElasticBlockPool(
        num_gpu_blocks=num_blocks,
        block_size=16,
        cell_size=128,
        num_layers=32,
        enable_caching=enable_caching,
        hash_block_size=16,
        pressure_threshold_pct=pressure_threshold_pct,
    )


def _make_hash(seed: int) -> BlockHash:
    global _hash_counter
    _hash_counter += 1
    raw = f"hash_{seed}_{_hash_counter}".encode().ljust(16, b"\x00")
    return BlockHash(raw)


class TestBasicAllocationAndFree(unittest.TestCase):

    def test_alloc_and_free(self):
        pool = _make_pool(num_blocks=50, enable_caching=False)
        initial_free = pool.get_num_free_blocks()
        blocks = pool.get_new_blocks(5)
        self.assertEqual(len(blocks), 5)
        self.assertEqual(pool.get_num_free_blocks(), initial_free - 5)
        for b in blocks:
            self.assertEqual(b.ref_cnt, 1)
            self.assertFalse(b.is_null)

        pool.free_blocks(blocks)
        for b in blocks:
            self.assertEqual(b.ref_cnt, 0)
        self.assertEqual(pool.get_num_free_blocks(), initial_free)

    def test_null_block_exists(self):
        pool = _make_pool()
        self.assertIsNotNone(pool.null_block)
        self.assertTrue(pool.null_block.is_null)

    def test_alloc_exceeds_capacity_raises(self):
        pool = _make_pool(num_blocks=10, enable_caching=False)
        free = pool.get_num_free_blocks()
        with self.assertRaises(ValueError):
            pool.get_new_blocks(free + 1)

    def test_get_usage(self):
        pool = _make_pool(num_blocks=100, enable_caching=False)
        self.assertAlmostEqual(pool.get_usage(), 0.0, places=1)
        blocks = pool.get_new_blocks(49)
        usage = pool.get_usage()
        self.assertGreater(usage, 0.0)
        self.assertLess(usage, 1.0)
        pool.free_blocks(blocks)


class TestPrefixCaching(unittest.TestCase):

    def test_cache_and_retrieve(self):
        pool = _make_pool(enable_caching=True)
        blocks = pool.get_new_blocks(3)
        bh = _make_hash(1)

        key = make_block_hash_with_group_id(bh, 0)
        blocks[0].block_hash = key
        pool.cached_block_hash_to_block.insert(key, blocks[0])

        result = pool.get_cached_block(bh, [0])
        self.assertIsNotNone(result)
        self.assertEqual(result[0].block_id, blocks[0].block_id)

    def test_cache_miss_returns_none(self):
        pool = _make_pool(enable_caching=True)
        bh = _make_hash(99)
        result = pool.get_cached_block(bh, [0])
        self.assertIsNone(result)

    def test_caching_disabled_returns_none(self):
        pool = _make_pool(enable_caching=False)
        bh = _make_hash(1)
        result = pool.get_cached_block(bh, [0])
        self.assertIsNone(result)


class TestEvictionQueue(unittest.TestCase):

    def test_freed_blocks_enter_eviction_queue(self):
        pool = _make_pool(num_blocks=50, enable_caching=True,
                          pressure_threshold_pct=0.0)
        blocks = pool.get_new_blocks(5)
        pool.free_blocks(blocks)
        self.assertEqual(pool._eviction_queue.num_free_blocks, 5)

    def test_eviction_queue_reused_before_new_alloc(self):
        pool = _make_pool(num_blocks=50, enable_caching=True,
                          pressure_threshold_pct=0.0)
        blocks = pool.get_new_blocks(5)
        old_ids = {b.block_id for b in blocks}
        pool.free_blocks(blocks)
        new_blocks = pool.get_new_blocks(3)
        reused_ids = {b.block_id for b in new_blocks}
        self.assertTrue(reused_ids.issubset(old_ids))

    def test_touch_removes_from_eviction_queue(self):
        pool = _make_pool(num_blocks=50, enable_caching=True,
                          pressure_threshold_pct=0.0)
        blocks = pool.get_new_blocks(3)
        pool.free_blocks(blocks)
        self.assertEqual(pool._eviction_queue.num_free_blocks, 3)
        pool.touch(blocks[:1])
        self.assertEqual(pool._eviction_queue.num_free_blocks, 2)
        self.assertEqual(blocks[0].ref_cnt, 1)


class TestMemoryPressure(unittest.TestCase):

    def test_pressure_flushes_eviction_queue(self):
        pool = _make_pool(num_blocks=100, enable_caching=True,
                          pressure_threshold_pct=0.05)
        blocks = pool.get_new_blocks(10)
        pool.free_blocks(blocks)
        self.assertEqual(pool._eviction_queue.num_free_blocks, 10)

        heavy = pool.get_new_blocks(90)
        small = pool.get_new_blocks(3)
        pool.free_blocks(small)

        self.assertEqual(pool._eviction_queue.num_free_blocks, 0)
        pool.free_blocks(heavy)

    def test_no_pressure_keeps_eviction_queue(self):
        pool = _make_pool(num_blocks=100, enable_caching=True,
                          pressure_threshold_pct=0.01)
        blocks = pool.get_new_blocks(5)
        pool.free_blocks(blocks)
        self.assertGreater(pool._eviction_queue.num_free_blocks, 0)


class TestSetCachingEnabled(unittest.TestCase):

    def test_disable_flushes_queue(self):
        pool = _make_pool(num_blocks=50, enable_caching=True,
                          pressure_threshold_pct=0.0)
        blocks = pool.get_new_blocks(5)

        for i, b in enumerate(blocks):
            bh = _make_hash(i + 1000)
            key = make_block_hash_with_group_id(bh, 0)
            b.block_hash = key
            pool.cached_block_hash_to_block.insert(key, b)

        pool.free_blocks(blocks)
        self.assertEqual(pool._eviction_queue.num_free_blocks, 5)

        pool.set_caching_enabled(False)
        self.assertFalse(pool.enable_caching)
        self.assertEqual(pool._eviction_queue.num_free_blocks, 0)
        for b in blocks:
            self.assertIsNone(b.block_hash)

    def test_enable_after_disable(self):
        pool = _make_pool(enable_caching=False)
        pool.set_caching_enabled(True)
        self.assertTrue(pool.enable_caching)

    def test_free_without_caching_goes_straight_to_manager(self):
        pool = _make_pool(num_blocks=50, enable_caching=False)
        free_before = pool.get_num_free_blocks()
        blocks = pool.get_new_blocks(5)
        pool.free_blocks(blocks)
        self.assertEqual(pool.get_num_free_blocks(), free_before)
        self.assertEqual(pool._eviction_queue.num_free_blocks, 0)


class TestResetPrefixCache(unittest.TestCase):

    def test_reset_clears_hashes(self):
        pool = _make_pool(num_blocks=50, enable_caching=True,
                          pressure_threshold_pct=0.0)
        blocks = pool.get_new_blocks(3)

        for i, b in enumerate(blocks):
            bh = _make_hash(i + 2000)
            key = make_block_hash_with_group_id(bh, 0)
            b.block_hash = key
            pool.cached_block_hash_to_block.insert(key, b)

        pool.free_blocks(blocks)
        result = pool.reset_prefix_cache()
        self.assertTrue(result)
        for b in pool._block_cache.values():
            self.assertIsNone(b.block_hash)


class TestEvictBlocks(unittest.TestCase):

    def test_evict_specific_blocks(self):
        pool = _make_pool(num_blocks=50, enable_caching=True,
                          pressure_threshold_pct=0.0)
        blocks = pool.get_new_blocks(3)

        bh = _make_hash(42)
        key = make_block_hash_with_group_id(bh, 0)
        blocks[0].block_hash = key
        pool.cached_block_hash_to_block.insert(key, blocks[0])

        pool.evict_blocks({blocks[0].block_id})
        self.assertIsNone(blocks[0].block_hash)
        result = pool.get_cached_block(bh, [0])
        self.assertIsNone(result)


class TestTakeEvents(unittest.TestCase):

    def test_returns_empty_list(self):
        pool = _make_pool()
        self.assertEqual(pool.take_events(), [])


class TestBlockPoolInterfaceCompleteness(unittest.TestCase):
    """Verify that PrefixCacheableElasticBlockPool implements all methods
    from vLLM's BlockPool."""

    def test_has_all_blockpool_methods(self):
        from vllm.v1.core.block_pool import BlockPool
        bp_methods = {
            m for m in dir(BlockPool)
            if not m.startswith("_") and callable(getattr(BlockPool, m))
        }
        pool_methods = {
            m for m in dir(PrefixCacheableElasticBlockPool)
            if not m.startswith("_") and callable(getattr(PrefixCacheableElasticBlockPool, m))
        }
        missing = bp_methods - pool_methods
        self.assertEqual(missing, set(), f"Missing BlockPool methods: {missing}")

    def test_has_required_attributes(self):
        pool = _make_pool()
        for attr in ("num_gpu_blocks", "enable_caching", "hash_block_size",
                      "null_block", "cached_block_hash_to_block",
                      "enable_kv_cache_events", "kv_event_queue",
                      "metrics_collector"):
            self.assertTrue(hasattr(pool, attr), f"Missing attribute: {attr}")


if __name__ == "__main__":
    unittest.main()
