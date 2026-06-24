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

"""Tests for prefix-affinity routing in the scheduler."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from arctic_inference.server.scheduler import (
    Scheduler,
    WorkerState,
    _Request,
    _compute_prefix_hash,
    least_loaded_routing,
    prefix_affinity_routing,
    strict_affinity_routing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workers(n: int, concurrency: int = 64) -> list[WorkerState]:
    return [
        WorkerState(handle=MagicMock(), concurrency_limit=concurrency)
        for _ in range(n)
    ]


def _make_request(prompt: str | list[int] = "hello", prefix_hash: int | None = None) -> _Request:
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    loop.close()
    return _Request(
        id=0,
        prompt=prompt,
        sampling_params={},
        future=future,
        prefix_hash=prefix_hash,
    )


# ---------------------------------------------------------------------------
# _compute_prefix_hash
# ---------------------------------------------------------------------------

class TestComputePrefixHash:
    def test_identical_strings_same_hash(self):
        assert _compute_prefix_hash("abc") == _compute_prefix_hash("abc")

    def test_identical_token_lists_same_hash(self):
        assert _compute_prefix_hash([1, 2, 3]) == _compute_prefix_hash([1, 2, 3])

    def test_full_hash_differs_with_different_suffix(self):
        base = "A" * 100
        a = base + "X"
        b = base + "Y"
        assert _compute_prefix_hash(a) != _compute_prefix_hash(b)

    def test_different_prefix_different_hash(self):
        assert _compute_prefix_hash("hello world") != _compute_prefix_hash("goodbye world")

    def test_token_full_hash_differs_with_different_suffix(self):
        base = list(range(100))
        a = base + [999]
        b = base + [888]
        assert _compute_prefix_hash(a) != _compute_prefix_hash(b)

    def test_empty_string(self):
        h = _compute_prefix_hash("")
        assert isinstance(h, int)

    def test_single_char(self):
        h = _compute_prefix_hash("x")
        assert isinstance(h, int)

    def test_single_token(self):
        h = _compute_prefix_hash([42])
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# prefix_affinity_routing
# ---------------------------------------------------------------------------

class TestPrefixAffinityRouting:
    def test_same_hash_routes_to_same_worker(self):
        """Requests with the same prefix_hash should land on the same worker."""
        workers = _make_workers(4)
        r1 = _make_request(prefix_hash=42)
        r2 = _make_request(prefix_hash=42)
        assert prefix_affinity_routing(r1, workers) == prefix_affinity_routing(r2, workers)

    def test_different_hash_can_route_differently(self):
        """Different hashes should (with enough workers) spread across workers."""
        workers = _make_workers(8)
        destinations = set()
        for h in range(100):
            r = _make_request(prefix_hash=h)
            destinations.add(prefix_affinity_routing(r, workers))
        assert len(destinations) > 1

    def test_no_hash_falls_back_to_least_loaded(self):
        """When prefix_hash is None, behavior matches least_loaded_routing."""
        workers = _make_workers(4)
        workers[0].active_requests = 10
        r = _make_request(prefix_hash=None)
        idx = prefix_affinity_routing(r, workers)
        expected = least_loaded_routing(r, workers)
        assert idx == expected

    def test_overloaded_preferred_spills_to_next(self):
        """When the preferred worker exceeds load threshold, the next in
        the hash ring is selected."""
        workers = _make_workers(4, concurrency=100)
        r = _make_request(prefix_hash=0)
        preferred = prefix_affinity_routing(r, workers)

        workers[preferred].active_requests = 90
        second = prefix_affinity_routing(r, workers)
        assert second != preferred

    def test_all_overloaded_falls_back_to_least_loaded(self):
        """When every candidate is above the load threshold, fall back to
        least_loaded_routing."""
        workers = _make_workers(3, concurrency=100)
        for ws in workers:
            ws.active_requests = 95
        workers[1].active_requests = 86

        r = _make_request(prefix_hash=7)
        idx = prefix_affinity_routing(r, workers)
        expected = least_loaded_routing(r, workers)
        assert idx == expected

    def test_no_available_workers_falls_back(self):
        """When all workers are unavailable, falls back to least_loaded_routing
        which itself falls back to all workers."""
        workers = _make_workers(2)
        for ws in workers:
            ws.available = False
        r = _make_request(prefix_hash=5)
        idx = prefix_affinity_routing(r, workers)
        expected = least_loaded_routing(r, workers)
        assert idx == expected

    def test_single_worker(self):
        workers = _make_workers(1)
        r = _make_request(prefix_hash=99)
        assert prefix_affinity_routing(r, workers) == 0

    def test_grpo_group_colocates(self):
        """Simulate GRPO: 8 identical prompts should all pick the same worker."""
        workers = _make_workers(4)
        prompt = "System: You are a math tutor.\nUser: Solve x+2=5"
        h = _compute_prefix_hash(prompt)
        indices = [
            prefix_affinity_routing(_make_request(prefix_hash=h), workers)
            for _ in range(8)
        ]
        assert len(set(indices)) == 1

    def test_different_groups_spread(self):
        """Different GRPO groups (different prompts) should spread."""
        workers = _make_workers(8)
        targets = set()
        for i in range(20):
            prompt = f"System prompt variant {i}. User: question"
            h = _compute_prefix_hash(prompt)
            idx = prefix_affinity_routing(_make_request(prefix_hash=h), workers)
            targets.add(idx)
        assert len(targets) > 1


# ---------------------------------------------------------------------------
# strict_affinity_routing
# ---------------------------------------------------------------------------

class TestStrictAffinityRouting:
    def test_same_hash_routes_to_same_worker(self):
        workers = _make_workers(4)
        r1 = _make_request(prefix_hash=42)
        r2 = _make_request(prefix_hash=42)
        assert strict_affinity_routing(r1, workers) == strict_affinity_routing(r2, workers)

    def test_overloaded_preferred_does_not_spill(self):
        """Strict mode pins to the keyed worker even when it's overloaded —
        the scheduler's outer back-off loop is responsible for waiting, not
        the routing function."""
        workers = _make_workers(4, concurrency=100)
        r = _make_request(prefix_hash=0)
        preferred = strict_affinity_routing(r, workers)
        workers[preferred].active_requests = 95  # past prefix's 85% threshold
        assert strict_affinity_routing(r, workers) == preferred

    def test_strict_and_soft_agree_when_underloaded(self):
        """When the preferred worker is below the load threshold, strict and
        prefix_affinity should pick the same worker."""
        workers = _make_workers(4)
        for h in range(20):
            r = _make_request(prefix_hash=h)
            assert strict_affinity_routing(r, workers) == prefix_affinity_routing(r, workers)

    def test_no_hash_falls_back_to_least_loaded(self):
        workers = _make_workers(3)
        workers[0].active_requests = 5
        r = _make_request(prefix_hash=None)
        assert strict_affinity_routing(r, workers) == least_loaded_routing(r, workers)

    def test_no_available_workers_falls_back(self):
        workers = _make_workers(2)
        for ws in workers:
            ws.available = False
        r = _make_request(prefix_hash=5)
        assert strict_affinity_routing(r, workers) == least_loaded_routing(r, workers)


# ---------------------------------------------------------------------------
# Scheduler.submit with routing_key
# ---------------------------------------------------------------------------


def _capture_requests(sch: Scheduler) -> list:
    """Replace _process_request with a no-op capture so we can inspect the
    _Request objects submit() builds, without racing real worker dispatch."""
    captured: list = []

    async def fake_process(req):
        captured.append(req)

    sch._process_request = fake_process  # type: ignore[method-assign]
    return captured


class TestSchedulerRoutingKey:
    def test_routing_key_replaces_prompt_hash(self):
        """Two prompts with different content but the same routing_key must
        produce equal prefix_hash values, so they hash to the same worker."""

        async def run() -> None:
            workers = [MagicMock() for _ in range(2)]
            sch = Scheduler(workers, enable_prefix_hash=True)
            try:
                captured = _capture_requests(sch)
                sch.submit("turn-1 prompt", {}, routing_key="g42")
                sch.submit("turn-2 very different prompt", {}, routing_key="g42")
                await asyncio.sleep(0)  # let create_task run capture
                assert len(captured) == 2
                assert captured[0].prefix_hash == captured[1].prefix_hash == hash("g42")
                # And critically, the prompt-derived hash differs — proving
                # routing_key overrode the prompt hash.
                assert _compute_prefix_hash("turn-1 prompt") != \
                       _compute_prefix_hash("turn-2 very different prompt")
            finally:
                await sch.shutdown()

        asyncio.run(run())

    def test_no_routing_key_uses_prompt_hash(self):
        async def run() -> None:
            workers = [MagicMock() for _ in range(2)]
            sch = Scheduler(workers, enable_prefix_hash=True)
            try:
                captured = _capture_requests(sch)
                sch.submit("abc", {})
                await asyncio.sleep(0)
                assert captured[0].prefix_hash == _compute_prefix_hash("abc")
                assert captured[0].strict is False
            finally:
                await sch.shutdown()

        asyncio.run(run())

    def test_strict_flag_propagates_to_request(self):
        async def run() -> None:
            workers = [MagicMock() for _ in range(2)]
            sch = Scheduler(workers)
            try:
                captured = _capture_requests(sch)
                sch.submit("p", {}, routing_key="k", strict=True)
                await asyncio.sleep(0)
                assert captured[0].strict is True
                assert captured[0].prefix_hash == hash("k")
            finally:
                await sch.shutdown()

        asyncio.run(run())

    def test_per_request_keys_produce_independent_hashes(self):
        """Different routing keys on different prompts each produce their own
        prefix_hash so the scheduler can spread them across replicas."""

        async def run() -> None:
            workers = [MagicMock() for _ in range(2)]
            sch = Scheduler(workers)
            try:
                captured = _capture_requests(sch)
                for prompt, key in zip(["a", "b", "c"], ["k0", "k1", "k0"]):
                    sch.submit(prompt, {}, routing_key=key)
                await asyncio.sleep(0)
                assert [r.prefix_hash for r in captured] == [
                    hash("k0"), hash("k1"), hash("k0"),
                ]
            finally:
                await sch.shutdown()

        asyncio.run(run())
