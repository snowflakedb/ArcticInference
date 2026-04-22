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
"""Numerical and wiring tests for MultiCacheDynamicNTKRotaryEmbedding.

These tests run on the GPU because that is the deployment target: the
forward re-uses vLLM's fused ``ops.rotary_embedding`` CUDA kernel, and
we want the tests to exercise the same device, dtype, and kernel paths
that production will.  Tests that depend on ``get_rope`` wiring are
device-independent and stay on CPU.

The ``forward_cuda`` path is a single elementwise add followed by the
same CUDA rope kernel the rest of vLLM uses.  That's what makes it
safe under CUDA graphs.  A few of the tests below additionally
exercise ``torch.compile(fullgraph=True)`` as a static proxy for
"no graph breaks", plus a real CUDA-graph capture/replay test for the
end-to-end safety property.
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

pytest.importorskip("vllm")


if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for multi-cache RoPE tests (they run on GPU "
        "to match the production code path).",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _set_default_dtype():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture(autouse=True)
def _default_vllm_config():
    """Provide a default ``VllmConfig`` for every test.

    ``RotaryEmbedding`` is a ``CustomOp`` whose ``__init__`` reads
    ``get_current_vllm_config()`` to pick the platform forward.  Outside
    a real engine that context is unset, so direct construction in tests
    raises ``AssertionError: Current vLLM config is not set.``.  This
    mirrors the ``default_vllm_config`` fixture in vLLM's own test suite.
    """
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_multi_cache(
    *,
    head_size: int = 64,
    rotary_dim: int = 64,
    max_position_embeddings: int = 2048,
    base: float = 10000.0,
    factors=None,
    is_neox_style: bool = True,
    dtype: torch.dtype = torch.float32,
    max_num_batched_tokens: int | None = None,
    device: torch.device | None = None,
):
    from arctic_inference.vllm.rope.multi_cache_ntk import (
        MultiCacheDynamicNTKRotaryEmbedding,
    )

    mod = MultiCacheDynamicNTKRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
        factors=factors,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    if device is not None:
        mod = mod.to(device)
    return mod


def _make_static_dynamic_ntk(
    *,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    scaling_factor: float,
    dtype: torch.dtype,
    device: torch.device,
):
    """Build vLLM's static :class:`DynamicNTKScalingRotaryEmbedding` as a
    reference.  Within any bucket our multi-cache must match this class
    bit-for-bit at the corresponding factor."""
    from vllm.model_executor.layers.rotary_embedding.dynamic_ntk_scaling_rope import (
        DynamicNTKScalingRotaryEmbedding,
    )

    mod = DynamicNTKScalingRotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        scaling_factor=scaling_factor,
        dtype=dtype,
    )
    return mod.to(device)


def _make_inputs(
    *,
    num_tokens: int,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_size: int = 64,
    seq_len_start: int = 0,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen_device = device or torch.device("cpu")
    gen = torch.Generator(device=gen_device).manual_seed(seed)
    positions = torch.arange(
        seq_len_start, seq_len_start + num_tokens,
        dtype=torch.long, device=gen_device,
    )
    query = torch.randn(
        num_tokens, num_heads * head_size, dtype=dtype,
        device=gen_device, generator=gen,
    )
    key = torch.randn(
        num_tokens, num_kv_heads * head_size, dtype=dtype,
        device=gen_device, generator=gen,
    )
    return positions, query, key


# --------------------------------------------------------------------------
# Cache construction
# --------------------------------------------------------------------------


def test_per_factor_caches_match_vllm_static_dynamic_ntk(device):
    """For each configured factor F, the slice of the concatenated cache
    belonging to F must be bit-identical to
    :class:`DynamicNTKScalingRotaryEmbedding(factor=F)`'s cache.
    """
    head_size = 64
    rotary_dim = 64
    max_pos = 2048
    base = 10000.0
    factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    multi = _make_multi_cache(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos, base=base,
        factors=factors, device=device,
    )
    multi_cache = multi.cos_sin_cache

    for F in factors:
        static = _make_static_dynamic_ntk(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_pos, base=base,
            is_neox_style=True, scaling_factor=F,
            dtype=torch.float32, device=device,
        )
        offset = multi.factor_offset(F)
        length = int(math.ceil(F * max_pos))
        multi_slice = multi_cache[offset: offset + length]
        assert multi_slice.shape == static.cos_sin_cache.shape, (
            f"cache shape mismatch for F={F}: "
            f"{tuple(multi_slice.shape)} vs {tuple(static.cos_sin_cache.shape)}"
        )
        torch.testing.assert_close(
            multi_slice, static.cos_sin_cache, rtol=0, atol=0,
        )


def test_cache_layout_offsets_are_cumulative(device):
    """``_per_factor_offsets`` must equal the prefix sums of the
    per-factor cache sizes, and the total cache length must equal the
    concatenated cache's length."""
    max_pos = 2048
    factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    multi = _make_multi_cache(
        max_position_embeddings=max_pos, factors=factors, device=device,
    )

    expected_sizes = [int(math.ceil(F * max_pos)) for F in factors]
    expected_offsets = [0]
    for sz in expected_sizes[:-1]:
        expected_offsets.append(expected_offsets[-1] + sz)

    assert multi._per_factor_offsets == expected_offsets
    assert multi._total_cache_len == sum(expected_sizes)
    assert multi.cos_sin_cache.shape[0] == sum(expected_sizes)


def test_factors_are_sorted_and_deduped(device):
    """Construction must sort + dedupe the factor list.  Downstream
    routing assumes strictly increasing unique factors."""
    multi = _make_multi_cache(
        factors=[4.0, 2.0, 4.0, 1.0, 3.0], device=device,
    )
    assert multi.factors == [1.0, 2.0, 3.0, 4.0]


def test_invalid_factors_raise():
    """Factors < 1.0 don't make sense (there's no factor smaller than
    the unscaled baseline) and must be rejected."""
    with pytest.raises(ValueError, match=">=.*1.0"):
        _make_multi_cache(factors=[0.5, 1.0])
    with pytest.raises(ValueError, match=">=.*1.0"):
        _make_multi_cache(factors=[])


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def test_bucket_for_seq_len_picks_smallest_covering_factor(device):
    """Scalar routing function:
    ``bucket = clamp(ceil(seq_len / m), 1, k) - 1``."""
    m = 2048
    multi = _make_multi_cache(
        max_position_embeddings=m,
        factors=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        device=device,
    )
    assert multi.bucket_for_seq_len(0) == 0  # degenerate safe-default
    assert multi.bucket_for_seq_len(1) == 0
    assert multi.bucket_for_seq_len(m) == 0  # exactly m fits factor 1
    assert multi.bucket_for_seq_len(m + 1) == 1
    assert multi.bucket_for_seq_len(2 * m) == 1
    assert multi.bucket_for_seq_len(2 * m + 1) == 2
    assert multi.bucket_for_seq_len(6 * m) == 5
    assert multi.bucket_for_seq_len(6 * m + 1) == 5  # clamped to last
    assert multi.bucket_for_seq_len(100 * m) == 5


def test_update_runtime_seq_lens_translates_to_offsets(device):
    """The vectorized update must produce the same offsets the scalar
    routing function returns."""
    m = 2048
    factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    multi = _make_multi_cache(
        max_position_embeddings=m, factors=factors,
        max_num_batched_tokens=64, device=device,
    )

    # Hand-picked seq_lens that exercise every bucket, including the
    # boundary + overflow cases.
    seq_lens_cpu = [
        0, 1, 1024, m,
        m + 1, m * 2,
        m * 2 + 1, m * 3,
        m * 3 + 1, m * 4,
        m * 4 + 1, m * 5,
        m * 5 + 1, m * 6, m * 6 + 1, 10_000_000,
    ]
    seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int32, device=device)
    multi.update_runtime_seq_lens(seq_lens)

    expected = [
        multi.factor_offset(factors[multi.bucket_for_seq_len(int(s))])
        for s in seq_lens_cpu
    ]
    got = multi.runtime_bucket_offsets[: len(seq_lens_cpu)].tolist()
    assert got == expected


def test_update_runtime_seq_lens_is_in_place(device):
    """The update must preserve the buffer's storage so CUDA graph
    replay sees new values without a re-capture."""
    multi = _make_multi_cache(
        max_num_batched_tokens=256, device=device,
    )
    storage_ptr_before = multi.runtime_bucket_offsets.data_ptr()
    buffer_id_before = id(multi.runtime_bucket_offsets)

    multi.update_runtime_seq_lens(
        torch.arange(1, 33, dtype=torch.int32, device=device),
    )

    assert multi.runtime_bucket_offsets.data_ptr() == storage_ptr_before
    assert id(multi.runtime_bucket_offsets) == buffer_id_before


def test_update_runtime_seq_lens_rejects_oversized_input(device):
    """Guard against silent truncation: too many tokens is an error."""
    multi = _make_multi_cache(max_num_batched_tokens=32, device=device)
    with pytest.raises(ValueError, match="runtime buffer was sized"):
        multi.update_runtime_seq_lens(
            torch.ones(64, dtype=torch.int32, device=device),
        )


def test_update_runtime_seq_lens_handles_empty_batch(device):
    """``seq_lens`` of shape [0] is a no-op (doesn't touch the buffer)."""
    multi = _make_multi_cache(
        max_num_batched_tokens=32, device=device,
    )
    before = multi.runtime_bucket_offsets.clone()
    multi.update_runtime_seq_lens(
        torch.empty(0, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(multi.runtime_bucket_offsets, before)


# --------------------------------------------------------------------------
# Forward correctness vs static DynamicNTK reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [1.0, 2.0, 4.0, 6.0])
def test_forward_single_bucket_matches_static_rope(device, factor):
    """All-tokens-same-seq_len batch: the output must match vLLM's
    :class:`DynamicNTKScalingRotaryEmbedding(factor=factor)` exactly.

    This is the core invariant: "within a bucket, we *are* static rope"."""
    head_size, rotary_dim = 64, 64
    max_pos = 2048
    base = 10000.0
    factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    num_tokens = 32
    # Pick a seq_len inside the bucket range.  For factor=F, the bucket
    # covers ((F-1)*m, F*m].  We aim mid-range: (F - 0.5) * m.
    seq_len = max(1, int((factor - 0.5) * max_pos))

    multi = _make_multi_cache(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos, base=base,
        factors=factors, max_num_batched_tokens=num_tokens,
        device=device,
    )
    static = _make_static_dynamic_ntk(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos, base=base,
        is_neox_style=True, scaling_factor=factor,
        dtype=torch.float32, device=device,
    )

    positions, q, k = _make_inputs(
        num_tokens=num_tokens, head_size=head_size,
        seq_len_start=max(0, seq_len - num_tokens), device=device,
    )

    multi.update_runtime_seq_lens(
        torch.full(
            (num_tokens,), seq_len, dtype=torch.int32, device=device,
        ),
    )
    got_q, got_k = multi.forward_native(positions, q.clone(), k.clone())
    ref_q, ref_k = static.forward_native(positions, q.clone(), k.clone())

    torch.testing.assert_close(got_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=0)


def test_forward_mixed_batch_routes_each_request_correctly(device):
    """A batch mixing two requests with different seq_lens (so different
    buckets) must rotate each request's slice of the output under its
    own bucket's static rope cache."""
    head_size, rotary_dim = 64, 64
    max_pos = 2048
    base = 10000.0
    factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    num_a, num_b = 17, 11
    seq_len_a = 1800  # bucket 0 (factor 1)
    seq_len_b = 7500  # bucket 3 (factor 4)

    multi = _make_multi_cache(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos, base=base,
        factors=factors, max_num_batched_tokens=num_a + num_b,
        device=device,
    )

    pos_a, q_a, k_a = _make_inputs(
        num_tokens=num_a, head_size=head_size,
        seq_len_start=0, seed=0, device=device,
    )
    pos_b, q_b, k_b = _make_inputs(
        num_tokens=num_b, head_size=head_size,
        seq_len_start=seq_len_b - num_b, seed=42, device=device,
    )

    positions = torch.cat([pos_a, pos_b], dim=0)
    query = torch.cat([q_a, q_b], dim=0)
    key = torch.cat([k_a, k_b], dim=0)
    per_token = torch.cat([
        torch.full((num_a,), seq_len_a, dtype=torch.int32, device=device),
        torch.full((num_b,), seq_len_b, dtype=torch.int32, device=device),
    ])

    multi.update_runtime_seq_lens(per_token)
    got_q, got_k = multi.forward_native(
        positions, query.clone(), key.clone(),
    )

    # Build per-bucket reference with the appropriate static factor.
    static_a = _make_static_dynamic_ntk(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos, base=base,
        is_neox_style=True,
        scaling_factor=factors[multi.bucket_for_seq_len(seq_len_a)],
        dtype=torch.float32, device=device,
    )
    static_b = _make_static_dynamic_ntk(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos, base=base,
        is_neox_style=True,
        scaling_factor=factors[multi.bucket_for_seq_len(seq_len_b)],
        dtype=torch.float32, device=device,
    )
    ref_q_a, ref_k_a = static_a.forward_native(
        pos_a, q_a.clone(), k_a.clone(),
    )
    ref_q_b, ref_k_b = static_b.forward_native(
        pos_b, q_b.clone(), k_b.clone(),
    )

    torch.testing.assert_close(got_q[:num_a], ref_q_a, rtol=0, atol=0)
    torch.testing.assert_close(got_k[:num_a], ref_k_a, rtol=0, atol=0)
    torch.testing.assert_close(got_q[num_a:], ref_q_b, rtol=0, atol=0)
    torch.testing.assert_close(got_k[num_a:], ref_k_b, rtol=0, atol=0)


def test_forward_cuda_matches_forward_native(device):
    """The fused CUDA path and the PyTorch path must produce the same
    output; that's the contract that lets us use the kernel with no
    further adaptation.  Use a mid-range seq_len and a non-trivial
    bucket to exercise a real offset."""
    head_size, rotary_dim = 64, 64
    max_pos = 2048
    num_tokens = 32
    seq_len = 5000  # bucket 2 (factor 3)

    multi = _make_multi_cache(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos,
        max_num_batched_tokens=num_tokens, device=device,
    )
    positions, q, k = _make_inputs(
        num_tokens=num_tokens, head_size=head_size,
        seq_len_start=seq_len - num_tokens, device=device,
    )

    multi.update_runtime_seq_lens(
        torch.full(
            (num_tokens,), seq_len, dtype=torch.int32, device=device,
        ),
    )

    q_native = q.clone()
    k_native = k.clone()
    got_q_native, got_k_native = multi.forward_native(
        positions, q_native, k_native,
    )

    q_cuda = q.clone()
    k_cuda = k.clone()
    # forward_cuda writes in place, so capture post-call tensors.
    got_q_cuda, got_k_cuda = multi.forward_cuda(
        positions, q_cuda, k_cuda,
    )

    torch.testing.assert_close(
        got_q_native, got_q_cuda, rtol=1e-4, atol=1e-5,
    )
    torch.testing.assert_close(
        got_k_native, got_k_cuda, rtol=1e-4, atol=1e-5,
    )


# --------------------------------------------------------------------------
# Buffer sizing + init state
# --------------------------------------------------------------------------


def test_runtime_buffer_sized_by_max_num_batched_tokens():
    multi = _make_multi_cache(
        max_position_embeddings=2048,
        max_num_batched_tokens=4096,
    )
    assert multi.runtime_bucket_offsets.shape == (4096,)


def test_runtime_buffer_has_conservative_default_when_nothing_specified():
    multi = _make_multi_cache(
        max_position_embeddings=2048,
        max_num_batched_tokens=None,
    )
    # Conservative default >= 1 and matches the module-private constant.
    from arctic_inference.vllm.rope import multi_cache_ntk as mcn

    assert multi.runtime_bucket_offsets.shape[0] == mcn._DEFAULT_BUFFER_SIZE


def test_runtime_buffer_initialized_to_factor_one_offset():
    """Before any update, all offsets should be 0 -- i.e. all tokens
    route to the unscaled factor-1 cache.  This is the safe regime for
    profile / dummy runs that bypass the model runner."""
    multi = _make_multi_cache(
        max_position_embeddings=2048, factors=[1.0, 2.0, 4.0],
        max_num_batched_tokens=128,
    )
    assert (multi.runtime_bucket_offsets == 0).all()


# --------------------------------------------------------------------------
# Graph-compatibility smoke tests
# --------------------------------------------------------------------------


def test_forward_compiles_without_graph_breaks(device):
    """``torch.compile(fullgraph=True)`` traces ``forward_native`` without
    hitting Python-level control flow on tensor values.  This is a
    static proxy for "safe to capture in a CUDA graph"."""
    pytest.importorskip("torch._dynamo")

    head_size, rotary_dim = 64, 64
    max_pos = 2048
    num_tokens = 16

    multi = _make_multi_cache(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos,
        max_num_batched_tokens=num_tokens, device=device,
    )
    multi.update_runtime_seq_lens(
        torch.full(
            (num_tokens,), 5000, dtype=torch.int32, device=device,
        ),
    )

    positions, q, k = _make_inputs(
        num_tokens=num_tokens, head_size=head_size,
        seq_len_start=4000, device=device,
    )

    import torch._dynamo as dynamo

    dynamo.reset()
    try:
        compiled = torch.compile(
            multi.forward_native,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        eager_q, eager_k = multi.forward_native(
            positions, q.clone(), k.clone(),
        )
        compiled_q, compiled_k = compiled(
            positions, q.clone(), k.clone(),
        )
    finally:
        dynamo.reset()

    torch.testing.assert_close(
        eager_q, compiled_q, rtol=1e-5, atol=1e-5,
    )
    torch.testing.assert_close(
        eager_k, compiled_k, rtol=1e-5, atol=1e-5,
    )


def test_compiled_forward_sees_fresh_buffer_values_between_calls(device):
    """After compilation, updating the buffer must still change the
    output on the next call.  If the buffer read were constant-folded
    into the graph, the second call would reuse the first call's
    values -- which is exactly the silent correctness bug we care
    about for CUDA graph replay."""
    pytest.importorskip("torch._dynamo")

    head_size, rotary_dim = 64, 64
    max_pos = 2048
    num_tokens = 16

    multi = _make_multi_cache(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos,
        max_num_batched_tokens=num_tokens, device=device,
    )
    positions, q, k = _make_inputs(
        num_tokens=num_tokens, head_size=head_size,
        seq_len_start=0, device=device,
    )

    import torch._dynamo as dynamo

    dynamo.reset()
    try:
        compiled = torch.compile(
            multi.forward_native,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )

        # First call: all tokens route to bucket 0 (factor 1).
        multi.update_runtime_seq_lens(
            torch.full(
                (num_tokens,), 1000, dtype=torch.int32, device=device,
            ),
        )
        out_a_q, out_a_k = compiled(positions, q.clone(), k.clone())

        # Second call: all tokens route to bucket 3 (factor 4).
        multi.update_runtime_seq_lens(
            torch.full(
                (num_tokens,), 7000, dtype=torch.int32, device=device,
            ),
        )
        out_b_q, out_b_k = compiled(positions, q.clone(), k.clone())
    finally:
        dynamo.reset()

    assert not torch.allclose(out_a_q, out_b_q)
    assert not torch.allclose(out_a_k, out_b_k)


def test_forward_survives_cuda_graph_capture_and_replay(device):
    """End-to-end validation of the CUDA-graph safety property.

    We capture the rotary forward into a CUDA graph while the runtime
    buffer routes tokens to bucket A, then update the buffer so they
    route to bucket B and replay.  Replay must produce the same output
    as running the forward eagerly under bucket B, proving the graph
    reads live values from the buffer rather than constant-folding.
    """
    head_size, rotary_dim = 64, 64
    max_pos = 2048
    num_tokens = 16

    multi = _make_multi_cache(
        head_size=head_size, rotary_dim=rotary_dim,
        max_position_embeddings=max_pos,
        max_num_batched_tokens=num_tokens, device=device,
    )
    positions, q, k = _make_inputs(
        num_tokens=num_tokens, head_size=head_size,
        seq_len_start=0, device=device,
    )

    # Static tensors backing the CUDA graph.
    q_buf = q.clone()
    k_buf = k.clone()
    out_q = torch.empty_like(q_buf)
    out_k = torch.empty_like(k_buf)

    # Warmup required by CUDA graph capture.
    multi.update_runtime_seq_lens(
        torch.full((num_tokens,), 1000, dtype=torch.int32, device=device),
    )
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            multi.forward_native(
                positions, q_buf.clone(), k_buf.clone(),
            )
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q_in = q_buf.clone()
        k_in = k_buf.clone()
        captured_q, captured_k = multi.forward_native(
            positions, q_in, k_in,
        )
        out_q.copy_(captured_q)
        out_k.copy_(captured_k)

    # Replay after routing to a different bucket.  The graph must pick
    # up the new offsets via the live buffer read.
    multi.update_runtime_seq_lens(
        torch.full((num_tokens,), 7000, dtype=torch.int32, device=device),
    )
    graph.replay()
    torch.cuda.synchronize()

    # Reference: eager forward under the new bucket.
    ref_q, ref_k = multi.forward_native(
        positions, q_buf.clone(), k_buf.clone(),
    )

    torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# ``get_rope`` wiring
# --------------------------------------------------------------------------


def test_get_rope_dispatches_multi_cache_type():
    from arctic_inference.vllm.rope import (
        MultiCacheDynamicNTKRotaryEmbedding,
        apply_rope_runtime_patches,
    )

    apply_rope_runtime_patches()
    from vllm.model_executor.layers.rotary_embedding import (
        get_rope as patched_get_rope,
    )

    mod = patched_get_rope(
        head_size=64,
        max_position=16384,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "multi_cache_ntk",
            "factors": [1.0, 2.0, 4.0],
            "rope_theta": 10000.0,
            "original_max_position_embeddings": 2048,
        },
        dtype=torch.float32,
    )
    assert isinstance(mod, MultiCacheDynamicNTKRotaryEmbedding)
    assert mod.factors == [1.0, 2.0, 4.0]
    assert mod._original_max_position_embeddings == 2048


def test_get_rope_uses_default_factors_when_factors_not_specified():
    from arctic_inference.vllm.rope import (
        DEFAULT_FACTORS,
        MultiCacheDynamicNTKRotaryEmbedding,
        apply_rope_runtime_patches,
    )

    apply_rope_runtime_patches()
    from vllm.model_executor.layers.rotary_embedding import (
        get_rope as patched_get_rope,
    )

    mod = patched_get_rope(
        head_size=64,
        max_position=16384,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "multi_cache_ntk",
            "rope_theta": 10000.0,
            "original_max_position_embeddings": 2048,
        },
        dtype=torch.float32,
    )
    assert isinstance(mod, MultiCacheDynamicNTKRotaryEmbedding)
    assert mod.factors == list(DEFAULT_FACTORS)


def test_get_rope_folds_legacy_factor_into_single_bucket():
    """Backward compat: ``rope_type=multi_cache_ntk`` with a scalar
    ``factor`` (and no ``factors``) should produce a single-bucket
    multi-cache at that factor.  This keeps configs written for the old
    static dynamic rope from silently changing behavior when the
    ``rope_type`` is changed."""
    from arctic_inference.vllm.rope import (
        MultiCacheDynamicNTKRotaryEmbedding,
        apply_rope_runtime_patches,
    )

    apply_rope_runtime_patches()
    from vllm.model_executor.layers.rotary_embedding import (
        get_rope as patched_get_rope,
    )

    mod = patched_get_rope(
        head_size=64,
        max_position=16384,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "multi_cache_ntk",
            "factor": 4.0,
            "rope_theta": 10000.0,
            "original_max_position_embeddings": 2048,
        },
        dtype=torch.float32,
    )
    assert isinstance(mod, MultiCacheDynamicNTKRotaryEmbedding)
    assert mod.factors == [4.0]


def test_get_rope_promotion_via_env(monkeypatch):
    from arctic_inference.vllm.rope import (
        DEFAULT_FACTORS,
        MultiCacheDynamicNTKRotaryEmbedding,
        apply_rope_runtime_patches,
    )

    apply_rope_runtime_patches()
    monkeypatch.setenv("ARCTIC_INFERENCE_MULTI_CACHE_ROPE", "1")

    from vllm.model_executor.layers.rotary_embedding import (
        get_rope as patched_get_rope,
    )

    mod = patched_get_rope(
        head_size=64,
        max_position=16384,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "dynamic",
            "factor": 4.0,
            "rope_theta": 10000.0,
        },
        dtype=torch.float32,
    )
    # Env-promoted: gets a single-bucket multi-cache using the ``factor``
    # from the legacy config (see ``_build_multi_cache_ntk`` semantics).
    assert isinstance(mod, MultiCacheDynamicNTKRotaryEmbedding)
    assert mod.factors == [4.0]
    # No ``factors`` key means the default list is NOT pulled in when
    # a legacy ``factor`` is present; explicit opt-in required.
    assert mod.factors != list(DEFAULT_FACTORS)


def test_get_rope_default_still_works_after_patching():
    """Non-multi-cache rope types must continue to work unchanged after
    our wrapper is installed."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from arctic_inference.vllm.rope import apply_rope_runtime_patches

    apply_rope_runtime_patches()
    from vllm.model_executor.layers.rotary_embedding import (
        get_rope as patched_get_rope,
    )

    mod = patched_get_rope(
        head_size=64,
        max_position=2048,
        is_neox_style=True,
        rope_parameters=None,
        dtype=torch.float32,
    )
    assert isinstance(mod, RotaryEmbedding)


def test_promotion_skips_alpha_variant(monkeypatch):
    """The legacy alpha variant uses a different base formula, so it
    must NOT be promoted to ``multi_cache_ntk`` even with the env flag
    set."""
    from arctic_inference.vllm.rope import (
        MultiCacheDynamicNTKRotaryEmbedding,
        apply_rope_runtime_patches,
    )

    apply_rope_runtime_patches()
    monkeypatch.setenv("ARCTIC_INFERENCE_MULTI_CACHE_ROPE", "1")
    from vllm.model_executor.layers.rotary_embedding import (
        get_rope as patched_get_rope,
    )

    mod = patched_get_rope(
        head_size=64,
        max_position=16384,
        is_neox_style=True,
        rope_parameters={
            "rope_type": "dynamic",
            "alpha": 1.0,
            "rope_theta": 10000.0,
        },
        dtype=torch.float32,
    )
    assert not isinstance(mod, MultiCacheDynamicNTKRotaryEmbedding)


def test_get_rope_threads_max_num_batched_tokens_from_vllm_config():
    """When a vLLM config is active the rope module should size its
    runtime buffer to match ``scheduler_config.max_num_batched_tokens``.
    """
    from arctic_inference.vllm.rope import (
        MultiCacheDynamicNTKRotaryEmbedding,
        apply_rope_runtime_patches,
    )
    from vllm.config import VllmConfig, set_current_vllm_config

    apply_rope_runtime_patches()
    from vllm.model_executor.layers.rotary_embedding import (
        get_rope as patched_get_rope,
    )

    cfg = VllmConfig()
    cfg.scheduler_config.max_num_batched_tokens = 7777
    with set_current_vllm_config(cfg):
        mod = patched_get_rope(
            head_size=64,
            max_position=32768,
            is_neox_style=True,
            rope_parameters={
                "rope_type": "multi_cache_ntk",
                "factors": [1.0, 2.0, 4.0],
                "rope_theta": 10000.0,
                "original_max_position_embeddings": 2048,
            },
            dtype=torch.float32,
        )
    assert isinstance(mod, MultiCacheDynamicNTKRotaryEmbedding)
    assert mod.runtime_bucket_offsets.shape[0] == 7777
