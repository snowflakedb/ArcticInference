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
"""Multi-cache dynamic NTK RoPE for ArcticInference.

This module implements :class:`MultiCacheDynamicNTKRotaryEmbedding`: a
RoPE layer that precomputes *multiple* cos/sin caches (one per static
scaling factor) and routes each token to the cache whose factor best
covers its current sequence length.  Within any given bucket, the
layer is bit-identical to vLLM's
:class:`~vllm.model_executor.layers.rotary_embedding.DynamicNTKScalingRotaryEmbedding`
built with that factor.  The module is therefore best understood as
"a seq_len-routed ensemble of static dynamic-NTK configurations".

Why this shape (vs. an on-the-fly per-token formula)?

* **Training distribution.**  Models are typically fine-tuned with a
  *single* static rope factor.  Matching that exact cache for requests
  within that factor's regime preserves the training distribution.
  The continuous (per-token) HF formula replaces the static eff_base
  with a seq_len-dependent one, which the model has never seen.
* **Performance.**  The forward reuses vLLM's existing fused rope CUDA
  kernel (``ops.rotary_embedding``) unchanged -- we just add a
  per-token offset tensor to ``positions`` before the kernel looks up
  ``cos_sin_cache``.  No new kernels required.  This is the same trick
  :class:`DeepseekScalingRotaryEmbedding` uses for YaRN offsets.

Cache layout
------------
The caches are concatenated along axis 0, exactly like
:class:`LinearScalingRotaryEmbedding`::

    cos_sin_cache = [  cache_F1 (size F1 * m) ,
                       cache_F2 (size F2 * m) ,
                       ...
                       cache_Fk (size Fk * m) ]

For a request at position ``p`` that the router assigns to factor
``F_i``, the kernel indexes ``cos_sin_cache[offsets[i] + p]``.
``offsets[i]`` is the cumulative size of caches strictly before
``cache_F_i``.

Routing
-------
``factor = clamp(ceil(seq_len / m), 1, max_factor)``: pick the smallest
factor whose cache covers the request's seq_len.  Routing is
per-*token*, so mixed batches (e.g. a short decode request alongside a
long prefill request) rotate each token under its own factor's cache.

CUDA-graph safety
-----------------
* The per-token offset tensor lives in a registered GPU buffer on the
  module (``runtime_bucket_offsets``).
* :meth:`update_runtime_seq_lens` writes into that buffer **in place**
  each forward.  The model runner calls this before invoking the
  model.
* The forward reads ``self.runtime_bucket_offsets[:num_tokens]`` with
  a fixed-shape slice; captured graphs record a load from this
  buffer's storage, and each replay picks up whatever was most
  recently written.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


_DEFAULT_BUFFER_SIZE = 2048

#: Default factor set (fixed integers 1..6, each bucket one ``m`` wide).
DEFAULT_FACTORS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


class MultiCacheDynamicNTKRotaryEmbedding(RotaryEmbedding):
    """RoPE with N concatenated static dynamic-NTK caches + per-token routing.

    Args:
        head_size: Attention head dimension.
        rotary_dim: Number of dimensions to rotate (``<= head_size``).
        max_position_embeddings: The **original** max position (``m``)
            the model was trained on.  Each factor ``F``'s cache spans
            ``ceil(F * m)`` positions.
        base: ``rope_theta`` from the HF config.
        is_neox_style: NeoX (halves) vs GPT-J (interleaved) rotation.
        dtype: Cache dtype (``cos_sin_cache`` is cast to this).
        factors: Sorted list of scaling factors.  Defaults to
            ``DEFAULT_FACTORS`` = ``(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)``.
            Each factor must be ``>= 1.0``.
        max_num_batched_tokens: Upper bound on ``num_tokens`` in any
            single forward.  Determines the size of
            ``runtime_bucket_offsets``.  The model runner writes actual
            per-token seq-lens (via :meth:`update_runtime_seq_lens`,
            which converts them to offsets) before each forward.
    """

    DEFAULT_FACTORS = DEFAULT_FACTORS

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        factors: Optional[Sequence[float]] = None,
        max_num_batched_tokens: Optional[int] = None,
    ) -> None:
        raw_factors = (
            list(factors) if factors is not None else list(DEFAULT_FACTORS)
        )
        # Dedupe + sort; validate monotone >= 1.0.
        normalized = sorted({float(f) for f in raw_factors})
        if not normalized or normalized[0] < 1.0:
            raise ValueError(
                "factors must be a non-empty list of values >= 1.0, got "
                f"{normalized}"
            )
        self.factors: list[float] = normalized
        self._original_max_position_embeddings = int(max_position_embeddings)

        # Per-factor cache sizes and cumulative offsets.
        self._per_factor_max_len: list[int] = [
            int(math.ceil(F * self._original_max_position_embeddings))
            for F in self.factors
        ]
        cumulative = 0
        per_factor_offsets: list[int] = []
        for mx in self._per_factor_max_len:
            per_factor_offsets.append(cumulative)
            cumulative += mx
        self._per_factor_offsets: list[int] = per_factor_offsets
        self._total_cache_len: int = cumulative

        # Runtime buffer sizing.
        if max_num_batched_tokens is None:
            buf_size = _DEFAULT_BUFFER_SIZE
        else:
            buf_size = int(max_num_batched_tokens)
        self._runtime_buffer_size = max(1, buf_size)

        # Parent __init__ calls ``_compute_cos_sin_cache`` and registers
        # ``cos_sin_cache``.  Our override concatenates per-factor
        # caches; it relies on ``self.factors`` et al already being set.
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

        # Per-token offset buffer.  Initialised to 0 = factor-1 cache,
        # which is the unscaled base.  This keeps profile / dummy runs
        # that bypass the model runner in a numerically safe regime.
        self.register_buffer(
            "runtime_bucket_offsets",
            torch.zeros(self._runtime_buffer_size, dtype=torch.long),
            persistent=False,
        )

        # Lookup table used by ``update_runtime_seq_lens`` to translate
        # a bucket index -> offset into the concatenated cache.
        # Registered as a buffer so it rides with the module to the
        # correct device (via ``.to(device)``).
        self.register_buffer(
            "_factor_offsets_tensor",
            torch.tensor(self._per_factor_offsets, dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Cache construction (per-factor dynamic-NTK caches, concatenated)
    # ------------------------------------------------------------------
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Build ``cos_sin_cache`` as the concatenation of per-factor caches.

        For each factor ``F``:

        1. ``max_len = ceil(F * m)`` positions (what vLLM's static
           ``DynamicNTKScalingRotaryEmbedding`` would allocate).
        2. ``eff_base = base * ((F * max_len / m) - (F - 1)) ** (d/(d-2))``
           (the dynamic NTK formula evaluated at the cache's own max
           seq_len, matching the static class's behavior).
        3. ``inv_freq = 1 / eff_base ** (arange(0, d, 2)/d)``.
        4. ``cache_F = concat(cos(t*inv_freq), sin(t*inv_freq))`` for
           ``t in [0, max_len)``.

        Concatenating all ``cache_F`` along dim 0 yields the unified
        ``cos_sin_cache``.  Per-factor starts are recorded in
        ``self._per_factor_offsets``.
        """
        m = self._original_max_position_embeddings
        d = self.rotary_dim
        caches: list[torch.Tensor] = []
        for F in self.factors:
            max_len = int(math.ceil(F * m))
            eff_base = self.base * (
                (F * max_len / m) - (F - 1.0)
            ) ** (d / (d - 2))
            inv_freq = 1.0 / (
                eff_base
                ** (torch.arange(0, d, 2, dtype=torch.float) / d)
            )
            t = torch.arange(max_len, dtype=torch.float)
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            caches.append(torch.cat((cos, sin), dim=-1))
        return torch.cat(caches, dim=0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def factor_offset(self, factor: float) -> int:
        """Return the offset into the concatenated cache for ``factor``."""
        try:
            idx = self.factors.index(float(factor))
        except ValueError as e:
            raise ValueError(
                f"factor {factor!r} not in configured factors "
                f"{self.factors!r}"
            ) from e
        return self._per_factor_offsets[idx]

    def bucket_for_seq_len(self, seq_len: int) -> int:
        """Scalar routing function: return the 0-indexed bucket for a seq_len.

        ``bucket = clamp(ceil(seq_len / m), 1, len(factors)) - 1``.
        Matches the vectorized routing in
        :meth:`update_runtime_seq_lens`.
        """
        m = self._original_max_position_embeddings
        k = len(self.factors)
        if seq_len <= 0:
            return 0
        bucket_one_indexed = min(k, max(1, math.ceil(seq_len / m)))
        return bucket_one_indexed - 1

    # ------------------------------------------------------------------
    # Public state update (called by the model runner each forward)
    # ------------------------------------------------------------------
    def update_runtime_seq_lens(
        self, seq_lens: torch.Tensor, non_blocking: bool = True
    ) -> None:
        """Convert per-token seq_lens to offsets, in place.

        ``seq_lens`` is shape ``[num_tokens]`` (on any device, any
        integer dtype).  For each token ``i``:

        .. code-block:: python

            bucket[i]  = clamp(ceil(seq_lens[i] / m), 1, k) - 1
            offset[i]  = factor_offsets[bucket[i]]

        The computed ``offset`` is written into
        ``self.runtime_bucket_offsets`` **in place**.  Preserving the
        buffer's storage is what makes captured CUDA graphs see fresh
        values on replay.
        """
        n = int(seq_lens.shape[0])
        if n > self._runtime_buffer_size:
            raise ValueError(
                f"MultiCacheDynamicNTKRotaryEmbedding: received {n} tokens "
                f"but runtime buffer was sized for {self._runtime_buffer_size}. "
                "Increase max_num_batched_tokens."
            )
        if n == 0:
            return

        m = self._original_max_position_embeddings
        max_bucket = len(self.factors)

        # Move to the buffer's device/dtype in one shot.  ``copy_`` on
        # the destination would work, but we need the ceil_div +
        # clamp arithmetic in int64 on-device anyway.
        sl = seq_lens.to(
            device=self.runtime_bucket_offsets.device,
            dtype=torch.long,
            non_blocking=non_blocking,
        )
        # Integer ceil_div for positive x: (x + m - 1) // m.
        bucket_idx = torch.clamp(
            (sl + m - 1) // m,
            min=1,
            max=max_bucket,
        ) - 1
        offsets = self._factor_offsets_tensor[bucket_idx]
        self.runtime_bucket_offsets[:n].copy_(offsets, non_blocking=non_blocking)

    # ------------------------------------------------------------------
    # Forward methods: delegate to vLLM's fused kernel with
    # ``positions + runtime_bucket_offsets`` as the effective positions.
    # No per-token formula recompute; just a tiny add + the same CUDA
    # rope kernel the rest of vLLM uses.
    # ------------------------------------------------------------------
    def _effective_positions(self, positions: torch.Tensor) -> torch.Tensor:
        num_tokens = positions.shape[0]
        # Use the bucket-offset slice matching the batch size.  The
        # slice is a view, not a copy; no allocation.
        offsets = self.runtime_bucket_offsets[:num_tokens]
        return torch.add(positions, offsets)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch-native forward (also used on CPU)."""
        effective = self._effective_positions(positions)
        return self.forward_static(
            effective,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            self.cos_sin_cache,
            self.is_neox_style,
        )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fused CUDA rope: same kernel as static rope, but with
        ``positions + per-token offset``.  This is the fast path."""
        if self.use_flashinfer:
            # Mirror DeepseekScalingRotaryEmbedding.forward_cuda: pass
            # the combined positions to flashinfer.  (Our base class
            # sets use_flashinfer = False by default, so this branch
            # is inactive unless explicitly enabled.)
            effective = self._effective_positions(positions)
            torch.ops.vllm.flashinfer_rotary_embedding(
                effective,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
            return query, key

        from vllm import _custom_ops as ops

        effective = self._effective_positions(positions)
        self._match_cos_sin_cache_dtype(query)
        # In-place kernel: writes into query / key.
        ops.rotary_embedding(
            effective,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Same as forward_cuda for our purposes (fused kernel handles
        both).  The ROCm Triton rope kernel doesn't take offsets, so
        route through the plain CUDA path."""
        return self.forward_cuda(positions, query, key)

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is None:
            # XPU kernel doesn't support key=None; fall back to native.
            return self.forward_native(positions, query, key)
        from vllm._ipex_ops import ipex_ops as ops

        effective = self._effective_positions(positions)
        self._match_cos_sin_cache_dtype(query)
        ops.rotary_embedding(
            effective,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        from vllm import _custom_ops as ops

        effective = self._effective_positions(positions)
        self._match_cos_sin_cache_dtype(query)
        ops.rotary_embedding(
            effective,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        parent = super().extra_repr()
        return (
            f"{parent}, factors={self.factors}, "
            f"orig_max_position={self._original_max_position_embeddings}, "
            f"runtime_buffer_size={self._runtime_buffer_size}, "
            f"total_cache_len={self._total_cache_len}, "
            f"multi_cache=True"
        )
