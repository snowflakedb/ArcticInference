"""Batched attention problem -- causal Llama-style GQA, prefill + decode.

This is the multi-user sibling of ``attention.py``: tensors carry a real batch
dimension and the same ``Reference`` forward covers both prompt prefill
(``Lq == Lk``) and decode (``Lq == 1`` against a per-user KV cache). The goal is
to reward kernels that batch independent requests efficiently without mixing
attention across batch elements.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from problem_loader import Config, pick_device

# Llama 3.1 70B attention dimensions (HF meta-llama/Meta-Llama-3.1-70B).
N_Q_HEADS = 64
N_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16

# Same heavy-tailed draw as the single-batch attention problem: aligned outlier
# channels plus leading attention-sink tokens. Latency is value-independent, but
# these inputs keep approximate kernels honest across each batch element.
N_OUTLIER_CHANNELS = 4
OUTLIER_SCALE = 4.0
N_SINK_TOKENS = 4
SINK_KEY_NORM = 14.0
SINK_QUERY_BIAS = 6.0

# Evaluator knobs (read by scripts/evaluate.py).
TOLERANCE = {"rtol": 3e-2, "atol": 3e-2}
WARMUP_ITERS = 3


def _repeat_kv(x: torch.Tensor, n_q_heads: int) -> torch.Tensor:
    """GQA expand (B, n_kv, T, D) -> (B, n_q, T, D)."""
    B, n_kv, T, D = x.shape
    rep = n_q_heads // n_kv
    if rep == 1:
        return x
    return x[:, :, None, :, :].expand(B, n_kv, rep, T, D).reshape(B, n_kv * rep, T, D)


def _causal_mask(Lq: int, Lk: int, device) -> torch.Tensor:
    """Bool mask (True = disallowed) for queries at the end of a length-Lk sequence."""
    q_pos = torch.arange(Lq, device=device).unsqueeze(1) + (Lk - Lq)
    k_pos = torch.arange(Lk, device=device).unsqueeze(0)
    return k_pos > q_pos


class Reference(nn.Module):
    """Naive batched attention in fp32, rounded once to bf16 on output."""

    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / math.sqrt(Q.shape[-1])
        Kf = _repeat_kv(K, Q.shape[1])
        Vf = _repeat_kv(V, Q.shape[1])
        scores = torch.matmul(Q.float(), Kf.float().transpose(-2, -1)) * scale
        if self.causal:
            scores = scores.masked_fill(_causal_mask(Q.shape[-2], K.shape[-2], Q.device), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, Vf.float()).to(DTYPE)


def _make_qkv(B: int, Lq: int, Lk: int) -> Callable[[], tuple]:
    device = pick_device()

    def factory() -> tuple:
        q = torch.randn((B, N_Q_HEADS, Lq, HEAD_DIM), device=device)
        k = torch.randn((B, N_KV_HEADS, Lk, HEAD_DIM), device=device)
        v = torch.randn((B, N_KV_HEADS, Lk, HEAD_DIM), device=device)

        chan = torch.argsort(torch.rand(HEAD_DIM, device=device))[:N_OUTLIER_CHANNELS]
        gain = torch.ones(HEAD_DIM, device=device)
        gain[chan] = OUTLIER_SCALE
        q = q * gain
        k = k * gain
        v[..., chan] = v[..., chan] * OUTLIER_SCALE

        u = torch.randn((B, HEAD_DIM), device=device)
        u = u / u.norm(dim=-1, keepdim=True)
        k[:, :, :N_SINK_TOKENS, :] = k[:, :, :N_SINK_TOKENS, :] + SINK_KEY_NORM * u[:, None, None, :]
        q = q + SINK_QUERY_BIAS * u[:, None, None, :]

        return (q.to(DTYPE), k.to(DTYPE), v.to(DTYPE))

    return factory


def make_configs() -> list[Config]:
    out: list[Config] = []
    # The checked prefill tier is capped below the single-B=1 8K case so the fp32
    # oracle remains practical while still stressing batch scheduling and head
    # grouping. A constant B*T holds the token budget fixed across the sweep, so
    # what varies is the batch/sequence split rather than the total work.

    # for B, T in ((2, 1024), (4, 1024), (2, 2048), (4, 2048)):
    #     out.append(Config(make_inputs=_make_qkv(B, T, T), init_kwargs={"causal": True}, name=f"prefill_B{B}_T{T}"))

    for T in (1024, 2048, 4096, 8192):
        B = 32_768 // T
        out.append(Config(make_inputs=_make_qkv(B, T, T), init_kwargs={"causal": True}, name=f"prefill_B{B}_T{T}"))

    # Same split sweep at 4x the token budget. The fp32 oracle is O(Lq*Lk) and
    # cannot hold T=16384, so these are perf_only: benchmarked, never correctness-
    # checked, and the reference is never built. One input set is ~2.7 GiB (q is 64
    # heads, k/v 8), drawn and freed per timed call. Names stay distinct from the
    # checked tier because B differs at every shared T.
    for T in (2048, 4096, 8192, 16384):
        B = 131_072 // T
        # No num_samples here: adaptive sampling is what this config needs. A fixed 12
        # was measured to stabilise it *in isolation* (rel_noise 0.99/1.76/1.57%) and then
        # still produced 8.94% in a real run, because it executes after four other
        # multi-millisecond configs and the part is hotter by then. The right count depends
        # on the thermal state, which only the evaluator can see.
        out.append(Config(make_inputs=_make_qkv(B, T, T), init_kwargs={"causal": True},
                          name=f"prefill_B{B}_T{T}", perf_only=True))

    # Batched decode models serving multiple users concurrently from independent
    # KV caches. Larger B trades sequence length for oracle memory footprint.
    for B, Lk in ((4, 8192), (8, 8192), (4, 32768), (8, 32768)):
        out.append(Config(make_inputs=_make_qkv(B, 1, Lk), init_kwargs={"causal": True}, name=f"decode_B{B}_kv{Lk}"))
    return out


def _valid_pairs(Lq: int, Lk: int) -> int:
    """Causal (q, k) pair count for queries occupying the last Lq positions."""
    return Lq * Lk - Lq * (Lq - 1) // 2


def flops(inputs: tuple, *, causal: bool = True, **_) -> float:
    """Useful exact-attention work: QK and PV, 4*D FLOPs per valid pair."""
    q, k, _ = inputs
    B, Hq, Lq, D = q.shape
    Lk = k.shape[-2]
    pairs = _valid_pairs(Lq, Lk) if causal else Lq * Lk
    return 4.0 * D * B * Hq * pairs


def bytes_moved(inputs: tuple, **_) -> float:
    """Minimum streaming HBM traffic: read Q/K/V once and write O once."""
    q, k, v = inputs
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[-2]
    elt = q.element_size()
    q_o = 2 * B * Hq * Lq * D
    kv = 2 * B * Hkv * Lk * D
    return float(elt * (q_o + kv))
