"""Attention problem -- causal Llama 3.1 70B attention, prefill + decode.

Grouped-query attention: 64 query heads, 8 KV heads, head_dim 128, bf16, causal.
`Reference` is the naive torch baseline -- the correctness oracle and the speedup
denominator (`speedup = reference_latency / solution_latency`), the agent's `v0`.
One forward serves both regimes, told apart by query length `Lq` vs key length
`Lk`: prefill (`Lq == Lk`, a whole prompt) and decode (`Lq == 1` against a long
KV cache).
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from problem_loader import Config, pick_device

# Llama 3.1 70B attention dimensions (HF meta-llama/Meta-Llama-3.1-70B).
N_Q_HEADS = 64    # num_attention_heads
N_KV_HEADS = 8    # num_key_value_heads (GQA: 8 query heads per KV head)
HEAD_DIM = 128    # hidden_size 8192 / 64
DTYPE = torch.bfloat16

# Inputs are heavy-tailed like trained-transformer attention: a few outlier
# feature channels (shared by Q/K, mirrored in V) and an attention sink (leading
# keys with a large norm along a shared direction every query projects onto).
# Logits reach ~15-30 rather than ~+-5. Latency is value-independent, so this
# only shapes the correctness draw.
N_OUTLIER_CHANNELS = 4
OUTLIER_SCALE = 4.0
N_SINK_TOKENS = 4
SINK_KEY_NORM = 14.0
SINK_QUERY_BIAS = 6.0

# Evaluator knobs (read by scripts/evaluate.py).
TOLERANCE = {"rtol": 3e-2, "atol": 3e-2}   # bf16 flash-attention precision band
WARMUP_ITERS = 3


def _repeat_kv(x: torch.Tensor, n_q_heads: int) -> torch.Tensor:
    """GQA expand (B, n_kv, T, D) -> (B, n_q, T, D); the naive baseline materializes it."""
    B, n_kv, T, D = x.shape
    rep = n_q_heads // n_kv
    if rep == 1:
        return x
    return x[:, :, None, :, :].expand(B, n_kv, rep, T, D).reshape(B, n_kv * rep, T, D)


def _causal_mask(Lq: int, Lk: int, device) -> torch.Tensor:
    """Bool mask (True = disallowed): the Lq queries are the last Lq positions of a
    length-Lk sequence, so query i attends to key j iff j <= Lk - Lq + i."""
    q_pos = torch.arange(Lq, device=device).unsqueeze(1) + (Lk - Lq)
    k_pos = torch.arange(Lk, device=device).unsqueeze(0)
    return k_pos > q_pos


class Reference(nn.Module):
    """Naive attention: materialize scores, softmax, then P@V -- all in fp32, with a
    single round to bf16 on the output."""

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


def _make_qkv(Lq: int, Lk: int) -> Callable[[], tuple]:
    device = pick_device()

    def factory() -> tuple:
        q = torch.randn((1, N_Q_HEADS, Lq, HEAD_DIM), device=device)
        k = torch.randn((1, N_KV_HEADS, Lk, HEAD_DIM), device=device)
        v = torch.randn((1, N_KV_HEADS, Lk, HEAD_DIM), device=device)

        # Outlier feature channels (shared by Q/K so they align in the dot product).
        chan = torch.argsort(torch.rand(HEAD_DIM, device=device))[:N_OUTLIER_CHANNELS]
        gain = torch.ones(HEAD_DIM, device=device)
        gain[chan] = OUTLIER_SCALE
        q = q * gain
        k = k * gain
        v[..., chan] = v[..., chan] * OUTLIER_SCALE

        # Attention sink on the leading keys, along a shared direction u.
        u = torch.randn(HEAD_DIM, device=device)
        u = u / u.norm()
        k[:, :, :N_SINK_TOKENS, :] = k[:, :, :N_SINK_TOKENS, :] + SINK_KEY_NORM * u
        q = q + SINK_QUERY_BIAS * u

        return (q.to(DTYPE), k.to(DTYPE), v.to(DTYPE))

    return factory


def make_configs() -> list[Config]:
    out: list[Config] = []
    # Single-user (B=1), causal. Prefill (Lq == Lk): production prompt lengths,
    # capped at 8K by the naive O(Lq*Lk) fp32 oracle's memory. Decode (Lq == 1):
    # one token against a KV cache up to Llama 3.1's 128K context.
    for T in (2048, 4096, 8192):
        out.append(Config(make_inputs=_make_qkv(T, T), init_kwargs={"causal": True}, name=f"prefill_T{T}"))
    for Lk in (2048, 8192, 32768, 131072):
        out.append(Config(make_inputs=_make_qkv(1, Lk), init_kwargs={"causal": True}, name=f"decode_kv{Lk}"))
    return out


def _valid_pairs(Lq: int, Lk: int) -> int:
    """Causal (q, k) pair count: the Lq queries are the last Lq of the sequence."""
    return Lq * Lk - Lq * (Lq - 1) // 2


def flops(inputs: tuple, *, causal: bool = True, **_) -> float:
    """Useful work for exact attention -- the two GEMMs Q@K^T and (softmax)@V: 4*D
    FLOPs per valid (q, k) pair over B*H_q head-batches (causal triangle if causal).
    Binds the compute-bound prefill regime (achieved TFLOP/s = this / latency)."""
    q, k, _ = inputs
    B, Hq, Lq, D = q.shape
    Lk = k.shape[-2]
    pairs = _valid_pairs(Lq, Lk) if causal else Lq * Lk
    return 4.0 * D * B * Hq * pairs


def bytes_moved(inputs: tuple, **_) -> float:
    """Min HBM traffic for a streaming kernel: read Q, K, V once and write O once
    (K/V are the GQA tensors, read once and reused across each query group). Binds
    the memory-bound decode regime (achieved GB/s = this / latency)."""
    q, k, v = inputs
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[-2]
    elt = q.element_size()
    q_o = 2 * B * Hq * Lq * D
    kv = 2 * B * Hkv * Lk * D
    return float(elt * (q_o + kv))
