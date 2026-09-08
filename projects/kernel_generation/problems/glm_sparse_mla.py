"""GLM-5.2 sparse MLA — the exact kernel SGLang runs for `nvidia/GLM-5.2-NVFP4`.

A correct solution here is meant to be **droppable into SGLang**. The signature is
the call at `srt/layers/attention/dsa_backend.py:3259`, inside `_forward_trtllm`:

    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q, kv_cache=kv, workspace_buffer=...,
        qk_nope_head_dim=..., kv_lora_rank=512, qk_rope_head_dim=64,
        block_tables=page_table_1.unsqueeze(1), seq_lens=..., max_seq_len=...,
        sparse_mla_top_k=2048, bmm1_scale=..., backend="trtllm-gen")

`Reference.forward(query, kv_cache, block_tables, seq_lens)` takes those four
tensors in SGLang's own shapes and dtypes, so substituting a solution is a
one-line swap needing no reshape or cast:

    problem input   SGLang local at that call site
    ------------    ------------------------------------------------------
    query           q_all.view(batch_size, 1, num_heads, head_dim)
    kv_cache        kv_cache.view(-1, 1, real_page_size, kv_cache_dim)
    block_tables    page_table_1.unsqueeze(1)
    seq_lens        metadata.dsa_cache_seqlens_int32

The scalars (`bmm1_scale`, `sparse_mla_top_k`, `max_seq_len`) are constructor
arguments, not inputs: they are configuration, fixed for a deployment, and keeping
them out of `forward` stops a solution from being tempted into a host sync such as
`seq_lens.max().item()`. Scratch is **not** an argument either — SGLang sizes
`workspace_buffer` for trtllm-gen specifically, so a different kernel must own its
own, allocated once outside the timed path.

Why these shapes
----------------
`GlmMoeDsaForCausalLM` subclasses `DeepseekV2ForCausalLM`, so attention is DeepSeek
**MLA** in its absorbed form: one latent entry per token holds `kv_lora_rank` (512)
compressed KV plus a shared `qk_rope_head_dim` (64) RoPE key, and the value is the
**first 512 dims of that same 576-dim entry**. Every head reads one latent stream,
so it is MQA at head_dim 576 with a 512-wide output. **DSA** makes it sparse: a
lightning indexer keeps the top 2048 keys per query, and this kernel consumes that
selection. `v_head_dim` (256) is *not* seen here — it appears only after `W_UV`, on
the way into `o_proj` (`attention_forward_methods/forward_mla.py:854-887`).

There is no batch dimension in the usual sense: `batch_size = page_table_1.shape[0]`
is the **query-token count**, with `q_len_per_request` pinned to 1. With MTP/EAGLE
verifying 6 draft tokens per step, `S = requests * 6`.

Verified contract (measured on a GB300, not inferred)
-----------------------------------------------------
* `seq_lens[i]` is the number of **valid** entries in `block_tables[i]`, and the
  kernel reads only that many columns. Probed by placing *poison* indices past the
  count: the output matched a first-`seq_lens`-only oracle and never picked them up.
  So the real invariant is **valid entries packed at the front**; `-1` is not a
  sentinel the kernel interprets, it is simply never dereferenced.
* `seq_lens = min(context_len, 2048)` — `compute_dsa_seqlens` is
  `original_seq_lens.clamp(max=index_topk)` (`dsa/utils.py:74`). At the low-latency
  context lengths it is uniformly 2048.
* `block_tables` holds **physical KV slot** indices. With the default
  `SGLANG_DSA_FUSE_TOPK=1` it is the indexer's output verbatim
  (`dsa_backend.py:724-729`), already physical.
* A fully padded row (all `-1`, `seq_lens=1`) returns exact zeros without error.
* Output is `bf16` even though both inputs are fp8.

`Reference` is the naive gather-then-softmax oracle in fp32 — the correctness
reference and the speedup denominator, the agent's `v0`. It is deliberately not a
tuned implementation.

Dtypes come from the checkpoint and the launch, and the two disagree in a way worth
stating: NVFP4 quantizes **only** the routed MoE experts — every layer's
`self_attn*` is in `quantization_config.ignore`, so q is *computed* in bf16. It
reaches this kernel as fp8 because `kv_cache_dtype` resolves to `fp8_e4m3` for any
DSA model on SM≥10 (`arg_groups/overrides.py:1812`), and SGLang then casts q to
match via `mla_quantize_and_rope_for_fp8`. That cast is **unit-scale**
(`quant_scale_q = quant_scale_kv = 1.0`, `kernels/ops/attention/utils.py:175`), i.e.
a straight e4m3 cast, which is why `bmm1_scale` carries no quantization factor.
RoPE is fused into the same call, so `query` here is already post-RoPE.

What this problem does NOT check
-------------------------------
* **Accuracy compounding.** fp8 forces a ~5e-2 band per call. Across 78 layers small
  per-layer error can compound: SGLang's own GLM-5.2 recipe documents a gfx950 bug
  where "the error was small per layer but compounded across all 78 layers and
  silently corrupted output — in-context reasoning broke (GSM8K ≈ 0) while short
  factual prompts still looked fine." Only an end-to-end eval catches that.
* **Contention.** The evaluator runs this kernel alone. In serving it competes for
  bandwidth and SMs with the MoE layers around it, so these latencies are a floor.
  The L2 flush between calls models the eviction that interleaving causes, but not
  the contention.

Notes on how it is timed
------------------------
The evaluator calls the module directly and takes latency from CUPTI device
timestamps, summing the merged intervals in which the GPU was busy. The number is
therefore GPU work rather than launch overhead -- this kernel is ~9-29us while
enqueuing it costs ~74-94us of Python, which an earlier event-pair harness reported
as 35-56us.

One module is built per constructor-argument set and handed every shape that shares
it, so `sparse_mla_top_k` and `bmm1_scale` are the only values a solution may treat
as compile-time constants. Each config is timed on `num_samples` independently
allocated input sets, rotated between calls, so the gather addresses move and
nothing cached during warmup stays valid. The LLC is flushed between calls too,
because a kernel that runs once per layer never finds its own data resident in
production.

Because the gaps between kernels are excluded rather than charged, a host sync costs
nothing in the reported number even though it would stall a real server. So
`seq_lens.tolist()` in the `Reference` below is timed like any other call. Serving
pays for that sync; this measurement does not see it.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from problem_loader import Config, pick_device

# ── nvidia/GLM-5.2-NVFP4 config.json (model_type "glm_moe_dsa") ──────────────
NUM_ATTENTION_HEADS = 64
KV_LORA_RANK = 512         # compressed KV latent per token
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64      # shared RoPE key per token
QK_HEAD_DIM = 256          # explicit in the config; == nope + rope
INDEX_TOPK = 2048          # DSA: keys the lightning indexer keeps per query

# ── the GB300 low-latency recipe ─────────────────────────────────────────────
# --tp 4 --quantization modelopt_fp4 --speculative-num-steps 5
# --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
# --max-running-requests 16 --cuda-graph-max-bs 16
TP_SIZE = 4
DRAFT_TOKENS = 6           # MTP/EAGLE tokens verified per step
MAX_REQUESTS = 16
PAGE_SIZE = 64             # "Setting page size to 64 for DeepSeek DSA"
MAX_CONTEXT = 1_048_576    # max_position_embeddings

# ── per-GPU shapes, all derived ──────────────────────────────────────────────
N_HEADS = NUM_ATTENTION_HEADS // TP_SIZE        # 16 query heads on this rank
D_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM          # 576, the latent entry width
HEAD_DIM_V = KV_LORA_RANK                       # 512; attn_mqa.v_head_dim = kv_lora_rank

# layer.scaling = qk_head_dim ** -0.5 (deepseek_v2.py:1774). rope_type is
# "default", so compute_mla_mscale_scaling returns it unchanged; q_scale and
# k_scale are both 1.0, hence bmm1_scale == layer.scaling.
BMM1_SCALE = QK_HEAD_DIM**-0.5                  # 0.0625

KV_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
INVALID = -1               # what SGLang leaves past seq_lens; never dereferenced

# Evaluator knobs (read by scripts/evaluate.py).
# fp8 e4m3 carries ~2 decimal digits; the measured median relative error of the
# real kernel against an fp32 oracle is 2.4-2.6% across every probed shape, so
# this band has roughly 2x headroom. It is much looser than the bf16 problems'
# 3e-2 — that is the price of the dtype production actually uses.
TOLERANCE = {"rtol": 5e-2, "atol": 5e-2}
WARMUP_ITERS = 3

# The indexer concentrates on an attention sink and on recent tokens rather than
# scattering uniformly. Gather locality is the binding cost here, so the draw
# models that structure.
N_SINK_TOKENS = 4
RECENT_WINDOW = 512
SINK_BONUS = 2.0
RECENT_BONUS = 1.0

# Heavy-tailed values, as in the other attention problems: outlier channels shared
# by q and the latent, plus large-norm sink rows. e4m3 saturates at 448, so the
# scales stay well inside range. Latency is value-independent; this shapes only the
# correctness draw.
N_OUTLIER_CHANNELS = 4
OUTLIER_SCALE = 4.0
SINK_ROW_NORM = 6.0
BASE_STD = 0.25


class Reference(nn.Module):
    """Naive sparse MLA: gather the selected latents, softmax, weight — fp32
    throughout, with one round to bf16 on the output.

    Implements the contract the real kernel was measured to have: read exactly
    `seq_lens[i]` entries from row `i`, which are packed at the front. Anything
    past that count is untouched, so no `-1` masking is needed — and a row whose
    count is 0 yields zeros rather than NaN.
    """

    def __init__(self, *, bmm1_scale: float, sparse_mla_top_k: int, max_seq_len: int) -> None:
        super().__init__()
        self.bmm1_scale = bmm1_scale
        self.sparse_mla_top_k = sparse_mla_top_k
        self.max_seq_len = max_seq_len

    def forward(
        self,
        query: torch.Tensor,        # (S, 1, N_HEADS, 576) fp8
        kv_cache: torch.Tensor,     # (pages, 1, PAGE_SIZE, 576) fp8
        block_tables: torch.Tensor,  # (S, 1, topk) int32, valid packed first
        seq_lens: torch.Tensor,     # (S,) int32, count of valid entries
    ) -> torch.Tensor:
        s_q = query.shape[0]
        # The paged cache is addressed by absolute slot, so flatten pages away.
        latents = kv_cache.reshape(-1, kv_cache.shape[-1])
        q = query.float()
        counts = seq_lens.tolist()

        out = torch.zeros(s_q, 1, query.shape[2], HEAD_DIM_V, device=query.device, dtype=torch.float32)
        for i in range(s_q):
            n = counts[i]
            if n <= 0:
                continue                      # fully padded row -> zeros
            sel = latents[block_tables[i, 0, :n].long()].float()   # (n, 576)
            logits = (q[i, 0] @ sel.transpose(0, 1)) * self.bmm1_scale   # (H, n)
            weights = torch.softmax(logits, dim=-1)
            out[i, 0] = weights @ sel[:, :HEAD_DIM_V]
        return out.to(OUT_DTYPE)


def _selection(n_req: int, ctx: int, pages_per_req: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical-slot selections plus their valid counts.

    Request `r` owns slots `[r*pages_per_req*PAGE_SIZE, ...)`. Its `j`-th verify
    token sits at local position `ctx - DRAFT_TOKENS + j`, so it may select local
    keys `<= that`. Selections are **packed at the front** and the tail is filled
    with `-1`, matching what SGLang leaves past `seq_lens`.

    Sorted ascending within a row. That is an assumption about the indexer's output
    order, not a measured property, and it matters for gather locality — an
    unsorted row would scatter more.
    """
    rows = n_req * DRAFT_TOKENS
    table = torch.full((rows, INDEX_TOPK), INVALID, dtype=torch.int32, device=device)
    counts = torch.empty(rows, dtype=torch.int32, device=device)
    base = (torch.arange(n_req, device=device) * pages_per_req * PAGE_SIZE).view(n_req, 1).to(torch.int32)

    for j in range(DRAFT_TOKENS):
        n_valid = ctx - DRAFT_TOKENS + j + 1        # causal: keys 0..pos
        take = min(n_valid, INDEX_TOPK)
        who = slice(j, None, DRAFT_TOKENS)          # token j of every request
        if take == n_valid:
            local = torch.arange(n_valid, dtype=torch.int32, device=device).view(1, n_valid)
            table[who, :take] = local + base
        else:
            score = torch.rand((n_req, n_valid), device=device)
            score[:, :N_SINK_TOKENS] += SINK_BONUS
            score[:, n_valid - RECENT_WINDOW:] += RECENT_BONUS
            local = score.topk(take, dim=-1).indices.sort(dim=-1).values.to(torch.int32)
            table[who, :take] = local + base
        counts[who] = take
    return table.unsqueeze(1), counts


def _make_inputs(n_req: int, ctx: int) -> Callable[[], tuple]:
    device = pick_device()
    s_q = n_req * DRAFT_TOKENS
    pages_per_req = math.ceil(ctx / PAGE_SIZE)
    pages = n_req * pages_per_req

    def factory() -> tuple:
        # Outlier channels shared by q and the latent so they survive the dot
        # product rather than averaging out.
        gain = torch.ones(D_QK, device=device)
        chan = torch.argsort(torch.rand(D_QK, device=device))[:N_OUTLIER_CHANNELS]
        gain[chan] = OUTLIER_SCALE

        q = torch.randn((s_q, 1, N_HEADS, D_QK), device=device) * BASE_STD * gain
        kv = torch.randn((pages, 1, PAGE_SIZE, D_QK), device=device) * BASE_STD * gain
        # Attention sink: each request's leading keys carry a larger norm.
        flat = kv.view(-1, D_QK)
        for r in range(n_req):
            lo = r * pages_per_req * PAGE_SIZE
            flat[lo : lo + N_SINK_TOKENS] *= SINK_ROW_NORM

        table, counts = _selection(n_req, ctx, pages_per_req, device)
        return (q.to(KV_DTYPE), kv.to(KV_DTYPE), table, counts)

    return factory



def make_configs() -> list[Config]:
    out: list[Config] = []
    # The low-latency envelope. `S = requests * 6`: a single user (S=6) is the
    # latency point, 16 concurrent requests (S=96) the recipe's ceiling.
    #
    # Past 2048 keys the selection saturates, so the kernel's arithmetic stops
    # growing with context and the sweep is really about how far the gather
    # ranges — at 1M the 2048 slots are scattered over a 576 MiB cache.
    #
    # ctx 1024 is the one short-context case: it drives `seq_lens < 2048` with a
    # packed prefix and a `-1` tail, which is a real serving state (a request that
    # has just started) and a distinct code path in any solution.
    #
    # 16 x 1M is left out deliberately: one input set would be 9 GiB of cache and
    # the benchmark holds warmup sets plus a timing ring.
    shapes = (
        (1, 1024),
        (1, 8192),
        (1, 131_072),
        (1, MAX_CONTEXT),
        (MAX_REQUESTS, 8192),
        (MAX_REQUESTS, 131_072),
    )
    for n_req, ctx in shapes:
        out.append(Config(
            make_inputs=_make_inputs(n_req, ctx),
            init_kwargs={
                "bmm1_scale": BMM1_SCALE,
                "sparse_mla_top_k": INDEX_TOPK,
                "max_seq_len": ctx,
            },
            name=f"verify_s{n_req * DRAFT_TOKENS}_ctx{ctx}",
        ))
    return out


def flops(inputs: tuple, **_) -> float:
    """Two GEMMs per selected key: q·latent over 576, then weight·value over 512.
    `2 * (576 + 512)` FLOPs per (head, valid entry). Only the first `seq_lens[i]`
    entries of each row are read, so short rows cost proportionally less."""
    query, _, _, seq_lens = inputs
    n_heads = query.shape[2]
    valid = int(seq_lens.sum().item())
    return 2.0 * (D_QK + HEAD_DIM_V) * n_heads * valid


def bytes_moved(inputs: tuple, **_) -> float:
    """Min HBM traffic: q in, output out, the index rows and their counts, and each
    **distinct** latent slot read once.

    Distinct is the point. All 16 heads share one gathered stream, and the 6 verify
    tokens of a request select heavily overlapping keys, so charging `S * topk`
    slots would overstate traffic several-fold and flatter the achieved bandwidth.
    Counted from the actual selections rather than assumed. fp8 makes a latent slot
    576 bytes, not 1152 — half the traffic of a bf16 cache.
    """
    query, kv_cache, block_tables, seq_lens = inputs
    elt = kv_cache.element_size()                     # 1 byte, fp8
    s_q, _, n_heads, _ = query.shape

    used = [block_tables[i, 0, : int(n)] for i, n in enumerate(seq_lens.tolist()) if int(n) > 0]
    distinct = int(torch.unique(torch.cat(used)).numel()) if used else 0

    q_bytes = elt * s_q * n_heads * D_QK
    out_bytes = OUT_DTYPE.itemsize * s_q * n_heads * HEAD_DIM_V
    kv_bytes = elt * distinct * kv_cache.shape[-1]
    meta_bytes = block_tables.element_size() * int(seq_lens.sum().item()) + seq_lens.numel() * 4
    return float(q_bytes + out_bytes + kv_bytes + meta_bytes)
