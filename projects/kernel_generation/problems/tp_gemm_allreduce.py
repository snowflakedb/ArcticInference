"""Tensor-parallel row-parallel linear: local GEMM then all-reduce, on 4 GPUs.

The shape is Llama-3.1-70B's MLP ``down_proj`` under TP4. The weight ``(intermediate,
hidden)`` is sharded on ``intermediate``, so rank ``r`` holds ``(intermediate/TP, hidden)``
and its GEMM produces a **partial sum** of the full ``(M, hidden)`` output. Every rank must
end up with the summed result, which is what the all-reduce is for. This is the op that
async-TP, Flux and TransformerEngine's userbuffers exist to optimise.

**The collective belongs to the solution, not the harness.** A solution is scored on the
GEMM and the all-reduce together, because the whole opportunity is in how they interact --
overlapping them, fusing them, or choosing a better collective for the message size. A
harness that did the reduce itself would be measuring only half the problem.

Why it is worth optimising. Measured on 4xB200 (NV18 all-to-all, ~956 GB/s per
GPU) with ``dist.all_reduce`` and a plain ``torch.mm``:

    M      GEMM    all-reduce   naive total   all-reduce share
    1      21.8us     19.1us        38.1us         50%
    128    22.4us     29.9us        52.1us         57%
    2048  161.1us    123.9us       275.2us         45%
    8192  596.7us    383.0us       970.1us         39%

Those are this configuration's own numbers, taken with the same shard and world size.
Note what changes if the world size does: the all-reduce always moves ``M*HIDDEN`` (message
size is independent of world size) but its ring cost is ``2*(W-1)/W``, so 1.5 here against
1.0 at TP2, while each rank's shard -- and therefore its GEMM -- halves as W doubles. TP4 is
thus the configuration where the collective's share of the total is *largest*, which is why
it is the harder one to hide.

The naive total is the sum of its parts to within a few percent at every size, i.e. **there
is no overlap** -- the collective sits on the critical path in full. Two separate wins are
available. At small M the all-reduce is pure latency (19us to move 16 KB, nowhere near
bandwidth-bound), so a one-shot collective over symmetric memory beats a ring. At large M it
is bandwidth-bound and the win is overlapping it with the GEMM. Measured here,
``torch.ops.symm_mem.two_shot_all_reduce_`` already beats ``dist.all_reduce`` by 2-10% up to
32 MB and loses by 5% at 128 MB, so even picking the right collective per size is worth
something.

Note there is no single best collective: vLLM's own measured table
(``references/vllm/vllm/distributed/device_communicators/all_reduce_utils.py:94-118``) has
the winner changing three times across message size at TP4, and the custom kernel switching
one-shot to two-shot at ``world_size <= 4 && bytes < 512KB``.

What this problem does NOT check
-------------------------------
* **Contention with the rest of the layer.** The evaluator runs this op alone. In serving it
  is one of several collectives per layer, competing for NVLink with them, so these numbers
  are a floor. NCCL caps itself at 36 blocks precisely to avoid that contention
  (``references/vllm/csrc/custom_all_reduce.cuh:229-235``).
* **Multi-node.** All four ranks are in one NVLink domain. Crossing a node boundary changes
  the collective's cost by an order of magnitude and its optimal algorithm with it.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from problem_loader import Config, pick_device, world_size

# Read by scripts/evaluate.py. >1 makes the evaluator re-exec itself under torchrun with
# this many ranks; every rank then runs the same evaluation and the reported latency is the
# max across them, because a collective is only finished when its slowest participant is.
NUM_GPUS = 4

# Llama-3.1-70B MLP dimensions (HF meta-llama/Meta-Llama-3.1-70B).
HIDDEN = 8192
INTERMEDIATE = 28672
DTYPE = torch.bfloat16

# Evaluator knobs (read by scripts/evaluate.py).
# Loose because this is a bf16 GEMM followed by a bf16 reduction over 4 ranks: the reference
# and a solution can legitimately sum the partials in a different order.
TOLERANCE = {"rtol": 5e-2, "atol": 5e-2}
WARMUP_ITERS = 3


class Reference(nn.Module):
    """Local GEMM, then NCCL all-reduce -- the obvious correct implementation, and the
    thing to beat. `weight` is a parameter so the evaluator's `load_state_dict` gives the
    solution the same shard this rank holds."""

    def __init__(self, *, hidden: int, shard: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shard, hidden, dtype=DTYPE))
        with torch.no_grad():
            self.weight.normal_(0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = torch.mm(x, self.weight)
        dist.all_reduce(partial)
        return partial


def _make_x(m: int, shard: int) -> callable:
    device = pick_device()

    def factory() -> tuple:
        # Each rank draws its own shard of the activation, because the evaluator seeds
        # torch per RANK. That difference is load-bearing, not cosmetic: were every rank
        # to draw the same values, all partial sums would be equal and `world_size *
        # mm(x, w)` would reproduce an all-reduce exactly -- so a solution could pass
        # correctness without communicating at all. Differing shards are what makes a
        # wrong reduction show up as a wrong sum.
        return (torch.randn(m, shard, device=device, dtype=DTYPE) * 0.02,)

    return factory


def make_configs() -> list[Config]:
    out: list[Config] = []
    shard = INTERMEDIATE // NUM_GPUS
    # M is the global token count: 1-8 is decode, where the collective is latency-bound and
    # costs as much as the GEMM; 2048+ is prefill, where it is bandwidth-bound and the win
    # is overlap. Both regimes are in the sweep because the right collective differs.
    for m in (1, 8, 128, 512, 2048, 8192):
        out.append(Config(make_inputs=_make_x(m, shard),
                          init_kwargs={"hidden": HIDDEN, "shard": shard},
                          name=f"tp{NUM_GPUS}_m{m}"))
    return out


def flops(inputs: tuple, *, shard: int = INTERMEDIATE // NUM_GPUS, **_) -> float:
    """The local GEMM only: 2*M*shard*HIDDEN. The all-reduce moves bytes but does almost no
    arithmetic, so FLOP/s here describes the compute half and understates the whole op --
    which is the point, since a solution that hides the collective raises this number."""
    x, = inputs
    m, k = x.shape
    return 2.0 * m * k * HIDDEN


def bytes_moved(inputs: tuple, *, shard: int = INTERMEDIATE // NUM_GPUS, **_) -> float:
    """HBM traffic for the GEMM plus the NVLink traffic a ring all-reduce must move.

    The collective term is `2*(W-1)/W * size`, the standard ring cost -- each of the W-1
    steps sends and receives one chunk in both the reduce-scatter and the all-gather phase.
    Counting it makes `pct_bandwidth` meaningful at large M, where the op is bound by the
    interconnect rather than by HBM.
    """
    x, = inputs
    m, k = x.shape
    elt = x.element_size()
    world = max(1, world_size())
    gemm = elt * (m * k + k * HIDDEN + m * HIDDEN)
    collective = 2.0 * (world - 1) / world * elt * m * HIDDEN
    return float(gemm + collective)
