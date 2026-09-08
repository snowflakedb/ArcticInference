"""FlashAttention 4 as a ceiling for `problems/attention.py` (dense causal GQA).

    SP=/path/to/fa4/site-packages
    PYTHONPATH=$SP:$SP/nvidia_cutlass_dsl/dsl_packages \\
    python scripts/evaluate.py problems/attention.py \\
        vendor_solutions/attention_fa4.py --stage full

This is **actual FA4**, not torch SDPA: `flash_attn.cute.flash_attn_func` from the
`flash-attn-4` package, the CuTe-DSL FlashAttention whose Blackwell kernels live in
`flash_fwd_sm100.py`. It is the same implementation SGLang reaches for — its
`kernels/ops/attention/flash_attention_v4.py` imports
`flash_attn.cute.flash_attn_varlen_func`, either from a vendored in-tree copy or,
with `SGLANG_INKLING_FA4_USE_PIP=1`, from this pip package.

Compare with `attention_torch_sdpa.py`, which is the *other* ceiling on the same
problem: that one dispatches to cuDNN's generated SM100 flash kernel. Two different
vendors' flash attention, same workload.

Install note: `flash_attn_4` is a **pure-Python** wheel (`py3-none-any`) — it JITs
via CuTe DSL rather than shipping a compiled extension, so it needs no source build
and cannot hit the libtorch ABI breakage that blocks prebuilt kernel wheels. It
does need `nvidia-cutlass-dsl`, whose `.pth` is only honoured for real
site-packages dirs, so `nvidia_cutlass_dsl/dsl_packages` must be on `PYTHONPATH`
explicitly. Deps installed `--no-deps` to keep the host's matching cu130 torch.

Layout: the problem hands over `(B, H, S, D)` (PyTorch attention convention) while
FA4 wants `(B, S, H, D)`. The transposes are views, not copies — head_dim stays
innermost either way — so the adapter costs no bandwidth. GQA needs no KV repeat:
FA4 takes `Hq != Hkv` directly and packs internally.

Causal alignment matches the problem by construction: FA4's `causal=True` masks
query `i` to keys `j <= i + (Lk - Lq)`, i.e. bottom-right aligned, which is exactly
what `problems/attention.py::_causal_mask` builds ("the Lq queries are the last Lq
positions of a length-Lk sequence"). So both the prefill (`Lq == Lk`) and decode
(`Lq == 1`) configs need no mask fixup. `softmax_scale` defaults to
`1/sqrt(head_dim)`, matching the reference.

**A measuring stick, not something the agent may see.** `build_mount_spec`
(src/main.rs:578-589) stages only `agent_ressources/docs`,
`agent_ressources/skills`, the problem file, and three *named* files out of
`scripts/`. `vendor_solutions/` is not mounted, which is why it lives here.
"""
import torch
import torch.nn as nn


class Solution(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal
        try:
            from flash_attn.cute import flash_attn_func
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "flash_attn.cute is not importable, so FlashAttention 4 cannot be "
                "benchmarked. Install with `uv pip install --no-deps "
                "--prerelease=allow flash-attn-4 einops quack-kernels "
                "torch-c-dlpack-ext` and put nvidia_cutlass_dsl/dsl_packages on "
                "PYTHONPATH. Not falling back to torch SDPA: that is a different "
                "kernel (cuDNN) and reporting it as FA4 would be wrong — see "
                "attention_torch_sdpa.py if that is what you want."
            ) from e
        self._fa = flash_attn_func

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # (B, H, S, D) -> (B, S, H, D); views, head_dim stays innermost.
        out = self._fa(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            causal=self.causal,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2)
