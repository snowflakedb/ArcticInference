"""torch SDPA as a ceiling for `problems/attention.py` (dense causal GQA).

    python scripts/evaluate.py problems/attention.py \\
        vendor_solutions/attention_torch_sdpa.py --stage full

Recovered from `fa4.py`, which had been sitting untracked in the repo root on the
GB300 box. The logic is unchanged; only this docstring is new.

**Despite the old filename, this is not FlashAttention 4.** It is
`F.scaled_dot_product_attention` with `enable_gqa`, and the backend is torch's
choice. Profiled on the GB300 (sm103, torch 2.13.0+cu130) the auto dispatch
launches **cuDNN's** generated flash-attention kernel, not the FlashAttention
repo's:

    prefill  T=8192           -> cudnn_generated_fort_native_sdpa_sm100_flash_
                                 fprop_f16_knob_7_128x128x128_4x1x1_cga1x1x1
    decode   Lq=1, Lk=131072  -> same family, 8x128x128, plus
                                 cudnn::fusion::lean_reduction_kernel (split-KV)

`flash`, `cudnn` and `math` all accept these shapes; `mem_efficient` does not. So
the number this reports is "what torch gives you for free on this GPU today" — a
moving target across torch and cuDNN versions, and worth re-profiling rather than
assuming it is still the same kernel.

**A measuring stick, not something the agent may see.** `build_mount_spec`
(src/main.rs:578-589) stages only `agent_ressources/docs`,
`agent_ressources/skills`, the problem file, and three *named* files out of
`scripts/`. `vendor_solutions/` is not mounted, which is why it lives here.
"""
import torch
import torch.nn.functional as F
from torch import nn


class Solution(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        Lq = Q.size(-2)
        Lk = K.size(-2)

        attn_mask = None
        is_causal = False

        if self.causal:
            if Lq == Lk:
                is_causal = True
            elif Lq > 1:
                q_pos = torch.arange(Lq, device=Q.device)[:, None] + (Lk - Lq)
                k_pos = torch.arange(Lk, device=Q.device)[None, :]
                attn_mask = k_pos <= q_pos
            # Lq == 1 needs no causal mask.

        return F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=Q.size(1) != K.size(1),
        )
