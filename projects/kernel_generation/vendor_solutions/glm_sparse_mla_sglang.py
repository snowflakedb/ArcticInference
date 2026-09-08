"""SGLang's production sparse-MLA kernel, wrapped as a solution for the evaluator.

Gives `problems/glm_sparse_mla.py` a real ceiling instead of a guess:

    SP=/path/to/flashinfer/site-packages
    PYTHONPATH=$SP:$SP/nvidia_cutlass_dsl/dsl_packages \\
    PATH=$SP/../bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \\
    python scripts/evaluate.py problems/glm_sparse_mla.py \\
        vendor_solutions/glm_sparse_mla_sglang.py --stage full

**A measuring stick, not something the agent may see.** `build_mount_spec`
(src/main.rs:578-589) stages exactly four things into the sandbox:
`agent_ressources/docs`, `agent_ressources/skills`, the problem file (as
`problem.py`), and three *named* files out of `scripts/` — `evaluate.py`,
`problem_loader.py`, `run_kernel.py`. `vendor_solutions/` is not among them, which
is why this directory is the right home for a tuned reference. Should a future
change ever mount whole directories, this file must not come along: handing the
agent a tuned implementation is exactly what the problem's naive `Reference` exists
to avoid.

This calls `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` with
`backend="trtllm-gen"`, which is what GB300 actually runs — resolved through
`kv_cache_dtype` auto→`fp8_e4m3` on SM≥10, then
`arg_groups/overrides.py:1913` picking `trtllm` for fp8 KV on `major >= 10`. Note a
same-model, same-flags run on SM120 takes a *different* kernel
(`flashinfer_sparse_mla`, via the `is_glm_sm12_fp8` branch), so this ceiling is
GB300-specific.

Environment notes, learned the hard way:
  * The PyPI `sgl-kernel` wheel cannot be used: it links `libnvrtc.so.12` (CUDA 12),
    and the cu130 build fails against torch 2.13 with
    `undefined symbol: ...c10_cuda_check_implementationEiPKcS2_ib` — an `int` vs
    `unsigned int` ABI break. FlashInfer JITs instead, so it sidesteps that.
  * `nvidia-cutlass-dsl` ships a `.pth`, which Python only processes for real
    site-packages dirs — not ones injected via `PYTHONPATH`. Its
    `nvidia_cutlass_dsl/dsl_packages` must be added explicitly or `import cutlass`
    fails.
  * `ninja` and `nvcc` must be on `PATH` for the JIT.
  * The first call fetches cubins and JITs. `--stage full` runs correctness before
    benchmarking, so that lands outside the timed path; a bare benchmark would
    otherwise trip `_bench_solo`'s 2 s `SLOW_FORWARD_MS` branch and report the
    compile as latency.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Workspace for trtllm-gen. Allocated once, never in `forward` — SGLang holds it on
# the backend object, and allocating per call would put a cudaMalloc in the
# measurement. Must be zero-initialized: the kernel keeps semaphore state in it.
WORKSPACE_BYTES = 256 * 1024 * 1024

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64


class Solution(nn.Module):
    """`trtllm_batch_decode_with_kv_cache_mla`, wrapped to the problem's signature.

    No padding or reshaping: the problem hands over SGLang's own shapes, which is
    the point — an adapter inside `forward` would land in the measurement.
    """

    def __init__(self, *, bmm1_scale: float, sparse_mla_top_k: int, max_seq_len: int) -> None:
        super().__init__()
        self.bmm1_scale = bmm1_scale
        self.sparse_mla_top_k = sparse_mla_top_k
        self.max_seq_len = max_seq_len
        self._ws: torch.Tensor | None = None
        self._counter: torch.Tensor | None = None
        self._tuned = False
        try:
            import flashinfer.utils
            from flashinfer.autotuner import autotune
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "flashinfer is not importable, so the production sparse-MLA kernel "
                "cannot be benchmarked. Install `flashinfer-python` plus its deps "
                "(minus torch, to keep the host's matching cu130 build) and put "
                "nvidia_cutlass_dsl/dsl_packages on PYTHONPATH. Not falling back to "
                "anything slower: this file exists to report a ceiling, and a quiet "
                "substitution would report the wrong one."
            ) from e
        self._fwd = trtllm_batch_decode_with_kv_cache_mla
        self._autotune = autotune
        self._utils = flashinfer.utils

    def _workspace(self, device: torch.device) -> torch.Tensor:
        # Lazy so it lands in warmup rather than construction, and cached so the
        # timed path never allocates.
        if self._ws is None or self._ws.device != device:
            self._ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        return self._ws

    def _kv_counter(self, device: torch.device, num_q_heads: int, batch_size: int) -> torch.Tensor:
        """Counter buffer for trtllm-gen's multi-CTA split-KV, sized by FlashInfer's
        own helper — the same grow-if-needed policy as SGLang's
        `grow_multi_ctas_kv_counter_buffer_if_needed` (trtllm_mla_backend.py:110).
        Zero-initialized once and reused; it must stay alive across launches."""
        need = self._utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(
            batch_size, num_q_heads, self._utils.get_device_sm_count(device)
        )
        if self._counter is None or self._counter.device != device or self._counter.numel() < need:
            self._counter = torch.zeros(need, dtype=torch.uint8, device=device)
        return self._counter

    def _call(self, query, kv_cache, block_tables, seq_lens) -> torch.Tensor:
        s_q, _, n_heads, _ = query.shape
        out = self._fwd(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self._workspace(query.device),
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=self.max_seq_len,
            sparse_mla_top_k=self.sparse_mla_top_k,
            bmm1_scale=self.bmm1_scale,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=self._kv_counter(query.device, n_heads, s_q),
        )
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        # Tune once, then serve — what SGLang does at startup
        # (`should_run_flashinfer_autotune` / `flashinfer_autotune_context`, on by
        # default since `disable_flashinfer_autotune=False`). Without this the kernel
        # runs its heuristic fallback tactic, which is NOT what production serves.
        # FlashInfer caches the winning tactic per shape signature process-wide, so
        # tuning during the correctness stage carries into the timed benchmark even
        # though the evaluator rebuilds the module between stages.
        if not self._tuned:
            with self._autotune(True):
                out = self._call(query, kv_cache, block_tables, seq_lens)
            self._tuned = True
            return out
        return self._call(query, kv_cache, block_tables, seq_lens)
