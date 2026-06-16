# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Based on vLLM v0.14.1 flash attention backend with Forest Cascade
# Attention (FCA) extensions.  Activated via the env var
# ARCTIC_INFERENCE_FOREST_CASCADE_ATTENTION=1.
"""Attention layer with FlashAttention and Forest Cascade Attention (FCA)."""

import copy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            return [16, 32, 64]
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(
        cls, kv_cache_dtype: CacheDType | None
    ) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(
        cls, capability: DeviceCapability
    ) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # For forest cascade attention (ragged groups).
    # When enabled, we reorder requests to pack groups with shared KV prefixes
    # contiguously, run a grouped prefix FA (groups as sequences, causal=False)
    # and a per-request suffix FA (causal=True), then merge and scatter back.
    use_forest_cascade: bool = False

    # Request permutation: packed_req_idx -> original_req_idx. Shape: [B].
    forest_perm: torch.Tensor | None = None

    # Token permutation for the packed Q tensor. Shape: [T].
    forest_token_perm: torch.Tensor | None = None

    # Prefix call: cu_seqlens_q in token space for packed Q. Shape: [G+1].
    forest_group_offsets: torch.Tensor | None = None

    # Group sizes in requests. Shape: [G].
    forest_group_sizes: torch.Tensor | None = None

    # Grouped prefix K/V metadata.
    forest_prefix_kv_lens: torch.Tensor | None = None
    forest_prefix_block_table: torch.Tensor | None = None

    # Suffix call: per-request K/V metadata in packed request order.
    forest_suffix_kv_lens: torch.Tensor | None = None
    forest_suffix_block_table: torch.Tensor | None = None

    # Suffix call: per-request cu_seqlens_q for packed Q. Shape: [B+1].
    forest_packed_query_start_loc: torch.Tensor | None = None

    # Cached maxima for kernel launch / AOT scheduler.
    forest_max_group_query_len: int | None = None
    forest_max_group_size: int | None = None
    forest_max_prefix_len: int | None = None
    forest_max_suffix_len: int | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[FlashAttentionMetadata]
):
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )

    # Forest-cascade uses derived per-group/per-request block tables.
    # Allow updating block tables without rebuilding the whole metadata.
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        # Forest cascade attention (ragged groups) tuning knobs,
        # configured via --forest-cascade-attn-configs CLI arg.
        fca_cfg = getattr(vllm_config, '_forest_cascade_attn_config', None)
        self.forest_cascade_enabled = fca_cfg is not None
        if fca_cfg is None:
            fca_cfg = {}
        self.forest_max_query_len = int(
            fca_cfg.get('max_query_len', 16))
        self.forest_min_batch_size = int(
            fca_cfg.get('min_batch_size', 8))
        self.forest_min_group_size = int(
            fca_cfg.get('min_group_size', 2))
        self.forest_min_additional_prefix_blocks = int(
            fca_cfg.get('min_additional_prefix_blocks', 1))
        self.forest_min_non_singleton_fraction = float(
            fca_cfg.get('min_non_singleton_fraction', 0.25))
        self.forest_max_non_singleton_groups = int(
            fca_cfg.get('max_non_singleton_groups', 256))

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            self.parallel_config
        )
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.has_piecewise_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        )
        self.max_cudagraph_size = (
            self.compilation_config.max_cudagraph_capture_size
        )

        # One-shot startup banner so operators can confirm whether FCA will
        # ever fire on this engine. Use print(..., flush=True) because the
        # InferenceWorker / EngineCore subprocesses only attach a logging
        # handler to the `vllm.*` namespace — module-level loggers under
        # `arctic_inference.*` produce no output there. The per-batch gate
        # also checks num_reqs, max_query_len, causal, and dcp_world_size;
        # those are still reported at DEBUG level in build() below.
        if self.forest_cascade_enabled and not self.has_piecewise_cuda_graph:
            print(
                f"[FCA] Forest Cascade Attention is CONFIGURED "
                f"(max_query_len={self.forest_max_query_len}, "
                f"min_batch_size={self.forest_min_batch_size}, "
                f"min_group_size={self.forest_min_group_size}, "
                f"min_additional_prefix_blocks={self.forest_min_additional_prefix_blocks}, "
                f"min_non_singleton_fraction={self.forest_min_non_singleton_fraction:.3f}, "
                f"max_non_singleton_groups={self.forest_max_non_singleton_groups}) "
                f"but will be DISABLED at runtime because "
                f"cudagraph_mode={self.compilation_config.cudagraph_mode} "
                f"captures no piecewise CUDA graphs. Set "
                f"compilation_config={{'cudagraph_mode': 'PIECEWISE'}} or "
                f"'FULL_AND_PIECEWISE' on the vLLM engine to let FCA fire.",
                flush=True,
            )
        elif self.forest_cascade_enabled:
            print(
                f"[FCA] Forest Cascade Attention ENABLED "
                f"(max_query_len={self.forest_max_query_len}, "
                f"min_batch_size={self.forest_min_batch_size}, "
                f"min_group_size={self.forest_min_group_size}, "
                f"min_additional_prefix_blocks={self.forest_min_additional_prefix_blocks}, "
                f"min_non_singleton_fraction={self.forest_min_non_singleton_fraction:.3f}, "
                f"max_non_singleton_groups={self.forest_max_non_singleton_groups}, "
                f"cudagraph_mode={self.compilation_config.cudagraph_mode}).",
                flush=True,
            )
        else:
            print(
                "[FCA] Forest Cascade Attention DISABLED "
                "(--forest-cascade-attn-configs not set).",
                flush=True,
            )

        if self.use_full_cuda_graph and self.aot_schedule:
            from vllm.utils.math_utils import round_up
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        self.aot_sliding_window: tuple[int, int] | None = None

    def will_forest_cascade_fire(
        self, num_reqs: int, max_query_len: int
    ) -> bool:
        """Cheap, side-effect-free pre-check used by the model runner to
        decide whether to set ``invalid_modes={FULL}`` on the cudagraph
        dispatcher. Mirrors the runtime gate in ``build()`` but uses only
        inputs that are known before metadata construction (``causal`` and
        ``common_prefix_len`` are intentionally omitted — false positives
        just dispatch PIECEWISE when FULL would also have worked).
        """
        return (
            self.forest_cascade_enabled
            and num_reqs >= int(self.forest_min_batch_size)
            and max_query_len <= int(self.forest_max_query_len)
            and self.dcp_world_size <= 1
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(
                    self.vllm_config
                )
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        def schedule(
            batch_size,
            cu_query_lens,
            max_query_len,
            seqlens,
            max_seq_len,
            causal,
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = (
                    FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                        cache_dtype
                    )
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        # Forest-cascade derived metadata.
        use_forest_cascade = False
        forest_perm = None
        forest_token_perm = None
        forest_packed_query_start_loc = None
        forest_group_offsets = None
        forest_group_sizes = None
        forest_prefix_kv_lens = None
        forest_prefix_block_table = None
        forest_suffix_kv_lens = None
        forest_suffix_block_table = None
        forest_max_group_query_len = None
        forest_max_group_size = None
        forest_max_prefix_len = None
        forest_max_suffix_len = None

        # Forest-cascade check: disabled when DCP is active. When the engine
        # captures FULL+PIECEWISE graphs, the cudagraph dispatcher selects
        # PIECEWISE for batches where this gate would be True, so FCA only
        # ever runs inside a piecewise replay.
        want_forest_cascade = (
            self.forest_cascade_enabled
            and causal
            and num_reqs >= int(self.forest_min_batch_size)
            and max_query_len <= int(self.forest_max_query_len)
            and self.dcp_world_size <= 1
        )

        forest_meta = None
        if want_forest_cascade:
            forest_meta = _try_build_forest_cascade_metadata(
                block_table=block_table_tensor[:num_reqs],
                seq_lens=seq_lens[:num_reqs],
                query_start_loc=query_start_loc[: num_reqs + 1],
                common_prefix_len=common_prefix_len,
                block_size=self.block_size,
                min_group_size=self.forest_min_group_size,
                min_additional_prefix_blocks=(
                    self.forest_min_additional_prefix_blocks
                ),
                min_non_singleton_fraction=(
                    self.forest_min_non_singleton_fraction
                ),
                max_non_singleton_groups=(
                    self.forest_max_non_singleton_groups
                ),
            )

        if self.dcp_world_size > 1:
            query_kv_lens = query_start_loc[1:] - query_start_loc[:-1]
            dcp_context_kv_lens = seq_lens - query_kv_lens

            dcp_context_kv_lens = get_dcp_local_seq_lens(
                dcp_context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            num_partitions = (
                self.dcp_world_size * self.cp_kv_cache_interleave_size
            )
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif forest_meta is not None:
            use_forest_cascade = True
            use_cascade = True

            # print(
            #     f"[FCA] path=forest_cascade num_reqs={num_reqs} "
            #     f"num_groups={int(forest_meta['group_sizes'].numel())} "
            #     f"max_group_size={int(forest_meta['max_group_size'])} "
            #     f"max_prefix_len={int(forest_meta['max_prefix_len'])} "
            #     f"max_suffix_len={int(forest_meta['max_suffix_len'])} "
            #     f"max_query_len={max_query_len}",
            #     flush=True,
            # )

            forest_perm = forest_meta["perm"]
            forest_token_perm = forest_meta["token_perm"]
            forest_packed_query_start_loc = forest_meta[
                "packed_query_start_loc"
            ]
            forest_group_offsets = forest_meta["group_offsets"]
            forest_group_sizes = forest_meta["group_sizes"]
            forest_prefix_kv_lens = forest_meta["prefix_kv_lens"]
            forest_prefix_block_table = forest_meta["prefix_block_table"]
            forest_suffix_kv_lens = forest_meta["suffix_kv_lens"]
            forest_suffix_block_table = forest_meta["suffix_block_table"]
            forest_max_group_query_len = forest_meta["max_group_query_len"]
            forest_max_group_size = forest_meta["max_group_size"]
            forest_max_prefix_len = forest_meta["max_prefix_len"]
            forest_max_suffix_len = forest_meta["max_suffix_len"]

            # Forest-cascade currently does not use the AOT scheduler.
            prefix_scheduler_metadata = None
            scheduler_metadata = None

        elif use_cascade:
            # print(
            #     f"[FCA] path=single_tree_cascade num_reqs={num_reqs} "
            #     f"common_prefix_len={common_prefix_len} "
            #     f"max_query_len={max_query_len}",
            #     flush=True,
            # )
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens],
                dtype=torch.int32,
                device=self.device,
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len],
                dtype=torch.int32,
                device=self.device,
            )
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            # print(
            #     f"[FCA] path=standard_flash num_reqs={num_reqs} "
            #     f"max_query_len={max_query_len} "
            #     f"want_forest_cascade={want_forest_cascade}",
            #     flush=True,
            # )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )

        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            use_forest_cascade=use_forest_cascade,
            forest_perm=forest_perm,
            forest_token_perm=forest_token_perm,
            forest_packed_query_start_loc=forest_packed_query_start_loc,
            forest_group_offsets=forest_group_offsets,
            forest_group_sizes=forest_group_sizes,
            forest_prefix_kv_lens=forest_prefix_kv_lens,
            forest_prefix_block_table=forest_prefix_block_table,
            forest_suffix_kv_lens=forest_suffix_kv_lens,
            forest_suffix_block_table=forest_suffix_block_table,
            forest_max_group_query_len=forest_max_group_query_len,
            forest_max_group_size=forest_max_group_size,
            forest_max_prefix_len=forest_max_prefix_len,
            forest_max_suffix_len=forest_max_suffix_len,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping

        # If we are using forest-cascade, the derived per-group/per-request
        # block tables depend on block ids, so we must rebuild them.
        if getattr(metadata, "use_forest_cascade", False):
            num_reqs = int(new_metadata.query_start_loc.numel() - 1)
            forest_meta = _try_build_forest_cascade_metadata(
                block_table=blk_table[:num_reqs],
                seq_lens=new_metadata.seq_lens[:num_reqs],
                query_start_loc=new_metadata.query_start_loc[
                    : num_reqs + 1
                ],
                common_prefix_len=new_metadata.common_prefix_len,
                block_size=self.block_size,
                min_group_size=self.forest_min_group_size,
                min_additional_prefix_blocks=(
                    self.forest_min_additional_prefix_blocks
                ),
                min_non_singleton_fraction=(
                    self.forest_min_non_singleton_fraction
                ),
                max_non_singleton_groups=(
                    self.forest_max_non_singleton_groups
                ),
            )

            if forest_meta is None:
                # Fall back to the most appropriate non-forest path.
                new_metadata.use_forest_cascade = False
                new_metadata.forest_perm = None
                new_metadata.forest_token_perm = None
                new_metadata.forest_packed_query_start_loc = None
                new_metadata.forest_group_offsets = None
                new_metadata.forest_group_sizes = None
                new_metadata.forest_prefix_kv_lens = None
                new_metadata.forest_prefix_block_table = None
                new_metadata.forest_suffix_kv_lens = None
                new_metadata.forest_suffix_block_table = None
                new_metadata.forest_max_group_query_len = None
                new_metadata.forest_max_group_size = None
                new_metadata.forest_max_prefix_len = None
                new_metadata.forest_max_suffix_len = None

                if new_metadata.common_prefix_len > 0:
                    new_metadata.use_cascade = True
                    if new_metadata.cu_prefix_query_lens is None:
                        num_actual_tokens = int(
                            new_metadata.num_actual_tokens
                        )
                        new_metadata.cu_prefix_query_lens = torch.tensor(
                            [0, num_actual_tokens],
                            dtype=torch.int32,
                            device=blk_table.device,
                        )
                    if new_metadata.prefix_kv_lens is None:
                        new_metadata.prefix_kv_lens = torch.tensor(
                            [new_metadata.common_prefix_len],
                            dtype=torch.int32,
                            device=blk_table.device,
                        )
                    if new_metadata.suffix_kv_lens is None:
                        new_metadata.suffix_kv_lens = (
                            new_metadata.seq_lens[:num_reqs]
                            - new_metadata.common_prefix_len
                        )
                else:
                    new_metadata.use_cascade = False
                    new_metadata.cu_prefix_query_lens = None
                    new_metadata.prefix_kv_lens = None
                    new_metadata.suffix_kv_lens = None

                new_metadata.prefix_scheduler_metadata = None
                new_metadata.scheduler_metadata = None
            else:
                new_metadata.use_forest_cascade = True
                new_metadata.use_cascade = True

                new_metadata.forest_perm = forest_meta["perm"]
                new_metadata.forest_token_perm = forest_meta["token_perm"]
                new_metadata.forest_packed_query_start_loc = forest_meta[
                    "packed_query_start_loc"
                ]
                new_metadata.forest_group_offsets = forest_meta[
                    "group_offsets"
                ]
                new_metadata.forest_group_sizes = forest_meta["group_sizes"]
                new_metadata.forest_prefix_kv_lens = forest_meta[
                    "prefix_kv_lens"
                ]
                new_metadata.forest_prefix_block_table = forest_meta[
                    "prefix_block_table"
                ]
                new_metadata.forest_suffix_kv_lens = forest_meta[
                    "suffix_kv_lens"
                ]
                new_metadata.forest_suffix_block_table = forest_meta[
                    "suffix_block_table"
                ]
                new_metadata.forest_max_group_query_len = forest_meta[
                    "max_group_query_len"
                ]
                new_metadata.forest_max_group_size = forest_meta[
                    "max_group_size"
                ]
                new_metadata.forest_max_prefix_len = forest_meta[
                    "max_prefix_len"
                ]
                new_metadata.forest_max_suffix_len = forest_meta[
                    "max_suffix_len"
                ]

                new_metadata.prefix_scheduler_metadata = None
                new_metadata.scheduler_metadata = None

                new_metadata.cu_prefix_query_lens = None
                new_metadata.prefix_kv_lens = None
                new_metadata.suffix_kv_lens = None

        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        if (
            is_quantized_kv_cache(self.kv_cache_dtype)
            and not flash_attn_supports_fp8()
        ):
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        attn_type = self.attn_type
        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(0)

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=sliding_window_size,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=self.sinks,
                )
                return output

        # Cascade attention (rare case).
        if getattr(attn_metadata, "use_forest_cascade", False):
            forest_cascade_attention(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                token_perm=attn_metadata.forest_token_perm,
                packed_query_start_loc=(
                    attn_metadata.forest_packed_query_start_loc
                ),
                group_offsets=attn_metadata.forest_group_offsets,
                group_sizes=attn_metadata.forest_group_sizes,
                prefix_kv_lens=attn_metadata.forest_prefix_kv_lens,
                prefix_block_table=attn_metadata.forest_prefix_block_table,
                suffix_kv_lens=attn_metadata.forest_suffix_kv_lens,
                suffix_block_table=attn_metadata.forest_suffix_block_table,
                max_query_len=attn_metadata.max_query_len,
                max_group_query_len=(
                    attn_metadata.forest_max_group_query_len
                ),
                max_group_size=attn_metadata.forest_max_group_size,
                max_prefix_len=attn_metadata.forest_max_prefix_len,
                max_suffix_len=attn_metadata.forest_max_suffix_len,
                softmax_scale=self.scale,
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                max_num_splits=attn_metadata.max_num_splits,
                fa_version=self.vllm_flash_attn_version,
                prefix_scheduler_metadata=(
                    attn_metadata.prefix_scheduler_metadata
                ),
                suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
                q_descale=layer._q_scale,
                k_descale=layer._k_scale,
                v_descale=layer._v_scale,
                s_aux=self.sinks,
            )
            return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=(
                attn_metadata.prefix_scheduler_metadata
            ),
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window)
            if self.sliding_window is not None
            else None
        )
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        context_attn_out_cor, context_lse_cor = cp_lse_ag_out_rs(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache."""
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,
            self.num_kv_heads,
        )

        sliding_window_size = (
            list(self.sliding_window)
            if self.sliding_window is not None
            else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention."""
    if common_prefix_len < 256:
        return False
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    if dcp_world_size > 1:
        return False

    num_queries_per_kv = num_query_heads // num_kv_heads
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        return True

    num_tokens = num_reqs
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return True
    #return cascade_time < flash_decoding_time


# -----------------------------------------------------------------------------
# Forest cascade attention (ragged groups)
# -----------------------------------------------------------------------------


def _recursive_lcp_split(
    adj_lcps, full_sorted, start_block,
    min_group_size, min_additional_prefix_blocks, B,
):
    """Partition sorted requests by recursively splitting at LCP minima."""

    def _prefix(a, b):
        if b - a <= 1:
            return int(full_sorted[a])
        sub_min_lcp = int(np.min(adj_lcps[a:b - 1]))
        sub_min_full = int(np.min(full_sorted[a:b]))
        return min(start_block + sub_min_lcp, sub_min_full)

    results = []
    stack = [(0, B)]

    while stack:
        s, e = stack.pop()
        n = e - s
        if n <= 0:
            continue
        if n == 1:
            results.append((s, e, int(full_sorted[s])))
            continue

        rng = adj_lcps[s:e - 1]
        min_idx_rel = int(np.argmin(rng))
        min_lcp = int(rng[min_idx_rel])
        max_lcp = int(np.max(rng))
        group_pf = _prefix(s, e)

        split_pos = s + min_idx_rel + 1
        left_n = split_pos - s
        right_n = e - split_pos

        left_pf = _prefix(s, split_pos)
        right_pf = _prefix(split_pos, e)

        unsplit_benefit = (
            n * max(group_pf - start_block, 0)
            if n >= min_group_size
            else 0
        )
        split_benefit = 0
        if left_n >= min_group_size:
            split_benefit += left_n * max(left_pf - start_block, 0)
        if right_n >= min_group_size:
            split_benefit += right_n * max(right_pf - start_block, 0)

        should_split = False
        if split_benefit > unsplit_benefit:
            should_split = True
        elif max_lcp > min_lcp * 2:
            should_split = True

        if should_split:
            stack.append((s, split_pos))
            stack.append((split_pos, e))
        else:
            results.append((s, e, group_pf))

    results.sort()
    return results


def _forest_sort_and_group(
    block_table: torch.Tensor,
    full_blocks: torch.Tensor,
    start_block: int,
    min_group_size: int,
    min_additional_prefix_blocks: int,
    min_non_singleton_fraction: float,
    max_non_singleton_groups: int,
) -> dict | None:
    """Sort requests by block table and form groups by LCP."""
    B = int(block_table.shape[0])
    max_cols = int(block_table.shape[1])
    device = block_table.device

    if B <= 1:
        return None

    effective_max_col = min(max_cols, int(full_blocks.max().item()))
    num_sort_cols = effective_max_col - start_block
    if num_sort_cols <= 0:
        return None

    bt_work = block_table[:, start_block:effective_max_col].contiguous()
    col_ids = torch.arange(
        start_block, effective_max_col, device=device, dtype=torch.int32
    )
    valid = col_ids.unsqueeze(0) < full_blocks.unsqueeze(1)
    bt_work = torch.where(valid, bt_work, torch.full_like(bt_work, -1))

    bt_cpu = bt_work.cpu().numpy().astype(np.int64)
    keys = tuple(bt_cpu[:, c] for c in reversed(range(num_sort_cols)))
    sorted_indices = np.lexsort(keys)
    perm = torch.from_numpy(sorted_indices.astype(np.int64)).to(
        device=device, dtype=torch.int64
    )

    bt_sorted = bt_work[perm]
    full_sorted = full_blocks[perm]

    same = bt_sorted[:-1] == bt_sorted[1:]
    raw_lcps = torch.cumprod(same.to(torch.int32), dim=1).sum(dim=1)

    min_full_pair = (
        torch.minimum(full_sorted[:-1], full_sorted[1:]) - start_block
    ).clamp(min=0).to(torch.int32)
    adj_lcps = torch.minimum(raw_lcps, min_full_pair)

    adj_lcps_cpu = adj_lcps.cpu().numpy().astype(np.int64)
    full_sorted_cpu = full_sorted.cpu().numpy().astype(np.int64)

    group_ranges = _recursive_lcp_split(
        adj_lcps_cpu,
        full_sorted_cpu,
        start_block,
        min_group_size,
        min_additional_prefix_blocks,
        B,
    )

    if not group_ranges:
        return None

    g_starts = np.array([r[0] for r in group_ranges], dtype=np.int64)
    g_ends = np.array([r[1] for r in group_ranges], dtype=np.int64)
    g_prefix = np.array([r[2] for r in group_ranges], dtype=np.int64)

    starts = torch.from_numpy(g_starts).to(device=device, dtype=torch.int64)
    ends = torch.from_numpy(g_ends).to(device=device, dtype=torch.int64)
    total_prefix_blocks_g = torch.from_numpy(g_prefix).to(
        device=device, dtype=torch.int32
    )
    group_sizes = (ends - starts).to(torch.int32)
    G = int(group_sizes.numel())
    if G <= 0:
        return None

    too_short = (
        (total_prefix_blocks_g - start_block)
        < min_additional_prefix_blocks
    )
    total_prefix_blocks_g[too_short] = start_block

    non_singleton = group_sizes >= int(min_group_size)
    num_non_singleton_groups = int(non_singleton.sum().item())
    if num_non_singleton_groups == 0:
        return None
    if num_non_singleton_groups > int(max_non_singleton_groups):
        return None

    num_non_singleton_reqs = int(group_sizes[non_singleton].sum().item())
    if (num_non_singleton_reqs / float(B)) < float(
        min_non_singleton_fraction
    ):
        return None

    has_additional = bool(
        ((total_prefix_blocks_g > start_block) & non_singleton).any().item()
    )
    if not has_additional:
        return None

    bt_p = block_table.index_select(0, perm)

    return {
        "perm": perm,
        "bt_p": bt_p,
        "starts": starts,
        "ends": ends,
        "group_sizes": group_sizes,
        "total_prefix_blocks_g": total_prefix_blocks_g,
    }


def _forest_make_suffix_block_table(
    full_block_table: torch.Tensor,
    prefix_blocks_per_req: torch.Tensor,
    suffix_blocks_per_req: torch.Tensor,
    max_suffix_blocks: int,
) -> torch.Tensor:
    """Create a per-request suffix block table after prefix blocks."""
    B, full_cols = full_block_table.shape
    if max_suffix_blocks <= 0:
        return full_block_table.new_empty((B, 0))

    device = full_block_table.device
    idx_in_suffix = (
        torch.arange(max_suffix_blocks, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(B, -1)
    )

    idx_in_full = idx_in_suffix + prefix_blocks_per_req.to(
        torch.int64
    ).unsqueeze(1)

    idx_in_full_clamped = idx_in_full.clamp(min=0, max=full_cols - 1)
    gathered = torch.gather(full_block_table, 1, idx_in_full_clamped)

    valid = (
        idx_in_suffix
        < suffix_blocks_per_req.to(torch.int64).unsqueeze(1)
    ) & (idx_in_full < full_cols)
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def _try_build_forest_cascade_metadata(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    common_prefix_len: int,
    block_size: int,
    min_group_size: int,
    min_additional_prefix_blocks: int,
    min_non_singleton_fraction: float,
    max_non_singleton_groups: int,
) -> dict | None:
    """Build forest-cascade metadata or return None if not beneficial."""
    B = int(block_table.shape[0])
    if B == 0:
        return None
    if common_prefix_len < 0:
        return None
    if common_prefix_len % block_size != 0:
        return None
    if query_start_loc.numel() != B + 1:
        return None

    device = block_table.device

    query_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(
        torch.int32
    )

    seq_lens_i32 = seq_lens.to(torch.int32)
    context_lens = (seq_lens_i32 - query_lens).to(torch.int32)
    if torch.any(context_lens < 0):
        return None

    prefix0_blocks = common_prefix_len // block_size

    min_context_blocks = int((context_lens.min().item()) // block_size)
    if prefix0_blocks > min_context_blocks:
        return None

    full_blocks = (context_lens // block_size).to(torch.int32)
    start_block = int(prefix0_blocks)

    group_result = _forest_sort_and_group(
        block_table=block_table,
        full_blocks=full_blocks,
        start_block=start_block,
        min_group_size=min_group_size,
        min_additional_prefix_blocks=min_additional_prefix_blocks,
        min_non_singleton_fraction=min_non_singleton_fraction,
        max_non_singleton_groups=max_non_singleton_groups,
    )
    if group_result is None:
        return None

    perm = group_result["perm"]
    bt_p = group_result["bt_p"]
    starts = group_result["starts"]
    ends = group_result["ends"]
    group_sizes = group_result["group_sizes"]
    total_prefix_blocks_g = group_result["total_prefix_blocks_g"]
    G = int(group_sizes.numel())

    query_lens_packed = query_lens.index_select(0, perm).to(torch.int32)

    packed_query_start_loc = torch.empty(
        (B + 1,), dtype=torch.int32, device=device
    )
    packed_query_start_loc[0] = 0
    packed_query_start_loc[1:] = torch.cumsum(query_lens_packed, dim=0)

    total_tokens = int(packed_query_start_loc[-1].item())
    if total_tokens <= 0:
        return None

    ql_prefix = packed_query_start_loc
    group_query_lens = (
        ql_prefix.index_select(0, ends)
        - ql_prefix.index_select(0, starts)
    ).to(torch.int32)

    group_offsets = torch.empty(
        (G + 1,), dtype=torch.int32, device=device
    )
    group_offsets[0] = 0
    group_offsets[1:] = torch.cumsum(group_query_lens, dim=0)

    max_group_query_len = int(group_query_lens.max().item())

    packed_req_ids = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int32),
        query_lens_packed.to(torch.int64),
    )
    start_pos = packed_query_start_loc.index_select(
        0, packed_req_ids.to(torch.int64)
    )
    pos_in_req = (
        torch.arange(total_tokens, device=device, dtype=torch.int32)
        - start_pos
    )

    orig_req_ids = perm.index_select(0, packed_req_ids.to(torch.int64))
    orig_start_pos = query_start_loc.index_select(0, orig_req_ids)
    token_perm = orig_start_pos.to(torch.int64) + pos_in_req.to(
        torch.int64
    )

    max_prefix_blocks = int(total_prefix_blocks_g.max().item())
    prefix_kv_lens = (total_prefix_blocks_g * block_size).to(torch.int32)
    max_prefix_len = int(prefix_kv_lens.max().item())
    max_group_size = int(group_sizes.max().item())

    prefix_block_table = (
        bt_p.index_select(0, starts)[:, :max_prefix_blocks].contiguous()
    )

    group_ids_req = torch.repeat_interleave(
        torch.arange(G, device=device, dtype=torch.int32),
        group_sizes.to(torch.int64),
    )
    prefix_blocks_req = total_prefix_blocks_g.index_select(
        0, group_ids_req.to(torch.int64)
    )
    prefix_lens_req = (prefix_blocks_req * block_size).to(torch.int32)

    seq_lens_packed = seq_lens_i32.index_select(0, perm).to(torch.int32)
    suffix_kv_lens = (seq_lens_packed - prefix_lens_req).to(torch.int32)

    if torch.any(suffix_kv_lens < 0):
        return None

    max_suffix_len = int(suffix_kv_lens.max().item())
    suffix_blocks_req = (
        (suffix_kv_lens + (block_size - 1)) // block_size
    ).to(torch.int32)
    max_suffix_blocks = int(suffix_blocks_req.max().item())

    suffix_block_table = _forest_make_suffix_block_table(
        bt_p, prefix_blocks_req, suffix_blocks_req, max_suffix_blocks
    ).contiguous()

    # print(
    #     f"[FCA] groups: prefix_kv_lens={prefix_kv_lens.cpu().tolist()} "
    #     f"group_sizes={group_sizes.cpu().tolist()}",
    #     flush=True,
    # )

    return {
        "perm": perm,
        "token_perm": token_perm,
        "packed_query_start_loc": packed_query_start_loc,
        "group_offsets": group_offsets,
        "group_sizes": group_sizes,
        "prefix_kv_lens": prefix_kv_lens,
        "prefix_block_table": prefix_block_table,
        "suffix_kv_lens": suffix_kv_lens,
        "suffix_block_table": suffix_block_table,
        "max_group_query_len": max_group_query_len,
        "max_group_size": max_group_size,
        "max_prefix_len": max_prefix_len,
        "max_suffix_len": max_suffix_len,
    }


def forest_cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    token_perm: torch.Tensor,
    packed_query_start_loc: torch.Tensor,
    group_offsets: torch.Tensor,
    group_sizes: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    prefix_block_table: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    suffix_block_table: torch.Tensor,
    max_query_len: int,
    max_group_query_len: int,
    max_group_size: int,
    max_prefix_len: int,
    max_suffix_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forest-cascade attention forward (supports multi-token query blocks)."""
    num_tokens = int(query.shape[0])
    B = int(suffix_kv_lens.numel())

    q_packed = torch.index_select(query, 0, token_perm)

    num_kv_heads = key_cache.shape[-2]
    num_groups = group_offsets.shape[0] - 1
    prefix_descale_shape = (num_groups, num_kv_heads)
    suffix_descale_shape = (B, num_kv_heads)

    effective_num_splits = (
        1 if vllm_is_batch_invariant() else max_num_splits
    )

    # Prefix attention: groups as sequences, causal=False.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=q_packed,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=group_offsets,
        max_seqlen_q=int(max_group_query_len),
        seqused_k=prefix_kv_lens,
        max_seqlen_k=int(max_prefix_len),
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        softcap=logits_soft_cap,
        alibi_slopes=alibi_slopes,
        block_table=prefix_block_table,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=(
            q_descale.expand(prefix_descale_shape)
            if q_descale is not None
            else None
        ),
        k_descale=(
            k_descale.expand(prefix_descale_shape)
            if k_descale is not None
            else None
        ),
        v_descale=(
            v_descale.expand(prefix_descale_shape)
            if v_descale is not None
            else None
        ),
        num_splits=effective_num_splits,
        s_aux=s_aux,
    )

    # Suffix attention: per-request, causal=True.
    if int(max_suffix_len) > 0:
        suffix_output, suffix_lse = flash_attn_varlen_func(
            q=q_packed,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=packed_query_start_loc,
            max_seqlen_q=int(max_query_len),
            seqused_k=suffix_kv_lens,
            max_seqlen_k=int(max_suffix_len),
            softmax_scale=softmax_scale,
            causal=True,
            window_size=list(sliding_window),
            softcap=logits_soft_cap,
            alibi_slopes=alibi_slopes,
            block_table=suffix_block_table,
            return_softmax_lse=True,
            scheduler_metadata=suffix_scheduler_metadata,
            fa_version=fa_version,
            q_descale=(
                q_descale.expand(suffix_descale_shape)
                if q_descale is not None
                else None
            ),
            k_descale=(
                k_descale.expand(suffix_descale_shape)
                if k_descale is not None
                else None
            ),
            v_descale=(
                v_descale.expand(suffix_descale_shape)
                if v_descale is not None
                else None
            ),
            num_splits=effective_num_splits,
        )
    else:
        suffix_output = torch.zeros_like(prefix_output)
        suffix_lse = torch.full_like(prefix_lse, float("-inf"))

    out_packed = torch.empty_like(prefix_output)
    merge_attn_states(
        out_packed, prefix_output, prefix_lse, suffix_output, suffix_lse
    )

    output.index_copy_(0, token_perm, out_packed)
    return output


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (
        cu_prefix_query_lens.shape[0] - 1,
        key_cache.shape[-2],
    )

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=(
            q_descale.expand(descale_shape)
            if q_descale is not None
            else None
        ),
        k_descale=(
            k_descale.expand(descale_shape)
            if k_descale is not None
            else None
        ),
        v_descale=(
            v_descale.expand(descale_shape)
            if v_descale is not None
            else None
        ),
        s_aux=s_aux,
        num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=(
            q_descale.expand(descale_shape)
            if q_descale is not None
            else None
        ),
        k_descale=(
            k_descale.expand(descale_shape)
            if k_descale is not None
            else None
        ),
        v_descale=(
            v_descale.expand(descale_shape)
            if v_descale is not None
            else None
        ),
        num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(
        output, prefix_output, prefix_lse, suffix_output, suffix_lse
    )
