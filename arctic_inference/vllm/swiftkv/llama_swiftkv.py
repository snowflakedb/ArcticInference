# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

import vllm.distributed.parallel_state as parallel_state
from vllm.attention import Attention, AttentionMetadata
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, is_pp_missing_parameter, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

import arctic_inference.vllm.model_runner as model_runner
from arctic_inference.common.swiftkv.configs import LlamaSwiftKVConfig
from arctic_inference.vllm.swiftkv.linear import SwiftKVLinear

logger = init_logger(__name__)


def get_attn_metadata_for_swiftkv():
    fwd_ctx = get_forward_context()
    if fwd_ctx.attn_metadata is None:
        return None
    meta = next(iter(fwd_ctx.attn_metadata.values()))
    assert all(m is meta for m in fwd_ctx.attn_metadata.values()), \
        "All attention metadata should be the same for LlamaSwiftKV."
    return meta


class LlamaSwiftKVAttention(nn.Module):

    def __init__(
        self,
        config: LlamaSwiftKVConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: Optional[AttentionType] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attn_type = attn_type

        self.q_proj_swiftkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_swiftkv",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=None,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj_swiftkv(hidden_states)
        q, _ = self.rotary_emb(positions, q, torch.empty_like(k))
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaSwiftKVDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaSwiftKVConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaSwiftKVAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            k=k_states,
            v=v_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class LlamaSwiftKVPrefillRunner(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, model: "LlamaSwiftKVModel",
                 prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self._model = [model]  # Box it to avoid recursive registration

    @property
    def model(self) -> "LlamaSwiftKVModel":
        return self._model[0]

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, IntermediateTensors,
               IntermediateTensors]:
        hidden_states = self.model.get_input_embeddings(input_ids)
        residual = None
        prefill_layers = self.model.layers[:self.config.num_key_value_layers]
        for idx, layer in enumerate(prefill_layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        return hidden_states, residual


@support_torch_compile
class LlamaSwiftKVDecodeRunner(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, model: "LlamaSwiftKVModel",
                 prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self._model = [model]  # Box it to avoid recursive registration

    @property
    def model(self) -> "LlamaSwiftKVModel":
        return self._model[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
    ) -> torch.Tensor:
        num_layers = (self.config.num_hidden_layers -
                      self.config.num_key_value_layers)
        k_split = torch.chunk(k_states, num_layers, dim=-1)
        v_split = torch.chunk(v_states, num_layers, dim=-1)
        for idx, layer in enumerate(
                self.model.layers[self.config.num_key_value_layers:]):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                k_split[idx],
                v_split[idx],
                residual,
            )
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states


class LlamaSwiftKVModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=self.quant_config,
        )
        self.layers = torch.nn.ModuleList([
            LlamaDecoderLayer(config=config,
                              cache_config=vllm_config.cache_config,
                              quant_config=vllm_config.quant_config,
                              prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_key_value_layers)
        ])
        with model_runner.set_shift_parallel_mode(True):
            self.layers.extend([
                LlamaSwiftKVDecoderLayer(config=config,
                                         cache_config=vllm_config.cache_config,
                                         quant_config=vllm_config.quant_config,
                                         prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_key_value_layers,
                                 config.num_hidden_layers)
            ])
            self.swiftkv_proj = SwiftKVLinear(
                num_layers=(config.num_hidden_layers - config.num_key_value_layers) // config.key_value_group_size,
                hidden_size=config.hidden_size,
                head_size=self.layers[0].self_attn.head_dim,
                total_num_kv_heads=self.layers[0].self_attn.total_num_kv_heads,
                bias=(getattr(config, "bias", False) or
                    getattr(config, "attention_bias", False)),
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.swiftkv_proj",
            )
            self.norm_swiftkv = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        for param in self.layers[config.num_key_value_layers:].parameters():
            param.shift_parallel_mode = True
        for param in self.swiftkv_proj.parameters():
            param.shift_parallel_mode = True

        self._init_prefill_runner(vllm_config)
        self._init_decode_runner(vllm_config)

        from arctic_inference.py_custom_ops import try_load_torch_library
        self.use_custom_ops = True if try_load_torch_library() else False

    def _init_prefill_runner(self, vllm_config: VllmConfig):
        vllm_config.compilation_config = copy.copy(
            vllm_config.compilation_config)
        vllm_config.compilation_config.inductor_compile_config = (
            vllm_config.compilation_config.inductor_compile_config.copy())
        self.prefill_runner = LlamaSwiftKVPrefillRunner(
            vllm_config=vllm_config, model=self)

    def _init_decode_runner(self, vllm_config: VllmConfig):
        vllm_config.compilation_config = copy.copy(
            vllm_config.compilation_config)
        vllm_config.compilation_config.inductor_compile_config = (
            vllm_config.compilation_config.inductor_compile_config.copy())
        self.decode_runner = LlamaSwiftKVDecodeRunner(
            vllm_config=vllm_config, model=self)

        config = vllm_config.model_config.hf_config
        if vllm_config.compilation_config.cudagraph_capture_sizes:
            self.cuda_graph_max_batch_size = max(
                vllm_config.compilation_config.cudagraph_capture_sizes)
            kv_size = sum(self.swiftkv_proj.output_partition_sizes) // 2
            self.decode_runner.inputs = {
                "hidden_states": torch.empty(self.cuda_graph_max_batch_size,
                                            config.hidden_size, device="cuda"),
                "residual": torch.empty(self.cuda_graph_max_batch_size,
                                        config.hidden_size, device="cuda"),
                "positions": torch.empty(self.cuda_graph_max_batch_size,
                                        dtype=torch.long, device="cuda"),
                "k_states": torch.empty(self.cuda_graph_max_batch_size,
                                        kv_size, device="cuda"),
                "v_states": torch.empty(self.cuda_graph_max_batch_size,
                                        kv_size, device="cuda"),
            }
        else:
            self.cuda_graph_max_batch_size = 0

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def swiftkv_gather(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sp_size = parallel_state._SP.world_size
        device_group = parallel_state._SP.device_group
        if sp_size > 1 and not model_runner.is_shift_parallel_mode():
            # All-gather hidden_states
            hidden_states_ = hidden_states.new_empty(
                hidden_states.size(0) * sp_size, hidden_states.size(1))
            op1 = torch.distributed.all_gather_into_tensor(
                hidden_states_, hidden_states, group=device_group, async_op=True)
            hidden_states = hidden_states_
            # All-gather residual
            residual_ = residual.new_empty(
                residual.size(0) * sp_size, residual.size(1))
            op2 = torch.distributed.all_gather_into_tensor(
                residual_, residual, group=device_group, async_op=True)
            residual = residual_
            # All-gather positions
            positions_ = positions.new_empty(positions.size(0) * sp_size)
            op3 = torch.distributed.all_gather_into_tensor(
                positions_, positions, group=device_group, async_op=True)
            positions = positions_
            # Wait for all-gathers to finish
            op1.wait()
            op2.wait()
            op3.wait()
        return hidden_states, residual, positions

    def swiftkv_select(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, IntermediateTensors,
               IntermediateTensors]:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = get_attn_metadata_for_swiftkv()
        if attn_metadata is None:
            return hidden_states, residual, positions, k_states, v_states

        if self.use_custom_ops:
            key_caches : List[torch.Tensor] = []
            value_caches : List[torch.Tensor] = []
            k_scales : List[torch.Tensor] = []
            v_scales : List[torch.Tensor] = []
            num_heads = self.layers[-1].self_attn.attn.num_kv_heads
            head_size = self.layers[-1].self_attn.attn.head_size
            for idx, layer in enumerate(
                    self.layers[self.config.num_key_value_layers:]):
                attn = layer.self_attn.attn
                kv_cache = attn.kv_cache[forward_context.virtual_engine]
                if kv_cache.numel():
                    key_caches.append(kv_cache[0])
                    value_caches.append(kv_cache[1])
                    k_scales.append(attn._k_scale)
                    v_scales.append(attn._v_scale)

            if len(key_caches) > 0:
                from arctic_inference.py_custom_ops import reshape_and_cache_flash_bulk
                reshape_and_cache_flash_bulk(
                    k_states, v_states, key_caches, value_caches, attn_metadata.slot_mapping,
                    attn.kv_cache_dtype, k_scales, v_scales, num_heads, head_size)
        else:
            num_layers = (self.config.num_hidden_layers -
                          self.config.num_key_value_layers)

            k_split = k_states.chunk(num_layers, dim=-1)
            v_split = v_states.chunk(num_layers, dim=-1)

            for idx, layer in enumerate(
                    self.layers[self.config.num_key_value_layers:]):
                attn = layer.self_attn.attn
                kv_cache = attn.kv_cache[forward_context.virtual_engine]
                if kv_cache.numel():
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        k_split[idx].view(-1, attn.num_kv_heads, attn.head_size),
                        v_split[idx].view(-1, attn.num_kv_heads, attn.head_size),
                        kv_cache[0],
                        kv_cache[1],
                        attn_metadata.slot_mapping,
                        attn.kv_cache_dtype,
                        attn._k_scale,
                        attn._v_scale,
                    )

        logits_indices = attn_metadata.swiftkv_logits_indices

        attn_metadata.num_actual_tokens = logits_indices.numel()
        attn_metadata.num_input_tokens = logits_indices.numel()
        attn_metadata.query_start_loc = torch.searchsorted(
            logits_indices, attn_metadata.query_start_loc, out_int32=True)
        attn_metadata.slot_mapping = attn_metadata.slot_mapping[logits_indices]

        # TODO: Make cascade attention work with SwiftKV
        attn_metadata.use_cascade = False
        attn_metadata.cu_prefix_query_lens = None
        attn_metadata.prefix_kv_lens = None
        attn_metadata.suffix_kv_lens = None

        hidden_states = hidden_states[logits_indices]
        residual = residual[logits_indices]
        positions = positions[logits_indices]
        k_states = k_states[logits_indices]
        v_states = v_states[logits_indices]
        return hidden_states, residual, positions, k_states, v_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states, residual = self.prefill_runner(
            input_ids,
            positions,
        )

        # Gather hidden states, residuals and positions
        hidden_states, residual, positions = self.swiftkv_gather(
            hidden_states,
            residual,
            positions,
        )

        with model_runner.set_shift_parallel_mode(True):

            # KV projection of all the remaining layers
            swiftkv_hidden_states = (
                self.norm_swiftkv.forward_cuda(hidden_states + residual))
            kv_states, _ = self.swiftkv_proj(swiftkv_hidden_states)

            # Update the KV cache of all the remaining layers. We concatenate all
            # the heads together to run it as a batch.
            q_states = hidden_states.new_empty(  # Just temporary buffer
                (hidden_states.size(0),
                self.swiftkv_proj.num_layers * hidden_states.size(1)))
            k_states, v_states = kv_states.split(kv_states.size(-1) // 2, dim=-1)
            rotary_emb = self.layers[0].self_attn.rotary_emb
            _, k_states = rotary_emb.forward_cuda(positions, q_states, k_states)

            orig_hidden_states = hidden_states
            hidden_states, residual, positions, k_states, v_states = (
                self.swiftkv_select(
                    hidden_states,
                    residual,
                    positions,
                    k_states,
                    v_states,
                )
            )

            size = hidden_states.shape[0]
            if size <= self.cuda_graph_max_batch_size:
                self.decode_runner.inputs["hidden_states"][:size].copy_(hidden_states)
                hidden_states = self.decode_runner.inputs["hidden_states"][:size]
                self.decode_runner.inputs["residual"][:size].copy_(residual)
                residual = self.decode_runner.inputs["residual"][:size]
                self.decode_runner.inputs["positions"][:size].copy_(positions)
                positions = self.decode_runner.inputs["positions"][:size]
                self.decode_runner.inputs["k_states"][:size].copy_(k_states)
                k_states = self.decode_runner.inputs["k_states"][:size]
                self.decode_runner.inputs["v_states"][:size].copy_(v_states)
                v_states = self.decode_runner.inputs["v_states"][:size]

            hidden_states = self.decode_runner(
                hidden_states,
                residual,
                positions,
                k_states,
                v_states,
            )

            attn_metadata = get_attn_metadata_for_swiftkv()
            if attn_metadata is not None:
                logits_indices = attn_metadata.swiftkv_logits_indices
                orig_hidden_states[logits_indices] = hidden_states[:size]

        return orig_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        for layer_idx in range(self.config.num_key_value_layers):
            stacked_params_mapping.extend([
                (f"layers.{layer_idx}.self_attn.qkv_proj",
                 f"layers.{layer_idx}.self_attn.q_proj", "q"),
                (f"layers.{layer_idx}.self_attn.qkv_proj",
                 f"layers.{layer_idx}.self_attn.k_proj", "k"),
                (f"layers.{layer_idx}.self_attn.qkv_proj",
                 f"layers.{layer_idx}.self_attn.v_proj", "v"),
            ])
        for i, layer_idx in enumerate(range(self.config.num_key_value_layers,
                                            self.config.num_hidden_layers,
                                            self.config.key_value_group_size)):
            stacked_params_mapping.extend([
                (f"swiftkv_proj",
                 f"layers.{layer_idx}.self_attn.k_proj_swiftkv", (i, "k")),
                (f"swiftkv_proj",
                 f"layers.{layer_idx}.self_attn.v_proj_swiftkv", (i, "v")),
            ])
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            orig_name = name
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                use_shift_mode = getattr(param, "shift_parallel_mode", None)
                with model_runner.set_shift_parallel_mode(use_shift_mode):
                    weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    logger.debug(f"Skip loading {orig_name}")
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                use_shift_mode = getattr(param, "shift_parallel_mode", None)
                with model_runner.set_shift_parallel_mode(use_shift_mode):
                    weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    logger.debug(f"Skip loading {orig_name}")
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                use_shift_mode = getattr(param, "shift_parallel_mode", None)
                with model_runner.set_shift_parallel_mode(use_shift_mode):
                    weight_loader(param, loaded_weight)


class LlamaSwiftKVForCausalLM(nn.Module):
    packed_modules_mapping = {
        "kv_proj_swiftkv": ["k_proj_swiftkv", "v_proj_swiftkv"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".k_proj_swiftkv.",
        ".v_proj_swiftkv.",
    ]

    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [
        ".q_proj_swiftkv.",
        ".down_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "k_proj_swiftkv": ("kv_proj_swiftkv", 1),
        "v_proj_swiftkv": ("kv_proj_swiftkv", 2),
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = LlamaSwiftKVModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                logit_scale)
        self.sampler = get_sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        assert intermediate_tensors is None and inputs_embeds is None
        model_output = self.model(input_ids, positions)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(weights)

