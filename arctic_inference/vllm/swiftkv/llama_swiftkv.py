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
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

import vllm.distributed.parallel_state as parallel_state
from vllm.attention.backends.abstract import AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.llama import (LlamaAttention,
                                              LlamaDecoderLayer,
                                              LlamaMLP)
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

import arctic_inference.vllm.model_runner as model_runner
from arctic_inference.common.swiftkv.configs import LlamaSwiftKVConfig

logger = init_logger(__name__)


def get_attn_metadata_for_swiftkv():
    fwd_ctx = get_forward_context()
    if fwd_ctx.attn_metadata is None:
        return None
    meta = next(iter(fwd_ctx.attn_metadata.values()))
    assert all(m is meta for m in fwd_ctx.attn_metadata.values()), \
        "All attention metadata should be the same for LlamaSwiftKV."
    return meta


class LlamaSwiftKVAttention(LlamaAttention):

    def __init__(
        self,
        config: LlamaSwiftKVConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=prefix,
            attn_type=attn_type)

        self.q_proj_swiftkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_swiftkv",
        )

        self.kv_proj_swiftkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj_swiftkv",
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        hidden_states = self.model.get_input_embeddings(input_ids)
        residual = None
        prefill_layers = self.model.layers[:self.config.num_key_value_layers]
        for idx, layer in enumerate(prefill_layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        sp_size = parallel_state._SP.world_size
        if sp_size > 1 and not model_runner.is_shift_parallel_mode():
            # All-gather across ulysses sequence parallel ranks
            hidden_states = parallel_state._SP.all_gather(hidden_states, dim=0)
            residual = parallel_state._SP.all_gather(residual, dim=0)
            positions = parallel_state._SP.all_gather(positions, dim=0)

        old_mode = model_runner.SP_TP_MODE
        old_tp_group = parallel_state.get_tp_group()
        model_runner.SP_TP_MODE = True
        parallel_state._TP = parallel_state._SP_TP

        # KV projection of all the remaining layers
        swiftkv_hidden_states = (
            self.model.norm_swiftkv(hidden_states + residual))

        k_states = []
        v_states = []
        rotary_emb = self.model.layers[0].self_attn.rotary_emb
        q = torch.empty_like(hidden_states)  # Just temporary buffer
        for layer in self.model.layers[self.config.num_key_value_layers:]:
            kv, _ = layer.self_attn.kv_proj_swiftkv(swiftkv_hidden_states)
            k, v = kv.chunk(2, dim=-1)
            _, k = rotary_emb(positions, q, k)
            k_states.append(k)
            v_states.append(v)
        k_states = torch.cat(k_states, dim=-1)
        v_states = torch.cat(v_states, dim=-1)

        model_runner.SP_TP_MODE = old_mode
        parallel_state._TP = old_tp_group

        return hidden_states, residual, positions, k_states, v_states


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
        # This is a hint for the compiler that v_states and k_states have
        # the same shape so that a single symbolic shape is inferred.
        torch._check(v_states.shape[0] == k_states.shape[0])
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

        self.vllm_config = vllm_config
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
            self.norm_swiftkv = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        for param in self.layers[config.num_key_value_layers:].parameters():
            param.shift_parallel_mode = True

        self._init_prefill_runner(vllm_config)
        self._init_decode_runner(vllm_config)

        from arctic_inference.py_custom_ops import try_load_torch_library
        self.use_custom_ops = True if try_load_torch_library() else False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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
            num_heads = self.layers[-1].self_attn.attn.num_kv_heads
            head_size = self.layers[-1].self_attn.attn.head_size
            num_kv = config.num_hidden_layers - config.num_key_value_layers
            kv_size = num_kv * num_heads * head_size
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

    def swiftkv_select(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = get_attn_metadata_for_swiftkv()
        if attn_metadata is None:
            # Graph capture or profiling mode.
            if hidden_states.shape[0] <= self.cuda_graph_max_batch_size:
                # Return the preallocated buffers so cuda graph is captured
                # correctly.
                inputs = self.decode_runner.inputs
                batch_size = hidden_states.shape[0]
                padded_size = self.vllm_config.pad_for_cudagraph(batch_size)
                return (inputs["hidden_states"][:padded_size],
                        inputs["residual"][:padded_size],
                        inputs["positions"][:padded_size],
                        inputs["k_states"][:padded_size],
                        inputs["v_states"][:padded_size])
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
        attn_metadata.query_start_loc = torch.searchsorted(
            logits_indices, attn_metadata.query_start_loc, out_int32=True)
        attn_metadata.slot_mapping = attn_metadata.slot_mapping[logits_indices]

        # TODO: Make cascade attention work with SwiftKV
        attn_metadata.use_cascade = False
        attn_metadata.cu_prefix_query_lens = None
        attn_metadata.prefix_kv_lens = None
        attn_metadata.suffix_kv_lens = None

        def index_fn(buffer_name: str, tensor: torch.Tensor,
                     indices: torch.LongTensor) -> torch.Tensor:
            # If the batch size is smaller than the maximum batch size
            # for cuda graph, we can use the preallocated buffer.
            batch_size = indices.numel()
            if batch_size <= self.cuda_graph_max_batch_size:
                buffer = self.decode_runner.inputs[buffer_name]
                torch.index_select(tensor, 0, indices, out=buffer[:batch_size])
                padded_size = self.vllm_config.pad_for_cudagraph(batch_size)
                return buffer[:padded_size]
            return tensor.index_select(0, indices)

        return (index_fn("hidden_states", hidden_states, logits_indices),
                index_fn("residual", residual, logits_indices),
                index_fn("positions", positions, logits_indices),
                index_fn("k_states", k_states, logits_indices),
                index_fn("v_states", v_states, logits_indices))

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states, residual, positions, k_states, v_states = (
            self.prefill_runner(input_ids, positions))

        orig_hidden_states = hidden_states
        hidden_states, residual, positions, k_states, v_states = (
            self.swiftkv_select(
                hidden_states,
                residual,
                positions,
                k_states,
                v_states))

        with model_runner.set_shift_parallel_mode(True):
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
            batch_size = logits_indices.numel()
            orig_hidden_states[logits_indices] = hidden_states[:batch_size]

        return orig_hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj.", ".q_proj.", "q"),
            (".qkv_proj.", ".k_proj.", "k"),
            (".qkv_proj.", ".v_proj.", "v"),
            (".gate_up_proj.", ".gate_proj.", 0),
            (".gate_up_proj.", ".up_proj.", 1),
            (".kv_proj_swiftkv.", ".k_proj_swiftkv.", "k"),
            (".kv_proj_swiftkv.", ".v_proj_swiftkv.", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                use_shift_mode = getattr(param, "shift_parallel_mode", None)
                with model_runner.set_shift_parallel_mode(use_shift_mode):
                    weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
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

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                use_shift_mode = getattr(param, "shift_parallel_mode", None)
                with model_runner.set_shift_parallel_mode(use_shift_mode):
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class LlamaSwiftKVForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj_swiftkv": ["k_proj_swiftkv", "v_proj_swiftkv"],
    }

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        self.unpadded_vocab_size = config.vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(
                self.model.embed_tokens)

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size,
                                                logit_scale)

    def _init_model(self,
                    vllm_config: VllmConfig,
                    prefix: str = ""):
        return LlamaSwiftKVModel(vllm_config=vllm_config, prefix=prefix)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
