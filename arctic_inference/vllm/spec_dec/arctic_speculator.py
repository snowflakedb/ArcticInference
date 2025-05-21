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

import collections
import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from arctic_inference.vllm.spec_dec.logits_processor_opt import LogitsProcessorOpt
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from arctic_inference.vllm.spec_dec.fp8 import Fp8ConfigWithEmbedding
from arctic_inference.vllm.spec_dec.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)

SQRT2 = 2**0.5

def padding_size(size: int) -> int:
    """Round up a size to the nearest multiple of 4."""
    mult = (1 << (size - 1).bit_length()) // 4
    if mult < 1:
        return size
    return (size + mult - 1) //  mult * mult


class MLPSpeculatorLayerNorm(nn.Module):
    """
    A L2 normalization implementation
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value
         fits in the range of your encoding scheme
         (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale_and_shift : bool
        Include a learned scaling and shift term after normalization.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale_and_shift=True,
    ):
        super().__init__()
        self.elementwise_scale_and_shift = elementwise_scale_and_shift
        if self.elementwise_scale_and_shift:
            self.weight = nn.Parameter(torch.empty(normalized_shape))
            self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps

    def forward(self, x):
        xf = x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale_and_shift:
            x = self.weight * x
            x = x + self.bias
        return x


def _generate_cg_key(padding_size: int, head_index: int):
    return (padding_size << 16) + head_index


class ArcticMLPSpeculator(nn.Module):
    """
    An implementation of the speculative models introduced in
    "Accelerating Production LLMs with Combined Token/Embedding
    Speculators"
    https://arxiv.org/pdf/2404.19124
    Trained speculators of this type are available on HF hub at:
    https://huggingface.co/ibm-fms and https://huggingface.co/ibm-granite
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.n_predict = config.n_predict
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.inner_dim = config.inner_dim if config.inner_dim != 0 else config.emb_dim

        self.max_speculative_tokens = config.num_lookahead_tokens

        self.tie_weights = config.tie_weights
        self.scale_input = config.scale_input

        self.quantize_lm_head = True

        quant_config = Fp8ConfigWithEmbedding() if self.quantize_lm_head else None

        self.qhead = None
        if self.tie_weights:
            assert (
                self.n_predict > 1
            ), "You cannot tie weights between stages when only 1 exists"
            embedding = VocabParallelEmbedding(
                config.vocab_size, self.inner_dim, org_num_embeddings=config.vocab_size
            )
            self.emb = nn.ModuleList([embedding] * self.max_speculative_tokens)

            # the initial projection from the base model may
            # have a different size, so that stays separate.
            proj_first = nn.Linear(self.emb_dim, self.inner_dim, bias=False)
            proj_tied = nn.Linear(self.inner_dim, self.inner_dim, bias=False)

            self.proj = nn.ModuleList(
                [proj_first] + [proj_tied] * (self.max_speculative_tokens - 1)
            )

            head = ParallelLMHead(
                self.vocab_size,
                self.inner_dim,
                bias=False,
                quant_config=quant_config,
                skip_quantization=True,
            )
            self.head = nn.ModuleList([head] * self.max_speculative_tokens)

            if self.quantize_lm_head:
                qhead = ParallelLMHead(
                    self.vocab_size,
                    self.inner_dim,
                    bias=False,
                    quant_config=quant_config,
                    skip_quantization=False,
                )
                qhead.quant_method = Fp8LinearMethod(quant_config=quant_config)
                self.qhead = nn.ModuleList([qhead] * self.max_speculative_tokens)

            ln = MLPSpeculatorLayerNorm(
                self.inner_dim, elementwise_scale_and_shift=True
            )
            self.ln = nn.ModuleList([ln] * self.max_speculative_tokens)

        else:
            self.emb = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        config.vocab_size,
                        self.inner_dim,
                        org_num_embeddings=config.vocab_size,
                    )
                    for _ in range(self.max_speculative_tokens)
                ]
            )

            self.proj = nn.ModuleList(
                [
                    nn.Linear(
                        (self.emb_dim if i == 0 else self.inner_dim),
                        self.inner_dim,
                        bias=False,
                    )
                    for i in range(self.max_speculative_tokens)
                ]
            )

            self.head = nn.ModuleList(
                [
                    ParallelLMHead(
                        self.vocab_size,
                        self.inner_dim,
                        bias=False,
                        quant_config=quant_config,
                    )
                    for _ in range(self.max_speculative_tokens)
                ]
            )
            self.ln = nn.ModuleList(
                [
                    MLPSpeculatorLayerNorm(
                        self.inner_dim, elementwise_scale_and_shift=True
                    )
                    for _ in range(self.max_speculative_tokens)
                ]
            )

        if self.scale_input:
            self.ln0 = MLPSpeculatorLayerNorm(
                self.emb_dim, elementwise_scale_and_shift=False
            )

        self.state_weight = 0.5 ** (0.5 / config.n_predict)
        self.emb_weight = math.sqrt((1 - self.state_weight**2) * (self.inner_dim / 2))
        self.activation = nn.GELU()
        self.config = config
        self.logits_processor = LogitsProcessorOpt(
            vocab_size=config.vocab_size,
            org_vocab_size=config.vocab_size,
            scale=1.0,
            skip_last_gather=True,
        )
        self.sampler = get_sampler()

        self.cuda_graph_max_batch_size = 0
        self.cuda_graph_mode = False
        if not vllm_config.model_config.enforce_eager:
            self.cuda_graph_mode = True
            self.cuda_graphs = {}
            self.cuda_graph_max_batch_size = padding_size(
                vllm_config.scheduler_config.max_num_seqs
            )
            self.static_cuda_buffers = {
                "last_tokens": torch.empty(
                    self.cuda_graph_max_batch_size, 1, dtype=torch.long
                ),
                "previous_hidden_states": torch.empty(
                    self.cuda_graph_max_batch_size, 1, self.emb_dim
                ),
                "next_tokens": [
                    torch.empty(self.cuda_graph_max_batch_size, 1, dtype=torch.long)
                    for _ in range(self.n_predict)
                ],
            }

    def _prepare_cuda_graph_ios(self, size, last_tokens, previous_hidden_states):
        self.static_cuda_buffers["last_tokens"][:size] = last_tokens
        if previous_hidden_states is not None:
            self.static_cuda_buffers["previous_hidden_states"][
                :size
            ] = previous_hidden_states

        padded_size = padding_size(size)

        static_last_tokens = self.static_cuda_buffers["last_tokens"][:padded_size]
        static_hidden_states = self.static_cuda_buffers["previous_hidden_states"][
            :padded_size
        ]
        return (padded_size, static_last_tokens, static_hidden_states)

    def generate_states(
        self,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        head_index: int,
    ) -> torch.Tensor:
        if head_index == 0 and self.scale_input:
            previous_hidden_states = self.ln0(previous_hidden_states) / SQRT2
        # Project and predict
        z = self.emb[head_index](last_tokens)  # b k d
        states = self.proj[head_index](previous_hidden_states)

        # Weighted add of state_weight*state and emb_weight*z
        # Let subsequent LN take care of denominator
        # state_weight is close to 1, so shouldn't be any precision issues
        states.add_(z, alpha=self.emb_weight / self.state_weight)

        states = self.activation(self.ln[head_index](states))  # b k d

        return states

    def generate_token_ids(
        self,
        batch_size: int,
        num_predict_tokens: int,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        next_tokens_tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        for head_index in range(num_predict_tokens):
            states = self.generate_states(
                last_tokens, previous_hidden_states, head_index
            )
            previous_hidden_states = states
            states = states.flatten(0, 1)
            head_weight = (
                self.qhead[head_index]
                if self.qhead is not None and batch_size <= 32
                else self.head[head_index]
            )
            logits = self.logits_processor(head_weight, states)

            if get_tensor_model_parallel_world_size() == 1:
                last_tokens = torch.argmax(logits, dim=-1).reshape(batch_size, -1)
            else:
                vals, indices = torch.topk(logits, 1, dim=-1)
                indices = indices + get_tensor_model_parallel_rank() * logits.shape[-1]
                vals = tensor_model_parallel_all_gather(vals)
                indices = tensor_model_parallel_all_gather(indices)
                argidx = torch.argmax(vals, -1).reshape(batch_size, -1)
                last_tokens = torch.gather(indices, -1, argidx)

            if next_tokens_tensors[head_index] == None:
                next_tokens_tensors[head_index] = last_tokens
            else:
                next_tokens_tensors[head_index].copy_(last_tokens)

    def generate_proposals(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> List[torch.tensor]:
        if num_predict_tokens > self.max_speculative_tokens:
            raise ValueError(
                f"Max speculative tokens for model is "
                f"{self.max_speculative_tokens}, but "
                f"{num_predict_tokens} were requested"
            )

        # b x 1 x d
        previous_hidden_states = previous_hidden_states.unsqueeze(1)

        # b x 1
        last_tokens = input_ids.unsqueeze(1)

        batch_size = input_ids.size(0)

        static_next_tokens = [None] * num_predict_tokens

        if self.cuda_graph_mode and batch_size <= self.cuda_graph_max_batch_size:
            padded_size, static_last_tokens, static_hidden_states = (
                self._prepare_cuda_graph_ios(
                    batch_size, last_tokens, previous_hidden_states
                )
            )
            cg_key = _generate_cg_key(padded_size, 0)
            g = self.cuda_graphs.get(cg_key)

            for i in range(num_predict_tokens):
                static_next_tokens[i] = self.static_cuda_buffers["next_tokens"][i][
                    :padded_size
                ]

            if g is None:
                from vllm.distributed.parallel_state import graph_capture

                device = torch.cuda.current_device()
                with graph_capture(device=device) as capture_context:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=capture_context.stream):
                        self.generate_token_ids(
                            padded_size,
                            num_predict_tokens,
                            static_last_tokens,
                            static_hidden_states,
                            static_next_tokens,
                        )

                self.cuda_graphs[cg_key] = g
            else:
                g.replay()
        else:
            self.generate_token_ids(
                batch_size,
                num_predict_tokens,
                last_tokens,
                previous_hidden_states,
                static_next_tokens,
            )

        next_tokens = []
        for i in range(num_predict_tokens):
            next_tokens.append(
                static_next_tokens[i][:batch_size])

        return torch.cat(next_tokens, dim=-1)

    def maybe_load_weight(self, param, loaded_weight):
        if param is not None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = name.replace("speculator.", "")
            param = params_dict.get(name)
            self.maybe_load_weight(param, loaded_weight)

            if name.startswith("head"):
                param = params_dict.get(name.replace("head", "qhead"))
                self.maybe_load_weight(param, loaded_weight)


class ArcticLSTMSpeculator(nn.Module):
    """
    An implementation of the speculative models introduced in
    "Accelerating Production LLMs with Combined Token/Embedding
    Speculators"
    https://arxiv.org/pdf/2404.19124
    Trained speculators of this type are available on HF hub at:
    https://huggingface.co/ibm-fms and https://huggingface.co/ibm-granite
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.n_predict = config.n_predict
        self.vocab_size = config.vocab_size
        self.input_hidden_dim = config.input_hidden_dim
        config.inner_dim = [int(i) for i in config.inner_dim.split(".")]
        self.inner_dim = config.inner_dim
        config.emb_dim = [int(i) for i in config.emb_dim.split(".")]
        self.emb_dim = config.emb_dim
        config.proj_dim = [int(i) for i in config.proj_dim.split(".")]
        self.proj_dim = config.proj_dim

        self.max_speculative_tokens = config.num_lookahead_tokens

        self.tie_weights = config.tie_weights
        self.tie_lstm_embs = config.tie_lstm_embs
        self.scale_input = config.scale_input
        self.quantize_lm_head = True

        quant_config = Fp8ConfigWithEmbedding() if self.quantize_lm_head else None
        self.method = getattr(config, "method", "sum_rnn")

        self.activation = nn.GELU()
        self.qhead = None
        if self.tie_weights:
            head = ParallelLMHead(
                self.vocab_size,
                self.inner_dim[-1],
                bias=False,
                quant_config=quant_config,
                skip_quantization=True,
            )
            self.head = nn.ModuleList([head] * self.max_speculative_tokens)

            if self.quantize_lm_head:
                qhead = ParallelLMHead(
                    self.vocab_size,
                    self.inner_dim[-1],
                    bias=False,
                    quant_config=quant_config,
                    skip_quantization=False,
                )
                qhead.quant_method = Fp8LinearMethod(quant_config=quant_config)
                self.qhead = nn.ModuleList([qhead] * self.max_speculative_tokens)
        else:
            self.head = nn.ModuleList(
                [
                    ParallelLMHead(
                        self.vocab_size,
                        self.inner_dim[-1],
                        bias=False,
                        quant_config=quant_config,
                    )
                    for _ in range(self.max_speculative_tokens)
                ]
            )

        if self.method == "sum_rnn":
            embs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [VocabParallelEmbedding(self.vocab_size, self.emb_dim[0])]
                    for i in range(1, len(self.emb_dim)):
                        print(f"ADDING ANOTHER EMB {i}")
                        seqs.append(
                            MLPSpeculatorLayerNorm(
                                self.emb_dim[i], elementwise_scale_and_shift=True
                            )
                        )
                        seqs.append(self.activation)
                        seqs.append(
                            nn.Linear(self.emb_dim[i - 1], self.emb_dim[i], bias=False)
                        )
                    embs.append(nn.Sequential(*seqs))
            self.emb = nn.ModuleList(embs)

            projs = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i <= 1:
                    seqs = [
                        nn.Linear(
                            (self.input_hidden_dim if n_i == 0 else self.inner_dim[-1]),
                            self.proj_dim[0],
                            bias=False,
                        )
                    ]
                    for i in range(1, len(self.proj_dim)):
                        print(f"ADDING ANOTHER PROJ {i}")
                        seqs.append(
                            MLPSpeculatorLayerNorm(
                                self.proj_dim[i], elementwise_scale_and_shift=True
                            )
                        )
                        seqs.append(self.activation)
                        seqs.append(
                            nn.Linear(
                                self.proj_dim[i - 1], self.proj_dim[i], bias=False
                            )
                        )
                    projs.append(nn.Sequential(*seqs))
            self.proj = nn.ModuleList(projs)

            lns = []
            for n_i in range(self.n_predict):
                if not self.tie_weights or n_i == 0:
                    seqs = [
                        MLPSpeculatorLayerNorm(
                            self.inner_dim[0], elementwise_scale_and_shift=True
                        )
                    ]
                    for i in range(1, len(self.inner_dim)):
                        seqs.append(self.activation)
                        seqs.append(
                            nn.Linear(
                                self.inner_dim[i - 1], self.inner_dim[i], bias=False
                            )
                        )
                        seqs.append(
                            MLPSpeculatorLayerNorm(
                                self.inner_dim[i], elementwise_scale_and_shift=True
                            )
                        )
                    lns.append(nn.Sequential(*seqs))
            self.ln = nn.ModuleList(lns)

        elif self.method == "sum_lstm":
            assert self.tie_weights
            self.forget_emb = nn.ModuleList(
                [nn.Embedding(self.vocab_size, self.emb_dim[0])]
            )
            if not self.tie_lstm_embs:
                self.input_emb = nn.ModuleList(
                    [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                )
                self.cell_emb = nn.ModuleList(
                    [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                )
                self.output_emb = nn.ModuleList(
                    [nn.Embedding(self.vocab_size, self.emb_dim[0])]
                )
            self.projs = nn.ModuleList(
                [
                    nn.Linear(self.input_hidden_dim, self.proj_dim[0] * 4, bias=False),
                    nn.Linear(self.inner_dim[-1], self.proj_dim[0] * 4, bias=False),
                ]
            )
            self.cell_ln = nn.ModuleList(
                [
                    MLPSpeculatorLayerNorm(
                        self.inner_dim[0], elementwise_scale_and_shift=True
                    )
                ]
            )
            self.state_ln = nn.ModuleList(
                [
                    MLPSpeculatorLayerNorm(
                        self.inner_dim[0], elementwise_scale_and_shift=True
                    )
                ]
            )

        if self.scale_input:
            self.ln0 = MLPSpeculatorLayerNorm(
                self.input_hidden_dim, elementwise_scale_and_shift=False
            )

        self.state_weight = 0.5 ** (0.5 / config.n_predict)
        self.emb_weight = math.sqrt(
            (1 - self.state_weight**2) * (self.inner_dim[0] / 2)
        )
        self.config = config
        self.logits_processor = LogitsProcessorOpt(
            vocab_size=config.vocab_size,
            org_vocab_size=config.vocab_size,
            scale=1.0,
            skip_last_gather=True,
        )
        self.sampler = get_sampler()

        self.cuda_graph_max_batch_size = 0
        self.cuda_graph_mode = False
        if not vllm_config.model_config.enforce_eager:
            self.cuda_graph_mode = True
            self.cuda_graphs = {}
            self.cuda_graph_max_batch_size = padding_size(
                vllm_config.scheduler_config.max_num_seqs
            )
            self.static_cuda_buffers = {
                "last_tokens": torch.empty(
                    self.cuda_graph_max_batch_size, 1, dtype=torch.long
                ),
                "previous_hidden_states": torch.empty(
                    self.cuda_graph_max_batch_size, 1, self.input_hidden_dim
                ),
                "cell_states": torch.empty(
                    self.cuda_graph_max_batch_size, 1, self.inner_dim[-1]
                ),
                "next_tokens": [
                    torch.empty(self.cuda_graph_max_batch_size, 1, dtype=torch.long)
                    for _ in range(self.n_predict)
                ],
            }
            if self.inner_dim[-1] != self.input_hidden_dim:
                print("CREATED NEXT PREVIOUS HIDDEN STATES")
                self.static_cuda_buffers["next_previous_hidden_states"] = torch.empty(
                    self.cuda_graph_max_batch_size, 1, self.inner_dim[-1]
                )

    def _prepare_cuda_graph_ios(
        self,
        size,
        last_tokens,
        previous_hidden_states,
        hidden_state_buffers,
        cell_states=None,
        use_lstm=False,
    ):
        self.static_cuda_buffers["last_tokens"][:size] = last_tokens
        if cell_states is not None:
            self.static_cuda_buffers["cell_states"][:size] = cell_states
        if previous_hidden_states is not None:
            hidden_state_buffers[:size] = previous_hidden_states

        padded_size = padding_size(size)

        static_last_tokens = self.static_cuda_buffers["last_tokens"][:padded_size]
        static_hidden_states = hidden_state_buffers[:padded_size]
        if use_lstm:
            static_cell_states = self.static_cuda_buffers["cell_states"][:padded_size]
            return (
                padded_size,
                static_last_tokens,
                static_hidden_states,
                static_cell_states,
            )
        else:
            return (padded_size, static_last_tokens, static_hidden_states)

    def generate_states(
        self,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        head_index: int,
        cell_states: torch.Tensor = None,
    ) -> torch.Tensor:

        if head_index == 0 and self.scale_input:
            previous_hidden_states = self.ln0(previous_hidden_states) / SQRT2

        actual_i = 0 if self.tie_weights else head_index
        actual_proj_i = 1 if self.tie_weights and head_index >= 2 else head_index

        if self.method == "sum_lstm":
            assert self.tie_lstm_embs

            prev_state = previous_hidden_states

            z = self.forget_emb[actual_i](last_tokens).repeat(1, 1, 4)  # b n d
            states = self.projs[actual_proj_i](prev_state)
            added_states = torch.add(
                states, z, alpha=self.emb_weight / self.state_weight
            )

            forget_input_output, cell_candidate = added_states.split(
                [self.proj_dim[0] * 3, self.proj_dim[0]], dim=-1
            )
            forget_gate, input_gate, output_gate = torch.sigmoid(
                forget_input_output
            ).split([self.proj_dim[0], self.proj_dim[0], self.proj_dim[0]], dim=-1)

            cell_candidate = self.activation(
                self.cell_ln[actual_i](cell_candidate)
            )  # b n d
            cell_candidate = cell_candidate * input_gate

            cell_states = cell_states * forget_gate
            cell_states = cell_states + cell_candidate

            state_candidate = self.activation(self.state_ln[actual_i](cell_states))
            state = state_candidate * output_gate

            return state, cell_states

        else:
            # Project and predict
            z = self.emb[actual_i](last_tokens)  # b k d
            states = self.proj[actual_proj_i](previous_hidden_states)

            # Weighted add of state_weight*state and emb_weight*z
            # Let subsequent LN take care of denominator
            # state_weight is close to 1, so shouldn't be any precision issues
            states.add_(z, alpha=self.emb_weight / self.state_weight)
            states = self.activation(self.ln[actual_i](states))  # b k d

            return states

    def generate_token_ids(
        self,
        batch_size: int,
        num_predict_tokens: int,
        last_tokens: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        next_tokens_tensors: List[torch.Tensor],
        cell_states: torch.Tensor = None,
    ) -> torch.Tensor:
        for head_index in range(num_predict_tokens):
            if self.method == "sum_lstm":
                states, cell_states = self.generate_states(
                    last_tokens, previous_hidden_states, head_index, cell_states
                )
            else:
                states = self.generate_states(
                    last_tokens, previous_hidden_states, head_index
                )
            previous_hidden_states = states
            states = states.flatten(0, 1)
            head_weight = (
                self.qhead[head_index]
                if self.qhead is not None and batch_size <= 32
                else self.head[head_index]
            )
            logits = self.logits_processor(head_weight, states)

            if get_tensor_model_parallel_world_size() == 1:
                last_tokens = torch.argmax(logits, dim=-1).reshape(batch_size, -1)
            else:
                vals, indices = torch.topk(logits, 1, dim=-1)
                indices = indices + get_tensor_model_parallel_rank() * logits.shape[-1]
                vals = tensor_model_parallel_all_gather(vals)
                indices = tensor_model_parallel_all_gather(indices)
                argidx = torch.argmax(vals, -1).reshape(batch_size, -1)
                last_tokens = torch.gather(indices, -1, argidx)

            if next_tokens_tensors[head_index] == None:
                next_tokens_tensors[head_index] = last_tokens
            else:
                next_tokens_tensors[head_index].copy_(last_tokens)

        return next_tokens_tensors

    def generate_proposals(
        self,
        input_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        num_predict_tokens: int,
    ) -> List[SamplerOutput]:
        if num_predict_tokens > self.max_speculative_tokens:
            raise ValueError(
                f"Max speculative tokens for model is "
                f"{self.max_speculative_tokens}, but "
                f"{num_predict_tokens} were requested"
            )

        # b x 1 x d
        previous_hidden_states = previous_hidden_states.unsqueeze(1)

        # b x 1
        last_tokens = input_ids.unsqueeze(1)

        batch_size = input_ids.size(0)
        state_shapes = list(previous_hidden_states.shape)
        state_shapes[-1] = self.inner_dim[-1]

        static_next_tokens = [None] * num_predict_tokens
        static_cell_states = None
        static_last_tokens = None
        static_hidden_states = None

        if self.cuda_graph_mode and batch_size <= self.cuda_graph_max_batch_size:
            static_states = self.static_cuda_buffers["previous_hidden_states"]
            if self.method == "sum_lstm":
                previous_cell_states = torch.zeros(
                    state_shapes,
                    device=previous_hidden_states.device,
                    dtype=previous_hidden_states.dtype,
                )
                (
                    padded_size,
                    static_last_tokens,
                    static_hidden_states,
                    static_cell_states,
                ) = self._prepare_cuda_graph_ios(
                    batch_size,
                    last_tokens,
                    previous_hidden_states,
                    static_states,
                    previous_cell_states,
                    use_lstm=True,
                )
            else:
                padded_size, static_last_tokens, static_hidden_states = (
                    self._prepare_cuda_graph_ios(
                        batch_size, last_tokens, previous_hidden_states, static_states
                    )
                )

            cg_key = _generate_cg_key(padded_size, 0)
            g = self.cuda_graphs.get(cg_key)

            static_states = (
                self.static_cuda_buffers["next_previous_hidden_states"]
                if self.inner_dim[-1] != self.input_hidden_dim
                else self.static_cuda_buffers["previous_hidden_states"]
            )

            for i in range(num_predict_tokens):
                static_next_tokens[i] = self.static_cuda_buffers["next_tokens"][i][
                    :padded_size
                ]

            if g is None:
                from vllm.distributed.parallel_state import graph_capture

                device = torch.cuda.current_device()
                with graph_capture(device=device) as capture_context:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=capture_context.stream):
                        if self.method == "sum_lstm":
                            self.generate_token_ids(
                                padded_size,
                                num_predict_tokens,
                                static_last_tokens,
                                static_hidden_states,
                                static_next_tokens,
                                cell_states=static_cell_states,
                            )
                        else:
                            self.generate_token_ids(
                                padded_size,
                                num_predict_tokens,
                                static_last_tokens,
                                static_hidden_states,
                                static_next_tokens,
                            )
                self.cuda_graphs[cg_key] = g
            else:
                g.replay()
        else:
            if self.method == "sum_lstm":
                self.generate_token_ids(
                    batch_size,
                    num_predict_tokens,
                    static_last_tokens,
                    static_hidden_states,
                    static_next_tokens,
                    cell_states=static_cell_states,
                )
            else:
                self.generate_token_ids(
                    batch_size,
                    num_predict_tokens,
                    static_last_tokens,
                    static_hidden_states,
                    static_next_tokens,
                )

        next_tokens = []
        for i in range(num_predict_tokens):
            next_tokens.append(
                static_next_tokens[i][:batch_size])

        return torch.cat(next_tokens, dim=-1)

    def maybe_load_weight(self, param, loaded_weight):
        if param is not None:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = collections.OrderedDict(weights)
        if self.method == "sum_lstm" and self.tie_lstm_embs:
            weights.pop("input_emb.0.weight")
            weights.pop("cell_emb.0.weight")
            weights.pop("output_emb.0.weight")
            for name, param in self.named_parameters():
                if "projs." in name:
                    print(f"REPLACING {name}")
                    forget_proj = weights.pop(name.replace("projs", "forget_proj"))
                    input_proj = weights.pop(name.replace("projs", "input_proj"))
                    output_proj = weights.pop(name.replace("projs", "output_proj"))
                    cell_proj = weights.pop(name.replace("projs", "cell_proj"))
                    weights[name] = torch.cat(
                        [forget_proj, input_proj, output_proj, cell_proj]
                    )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights.items():
            print(f"LOADING {name}")
            name = name.replace("speculator.", "")
            param = params_dict.get(name)
            self.maybe_load_weight(param, loaded_weight)

            if name.startswith("head"):
                param = params_dict.get(name.replace("head", "qhead"))
                self.maybe_load_weight(param, loaded_weight)
