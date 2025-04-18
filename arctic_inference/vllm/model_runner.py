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

import time
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn

import vllm
from vllm.config import VllmConfig, SpeculativeConfig
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, logger

from arctic_inference.patching import ArcticPatch


class GPUModelRunnerPatch(ArcticPatch[GPUModelRunner]):

    _orig_prepare_inputs = GPUModelRunner._prepare_inputs
    _orig_load_model = GPUModelRunner.load_model
    _orig_init = GPUModelRunner.__init__

    def __init__(self, *args, **kwargs):
        self._orig_init(*args, **kwargs)

        from arctic_inference.vllm.spec_dec.arctic_proposer import ArcticProposer
        self.mlp_drafter = ArcticProposer()

    def _prepare_inputs(self, *args, **kwargs):
        attn_metadata, logits_indices, *rest = (
            self._orig_prepare_inputs(*args, **kwargs))
        # SwiftKV requires knowing the logits indices from inside the model
        # definition in order to early-stop the prefill tokens.
        attn_metadata.swiftkv_logits_indices = logits_indices
        return attn_metadata, logits_indices, *rest

    def monkeypatch_forward(self):
        from vllm.distributed.parallel_state import _SP
        SP_size = _SP.world_size
        SP_rank = _SP.rank_in_group
        device_group = _SP.device_group
        model_forward = self.model.forward

        def ulysses_forward(*args, **kwargs):
            # update inputs
            input_ids = kwargs['input_ids']
            positions = kwargs['positions']
            # Ulysses parameters
            N = input_ids.shape[0]
            N_ulysses = N // SP_size
            N_offset = N_ulysses * SP_rank
            # narrow the input
            kwargs['input_ids'] = input_ids[N_offset:N_offset + N_ulysses]
            kwargs['positions'] = positions[N_offset:N_offset + N_ulysses]
            # original forward
            output = model_forward(*args, **kwargs)
            # all-gather model_output
            model_output = torch.empty((N, self.model.config.hidden_size),
                                       dtype=output.dtype,
                                       device=output.device)
            torch.distributed.all_gather_into_tensor(model_output,
                                                     output,
                                                     group=device_group)
            return model_output

        self.model.forward = ulysses_forward

    def _load_spec_model(
        self,
        vllm_config: VllmConfig,
        speculative_config: SpeculativeConfig,
    ) -> nn.Module:
        import copy
        from vllm.config import VllmConfig
        draft_worker_config = copy.deepcopy(vllm_config)
        draft_worker_config.model_config = speculative_config.draft_model_config
        draft_worker_config.quant_config = VllmConfig._get_quantization_config(
            draft_worker_config.model_config,
            vllm_config.load_config,
        )
        speculative_config.draft_parallel_config.worker_cls =\
            draft_worker_config.parallel_config.sd_worker_cls
        draft_worker_config.parallel_config = speculative_config.draft_parallel_config

        return get_model(vllm_config=draft_worker_config)

    def load_model(self, *args, **kwargs):
        self._orig_load_model(*args, **kwargs)
        if self.parallel_config.sequence_parallel_size > 1:
            self.monkeypatch_forward()
        if self.speculative_config:
            self.draft_model = self._load_spec_model(vllm_config=self.vllm_config,
                                                     speculative_config=self.speculative_config)
            self.mlp_drafter.link_model(self.draft_model)
    
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_encoder(scheduler_output)
            encoder_outputs = self._gather_encoder_outputs(scheduler_output)
        else:
            encoder_outputs = []

        # Prepare the decoder inputs.
        attn_metadata, logits_indices, spec_decode_metadata = (
            self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # add padding to the batch size to make it a multiple of SP
        SP = self.parallel_config.sequence_parallel_size
        num_input_tokens = (num_scheduled_tokens + SP - 1) // SP * SP
        if (self.use_cuda_graph
                and num_input_tokens // SP <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = SP * self.vllm_config.pad_for_cudagraph(
                num_input_tokens // SP)
        else:
            # Eager mode.
            pass
        attn_metadata.num_input_tokens = num_input_tokens

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if encoder_outputs:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, encoder_outputs)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        from vllm.distributed.parallel_state import get_pp_group
        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            assert self.intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                self.intermediate_tensors[k][:num_input_tokens].copy_(
                    v[:num_input_tokens], non_blocking=True)
            intermediate_tensors = IntermediateTensors({
                k: v[:num_input_tokens]
                for k, v in self.intermediate_tensors.items()
            })

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        from vllm.forward_context import set_forward_context
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            return hidden_states

        hidden_states = hidden_states[:num_scheduled_tokens]
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # TODO(woosuk): Optimize the memory usage.
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.model.sample(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # TODO(woosuk): Optimize the memory usage.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        for i, generator in self.input_batch.generators.items():
            req_id = self.input_batch.req_ids[i]
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator.set_offset(generator.get_offset() - 4)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states,
            scheduler_output,
        )

        # Get the valid generated tokens.
        previous_hidden_states = None
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
            previous_hidden_states = sample_hidden_states
        else:
            valid_mask = sampled_token_ids != -1
            gen_lens = valid_mask.sum(dim=1)
            num_sampled_tokens = np.array(spec_decode_metadata.num_draft_tokens)
            num_sampled_tokens = torch.tensor(num_sampled_tokens, device=gen_lens.device) + 1
            hidden_states_idx = (gen_lens - 1) + torch.cumsum(num_sampled_tokens, 0) - num_sampled_tokens
            previous_hidden_states = sample_hidden_states[hidden_states_idx]

            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids, self.input_batch.vocab_size)

        disable_spec_decode = (
            self.speculative_config and 
            self.speculative_config.speculative_disable_by_batch_size and
            len(self.input_batch.req_ids) > self.speculative_config.speculative_disable_by_batch_size
        )   

        if not self.use_spec_decode or disable_spec_decode:
            spec_token_ids = None
        else:
            spec_token_ids = self.generate_draft_token_ids_arctic(
                valid_sampled_token_ids, previous_hidden_states)

        from vllm.v1.outputs import ModelRunnerOutput
        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
        )
    
    def generate_draft_token_ids_arctic(
        self,
        sampled_token_ids: list[list[int]],
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        last_tokens : list[int] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            assert num_sampled_ids >= 1

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
            last_tokens.append(self.input_batch.token_ids_cpu[i, end_idx - 1])

        drafter_output = self.mlp_drafter.propose(
            last_tokens,
            previous_hidden_states=previous_hidden_states,
        )

        draft_token_ids = drafter_output.tolist()

        return draft_token_ids
    
    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            vllm.v1.worker.gpu_model_runner.logger.warning(
                "Skipping CUDA graph capture. Please add "
                "-O %s to use CUDA graphs.", CompilationLevel.PIECEWISE)
            return
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        SP = self.parallel_config.sequence_parallel_size
        from vllm.distributed.parallel_state import graph_capture
        with graph_capture(device=self.device):
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens * SP)
                self._dummy_run(num_tokens * SP)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))
