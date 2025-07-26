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

import contextlib
import copy
import time
from typing import Any, Union, Optional, TYPE_CHECKING
from itertools import tee

import numpy as np
import torch
import vllm.distributed.parallel_state as parallel_state
import vllm.envs as envs
from tqdm import tqdm
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.config import CompilationLevel
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             is_global_first_rank)
from vllm.forward_context import set_forward_context
from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN, RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from arctic_inference.common.suffix_cache import SuffixCache
from arctic_inference.patching import ArcticPatch
from arctic_inference.vllm.spec_dec.arctic_proposer import ArcticProposer
from arctic_inference.common.suffix_cache import SuffixSpecResult

SP_TP_MODE = None


@contextlib.contextmanager
def set_shift_parallel_mode(mode: Optional[bool]):
    if mode is None:
        yield
        return

    global SP_TP_MODE

    if not is_shift_parallel_mode():
        assert not parallel_state._TP_STATE_PATCHED
        parallel_state._ORIG_TP = parallel_state._TP

    old_mode = SP_TP_MODE
    old_tp_group = parallel_state.get_tp_group()
    SP_TP_MODE = mode

    parallel_state._TP = (parallel_state._SP_TP if mode
                          else parallel_state._ORIG_TP)

    try:
        yield
    finally:
        # restore the original state
        SP_TP_MODE = old_mode
        parallel_state._TP = old_tp_group


def is_shift_parallel_mode() -> bool:
    """Check if the shift parallel mode is enabled."""
    global SP_TP_MODE
    return SP_TP_MODE is True


class GPUModelRunnerPatch(ArcticPatch[GPUModelRunner]):

    _orig_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    _orig_prepare_inputs = GPUModelRunner._prepare_inputs
    _orig_profile_run = GPUModelRunner.profile_run
    _orig_load_model = GPUModelRunner.load_model
    _orig_propose_draft_token_ids = GPUModelRunner.propose_draft_token_ids
    _orig_init = GPUModelRunner.__init__

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # Ulysses sequence parallelism
        if vllm_config.parallel_config.ulysses_sequence_parallel_size > 1:
            self.use_ulysses = True
            pass_config = vllm_config.compilation_config.pass_config
            if pass_config.enable_sequence_parallelism:
                raise ValueError(
                    "Ulysses sequence parallelism is incompatible with native "
                    "sequence parallelism. Set enable_sequence_parallelism "
                    "to False in the pass config to use Ulysses.")
        else:
            self.use_ulysses = False

        # Speculative decoding
        # TODO: Use "arctic" as an umbrella method that also covers the Arctic
        # Inverence version of "mlp_speculator".
        if (vllm_config.speculative_config is not None and \
                vllm_config.speculative_config.method in (
                    "arctic", "suffix", "mlp_speculator")):
            # Delay the creation of the drafter until
            # after the child class has been initialized.
            arctic_speculative_config = vllm_config.speculative_config
            vllm_config.speculative_config = None
        else:
            arctic_speculative_config = None

        self._orig_init(vllm_config, device)

        # Set up speculative decoding.
        self._suffix_cache = None
        if arctic_speculative_config is not None:
            # Restore the speculative config.
            self.vllm_config.speculative_config = arctic_speculative_config
            self.speculative_config = arctic_speculative_config

            if get_pp_group().is_last_rank:
                if (self.speculative_config.method == "arctic" or
                      self.speculative_config.method == "mlp_speculator"):
                    self.drafter = ArcticProposer(self.vllm_config)
                elif self.speculative_config.method != "suffix":
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")

                self.rejection_sampler = RejectionSampler()

        if (self.speculative_config is not None and
                self.speculative_config.enable_suffix_decoding):
            if self.speculative_config.method not in (
                    "arctic", "suffix", "mlp_speculator"):
                raise ValueError(
                    "Suffix decoding is only supported with the 'arctic', "
                    "'mlp_speculator' or 'suffix' spec decoding methods.")
            self._suffix_cache = SuffixCache(
                self.speculative_config.suffix_cache_max_depth)

    def profile_run(self) -> None:
        self._orig_profile_run()
        if self.shift_model is not None:
            # Run the shift model to trigger compilation.
            orig_model, self.model = self.model, self.shift_model
            try:
                with set_shift_parallel_mode(True):
                    self._dummy_run(self.max_num_tokens, is_profile=True)
            finally:
                self.model = orig_model

    def _prepare_inputs(self, *args, **kwargs):
        attn_metadata, attention_cuda_graphs, logits_indices, *rest = (
            self._orig_prepare_inputs(*args, **kwargs))
        # SwiftKV requires knowing the logits indices from inside the model
        # definition in order to early-stop the prefill tokens.
        for meta in attn_metadata.values():
            meta.swiftkv_logits_indices = logits_indices
        return attn_metadata, attention_cuda_graphs, logits_indices, *rest

    def monkeypatch_forward(self: GPUModelRunner):
        sp_size = parallel_state._SP.world_size
        sp_rank = parallel_state._SP.rank_in_group
        device_group = parallel_state._SP.device_group
        model_forward = self.model.forward
        input_key = 'inputs_embeds' if  self.is_multimodal_model else 'input_ids'

        def ulysses_forward(*args, **kwargs):
            # update inputs
            input_tensor = kwargs[input_key]
            positions = kwargs['positions']
            # Ulysses parameters
            N = input_tensor.shape[0]

            N_ulysses = N // sp_size
            N_offset = N_ulysses * sp_rank

            # narrow the input
            kwargs[input_key] = input_tensor[N_offset:N_offset + N_ulysses]
            kwargs['positions'] = positions[N_offset:N_offset + N_ulysses]

            with set_shift_parallel_mode(False):
                output = model_forward(*args, **kwargs)

            if output.size(0) == N_ulysses:
                # all-gather model_output
                model_output = torch.empty((N, self.hidden_size),
                                        dtype=output.dtype,
                                        device=output.device)
                torch.distributed.all_gather_into_tensor(model_output,
                                                        output,
                                                        group=device_group)
            else:
                # SwiftKV models will already have all-gathered the output.
                assert output.size(0) == N
                model_output = output
            return model_output

        self.model.forward = ulysses_forward

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)

        # Prepare the decoder inputs.
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata,
         num_scheduled_tokens_np) = (self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        use_shift_model = (
            self.use_ulysses and self.shift_model is not None and
            num_scheduled_tokens <= self.shift_parallel_threshold)
        if self.use_ulysses and not use_shift_model:
            # add padding to the batch size to make it a multiple of SP
            sp_size = self.parallel_config.ulysses_sequence_parallel_size
            num_input_tokens = round_up(num_scheduled_tokens, sp_size)
            if (self.use_cuda_graph and num_input_tokens // sp_size
                    <= self.cudagraph_batch_sizes[-1]):
                num_input_tokens = self.vllm_config.pad_for_cudagraph(
                    num_input_tokens // sp_size) * sp_size
        elif (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
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

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, but we
        # compiled with full CUDA graphs, we have to skip them entirely.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ):
            self.maybe_setup_kv_connector(scheduler_output)

            model = self.shift_model if use_shift_model else self.model
            with set_shift_parallel_mode(use_shift_model):
                model_output = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, finished_sending,
                                  finished_recving)

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(valid_sampled_token_ids):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        if self._suffix_cache is not None:
            self._update_suffix_cache(valid_sampled_token_ids)

        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        else:
            spec_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampler_output.sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                attn_metadata,
            )

        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            num_nans_in_logits=num_nans_in_logits,
        )

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        original_sampled_token_ids: np.ndarray,
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        attn_metadata: dict[str, Any],
    ) -> list[list[int]]:
        disable_spec_decode = (
            self.speculative_config and
            self.speculative_config.disable_by_batch_size and
            len(self.input_batch.req_ids) > self.speculative_config.disable_by_batch_size
        )
        if disable_spec_decode:
            # No speculative decoding is enabled.
            return [[] for _ in sampled_token_ids]

        suffix_spec_token_ids = None
        new_sampled_token_ids = sampled_token_ids.copy()
        if self._suffix_cache is not None:
            results = self.propose_suffix_draft_token_ids(
                new_sampled_token_ids)
            suffix_spec_token_ids = []
            # The score is an estimate of the acceptance length. Thus, the
            # heuristic is to use the suffix decoded tokens if the score is
            # greater than the # of tokens we would speculate otherwise.
            min_score = (self.speculative_config.num_speculative_tokens
                         if self.speculative_config.method != "suffix" else 0)
            min_score = (0 if self.speculative_config.method == "suffix"
                         else self.speculative_config.num_speculative_tokens)
            for i, result in enumerate(results):
                if result.score >= min_score:
                    # Use suffix decoded tokens, disable other speculation
                    # methods for this request.
                    new_sampled_token_ids[i] = []
                    suffix_spec_token_ids.append(result.token_ids)
                else:
                    suffix_spec_token_ids.append([])

        spec_token_ids = None
        if self.speculative_config.method == "suffix":
            pass
        elif (self.speculative_config.method == "arctic" or 
              self.speculative_config.method == "mlp_speculator"):
            assert isinstance(self.drafter, ArcticProposer)
            previous_hidden_states = self.drafter.prepare_hidden_states(
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=original_sampled_token_ids,
                spec_decode_metadata=spec_decode_metadata,
            )
            spec_token_ids = self.propose_arctic_draft_token_ids(
                scheduler_output,
                new_sampled_token_ids, 
                previous_hidden_states=previous_hidden_states)
        else:
            spec_token_ids = self._orig_propose_draft_token_ids(
                scheduler_output,
                new_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                attn_metadata,
            )

        if spec_token_ids is None:
            spec_token_ids = suffix_spec_token_ids
        elif suffix_spec_token_ids is not None:
            spec_token_ids = [
                suffix_spec_token_ids[i] or spec_token_ids[i]
                for i in range(len(suffix_spec_token_ids))
            ]

        return spec_token_ids

    def propose_arctic_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        last_tokens : list[int] = []
        max_spec_tokens = self.speculative_config.num_speculative_tokens
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            
            if (num_sampled_ids == 0):
                if self.speculative_config.enable_suffix_decoding:
                    return [[]] * len(sampled_token_ids)
                req_id = self.input_batch.req_ids[i]
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                sampled_ids = [req_state.get_token_id(seq_len)]

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids

            max_spec_tokens = min(
                max_spec_tokens, self.max_model_len - end_idx - 1,
            )
            if max_spec_tokens <= 0:
                continue

            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids[-1]
            last_tokens.append(self.input_batch.token_ids_cpu[i, end_idx - 1])

        if max_spec_tokens <= 0:
            return [[] for _ in sampled_token_ids]

        drafter_output = self.drafter.propose(
            last_tokens,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=max_spec_tokens,
        )

        draft_token_ids = drafter_output.tolist()

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids[i] = []

        return draft_token_ids

    def _update_suffix_cache(self, sampled_token_ids: list[list[int]]) -> None:
        seen_req_ids = set()
        for i, sampled_ids in enumerate(sampled_token_ids):
            req_id = self.input_batch.req_ids[i]
            seen_req_ids.add(req_id)

            if not sampled_ids:
                continue

            index = self.input_batch.req_id_to_index[req_id]
            if not self._suffix_cache.has_cached_prompt(req_id):
                num_prompt_tokens = self.input_batch.num_prompt_tokens[index]
                prompt_token_ids = (
                    self.input_batch.token_ids_cpu[index, :num_prompt_tokens])
                self._suffix_cache.cache_prompt(req_id, prompt_token_ids)

            self._suffix_cache.update_response(req_id, sampled_ids)

        # Evict prompts that are not seen
        for req_id in self._suffix_cache.cached_prompt_ids():
            if req_id not in seen_req_ids:
                self._suffix_cache.evict_prompt(req_id)

    def propose_suffix_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        spec_token_ids: Optional[list[list[int]]] = None,
    ) -> list[list[int]]:
        config = self.speculative_config
        results = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            spec_ids = spec_token_ids[i] if spec_token_ids is not None else []
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                results.append(SuffixSpecResult())
                continue

            req_id = self.input_batch.req_ids[i]

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + len(sampled_ids)
            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids

            size = min(end_idx, config.suffix_cache_max_depth)
            pattern = self.input_batch.token_ids_cpu[i, end_idx-size:end_idx]
            pattern = pattern.tolist() + spec_ids
            if len(pattern) > config.suffix_cache_max_depth:
                pattern = pattern[-config.suffix_cache_max_depth:]
            max_spec_tokens = min(MAX_SPEC_LEN - len(spec_ids),
                                  config.suffix_cache_max_depth,
                                  self.max_model_len - end_idx - 1)
            # max_spec_offset is modified to mimic the behavior of the original
            # max_spec_factor and max_spec_offset as if the speculative tokens
            # were generated by suffix decoding. For example, if:
            #   - max_spec_factor = 2
            #   - max_spec_offset = -1
            #   - we've already speculated 3 tokens
            #   - and the suffix match length is 6
            # Then:
            #   - The match length before the already-speculated tokens is 3
            #   - The original config allow up to 5 speculated tokens total
            #   - Already speculated 3 tokens, so should allow 2 more tokens
            # So the new config should map match length 6 to 2 max spec tokens.
            max_spec_factor = config.suffix_max_spec_factor
            max_spec_offset = (config.suffix_max_spec_offset -
                              len(spec_ids) * (max_spec_factor + 1))
            result = self._suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=max_spec_factor,
                max_spec_offset=max_spec_offset,
                min_token_prob=config.suffix_min_token_prob)

            results.append(result)

        return results

    def load_model(self) -> None:
        load_shift_model = (
            self.vllm_config.parallel_config.enable_shift_parallel)

        if load_shift_model:
            # Make a deep copy of the config before loading the model.
            shift_config = copy.deepcopy(self.vllm_config)

        self._orig_load_model()

        if self.parallel_config.ulysses_sequence_parallel_size > 1:
            self.monkeypatch_forward()

        if load_shift_model:
            shift_config.parallel_config.tensor_parallel_size *= (
                shift_config.parallel_config.ulysses_sequence_parallel_size)
            shift_config.parallel_config.ulysses_sequence_parallel_size = 1
            with set_shift_parallel_mode(True):
                self.shift_model = get_model(vllm_config=shift_config)
            self.shift_parallel_threshold = (
                shift_config.parallel_config.shift_parallel_threshold)
            if "SwiftKV" in self.model.__class__.__name__:
                # HACK: Replace the decode-runner since it always runs in full
                # TP, but the original model is captured using SP * BATCH_SIZE,
                # which does not cover all its cuda graph sizes. The shift-mode
                # model should have all its cuda graphs captured correctly.
                self.model.model.decode_runner = (
                    self.shift_model.model.decode_runner)
        else:
            self.shift_model = None
            self.shift_parallel_threshold = 0

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "set -O %s and ensure `use_cudagraph` was not manually set to "
                "False", CompilationLevel.PIECEWISE)
            return

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with parallel_state.graph_capture(device=self.device):
            sp_size = self.parallel_config.ulysses_sequence_parallel_size
            full_cg = self.full_cuda_graph
            # capture original model shapes
            compilation_cases = (shape for shape in reversed(self.cudagraph_batch_sizes)
                if shape * sp_size > self.shift_parallel_threshold
                and shape * sp_size <= self.max_num_tokens)
            # Only rank 0 should print progress bar during capture
            if is_global_first_rank():
                print_cases, compilation_cases = tee(compilation_cases)
                logger.info(f"original model shapes {list(print_cases)}")
                compilation_cases = tqdm(list(compilation_cases),
                                         desc="Capturing CUDA graph shapes of original model")
            for num_tokens in compilation_cases:
                # We skip EPLB here since we don't want to record dummy metrics
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens * sp_size,
                                    capture_attn_cudagraph=full_cg,
                                    skip_eplb=True)
                self._dummy_run(num_tokens * sp_size,
                                capture_attn_cudagraph=full_cg,
                                skip_eplb=True)

            # Capture shift model shapes
            if self.shift_model is not None:
                orig_model, self.model = self.model, self.shift_model
                # Reset compilation cases
                compilation_cases = (shape for shape in reversed(self.cudagraph_batch_sizes)
                    if shape <= self.shift_parallel_threshold
                    or "SwiftKV" in self.model.__class__.__name__)
                # Note: We want to capture all shapes for the SwiftKV shift model.
                # This is necessary since SwiftKV always uses full TP for the decode runner.
                # For all other models, we only capture necessary shapes for the SP_TP mode,
                # yielding less setup time.
                if is_global_first_rank():
                    print_cases, compilation_cases = tee(compilation_cases)
                    logger.info(f"shift model shapes {list(print_cases)}")
                    compilation_cases = tqdm(list(compilation_cases),
                                             desc="Capturing CUDA graph shapes of shift model")
                with set_shift_parallel_mode(True):
                    for num_tokens in compilation_cases:
                        for _ in range(self.vllm_config.compilation_config.
                                       cudagraph_num_of_warmups):
                            self._dummy_run(num_tokens,
                                            capture_attn_cudagraph=full_cg,
                                            skip_eplb=True)
                        self._dummy_run(num_tokens,
                                        capture_attn_cudagraph=full_cg,
                                        skip_eplb=True)
                self.model = orig_model

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        self._orig_initialize_kv_cache(kv_cache_config)

        if self.shift_model is not None:
            # Bind the KV caches to the shift parallel model.
            forward_context = (
                self.vllm_config.compilation_config.static_forward_context)
            for mod in self.shift_model.modules():
                if isinstance(mod, Attention):
                    mod.kv_cache = forward_context[mod.layer_name].kv_cache
