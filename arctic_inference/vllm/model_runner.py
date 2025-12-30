# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import copy
import gc
import time
from typing import Any, Optional, TYPE_CHECKING, Union

import numpy as np
import torch
from tqdm import tqdm

import vllm.distributed.parallel_state as parallel_state
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, CompilationLevel, VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             is_global_first_rank)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils import round_up, cdiv
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN, RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    logger,
    AsyncGPUModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from arctic_inference.patching import ArcticPatch
from arctic_inference.suffix_decoding import (SuffixDecodingCache,
                                              SuffixDecodingDraft)
from arctic_inference.vllm.spec_dec.arctic_proposer import (ArcticProposer,
                                                            SuffixProposer)

SP_TP_MODE = None


@contextlib.contextmanager
def set_shift_parallel_mode(mode: Optional[bool]):
    """
    Swap the tensor-parallel group to an SP-compatible variant when 'mode' is True.
    """
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

    parallel_state._TP = (parallel_state._SP_TP
                          if mode else parallel_state._ORIG_TP)

    try:
        yield
    finally:
        SP_TP_MODE = old_mode
        parallel_state._TP = old_tp_group


def is_shift_parallel_mode() -> bool:
    """Check if the shift parallel mode is enabled."""
    global SP_TP_MODE
    return SP_TP_MODE is True


class GPUModelRunnerPatch(ArcticPatch[GPUModelRunner]):
    """
    Extension of vLLM's GPUModelRunner that adds:
      - Ulysses sequence parallel + shift-parallel dual-model routing
      - Arctic / suffix speculative decoding hooks
      - SwiftKV metadata propagation
      - Async scheduling correctness (fences + async output wrapper)
    """

    _orig_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    _orig_capture_cudagraphs = GPUModelRunner._capture_cudagraphs
    _orig_prepare_inputs = GPUModelRunner._prepare_inputs
    _orig_profile_run = GPUModelRunner.profile_run
    _orig_load_model = GPUModelRunner.load_model
    _orig_propose_draft_token_ids = GPUModelRunner.propose_draft_token_ids
    _orig_dummy_run = GPUModelRunner._dummy_run
    _orig_init = GPUModelRunner.__init__

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        if vllm_config.parallel_config.ulysses_sequence_parallel_size > 1:
            self.use_ulysses = True
            pass_config = vllm_config.compilation_config.pass_config
            if pass_config.enable_sequence_parallelism:
                raise ValueError(
                    "Ulysses sequence parallelism is incompatible with native "
                    "sequence parallelism. Set enable_sequence_parallelism "
                    "to False in the pass config to use Ulysses."
                )
        else:
            self.use_ulysses = False

        if (vllm_config.speculative_config is not None and
                vllm_config.speculative_config.method in ("arctic", "suffix", "mlp_speculator")):
            arctic_speculative_config = vllm_config.speculative_config
            vllm_config.speculative_config = None
        else:
            arctic_speculative_config = None

        self._orig_init(vllm_config, device)
        
        self._suffix_cache = None
        if arctic_speculative_config is not None:
            self.vllm_config.speculative_config = arctic_speculative_config
            self.speculative_config = arctic_speculative_config

            if get_pp_group().is_last_rank:
                if self.speculative_config.method in ("arctic", "mlp_speculator"):
                    self.drafter = ArcticProposer(self.vllm_config)
                elif self.speculative_config.method != "suffix":
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")
                else:
                    self.drafter = SuffixProposer()
                self.rejection_sampler = RejectionSampler()

        if (self.speculative_config is not None and
                self.speculative_config.enable_suffix_decoding):
            if self.speculative_config.method not in ("arctic", "suffix", "mlp_speculator"):
                raise ValueError(
                    "Suffix decoding is only supported with the 'arctic', "
                    "'mlp_speculator' or 'suffix' spec decoding methods."
                )
            spec_cfg = self.speculative_config
            self._suffix_cache = SuffixDecodingCache(
                max_tree_depth=spec_cfg.suffix_cache_max_depth,
                max_cached_requests=spec_cfg.suffix_cache_max_requests
            )
            

    def profile_run(self) -> None:
        self._orig_profile_run()
        if getattr(self, "shift_model", None) is not None:
            orig_model, self.model = self.model, self.shift_model
            try:
                with set_shift_parallel_mode(True):
                    self._dummy_run(self.max_num_tokens, is_profile=True)
            finally:
                self.model = orig_model

    def _prepare_inputs(self, *args, **kwargs):
        """
        Forward to upstream _prepare_inputs, then add SwiftKV-specific metadata.
        """
        (attn_metadata, logits_indices, spec_decode_metadata,
         num_scheduled_tokens_np, spec_decode_common_attn_metadata,
         max_query_len, ubatch_slices, num_tokens_after_padding) = (
            self._orig_prepare_inputs(*args, **kwargs)
        )

        if isinstance(attn_metadata, list):
            for ubatch_attn_metadata in attn_metadata:
                for meta in ubatch_attn_metadata.values():
                    meta.swiftkv_logits_indices = logits_indices
        else:
            for meta in attn_metadata.values():
                meta.swiftkv_logits_indices = logits_indices

        return (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                max_query_len, ubatch_slices, num_tokens_after_padding)

    def monkeypatch_forward(self: GPUModelRunner):
        """
        Slice the batch across Ulysses SP ranks for forward, then all-gather.
        """
        sp_size = parallel_state._SP.world_size
        sp_rank = parallel_state._SP.rank_in_group
        device_group = parallel_state._SP.device_group
        model_forward = self.model.forward
        input_key = 'inputs_embeds' if self.supports_mm_inputs else 'input_ids'

        def ulysses_forward(*args, **kwargs):
            input_tensor = kwargs[input_key]
            positions = kwargs['positions']

            N = input_tensor.shape[0]
            N_ulysses = N // sp_size
            N_offset = N_ulysses * sp_rank

            kwargs[input_key] = input_tensor[N_offset:N_offset + N_ulysses]
            kwargs['positions'] = positions[N_offset:N_offset + N_ulysses]

            with set_shift_parallel_mode(False):
                output = model_forward(*args, **kwargs)

            if output.size(0) == N_ulysses:
                model_output = torch.empty((N, self.hidden_size),
                                           dtype=output.dtype,
                                           device=output.device)
                torch.distributed.all_gather_into_tensor(model_output,
                                                         output,
                                                         group=device_group)
            else:
                assert output.size(0) == N
                model_output = output
            return model_output

        self.get_model().forward = ulysses_forward

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
        """
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = num_tokens // 2
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [
                num_prefill_tokens
            ]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())

        num_scheduled_tokens_for_logits = num_scheduled_tokens[num_scheduled_tokens > 0]
        if num_scheduled_tokens_for_logits.size == 0:
            # Handle edge case: all requests have 0 tokens.
            logits_indices = torch.empty(0, dtype=torch.long, device=self.device)
            # Use numpy for the final return indexing
            logit_indices_np = np.array([], dtype=int)
        else:
            logits_indices_cpu = np.cumsum(num_scheduled_tokens_for_logits) - 1
            logits_indices = torch.from_numpy(logits_indices_cpu).to(self.device)
            # Use numpy for the final return indexing
            logit_indices_np = np.cumsum(num_scheduled_tokens) - 1

        ubatch_slices = None
        num_tokens_after_padding = None

        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        if self.parallel_config.enable_dbo and allow_microbatching:
            ubatch_slices, ubatch_num_tokens_after_padding = ubatch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            # Currently when DBO is enabled `ubatch_split` returns
            # the num_tokens_after_padding for a single ubatch, but we have 2
            # TODO(sage,lucas): this is cruft that should be addressed in the
            # padding refactor.
            if ubatch_num_tokens_after_padding is not None:
                num_tokens_after_padding = ubatch_num_tokens_after_padding * 2

        # If we failed to microbatch, currently need to resynchronize
        # TODO(lucas,sage): we should be able to avoid this second sync by
        #  refactoring `get_dp_padding_ubatch` and `get_dp_padding` into
        #  a single `coordinate_batch_across_dp` function.
        if num_tokens_after_padding is None:
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
            num_tokens_after_padding = num_tokens + num_pad
        else:
            num_tokens_across_dp = num_tokens_after_padding
            num_tokens_after_padding = int(num_tokens_after_padding[0].item())

        attn_metadata: Optional[PerLayerAttnMetadata] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            attn_metadata = {}
            if ubatch_slices is not None:
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]

            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(
                num_scheduled_tokens)
            self.query_start_loc.np[1:num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    max_seq_len=self.max_model_len,
                    block_table_tensor=self.input_batch.
                    block_table[kv_cache_group_id].get_device_tensor(num_reqs),
                    slot_mapping=self.input_batch.block_table[
                        kv_cache_group_id].slot_mapping.gpu[:num_tokens],
                    causal=True)
                for attn_group in self.attn_groups[kv_cache_group_id]:
                    if ubatch_slices is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            ubatch_slices, common_attn_metadata)
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = (attn_group\
                                               .get_metadata_builder(ubatch_id=ubid)\
                                               .build_for_cudagraph_capture(common_attn_metadata))
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        assert type(attn_metadata) is dict
                        attn_metadata_i = attn_group.get_metadata_builder()\
                            .build_for_cudagraph_capture(common_attn_metadata)
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i

        if attn_metadata is not None:
            if isinstance(attn_metadata, list):
                for ubatch_attn_metadata in attn_metadata:
                    for meta in ubatch_attn_metadata.values():
                        meta.swiftkv_logits_indices = logits_indices
            else:
                for meta in attn_metadata.values():
                    meta.swiftkv_logits_indices = logits_indices

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens, remove_lora):
            model_kwargs = self._init_model_kwargs(num_tokens)
            if (self.supports_mm_inputs
                    and not self.model_config.is_encoder_decoder):
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = self._init_model_kwargs(num_tokens)
            else:
                input_ids = self.input_ids.gpu[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens]
            else:
                positions = self.positions.gpu[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)

            # filter out the valid batch descriptor
            _cg_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                BatchDescriptor(num_tokens=num_tokens_after_padding,
                                uniform_decode=uniform_decode)) \
                if not is_profile else (CUDAGraphMode.NONE, None)
            if cudagraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for cudagraph capture
                assert cudagraph_runtime_mode == CUDAGraphMode.NONE or \
                    cudagraph_runtime_mode == _cg_mode, (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}.")
            else:
                cudagraph_runtime_mode = _cg_mode

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    ubatch_slices=ubatch_slices):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[
        ModelRunnerOutput, AsyncGPUModelRunnerOutput, IntermediateTensors
    ]:
        """
        This override preserves the upstream execution order but:
          - wraps input prep with the async fence (fixes hangs),
          - returns AsyncGPUModelRunnerOutput when async scheduling is enabled,
          - routes forward() to shift_model when under the threshold,
          - computes logits with the same model used for forward().
        """
        with self.synchronize_input_prep():
            self._update_states(scheduler_output)

            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)

            (attn_metadata, logits_indices, spec_decode_metadata,
             num_scheduled_tokens_np, spec_decode_common_attn_metadata,
             max_query_len, ubatch_slices, num_tokens_after_padding) = (
                self._prepare_inputs(scheduler_output)
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        use_shift_model = (
            getattr(self, "use_ulysses", False)
            and getattr(self, "shift_model", None) is not None
            and num_scheduled_tokens <= self.shift_parallel_threshold
        )

        if self.use_ulysses and not use_shift_model:
            sp_size = self.parallel_config.ulysses_sequence_parallel_size
            num_input_tokens = round_up(num_scheduled_tokens, sp_size)
            if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                    and num_input_tokens // sp_size <= self.cudagraph_batch_sizes[-1]):
                num_input_tokens = (
                    self.vllm_config.pad_for_cudagraph(num_input_tokens // sp_size) * sp_size
                )
        elif (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
              and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens
            )
        else:
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if (self.compilation_config.pass_config.enable_sequence_parallelism
                    and tp_size > 1):
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        if self.supports_mm_inputs:
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.supports_mm_inputs and get_pp_group().is_first_rank:
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled
            )
            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        else:
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)

        if (self.model_config.is_encoder_decoder
                and scheduler_output.scheduled_encoder_inputs):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        uniform_decode = (
            max_query_len == self.uniform_decode_query_len
            and num_scheduled_tokens == self.input_batch.num_reqs * max_query_len
        )
        batch_descriptor = BatchDescriptor(
            num_tokens=num_input_tokens, uniform_decode=uniform_decode
        )
        cudagraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(batch_descriptor)

        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            ubatch_slices=ubatch_slices,
        ), self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output:
            model = self.shift_model if use_shift_model else self.model
            with set_shift_parallel_mode(use_shift_model):
                model_output = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 0
        )

        if not get_pp_group().is_last_rank:
            assert isinstance(hidden_states, IntermediateTensors)
            if not broadcast_pp_output:
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states
            get_pp_group().send_tensor_dict(
                hidden_states.tensors,
                all_gather_group=get_tp_group(),
            )
            logits = None
        else:
            if self.input_batch.pooling_params:
                output = self._pool(
                    hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
                )
                output.kv_connector_output = kv_connector_output
                return output

            sample_hidden_states = hidden_states[logits_indices]
            with set_shift_parallel_mode(False):
                logits = self.model.compute_logits(sample_hidden_states)

        if broadcast_pp_output:
            model_output_broadcast_data = {}
            if logits is not None:
                model_output_broadcast_data["logits"] = logits.contiguous()
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
            )
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        if scheduler_output.grammar_bitmask is not None:
            from vllm.v1.structured_output.utils import apply_grammar_bitmask
            apply_grammar_bitmask(scheduler_output, self.input_batch,
                                  logits, self.device)

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        with record_function_or_nullcontext("Bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output, sampler_output, logits, hidden_states, num_scheduled_tokens
            )

        if self._suffix_cache is not None:
            self._update_suffix_cache(valid_sampled_token_ids)

        sampling_metadata = self.input_batch.sampling_metadata

        if not self.speculative_config:
            self._draft_token_ids = None
        else:
            assert spec_decode_common_attn_metadata is not None
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                sampler_output.sampled_token_ids,
                sampling_metadata,
                hidden_states,
                hidden_states[logits_indices], 
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

        self.eplb_step()

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
        )

        if self.use_async_scheduling:
            return AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
            )
        return output

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        original_sampled_token_ids: Union[np.ndarray, torch.Tensor],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[list[int]]:
        disable_spec_decode = (
            self.speculative_config
            and self.speculative_config.disable_by_batch_size
            and len(self.input_batch.req_ids) > self.speculative_config.disable_by_batch_size
        )
        if disable_spec_decode:
            return [[] for _ in sampled_token_ids]

        suffix_spec_token_ids = None
        new_sampled_token_ids = sampled_token_ids.copy()
        if self._suffix_cache is not None:
            results = self.propose_suffix_draft_token_ids(new_sampled_token_ids)
            suffix_spec_token_ids = []
            min_score = 0 if self.speculative_config.method == "suffix" \
                else self.speculative_config.num_speculative_tokens
            for i, result in enumerate(results):
                if result.score >= min_score:
                    new_sampled_token_ids[i] = []
                    suffix_spec_token_ids.append(result.token_ids)
                else:
                    suffix_spec_token_ids.append([])

        spec_token_ids = None
        if self.speculative_config.method == "suffix":
            pass
        elif self.speculative_config.method in ("arctic", "mlp_speculator"):
            assert isinstance(self.drafter, ArcticProposer)
            if isinstance(original_sampled_token_ids, np.ndarray):
                original_sampled_token_ids_tensor = torch.from_numpy(
                    original_sampled_token_ids
                ).to(self.device)
            else:
                original_sampled_token_ids_tensor = original_sampled_token_ids

            previous_hidden_states = self.drafter.prepare_hidden_states(
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=original_sampled_token_ids_tensor,
                spec_decode_metadata=spec_decode_metadata,
            )
            spec_token_ids = self.propose_arctic_draft_token_ids(
                scheduler_output,
                new_sampled_token_ids,
                previous_hidden_states=previous_hidden_states,
            )
        else:
            spec_token_ids = self._orig_propose_draft_token_ids(
                scheduler_output,
                new_sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                common_attn_metadata,
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
        last_tokens = []
        max_spec_tokens = self.speculative_config.num_speculative_tokens
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)

            if num_sampled_ids == 0:
                if self.speculative_config.enable_suffix_decoding:
                    return [[]] * len(sampled_token_ids)
                req_id = self.input_batch.req_ids[i]
                req_state = self.requests[req_id]
                seq_len = (req_state.num_computed_tokens +
                           scheduler_output.num_scheduled_tokens[req_id])
                sampled_ids = [req_state.get_token_id(seq_len)]
                num_sampled_ids = len(sampled_ids)

            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids

            current_max_spec_tokens = min(
                max_spec_tokens,
                self.max_model_len - end_idx - 1,
            )
            if current_max_spec_tokens <= 0:
                last_tokens.append(None)
                continue

            if num_sampled_ids > 0:
                self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
                last_tokens.append(
                    self.input_batch.token_ids_cpu[i, end_idx - 1].item()
                )
            else:
                last_tokens.append(None)
                continue

        valid_last_tokens = [t for t in last_tokens if t is not None]
        if not valid_last_tokens:
            return [[] for _ in sampled_token_ids]

        if previous_hidden_states is not None:
            indices = [i for i, t in enumerate(last_tokens) if t is not None]
            previous_hidden_states = previous_hidden_states[indices]

        final_max_spec_tokens = min(
            max_spec_tokens,
            self.max_model_len - self.input_batch.num_tokens_no_spec.max() - 1,
        )
        if final_max_spec_tokens <= 0:
            return [[] for _ in sampled_token_ids]

        drafter_output = self.drafter.propose(
            valid_last_tokens,
            previous_hidden_states=previous_hidden_states,
            num_predict_tokens=final_max_spec_tokens,
        )

        draft_token_ids_list = drafter_output.tolist()
        final_draft_token_ids = []
        draft_iter = iter(draft_token_ids_list)
        for t in last_tokens:
            if t is not None:
                final_draft_token_ids.append(next(draft_iter))
            else:
                final_draft_token_ids.append([])

        return final_draft_token_ids

    def _update_suffix_cache(self, sampled_token_ids: list[list[int]]) -> None:
        seen_req_ids = set()
        for i, sampled_ids in enumerate(sampled_token_ids):
            req_id = self.input_batch.req_ids[i]
            seen_req_ids.add(req_id)

            if not sampled_ids:
                continue

            index = self.input_batch.req_id_to_index[req_id]
            if req_id not in self._suffix_cache.active_requests:
                if req_id in self._suffix_cache.cached_requests:
                    self._suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = self.input_batch.num_prompt_tokens[index]
                prompt_token_ids = self.input_batch.token_ids_cpu[index, :num_prompt_tokens]
                self._suffix_cache.start_request(req_id, prompt_token_ids.tolist())

            self._suffix_cache.add_active_response(req_id, sampled_ids)

        for req_id in list(self._suffix_cache.active_requests):
            if req_id not in seen_req_ids:
                self._suffix_cache.stop_request(req_id)

    def propose_suffix_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
    ) -> list[SuffixDecodingDraft]:
        config = self.speculative_config
        results = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                results.append(SuffixDecodingDraft())
                continue

            req_id = self.input_batch.req_ids[i]
            if req_id in self.input_batch.spec_decode_unsupported_reqs:
                results.append(SuffixDecodingDraft())
                continue

            num_tokens = self.input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                results.append(SuffixDecodingDraft())
                continue

            start = max(0, num_tokens - config.suffix_cache_max_depth)
            pattern = self.input_batch.token_ids_cpu[i, start:num_tokens].tolist()
            result = self._suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    MAX_SPEC_LEN, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=config.suffix_max_spec_factor,
                max_spec_offset=config.suffix_max_spec_offset,
                min_token_prob=config.suffix_min_token_prob,
            )

            results.append(result)

        return results

    def load_model(self, eep_scale_up: bool = False) -> None:
        load_shift_model = (
            self.vllm_config.parallel_config.enable_shift_parallel
        )

        if load_shift_model:
            shift_config = copy.deepcopy(self.vllm_config)

        self._orig_load_model(eep_scale_up)

        if self.parallel_config.ulysses_sequence_parallel_size > 1:
            self.monkeypatch_forward()

        if load_shift_model:
            shift_config.parallel_config.tensor_parallel_size *= (
                shift_config.parallel_config.ulysses_sequence_parallel_size
            )
            shift_config.parallel_config.ulysses_sequence_parallel_size = 1
            with set_shift_parallel_mode(True):
                self.shift_model = get_model(vllm_config=shift_config)
            self.shift_parallel_threshold = (
                shift_config.parallel_config.shift_parallel_threshold
            )
            if "SwiftKV" in self.model.__class__.__name__:
                if hasattr(self.model, "model") and hasattr(self.model.model, "decode_runner"):
                    self.model.model.decode_runner = self.shift_model.model.decode_runner
                else:
                    logger.warning("Could not apply SwiftKV HACK: "
                                   "model.model.decode_runner not found.")
        else:
            self.shift_model = None
            self.shift_parallel_threshold = 0

    def _capture_cudagraphs(self, compilation_cases: list[int],
                            cudagraph_runtime_mode: CUDAGraphMode,
                            uniform_decode: bool):
        """
        Capture CUDA graphs for both base (SP) and shift (TP) variants, splitting
        shapes by threshold so both models have required graphs captured.
        """
        # make sure it is a valid mode
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE and \
            cudagraph_runtime_mode in [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]

        sp_size = parallel_state._SP.world_size
        tp_size = parallel_state._TP.world_size

        # capture shapes for the base model
        compilation_cases_base = [
            min(shape * sp_size, self.max_num_tokens)
            for shape in compilation_cases
            if shape * sp_size > self.shift_parallel_threshold
        ]
        # filter again for FULL mode
        if cudagraph_runtime_mode == CUDAGraphMode.FULL:
            compilation_cases_base = [shape for shape in compilation_cases_base if
                                      shape <= self.scheduler_config.max_num_seqs]

        if is_global_first_rank():
            logger.info(f"base model (SP={sp_size}, TP={tp_size}) shapes {compilation_cases_base}, cudagraph mode {cudagraph_runtime_mode}")

        # capture the base model graphs
        if compilation_cases_base:
            self._orig_capture_cudagraphs(
                compilation_cases_base, cudagraph_runtime_mode, uniform_decode
            )

        if getattr(self, "shift_model", None) is not None:

            # capture shapes for the shift model
            compilation_cases_shift = [
                shape for shape in compilation_cases
                if shape <= self.shift_parallel_threshold
                or "SwiftKV" in self.model.__class__.__name__
            ]
            # filter again for FULL mode
            if cudagraph_runtime_mode == CUDAGraphMode.FULL:
                compilation_cases_shift = [shape for shape in compilation_cases_shift if
                                           shape <= self.scheduler_config.max_num_seqs]

            if is_global_first_rank():
                logger.info(f"shift model (SPxTP={sp_size * tp_size}) shapes {compilation_cases_shift}, cudagraph mode {cudagraph_runtime_mode}")

            # capture the shift model graphs
            if compilation_cases_shift:
                orig_model, self.model = self.model, self.shift_model
                try:
                    with set_shift_parallel_mode(True):
                        # capture the shift model graphs
                        self._orig_capture_cudagraphs(
                            compilation_cases_shift, cudagraph_runtime_mode, uniform_decode
                        )
                finally:
                    self.model = orig_model

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache for base model, then bind the same buffers into
        the shift-parallel model so they share cache/state.
        """
        self._orig_initialize_kv_cache(kv_cache_config)

        if getattr(self, "shift_model", None) is not None:
            forward_context = self.vllm_config.compilation_config.static_forward_context

            try:
                from vllm.attention import Attention
                for mod in self.shift_model.modules():
                    if isinstance(mod, Attention):
                        if hasattr(mod, "layer_name"):
                            ln = mod.layer_name
                            if ln in forward_context:
                                mod.kv_cache = forward_context[ln].kv_cache
                            else:
                                logger.warning(f"Could not find {ln} in forward_context for shift_model.")
                        else:
                            logger.warning("Could not bind KV cache for shift_model: "
                                           "Attention module missing 'layer_name'.")
            except ImportError:
                logger.warning("Could not import Attention to bind KV cache for shift_model.")

            try:
                from vllm.model_executor.layers.mamba.abstract import MambaBase
                for mod in self.shift_model.modules():
                    if isinstance(mod, MambaBase):
                        if hasattr(mod, "layer_name"):
                            ln = mod.layer_name
                            if ln in forward_context:
                                mod.state = forward_context[ln].state
                            else:
                                logger.warning(f"Could not find {ln} in forward_context for shift_model.")
                        else:
                            logger.warning("Could not bind Mamba state for shift_model: "
                                           "Mamba module missing 'layer_name'.")
            except ImportError:
                pass