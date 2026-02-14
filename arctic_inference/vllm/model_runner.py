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
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             is_global_first_rank)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import round_up, cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput,
                              SamplerOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN, RejectionSampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    logger,
    AsyncGPUModelRunnerOutput,
)
from vllm.v1.structured_output.utils import apply_grammar_bitmask

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
    Rebased GPUModelRunnerPatch for vLLM v14.
    """

    _orig_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    _orig_capture_cudagraphs = GPUModelRunner._capture_cudagraphs
    _orig_prepare_inputs = GPUModelRunner._prepare_inputs
    _orig_profile_run = GPUModelRunner.profile_run
    _orig_load_model = GPUModelRunner.load_model
    _orig_propose_draft_token_ids = GPUModelRunner.propose_draft_token_ids
    _orig_dummy_run = GPUModelRunner._dummy_run
    _orig_init = GPUModelRunner.__init__
    _orig_build_attention_metadata = GPUModelRunner._build_attention_metadata
    _orig_execute_model = GPUModelRunner.execute_model
    # _orig_pad_for_sequence_parallelism = GPUModelRunner._pad_for_sequence_parallelism

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        if vllm_config.parallel_config.ulysses_sequence_parallel_size > 1:
            self.use_ulysses = True
            pass_config = vllm_config.compilation_config.pass_config
            if pass_config.enable_sp:
                raise ValueError(
                    "Ulysses sequence parallelism is incompatible with native "
                    "sequence parallelism. Set enable_sequence_parallelism "
                    "to False in the pass config to use Ulysses."
                )
        else:
            self.use_ulysses = False

        arctic_methods = ("arctic", "suffix", "mlp_speculator")
        is_arctic_spec = (vllm_config.speculative_config is not None and 
                          vllm_config.speculative_config.method in arctic_methods)

        arctic_speculative_config = None
        if is_arctic_spec:
            arctic_speculative_config = vllm_config.speculative_config
            vllm_config.speculative_config = None

        self._orig_init(vllm_config, device)

        self._suffix_cache: Optional[SuffixDecodingCache] = None
        
        if is_arctic_spec:
            self.vllm_config.speculative_config = arctic_speculative_config
            self.speculative_config = arctic_speculative_config
            self.num_spec_tokens = getattr(self.speculative_config, 
                                           "num_speculative_tokens", 0)
            self.uniform_decode_query_len = 1 + self.num_spec_tokens

            if not hasattr(self, "draft_token_ids_cpu") or self.draft_token_ids_cpu is None:
                self.draft_token_ids_event = torch.Event()
                self.draft_token_ids_copy_stream = torch.cuda.Stream()
                self.draft_token_ids_cpu = torch.empty(
                    (self.max_num_reqs, self.num_spec_tokens),
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

            if (self.use_async_scheduling
                    and self.speculative_config.method in ("arctic", "mlp_speculator", "suffix")):
                if not hasattr(self, "valid_sampled_token_count_cpu") or self.valid_sampled_token_count_cpu is None:
                    self.valid_sampled_token_count_event = torch.Event()
                    self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                    self.valid_sampled_token_count_cpu = torch.empty(
                        self.max_num_reqs,
                        dtype=torch.int64,
                        device="cpu",
                        pin_memory=self.pin_memory,
                    )

            if get_pp_group().is_last_rank:
                if self.speculative_config.method in ("arctic", "mlp_speculator"):
                    self.drafter = ArcticProposer(self.vllm_config)
                elif self.speculative_config.method == "suffix":
                    self.drafter = SuffixProposer()
                else:
                    raise ValueError(f"Unknown speculative decoding method: {self.speculative_config.method}")
                
                self.rejection_sampler = RejectionSampler(self.sampler)

        if (self.speculative_config is not None and 
                getattr(self.speculative_config, "enable_suffix_decoding", False)):
            
            if self.speculative_config.method not in arctic_methods:
                raise ValueError(
                    "Suffix decoding is only supported with the 'arctic', "
                    "'mlp_speculator' or 'suffix' spec decoding methods."
                )
            spec_cfg = self.speculative_config
            self._suffix_cache = SuffixDecodingCache(
                max_tree_depth=spec_cfg.suffix_cache_max_depth,
                max_cached_requests=spec_cfg.suffix_cache_max_requests
            )

        # Async suffix decoding infrastructure: a dedicated CUDA stream and
        # pinned buffer for copying sampled token IDs to CPU *without*
        # serialising behind Arctic GPU drafting work on the default stream.
        if self._suffix_cache is not None and self.use_async_scheduling:
            self.suffix_copy_stream = torch.cuda.Stream()
            self.suffix_copy_done_event = torch.Event()
            max_gen_len = 1 + self.num_spec_tokens
            self.suffix_sampled_ids_pinned = torch.empty(
                (self.max_num_reqs, max_gen_len),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            # Pinned buffer for suffix merge results.  Using pinned memory
            # for H2C copies avoids the implicit default-stream
            # synchronisation that cudaMemcpyAsync performs with pageable
            # (non-pinned) source memory.  Without this, the merge step
            # blocks the CPU until ALL pending GPU work (including Arctic
            # drafting) completes, destroying the async overlap.
            self._suffix_merge_pinned = torch.zeros(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory,
            )

        # Per-request response tokens for suffix pattern building in async
        # mode.  In async scheduling, _bookkeeping_sync writes -1 placeholders
        # to token_ids_cpu instead of real values, corrupting the pattern that
        # propose_suffix_draft_token_ids reads.  We keep a clean copy here.
        self._suffix_response_tokens: dict[str, list[int]] = {}

        # Backup-token buffer used by suffix-only async rejection sampling.
        # The arctic proposer has its own buffer; this one covers the case
        # where no arctic drafter is present.
        self._suffix_backup_tokens_gpu: Optional[torch.Tensor] = None
        if (self._suffix_cache is not None
                and self.use_async_scheduling
                and self.speculative_config.method not in ("arctic",
                                                           "mlp_speculator")):
            self._suffix_backup_tokens_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device,
            )

    def _suffix_only_rejection_sample(
        self,
        sampled_token_ids: torch.Tensor,
        common_attn_metadata: "CommonAttentionMetadata",
    ) -> None:
        """Rejection-sample accepted tokens for suffix-only async scheduling.

        EAGLE / arctic do this inside propose_draft_token_ids via
        prepare_next_token_ids_padded.  For suffix-only there is no
        drafter with that method, so we call the same Triton kernel
        directly and feed the results into _copy_valid_sampled_token_count.
        """
        from vllm.triton_utils import triton
        from vllm.v1.spec_decode.utils import (
            eagle_prepare_next_token_padded_kernel,
        )

        num_reqs = self.input_batch.num_reqs
        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        # Compute backup tokens (last accepted token per request) on CPU,
        # then copy to GPU in one shot to avoid per-element synchronisation.
        backup = self._suffix_backup_tokens_gpu
        assert backup is not None
        backup_np = np.empty(num_reqs, dtype=np.int32)
        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]
            seq_len = int(common_attn_metadata.seq_lens_cpu[i].item())
            backup_np[i] = self.requests[req_id].get_token_id(seq_len)
        # Copy directly from CPU numpy-backed tensor to GPU; avoids
        # creating an intermediate GPU tensor via .to(device).
        backup[:num_reqs].copy_(
            torch.from_numpy(backup_np), non_blocking=True,
        )

        next_token_ids = torch.empty(batch_size, dtype=torch.int32,
                                     device=device)
        valid_counts = torch.empty(batch_size, dtype=torch.int32,
                                   device=device)

        BLOCK_SIZE_TOKENS = triton.next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[(batch_size,)](
            sampled_token_ids,
            self.discard_request_mask.gpu,
            backup,
            next_token_ids,
            valid_counts,
            self.model_config.get_vocab_size(),
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        self._copy_valid_sampled_token_count(next_token_ids, valid_counts)

    def _build_attention_metadata(self, *args, **kwargs):
        attn_metadata, spec_decode_common_attn_metadata = \
            self._orig_build_attention_metadata(*args, **kwargs)

        logits_indices = kwargs.get("logits_indices", None)
        if logits_indices is not None:
            if isinstance(attn_metadata, list):
                for ub in attn_metadata:
                    for meta in ub.values():
                        meta.swiftkv_logits_indices = logits_indices
            else:
                for meta in attn_metadata.values():
                    meta.swiftkv_logits_indices = logits_indices

        return attn_metadata, spec_decode_common_attn_metadata

    # set padding for SP here
    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:

        sp_size = self.parallel_config.ulysses_sequence_parallel_size
        num_input_tokens = round_up(num_scheduled_tokens, sp_size)

        #if torch.distributed.get_rank() == 0:
        #    print(f"padding num_scheduled_tokens {num_scheduled_tokens} -> num_input_tokens {num_input_tokens}")

        return num_input_tokens


    def profile_run(self) -> None:
        self._orig_profile_run()
        if getattr(self, "shift_model", None) is not None:
            orig_model, self.model = self.model, self.shift_model
            try:
                with set_shift_parallel_mode(True):
                    self._dummy_run(self.max_num_tokens, is_profile=True)
            finally:
                self.model = orig_model


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

            if torch.distributed.get_rank() == 0:
                print(f"N {N}, N_ulysses {N_ulysses}")

            kwargs[input_key] = input_tensor[N_offset:N_offset + N_ulysses]
            kwargs['positions'] = positions[N_offset:N_offset + N_ulysses]

            with set_shift_parallel_mode(False):
                output = model_forward(*args, **kwargs)

            if output.size(0) == N_ulysses:
                model_output = torch.empty((N, output.shape[1]),
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
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        from vllm.v1.worker.gpu_model_runner import supports_mm_encoder_only
        if supports_mm_encoder_only(self.model):
            return torch.tensor([]), torch.tensor([])

        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        
        if create_mixed_batch:
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        if torch.distributed.get_rank() == 0:
            print(f"num_tokens_unpadded: {num_tokens_unpadded}, num_reqs: {num_reqs}")

        _cg_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                force_uniform_decode=uniform_decode,
                force_has_lora=activate_lora,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cg_mode
        
        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs

        from vllm.v1.worker.gpu_model_runner import maybe_create_ubatch_slices
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            self.vllm_config.parallel_config.num_ubatches,
        )

        logits_indices_cpu = np.cumsum(num_scheduled_tokens) - 1
        logits_indices = torch.from_numpy(logits_indices_cpu).to(self.device)

        attn_metadata = None
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                seq_lens_list = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens_list = [max_query_len] * num_reqs # simplified
                
            self.seq_lens.np[:num_reqs] = seq_lens_list
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            pad_attn = (cudagraph_runtime_mode == CUDAGraphMode.FULL)
            attn_metadata, _ = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs_padded,
                max_query_len=max_query_len,
                ubatch_slices=ubatch_slices_padded if pad_attn else ubatch_slices,
                for_cudagraph_capture=is_graph_capturing,
            )

        if attn_metadata is not None:
            if isinstance(attn_metadata, list):
                for ub_meta in attn_metadata:
                    for meta in ub_meta.values():
                        meta.swiftkv_logits_indices = logits_indices
            else:
                for meta in attn_metadata.values():
                    meta.swiftkv_logits_indices = logits_indices

        with self.maybe_dummy_run_with_lora(self.lora_config, num_scheduled_tokens, 
                                            num_sampled_tokens, activate_lora, remove_lora):
            
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)
                model_kwargs.update(self._dummy_mm_kwargs(num_reqs))
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            positions = self.positions.gpu[:num_tokens_padded]
            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]

            intermediate_tensors = None
            if not get_pp_group().is_first_rank:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens, dtype=self.model_config.dtype, device=self.device)
                intermediate_tensors = self.sync_and_slice_intermediate_tensors(num_tokens_padded, None, False)

            target_num_tokens = num_tokens_padded
            if ubatch_slices_padded is not None:
                target_num_tokens = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = target_num_tokens

            with self.maybe_randomize_inputs(input_ids, inputs_embeds), set_forward_context(
                attn_metadata, self.vllm_config, num_tokens=target_num_tokens,
                num_tokens_across_dp=num_tokens_across_dp, cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc, ubatch_slices=ubatch_slices_padded):
                
                outputs = self.model(input_ids=input_ids, positions=positions, 
                                     intermediate_tensors=intermediate_tensors, 
                                     inputs_embeds=inputs_embeds, **model_kwargs)

            hidden_states = outputs[0] if self.use_aux_hidden_state_outputs else outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                self.drafter.dummy_run(num_tokens, use_cudagraphs=False, is_graph_capturing=is_graph_capturing)

        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        return hidden_states, hidden_states[logits_indices]

    # ------------------------------------------------------------------
    # _sample override: bypass the rejection sampler when
    # disable_by_batch_size is active.
    #
    # In async scheduling the AsyncScheduler always adds placeholder
    # spec tokens for every decode request, so spec_decode_metadata is
    # never None.  When the batch exceeds disable_by_batch_size, draft
    # proposal is correctly skipped (propose_draft_token_ids returns
    # empty), but the *current* step still runs the rejection sampler
    # on the (large) logits tensor → logits.sort inside top-k/top-p
    # OOMs.
    #
    # Fix: when disable_by_batch_size triggers, sample only the first
    # (target) logit per request with the regular sampler and return an
    # "all-drafts-rejected" output.  The scheduler's accounting stays
    # correct because update_from_output sees num_accepted=0 and
    # adjusts num_computed_tokens / num_output_placeholders accordingly.
    # ------------------------------------------------------------------
    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        if (
            spec_decode_metadata is not None
            and self.speculative_config
            and self.speculative_config.disable_by_batch_size
            and len(self.input_batch.req_ids)
            > self.speculative_config.disable_by_batch_size
        ):
            sampling_metadata = self.input_batch.sampling_metadata
            self.input_batch.update_async_output_token_ids()

            # Replay the async-spec-token update the base _sample does
            # so that penalties / bad-words see the right token history.
            if (self.use_async_scheduling
                    and self._draft_token_req_ids is not None):
                draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
                self.input_batch.update_async_spec_token_ids(
                    draft_token_ids_cpu)

            # Extract the first (target) logit per request.
            cu = spec_decode_metadata.cu_num_sampled_tokens
            batch_size = len(spec_decode_metadata.num_draft_tokens)
            first_indices = torch.zeros(
                batch_size, dtype=cu.dtype, device=cu.device)
            if batch_size > 1:
                first_indices[1:] = cu[:-1]
            assert logits is not None
            first_logits = logits[first_indices]

            regular_output = self.sampler(
                logits=first_logits,
                sampling_metadata=sampling_metadata,
            )

            # Build "all-drafts-rejected" output in the shape the
            # rejection sampler would return:
            #   (batch_size, max_spec_len + 1)
            # Position 0 = accepted token, positions 1.. = placeholder.
            max_spec_len = spec_decode_metadata.max_spec_len
            PLACEHOLDER = -1  # matches PLACEHOLDER_TOKEN_ID in
            #                   rejection_sampler.py
            output_token_ids = torch.full(
                (batch_size, max_spec_len + 1),
                PLACEHOLDER,
                device=regular_output.sampled_token_ids.device,
                dtype=regular_output.sampled_token_ids.dtype,
            )
            output_token_ids[:, 0] = (
                regular_output.sampled_token_ids.squeeze(-1))

            self._update_states_after_model_execute(output_token_ids)
            return SamplerOutput(
                sampled_token_ids=output_token_ids,
                logprobs_tensors=None,
            )

        # Inline the base GPUModelRunner._sample logic here because the
        # class is monkey-patched at runtime, making both super() and
        # GPUModelRunner._sample(self, ...) resolve back to this method.
        sampling_metadata = self.input_batch.sampling_metadata
        self.input_batch.update_async_output_token_ids()
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits, sampling_metadata=sampling_metadata)

        if (self.use_async_scheduling
                and self._draft_token_req_ids is not None):
            draft_token_ids_cpu, _ = self._get_draft_token_ids_cpu()
            self.input_batch.update_async_spec_token_ids(
                draft_token_ids_cpu)

        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            logits,
            sampling_metadata,
        )
        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids)
        return sampler_output

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[
        ModelRunnerOutput, AsyncGPUModelRunnerOutput, IntermediateTensors
    ]:
        num_scheduled_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
        if num_scheduled_tokens is None:
            try:
                num_scheduled_tokens = int(
                    sum(scheduler_output.num_scheduled_tokens.values()) 
                )
            except Exception:
                num_scheduled_tokens = 0

        use_shift_model = (
            getattr(self, "use_ulysses", False)
            and getattr(self, "shift_model", None) is not None
            and num_scheduled_tokens <= int(getattr(self, "shift_parallel_threshold", 0))
        )

        if not use_shift_model:
            return self._orig_execute_model(scheduler_output, intermediate_tensors) 

        orig_model = self.model
        try:
            self.model = self.shift_model
            with set_shift_parallel_mode(True):
                return self._orig_execute_model(scheduler_output, intermediate_tensors) 
        finally:
            self.model = orig_model

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[list[int]] | torch.Tensor:
        drafting_disabled = (
            self.speculative_config
            and self.speculative_config.disable_by_batch_size
            and len(self.input_batch.req_ids) > self.speculative_config.disable_by_batch_size
        )

        use_async_path = (
            self.speculative_config.method in ("arctic", "mlp_speculator")
            and isinstance(sampled_token_ids, torch.Tensor)
            and self.use_async_scheduling
            and common_attn_metadata is not None
        )

        if drafting_disabled and not use_async_path:
            # Non-async: safe to return early; bookkeeping runs before
            # the draft result is consumed.
            if self.speculative_config.method in ("arctic", "mlp_speculator"):
                 batch_size = len(self.input_batch.req_ids)
                 return torch.empty((batch_size, 0), dtype=torch.int64, device=self.device)
            return [[] for _ in range(len(self.input_batch.req_ids))]

        if use_async_path:
            assert isinstance(sampled_token_ids, torch.Tensor)
            
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                )
            )
            self._copy_valid_sampled_token_count(
                next_token_ids, valid_sampled_tokens_count
            )

            # Check disable_by_batch_size AFTER setting
            # prev_sampled_token_ids / valid_sampled_token_count
            # (via _copy_valid_sampled_token_count above) so that
            # _bookkeeping_sync and _update_states still work.
            if drafting_disabled:
                batch_size = len(self.input_batch.req_ids)
                return torch.empty((batch_size, 0), dtype=torch.int64, device=self.device)

            target_hidden_states = self.drafter.prepare_hidden_states(
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=sampled_token_ids,
                spec_decode_metadata=spec_decode_metadata,
            )

            draft = self.drafter.propose(
                context_token_ids=next_token_ids,
                previous_hidden_states=target_hidden_states,
                num_predict_tokens=self.drafter.model.n_predict,
            )
            # Pad to full num_spec_tokens width so the tensor matches
            # the shape expected by _copy_draft_token_ids_to_cpu and
            # _prepare_input_ids, and so suffix merge has room to write
            # longer sequences.
            if draft.shape[1] < self.num_spec_tokens:
                draft = torch.nn.functional.pad(
                    draft, (0, self.num_spec_tokens - draft.shape[1]),
                    value=0,
                )
            return draft

        if isinstance(sampled_token_ids, torch.Tensor):
            vocab_size = self.model_config.get_vocab_size()
            sampled_token_ids_list = [
                [t for t in seq if t != -1 and t < vocab_size]
                for seq in sampled_token_ids.tolist()
            ]
            sampled_token_ids_tensor = sampled_token_ids
        else:
            sampled_token_ids_list = sampled_token_ids
            sampled_token_ids_tensor = None

        arctic_spec_token_ids = None
        suffix_spec_token_ids = None

        if self.speculative_config.method in ("arctic", "mlp_speculator"):
            if sampled_token_ids_tensor is None:
                import numpy as np
                sampled_token_ids_tensor = torch.tensor(sampled_token_ids_list, device=self.device)

            previous_hidden_states = self.drafter.prepare_hidden_states(
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=sampled_token_ids_tensor,
                spec_decode_metadata=spec_decode_metadata,
            )

            next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                sampled_token_ids_list,
                self.requests,
                self.input_batch,
                scheduler_output.num_scheduled_tokens,
            )
            
            arctic_output_tensor = self.drafter.propose(
                context_token_ids=next_token_ids,
                previous_hidden_states=previous_hidden_states,
                num_predict_tokens=self.drafter.model.n_predict,
            )
            
            arctic_spec_token_ids = arctic_output_tensor.tolist()

        if self._suffix_cache is not None:
            self._update_suffix_cache(sampled_token_ids_list)
            results = self.propose_suffix_draft_token_ids(sampled_token_ids_list)
            
            suffix_spec_token_ids = []
            min_score = 0 if self.speculative_config.method == "suffix" \
                else self.drafter.model.n_predict
                
            for result in results:
                if result.score >= min_score:
                    suffix_spec_token_ids.append(result.token_ids)
                else:
                    suffix_spec_token_ids.append([])

        spec_token_ids = None
        if suffix_spec_token_ids is not None and arctic_spec_token_ids is not None:
            spec_token_ids = [
                s_tokens if s_tokens else a_tokens
                for s_tokens, a_tokens in zip(suffix_spec_token_ids, arctic_spec_token_ids)
            ]
        elif suffix_spec_token_ids is not None:
            spec_token_ids = suffix_spec_token_ids
        elif arctic_spec_token_ids is not None:
            spec_token_ids = arctic_spec_token_ids
        else:
            spec_token_ids = self._orig_propose_draft_token_ids(
                scheduler_output,
                sampled_token_ids_list,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                common_attn_metadata,
            )

        if spec_token_ids is None:
             spec_token_ids = [[] for _ in range(len(self.input_batch.req_ids))]

        return spec_token_ids

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
                self._suffix_response_tokens[req_id] = []

            self._suffix_cache.add_active_response(req_id, sampled_ids)
            self._suffix_response_tokens[req_id].extend(sampled_ids)

        for req_id in list(self._suffix_cache.active_requests):
            if req_id not in seen_req_ids:
                self._suffix_cache.stop_request(req_id)
                self._suffix_response_tokens.pop(req_id, None)

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
            index = self.input_batch.req_id_to_index[req_id]

            # In async mode, token_ids_cpu contains -1 placeholders at
            # decoded positions (written by _bookkeeping_sync).  Build the
            # pattern from the clean _suffix_response_tokens instead.
            if (self.use_async_scheduling
                    and req_id in self._suffix_response_tokens):
                response = self._suffix_response_tokens[req_id]
                num_prompt = int(
                    self.input_batch.num_prompt_tokens[index])
                num_tokens = num_prompt + len(response)
                if num_tokens >= self.max_model_len:
                    results.append(SuffixDecodingDraft())
                    continue
                # Take up to suffix_cache_max_depth tokens from the tail.
                depth = config.suffix_cache_max_depth
                if len(response) >= depth:
                    pattern = response[-depth:]
                else:
                    need = depth - len(response)
                    prompt_start = max(0, num_prompt - need)
                    prompt_part = self.input_batch.token_ids_cpu[
                        index, prompt_start:num_prompt].tolist()
                    pattern = prompt_part + response
            else:
                num_tokens = self.input_batch.num_tokens_no_spec[i]
                if num_tokens >= self.max_model_len:
                    results.append(SuffixDecodingDraft())
                    continue
                start = max(0, num_tokens - config.suffix_cache_max_depth)
                pattern = self.input_batch.token_ids_cpu[
                    i, start:num_tokens].tolist()

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


    def _start_suffix_copy(
        self,
        sampled_token_ids: torch.Tensor,
    ) -> None:
        """Initiate an async D2H copy of sampled token IDs for suffix decoding.

        Copies ``sampled_token_ids`` to a pinned CPU buffer on a dedicated
        CUDA stream (``suffix_copy_stream``) that only waits for prior work
        on the default stream (i.e. sampling).

        **This MUST be called BEFORE launching Arctic GPU work on the default
        stream** so that the copy is not ordered behind Arctic kernels.  The
        resulting timeline is::

            Default stream:  [sample] -> [arctic prepare / propose  ...]
            Suffix stream :  [sample] -> [D2H copy] -> [event]
            CPU            :                            wait -> suffix logic

        The companion ``_finish_suffix_copy`` synchronises on the copy event
        and returns the materialised CPU list.
        """
        n_rows = sampled_token_ids.shape[0]
        n_cols = sampled_token_ids.shape[-1]
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.suffix_copy_stream):
            self.suffix_copy_stream.wait_stream(default_stream)
            self.suffix_sampled_ids_pinned[:n_rows, :n_cols].copy_(
                sampled_token_ids, non_blocking=True,
            )
            self.suffix_copy_done_event.record()
        self._suffix_copy_shape = (n_rows, n_cols)

    def _finish_suffix_copy(self) -> list[list[int]]:
        """Wait for the suffix copy and return sampled token IDs as CPU lists.

        Synchronises on the copy event recorded by ``_start_suffix_copy``,
        then parses the pinned buffer into ``list[list[int]]``, applying
        rejection-sampling for spec-decode batches and masking discarded
        (still-in-prefill) requests.
        """
        self.suffix_copy_done_event.synchronize()
        n_rows, n_cols = self._suffix_copy_shape
        pinned = self.suffix_sampled_ids_pinned[:n_rows, :n_cols]

        num_reqs = self.input_batch.num_reqs
        discard_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]

        if n_cols == 1:
            # No spec decode tokens -- simple tolist on pinned buffer.
            result = pinned.tolist()
            for i in discard_indices:
                result[int(i)].clear()
        else:
            # Spec decode tokens present -- rejection-sample on CPU.
            # parse_output does .cpu().numpy(); on an already-CPU tensor the
            # .cpu() is a no-op, so this efficiently reuses the pinned buffer.
            result, _ = RejectionSampler.parse_output(
                pinned,
                self.input_batch.vocab_size,
                discard_indices,
            )

        return result

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> Union[ModelRunnerOutput, AsyncGPUModelRunnerOutput, IntermediateTensors]:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # noqa

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(
            sampled_token_ids: torch.Tensor | list[np.ndarray],
        ) -> None:
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )
            # When drafting is disabled (e.g. disable_by_batch_size), the
            # result is a 0-column tensor.  Don't copy it to CPU or the
            # shape will mismatch draft_token_ids_cpu; instead signal
            # "no draft" so the next step runs without speculation.
            if (isinstance(self._draft_token_ids, torch.Tensor)
                    and self._draft_token_ids.shape[1] == 0):
                self._draft_token_ids = None
            else:
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        # --- Draft proposal orchestration ---
        #
        # There are three drafting modes depending on async scheduling and
        # which drafters are active:
        #
        # A) Arctic-only + async:
        #      Pre-bookkeeping.  Arctic runs entirely on GPU tensors
        #      (the existing EAGLE-like async path).
        #
        # B) Suffix (with or without arctic) + async:
        #      Pre-bookkeeping.  Uses a *dedicated* suffix_copy_stream so
        #      that the D2H transfer of sampled token IDs only waits on
        #      "sampling complete" -- NOT on Arctic GPU work.  Timeline:
        #
        #        Default stream:  [sample] -> [arctic GPU work ----------->]
        #        Suffix  stream:  [sample] -> [D2H copy] -> [event]
        #        CPU            :                           wait -> suffix
        #
        #      At merge time, if any requests need arctic fallback, we
        #      sync on the arctic result (.tolist()); by then the GPU
        #      kernels have had the full suffix-CPU window to finish.
        #
        # C) Non-async (any combination):
        #      Post-bookkeeping.  propose_draft_token_ids handles everything
        #      using CPU valid_sampled_token_ids from bookkeeping.

        has_suffix = self._suffix_cache is not None
        is_arctic_method = (
            self.speculative_config is not None
            and self.speculative_config.method in ("arctic", "mlp_speculator")
        )
        input_fits_in_drafter = spec_decode_common_attn_metadata is not None
        sampled_token_ids = sampler_output.sampled_token_ids

        # Determine if we should use async pre-bookkeeping drafting.
        use_async_spec = (
            self.use_async_scheduling
            and self.speculative_config is not None
            and (is_arctic_method or has_suffix)
            and input_fits_in_drafter
        )

        if use_async_spec:
            if is_arctic_method and not has_suffix:
                # (A) Arctic-only async: use existing closure which calls
                #     propose_draft_token_ids (the method) with a GPU tensor
                #     and triggers the async GPU path internally.
                propose_draft_token_ids(sampled_token_ids)
            elif is_arctic_method:
                # (B) Arctic + suffix async.
                #     D2H copy of sampled tokens runs on a dedicated
                #     suffix_copy_stream that only waits for sampling,
                #     NOT for Arctic GPU kernels enqueued afterwards.
                #     Suffix CPU work then overlaps with Arctic GPU.

                # Step 1: Initiate async D2H copy before Arctic GPU work.
                self._start_suffix_copy(sampled_token_ids)

                # Step 2: Launch arctic on the default stream.
                with record_function_or_nullcontext(
                    "gpu_model_runner: draft (arctic)"
                ):
                    arctic_draft = self.propose_draft_token_ids(
                        scheduler_output,
                        sampled_token_ids,
                        self.input_batch.sampling_metadata,
                        hidden_states,
                        sample_hidden_states,
                        aux_hidden_states,
                        spec_decode_metadata,
                        spec_decode_common_attn_metadata,
                    )

                # Step 3: Wait for D2H copy, then run suffix on CPU.
                #         This overlaps with remaining Arctic GPU work.
                with record_function_or_nullcontext(
                    "gpu_model_runner: draft (suffix)"
                ):
                    sampled_cpu = self._finish_suffix_copy()
                    self._update_suffix_cache(sampled_cpu)
                    suffix_results = self.propose_suffix_draft_token_ids(
                        sampled_cpu
                    )

                    min_score = self.drafter.model.n_predict
                    suffix_draft = [
                        result.token_ids if result.score >= min_score
                        else []
                        for result in suffix_results
                    ]

                # Step 4: Merge arctic + suffix results.
                #         Suffix takes priority when available.
                #
                #         The merged result MUST be a torch.Tensor on GPU
                #         because upstream _prepare_input_ids (next iteration)
                #         asserts isinstance(self._draft_token_ids, torch.Tensor)
                #         and uses it for on-device scatter.

                # Detect whether Arctic drafting was disabled (0-column
                # tensor from disable_by_batch_size).
                arctic_disabled = (
                    isinstance(arctic_draft, torch.Tensor)
                    and arctic_draft.shape[1] == 0
                )

                # Collect suffix rows that have results.
                suffix_indices = [
                    i for i, s in enumerate(suffix_draft) if s
                ]

                if arctic_disabled:
                    # disable_by_batch_size triggered -- skip ALL
                    # speculation (including suffix) to avoid OOM
                    # under high batch pressure.  Suffix CPU work
                    # already ran but we discard its results.
                    self._draft_token_ids = None
                else:
                    if isinstance(arctic_draft, torch.Tensor):
                        merged = arctic_draft
                    else:
                        # Shouldn't happen in async path, but handle
                        # gracefully.
                        k = self.num_spec_tokens
                        n = len(arctic_draft)
                        pin = self._suffix_merge_pinned[:n, :k]
                        pin_np = pin.numpy()
                        pin_np[:] = 0
                        for i, row in enumerate(arctic_draft):
                            t = row[:k]
                            if t:
                                pin_np[i, :len(t)] = t
                        merged = pin.to(
                            device=self.device, non_blocking=True)

                    # When Arctic drafting is disabled (0-column), pad
                    # to full num_spec_tokens width so that suffix merge
                    # has room to write and _copy_draft_token_ids_to_cpu
                    # gets the shape it expects.
                    if merged.shape[1] < self.num_spec_tokens:
                        merged = torch.nn.functional.pad(
                            merged,
                            (0, self.num_spec_tokens - merged.shape[1]),
                            value=0,
                        )

                    # Batch-overwrite rows that have suffix results.
                    # CRITICAL: use the pre-allocated *pinned* merge
                    # buffer so that cudaMemcpyAsync does NOT synchronise
                    # the default stream.  With pageable (non-pinned)
                    # memory CUDA must sync the stream first, blocking the
                    # CPU until Arctic GPU work finishes -- destroying
                    # the async overlap.
                    width = merged.shape[1]
                    if suffix_indices:
                        n_sfx = len(suffix_indices)
                        overlay_pin = \
                            self._suffix_merge_pinned[:n_sfx, :width]
                        overlay_np = overlay_pin.numpy()
                        overlay_np[:] = 0
                        for j, idx in enumerate(suffix_indices):
                            s = suffix_draft[idx]
                            slen = min(len(s), width)
                            overlay_np[j, :slen] = s[:slen]
                        # Pinned H2C -- truly non-blocking.
                        overlay_t = overlay_pin.to(
                            device=self.device, non_blocking=True)
                        idx_pin = torch.tensor(
                            suffix_indices, dtype=torch.long
                        ).pin_memory()
                        idx_t = idx_pin.to(
                            device=self.device, non_blocking=True)
                        merged.index_copy_(0, idx_t, overlay_t)

                    self._draft_token_ids = merged
                    self._copy_draft_token_ids_to_cpu(scheduler_output)
            else:
                # (B2) Suffix-only async.
                #      No Arctic GPU work to overlap with, so skip the
                #      copy stream (its overhead exceeds the marginal
                #      overlap with the lightweight rejection kernel).
                #      Instead: rejection sample → parse tokens directly
                #      → suffix CPU work → build GPU tensor via pinned buf.
                self._suffix_only_rejection_sample(
                    sampled_token_ids,
                    spec_decode_common_attn_metadata,
                )

                # Parse sampled tokens to CPU lists.  .cpu() syncs the
                # default stream (waits for the rejection kernel, which
                # is lightweight), then parse_output performs CPU-side
                # rejection to extract accepted token IDs.
                with record_function_or_nullcontext(
                    "gpu_model_runner: draft (suffix)"
                ):
                    num_reqs = self.input_batch.num_reqs
                    discard_indices = np.nonzero(
                        self.discard_request_mask.np[:num_reqs]
                    )[0]
                    n_cols = sampled_token_ids.shape[-1]
                    if n_cols == 1:
                        sampled_cpu = sampled_token_ids.tolist()
                        for idx in discard_indices:
                            sampled_cpu[int(idx)].clear()
                    else:
                        sampled_cpu, _ = RejectionSampler.parse_output(
                            sampled_token_ids.cpu(),
                            self.input_batch.vocab_size,
                            discard_indices,
                        )

                    self._update_suffix_cache(sampled_cpu)
                    suffix_results = self.propose_suffix_draft_token_ids(
                        sampled_cpu
                    )
                    suffix_draft = [
                        result.token_ids if result.score >= 0
                        else []
                        for result in suffix_results
                    ]

                # Build GPU tensor from suffix lists via pinned buffer.
                k = self.num_spec_tokens
                n = len(suffix_draft)
                pin = self._suffix_merge_pinned[:n, :k]
                pin_np = pin.numpy()
                pin_np[:] = 0
                for i, s in enumerate(suffix_draft):
                    if s:
                        slen = min(len(s), k)
                        pin_np[i, :slen] = s[:slen]
                self._draft_token_ids = pin.to(
                    device=self.device, non_blocking=True)
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        # --- Bookkeeping ---
        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        # (C) Non-async drafting: run after bookkeeping.
        # Pass the raw GPU sampled_token_ids tensor (not the cleaned
        # valid_sampled_token_ids list) so that propose_draft_token_ids
        # gets a properly shaped 2-D tensor with -1 markers for rejected
        # tokens.  The method already handles tensors correctly: it
        # filters out -1 to build the CPU list for
        # prepare_next_token_ids_cpu and keeps the raw tensor for
        # prepare_hidden_states.
        if (
            self.speculative_config is not None
            and not use_async_spec
            and input_fits_in_drafter
        ):
            propose_draft_token_ids(sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.model_config.enable_return_routed_experts:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")

            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )

        if not self.use_async_scheduling:
            return output

        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output


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


    from vllm.forward_context import BatchDescriptor
    def _case_bs(self, case) -> int:
        # vLLM can pass ints, tuples, or sometimes BatchDescriptor-like objects
        if isinstance(case, int):
            return case
        if isinstance(case, BatchDescriptor):
            return int(case.num_tokens)
        if isinstance(case, tuple):
            return int(case[0])
        # last resort
        return int(getattr(case, "num_tokens"))

    def _with_bs(self, case, new_bs: int):
        if isinstance(case, tuple):
            return (new_bs, *case[1:])
        if isinstance(case, BatchDescriptor):
            # Best-effort reconstruction; adjust if your BatchDescriptor signature differs.
            return BatchDescriptor(
                num_tokens=new_bs,
                num_reqs=case.num_reqs,
                uniform=case.uniform,
                has_lora=case.has_lora,
            )
        return new_bs


    def _capture_cudagraphs(
        self,
        compilation_cases: list[tuple[int, bool]],
        cudagraph_runtime_mode: CUDAGraphMode,
        uniform_decode: bool,
    ):
        """
        Capture CUDA graphs for both base (SP) and shift (TP) variants, splitting
        shapes by threshold so both models have required graphs captured.
        """
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE and \
            cudagraph_runtime_mode in [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]

        sp_size = parallel_state._SP.world_size
        tp_size = parallel_state._TP.world_size

        # if torch.distributed.get_rank() == 0:
        #     print(f"compilation_cases: {compilation_cases}")

        # Base model (SP): scale only the batch-size field.
        compilation_cases_base = []
        for case in compilation_cases:
            bs = self._case_bs(case)
            scaled_bs = min(bs * sp_size, self.max_num_tokens)
            if scaled_bs > int(getattr(self, "shift_parallel_threshold", 0)):
                compilation_cases_base.append(self._with_bs(case, scaled_bs))

        compilation_cases_base = compilation_cases

        if is_global_first_rank():
            logger.info(
                "base model (SP=%s, TP=%s) cudagraph mode %s shapes %s",
                sp_size, tp_size, cudagraph_runtime_mode, [ self._case_bs(c) for c in compilation_cases_base ]
            )
            print(f"compilation_cases_base: {compilation_cases_base}")

        if compilation_cases_base:
            self._orig_capture_cudagraphs(
                compilation_cases_base, cudagraph_runtime_mode, uniform_decode
            )

        if is_global_first_rank():
            print(f"compilation_cases_base: {compilation_cases_base}")

        # Shift model (SP*TP but configured as TP-only variant in your routing):
        if getattr(self, "shift_model", None) is not None:
            compilation_cases_shift = []
            for case in compilation_cases:
                bs = self._case_bs(case)
                if (bs <= int(getattr(self, "shift_parallel_threshold", 0))
                        or "SwiftKV" in self.model.__class__.__name__):
                    compilation_cases_shift.append(case)

            if is_global_first_rank():
                logger.info(
                    "shift model (SPxTP=%s) shapes %s",
                    sp_size * tp_size, [ self._case_bs(c) for c in compilation_cases_shift ]
                )

            if compilation_cases_shift:
                orig_model, self.model = self.model, self.shift_model
                try:
                    with set_shift_parallel_mode(True):
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
                from vllm.model_executor.layers.attention import Attention
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