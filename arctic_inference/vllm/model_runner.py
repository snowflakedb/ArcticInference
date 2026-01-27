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
    _orig_build_attention_metadata = GPUModelRunner._build_attention_metadata
    _orig_execute_model = GPUModelRunner.execute_model
    _orig_pad_for_sequence_parallelism = GPUModelRunner._pad_for_sequence_parallelism

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
    _orig_pad_for_sequence_parallelism = GPUModelRunner._pad_for_sequence_parallelism

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

            if self.use_async_scheduling:
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


    def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
        use_shift_model = (
            getattr(self, "use_ulysses", False)
            and getattr(self, "shift_model", None) is not None
            and num_scheduled_tokens <= int(getattr(self, "shift_parallel_threshold", 0))
        )

        if getattr(self, "use_ulysses", False) and not use_shift_model:
            sp_size = self.parallel_config.ulysses_sequence_parallel_size
            num_input_tokens = round_up(num_scheduled_tokens, sp_size)

            if (
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                and hasattr(self, "cudagraph_batch_sizes")
                and self.cudagraph_batch_sizes
                and (num_input_tokens // sp_size) <= self.cudagraph_batch_sizes[-1]
            ):
                num_input_tokens = (
                    self.vllm_config.pad_for_cudagraph(num_input_tokens // sp_size) * sp_size
                )

            return num_input_tokens

        return self._orig_pad_for_sequence_parallelism(num_scheduled_tokens)


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
        sampled_token_ids: Union[list[list[int]], torch.Tensor],
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
            if isinstance(sampled_token_ids, torch.Tensor):
                return [[] for _ in range(sampled_token_ids.shape[0])]
            return [[] for _ in sampled_token_ids]
        
        spec_token_ids: Optional[list[list[int]]] = None
        
        used_async_path = (
            self.speculative_config.method in ("arctic", "mlp_speculator")
            and self._suffix_cache is None
            and isinstance(sampled_token_ids, torch.Tensor)
            and self.use_async_scheduling
            and common_attn_metadata is not None
        )

        if used_async_path:
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

            target_hidden_states = self.drafter.prepare_hidden_states(
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=sampled_token_ids,
                spec_decode_metadata=spec_decode_metadata,
            )

            spec_token_ids = self.propose_arctic_draft_from_next_tokens(
                next_token_ids=next_token_ids,
                previous_hidden_states=target_hidden_states,
            ) 

            return spec_token_ids       
        
        suffix_spec_token_ids = None
        
        if isinstance(sampled_token_ids, torch.Tensor):
            new_sampled_token_ids = sampled_token_ids.tolist()
        else:
            new_sampled_token_ids = sampled_token_ids.copy()

        if self._suffix_cache is not None:
            self._update_suffix_cache(new_sampled_token_ids)
            
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

        if self.speculative_config.method == "suffix":
            pass
        elif self.speculative_config.method in ("arctic", "mlp_speculator"):
            assert isinstance(self.drafter, ArcticProposer)
            
            if isinstance(sampled_token_ids, torch.Tensor):
                sampled_token_ids_tensor = sampled_token_ids
            else:
                import numpy as np
                if isinstance(sampled_token_ids, np.ndarray):
                     sampled_token_ids_tensor = torch.from_numpy(sampled_token_ids).to(self.device)
                else:
                     sampled_token_ids_tensor = sampled_token_ids

            previous_hidden_states = self.drafter.prepare_hidden_states(
                sample_hidden_states=sample_hidden_states,
                sampled_token_ids=sampled_token_ids_tensor,
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

    def propose_arctic_draft_from_next_tokens(
        self,
        next_token_ids: torch.Tensor,                 # [B] device
        previous_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fast pre-bookkeeping Arctic path:
        - next_token_ids are computed on device (no CPU sync)
        - produce a torch tensor stacked on dim 1

        TODO: handle corner cases like max length reached
        """
        assert isinstance(self.drafter, ArcticProposer)

        max_spec_tokens = self.speculative_config.num_speculative_tokens

        drafter_output = self.drafter.propose(
            context_token_ids=next_token_ids,                 
            previous_hidden_states=previous_hidden_states,    
            num_predict_tokens=max_spec_tokens,
        )

        # [batch_size, num_speculative_tokens]
        return drafter_output

    def propose_arctic_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        previous_hidden_states: Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        last_tokens: list[Optional[int]] = []
        max_spec_tokens = self.speculative_config.num_speculative_tokens
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)

            if num_sampled_ids == 0:
                if self.speculative_config.enable_suffix_decoding:
                    # Avoid aliased inner lists
                    return [[] for _ in range(len(sampled_token_ids))]
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
        final_draft_token_ids: list[list[int]] = []
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
            cudagraph_stats,  # Fix: Unpack cudagraph_stats
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
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        use_async_sched_for_arctic = (
            self.use_async_scheduling
            and self.speculative_config
            and self.speculative_config.method in ("arctic", "mlp_speculator")
        )

        # For arctic spec decoding, it can treat arbitrary input lengths
        input_fits_in_drafter = spec_decode_common_attn_metadata and True
        sampled_token_ids = sampler_output.sampled_token_ids
        if use_async_sched_for_arctic:
            if input_fits_in_drafter:
                propose_draft_token_ids(sampled_token_ids)

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

        if (
            self.speculative_config
            and not use_async_sched_for_arctic
            and input_fits_in_drafter
        ):
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,  # Fix: Pass cudagraph_stats to output
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


    def _capture_cudagraphs(self, compilation_cases,
                            cudagraph_runtime_mode: CUDAGraphMode,
                            uniform_decode: bool):
        """
        Capture CUDA graphs for both base (SP) and shift (TP) variants, splitting
        shapes by threshold so both models have required graphs captured.
        """
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE and \
            cudagraph_runtime_mode in [CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]

        sp_size = parallel_state._SP.world_size
        tp_size = parallel_state._TP.world_size

        # Base model (SP): scale only the batch-size field.
        compilation_cases_base = []
        for case in compilation_cases:
            bs = self._case_bs(case)
            scaled_bs = min(bs * sp_size, self.max_num_tokens)
            if scaled_bs > int(getattr(self, "shift_parallel_threshold", 0)):
                compilation_cases_base.append(self._with_bs(case, scaled_bs))

        if is_global_first_rank():
            logger.info(
                "base model (SP=%s, TP=%s) shapes %s",
                sp_size, tp_size, [ self._case_bs(c) for c in compilation_cases_base ]
            )

        if compilation_cases_base:
            self._orig_capture_cudagraphs(
                compilation_cases_base, cudagraph_runtime_mode, uniform_decode
            )

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