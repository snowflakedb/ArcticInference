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
#from vllm.config import CUDAGraphMode, CompilationLevel, VllmConfig
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             is_global_first_rank)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.sequence import IntermediateTensors
#from vllm.utils import round_up, cdiv
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
from vllm.config import (
    VllmConfig,
)

# Try to import the grammar bitmask helper; degrade gracefully if unavailable.
apply_grammar_bitmask: Any
try:
    from vllm.v1.structured.output import apply_grammar_bitmask  # type: ignore
except Exception:
    try:
        from vllm.v1.structured.grammar import apply_grammar_bitmask  # type: ignore
    except Exception:
        apply_grammar_bitmask = None  # type: ignore

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

from arctic_inference.patching import ArcticPatch
from arctic_inference.suffix_decoding import (SuffixDecodingCache,
                                              SuffixDecodingDraft)
from arctic_inference.vllm.spec_dec.arctic_proposer import (ArcticProposer,
                                                            SuffixProposer)


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

        # Intercept Arctic/suffix/mlp_speculator configs so we can manage drafter lifecycle.
        if (vllm_config.speculative_config is not None and
                vllm_config.speculative_config.method in ("arctic", "suffix", "mlp_speculator")):
            arctic_speculative_config = vllm_config.speculative_config
            vllm_config.speculative_config = None
        else:
            arctic_speculative_config = None

        self._orig_init(vllm_config, device)

        self._suffix_cache: Optional[SuffixDecodingCache] = None
        if arctic_speculative_config is not None:
            self.vllm_config.speculative_config = arctic_speculative_config
            self.speculative_config = arctic_speculative_config

            # re-derive async-spec state because base __init__
            # was run with speculative_config=None.
            # 1) num_spec_tokens / uniform decode length
            self.num_spec_tokens = getattr(self.speculative_config,
                                           "num_speculative_tokens", 0)
            self.uniform_decode_query_len = 1 + self.num_spec_tokens
            # 2) padded-spec async copy infra for the fast path
            if self.use_async_scheduling and self.num_spec_tokens:
                if self.valid_sampled_token_count_event is None:
                    self.valid_sampled_token_count_event = torch.cuda.Event()
                if self.valid_sampled_token_count_copy_stream is None:
                    self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()

            if get_pp_group().is_last_rank:
                if self.speculative_config.method in ("arctic", "mlp_speculator"):
                    self.drafter = ArcticProposer(self.vllm_config)
                elif self.speculative_config.method != "suffix":
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")
                else:
                    self.drafter = SuffixProposer()
                self.rejection_sampler = RejectionSampler(self.sampler)

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
            return [[] for _ in sampled_token_ids]
        
        spec_token_ids: Optional[list[list[int]]] = None
        
        used_async_path = (
            self.speculative_config.method in ("arctic", "mlp_speculator")
            and self._suffix_cache is None
            and isinstance(sampled_token_ids, torch.Tensor)
            and self.use_async_scheduling
            and common_attn_metadata is not None
        )

        print("177: sampled_token_ids:", sampled_token_ids)

        if used_async_path:
            # --- Arctic GPU fast path (pre-bookkeeping) ---
            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_indices.gpu,
                    self.num_discarded_requests,
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

        if self.speculative_config.method == "suffix":
            pass
        elif self.speculative_config.method in ("arctic", "mlp_speculator"):
            assert isinstance(self.drafter, ArcticProposer)
            if isinstance(sampled_token_ids, np.ndarray):
                sampled_token_ids_tensor = torch.from_numpy(
                    sampled_token_ids
                ).to(self.device)
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

        print("next_token_ids:", next_token_ids)
        print("previous_hidden_states:", previous_hidden_states.shape)
        print("max_spec_tokens:", max_spec_tokens)
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
        results: list[SuffixDecodingDraft] = []
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

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> Union[ModelRunnerOutput, AsyncGPUModelRunnerOutput, IntermediateTensors]:
        print("GPUModelRunnerPatch: sample_tokens called")
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

        print("475 sampler_output.sampled_token_ids:", sampler_output.sampled_token_ids)

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
                print("proposed draft token ids:", self._draft_token_ids)

        use_async_sched_for_arctic = (
            self.use_async_scheduling
            and self.speculative_config
            and self.speculative_config.method in ("arctic", "mlp_speculator")
        )
        print("499 use_async_sched_for_arctic:", use_async_sched_for_arctic)

        # For arctic spec decoding, it can treat aribtrary input lengths
        input_fits_in_drafter = spec_decode_common_attn_metadata and True
        sampled_token_ids = sampler_output.sampled_token_ids
        print("504: sampled_token_ids:", sampled_token_ids)
        if input_fits_in_drafter:
            print("505 input fits in drafter, proposing draft token ids")
            propose_draft_token_ids(sampled_token_ids)

        print("508: sampled_token_ids:", sampled_token_ids)
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
