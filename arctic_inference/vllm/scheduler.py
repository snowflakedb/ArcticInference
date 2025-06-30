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

from vllm.v1.core.sched.scheduler import Scheduler

from arctic_inference.patching import ArcticPatch

class SchedulerPatch(ArcticPatch[Scheduler]):

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: list[Request] = []
        outputs: list[EngineCoreOutput] = []
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    mm_positions = request.mm_positions[input_id]
                    start_pos = mm_positions.offset
                    num_tokens = mm_positions.length
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None

            # Append generated tokens and check for stop. Note that if
            # a request is still being prefilled, we expect the model runner
            # to return empty token ids for the request.
            for num_new, output_token_id in enumerate(new_token_ids, 1):
                request.append_output_token_ids(output_token_id)

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = check_stop(request, self.max_model_len)
                if stopped:
                    kv_transfer_params = self._free_request(request)
                    del new_token_ids[num_new:]  # Trim new tokens if needed.
                    break

            # Extract sample logprobs if needed.
            if request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                # if self.structured_output_manager.should_advance(request):
                #     metadata = request.structured_output_request
                #     # Needs to happen after new_token_ids are accepted.
                #     request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                #         spec_token_ids[req_index])
                if request.use_structured_output:
                    request.spec_token_ids = []
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    ))

            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

            if not stopped:
                new_running.append(request)

        # P/D: update state for finished KV Transfers.
        self._update_from_kv_xfer_finished(model_runner_output)

        # Return the cached request data to the queue so they can be reused.
        for req_data in scheduler_output.scheduled_cached_reqs:
            # NOTE(rob): since we free stopped reqs above, adding stopped reqs
            # to _cached_reqs_data will cause a memory leak.
            if req_data.req_id not in self.finished_req_ids:
                self._cached_reqs_data[req_data.req_id].append(req_data)

        self.running = new_running
        engine_core_outputs = EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(spec_decoding_stats),
        )
        if self.include_finished_set:
            #TODO currently sending duplicates here, improve this
            engine_core_outputs.finished_requests = (
                scheduler_output.finished_req_ids | self.finished_req_ids)

        return engine_core_outputs