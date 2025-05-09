# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import AsyncGenerator, AsyncIterator, Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple, Union, cast

from fastapi import Request
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              ErrorResponse,
                                              RequestResponseMetadata,
                                              UsageInfo)
# yapf: enable
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import merge_async_iterators
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
logger = init_logger(__name__)


class AdaptiveOpenAIServingCompletion(OpenAIServingCompletion):

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        result_generator: AsyncIterator[Tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_text_lens = [0] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts
        num_prompt_tokens = [0] * num_prompts

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage
            include_continuous_usage = (
                include_usage and stream_options.continuous_usage_stats
            )
        else:
            include_usage, include_continuous_usage = False, False

        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                prompt_logprobs = res.prompt_logprobs
                prompt_text = res.prompt

                # Prompt details are excluded from later streamed outputs
                if res.prompt_token_ids is not None:
                    num_prompt_tokens[prompt_idx] = len(res.prompt_token_ids)

                delta_token_ids: GenericSequence[int]
                out_logprobs: Optional[GenericSequence[Optional[Dict[int, Logprob]]]]

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices

                    assert request.max_tokens is not None
                    if request.echo and not has_echoed[i]:
                        assert prompt_token_ids is not None
                        assert prompt_text is not None
                        if request.max_tokens == 0:
                            # only return the prompt
                            delta_text = prompt_text
                            delta_token_ids = prompt_token_ids
                            out_logprobs = prompt_logprobs
                        else:
                            assert prompt_logprobs is not None
                            # echo the prompt and first token
                            delta_text = prompt_text + output.text
                            delta_token_ids = [*prompt_token_ids, *output.token_ids]
                            out_logprobs = [
                                *prompt_logprobs,
                                *(output.logprobs or []),
                            ]
                        has_echoed[i] = True
                    else:
                        # return just the delta
                        delta_text = output.text
                        delta_token_ids = output.token_ids
                        out_logprobs = output.logprobs

                        if (
                            not delta_text
                            and not delta_token_ids
                            and not previous_num_tokens[i]
                        ):
                            # Chunked prefill case, don't return empty chunks
                            continue

                    if request.logprobs is not None:
                        assert out_logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=previous_text_lens[i],
                        )
                    else:
                        logprobs = None

                    previous_text_lens[i] += len(output.text)
                    previous_num_tokens[i] += len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                            )
                        ],
                    )
                    if include_continuous_usage:
                        prompt_tokens = num_prompt_tokens[prompt_idx]
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    yield chunk
                    # response_json = chunk.model_dump_json(exclude_unset=False)
                    # yield f"data: {response_json}\n\n"

            total_prompt_tokens = sum(num_prompt_tokens)
            total_completion_tokens = sum(previous_num_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            )

            if include_usage:
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage_info,
                )
                # final_usage_data = (final_usage_chunk.model_dump_json(
                #     exclude_unset=False, exclude_none=True))
                # yield f"data: {final_usage_data}\n\n"
                yield final_usage_chunk

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = final_usage_info

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            import json

            yield json.loads(data)
            # yield f"data: {data}\n\n"
        # yield "data: [DONE]\n\n"