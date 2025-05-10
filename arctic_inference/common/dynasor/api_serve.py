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
"""
Adopted from vLLM openai server
"""

import uvloop
import asyncio
import importlib
from typing import Dict, Optional, Set, Tuple, Union, Annotated, Union, AsyncGenerator, Any
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.serving_completion import CompletionStreamResponse
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.entrypoints.openai.api_server import with_cancellation
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, ChatCompletionRequest, ChatCompletionResponse,
    ErrorResponse
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.utils import (cli_env_setup, load_aware_call,
                                    with_cancellation)
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION
from vllm.logger import init_logger
from vllm.entrypoints.chat_utils import (load_chat_template,
                                         resolve_hf_chat_template,
                                         resolve_mistral_chat_template)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.version import __version__ as VLLM_VERSION


from arctic_inference.common.dynasor.serving_chat import AdaptiveOpenAIServingChat
from arctic_inference.common.dynasor.serving_completion import AdaptiveOpenAIServingCompletion
from .entropy import (
    obtain_answer,
    should_early_exit,
    uncertain_words,
    is_certain_answer,
)

#overriding the original init_app_state class from vllm's api_server
_orig_init_app_state = api_server.init_app_state
logger = init_logger("vllm.entrypoints.openai.api_server")

def adaptive_chat(request: Request) -> Optional[AdaptiveOpenAIServingChat]:
    return request.app.state.adaptive_openai_serving_chat


def completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.openai_serving_completion


def adaptive_compute_completion(request: Request) -> Optional[OpenAIServingCompletion]:
    return request.app.state.adaptive_compute_openai_serving_completion


async def init_app_state_override(engine_client, model_config, state, args):
    await _orig_init_app_state(engine_client, model_config, state, args)

    # overriding just the adaptive_compute handlers
    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    resolved_chat_template = load_chat_template(args.chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = await engine_client.get_tokenizer()

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(
                chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer,
                chat_template=None,
                tools=None,
                trust_remote_code=model_config.trust_remote_code)

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: %s\n"
                    "It is different from official chat template '%s'. "
                    "This discrepancy may lead to performance degradation.",
                    resolved_chat_template, args.model)

    state.adaptive_compute_openai_serving_completion = AdaptiveOpenAIServingCompletion(
        engine_client=engine_client,
        model_config=model_config,
        models=state.openai_serving_models,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    ) if model_config.runner_type == "generate" else None

    state.adaptive_openai_serving_chat = AdaptiveOpenAIServingChat(
        engine_client,
        model_config,
        state.openai_serving_models,
        args.response_role,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
        enable_reasoning=args.enable_reasoning,
        reasoning_parser=args.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
    ) if model_config.runner_type == "generate" else None


api_server.init_app_state = init_app_state_override

done_struct = "data: [DONE]\n\n"


async def _handle_adaptive_compute(request: CompletionRequest, raw_request: Request) -> AsyncGenerator[Tuple[CompletionStreamResponse, Dict[str, Any]], None]:
    """
    Handle adaptive compute for `/v1/completions` requests.
    """
    handler = adaptive_compute_completion(raw_request)
    assert handler is not None

    original_request = request.model_copy()
    adaptive_compute = request.adaptive_compute

    # Setup basic configurations (will be updated)
    remaining_tokens = request.max_tokens
    sending_prompt = request.prompt
    output_var_template: Optional[CompletionStreamResponse] = None
    answers = []
    is_certains = []

    # Setup basic constants
    probe_text = adaptive_compute.get("probe_text", None)
    probe_text_end = adaptive_compute.get("probe_text_end", None)
    certainty_window = adaptive_compute.get("certainty_window", 2)
    token_interval = adaptive_compute.get("token_interval", 32)
    token_count = 0
    assert probe_text is not None

    # Coordination state - for stop condition
    certainty_event = asyncio.Event()
    final_answer_box = {"answer": None}

    #probing function
    async def launch_probe(prompt_snapshot: str) -> None:
        """
        Launch a probe task asynchronously.
        """
        probe_request = original_request.model_copy()
        # TODO(GindaChen): inefficient string manipulation
        probe_prompt = prompt_snapshot + probe_text
        # TODO(GindaChen): Make these configurable
        probe_request.prompt = probe_prompt
        probe_request.stream = True
        probe_request.max_tokens = 20
        probe_request.temperature = 0.6
        probe_request.top_p = 0.95
        probe_request.adaptive_compute = None

        try:
            probe_generator = await handler.create_completion(probe_request, raw_request)
            probe_output_text = ""
            async for output in probe_generator:
                token = output.choices[0].text
                probe_output_text += token
            answer = obtain_answer(probe_output_text)
            is_certain = is_certain_answer(probe_output_text, uncertain_words)
            answers.append(answer)
            is_certains.append(is_certain)
            if should_early_exit(answers, probe_output_text, uncertain_words, certainty_window, is_certains):
                final_answer_box["answer"] = answer
                certainty_event.set()
                # logger.debug("\033[91mdetected early exit\033[0m")
        except Exception as e:
            logger.warning(f"Probe task failed: {e}")
        return

    # Main decoding loop
    while remaining_tokens > 0:
        decoding_request = original_request.model_copy()
        decoding_request.prompt = sending_prompt
        decoding_request.stream = True
        decoding_request.max_tokens = min(token_interval, remaining_tokens)
        decoding_request.adaptive_compute = None

        generator = await handler.create_completion(decoding_request, raw_request)
        if isinstance(generator, ErrorResponse):
            # error_response = generator.model_dump()
            # yield f"data: {error_response}\n\n"
            yield generator, dict()
            # yield "data: [DONE]\n\n"
            yield done_struct, dict()
            return

        # TODO(GindaChen): inefficient string buffering
        token_buffer = ""

        # Send the actual query
        async for output in generator:
            if certainty_event.is_set():
                break  # Stop if model was confident

            token = output.choices[0].text
            token_buffer += token
            remaining_tokens -= 1
            token_count += 1

            if output_var_template is None:
                output_var_template = output
            
            # response_json = output.model_dump_json(exclude_unset=False, exclude_none=True)
            # yield f"data: {response_json}\n\n"
            yield output, dict(exclude_unset=False, exclude_none=True)

            if token_count % token_interval == 0:
                # Launch probe task asynchronously
                asyncio.create_task(launch_probe(sending_prompt + token_buffer))

            if remaining_tokens <= 0:
                break

        sending_prompt += token_buffer

        if certainty_event.is_set(): #early exit condition is met
            answer = final_answer_box["answer"]
            logger.debug(f"Early exit on answer: {answer = }")

            response_obj = output_var_template.model_copy()
            response_obj.choices[0].text = probe_text + answer + probe_text_end
            response_obj.choices[0].finish_reason = "stop"
            response_obj.choices[0].stop_reason = "stop"

            # response_json = response_obj.model_dump_json(
            #     exclude_unset=False, exclude_none=True
            # )
            # yield f"data: {response_json}\n\n"
            yield response_obj, dict(exclude_unset=False, exclude_none=True)
            # yield "data: [DONE]\n\n"
            yield done_struct, dict()
            return
        
        if remaining_tokens <= 0:
            logger.debug(f"Early exit: Remaining tokens <= 0")
            # yield f"data: [DONE]\n\n"
            yield done_struct, dict()
            break

    # yield "data: [DONE]\n\n"
    yield done_struct, dict()


async def handle_adaptive_compute(request: CompletionRequest, raw_request: Request):
    generator = _handle_adaptive_compute(request, raw_request)
    async for response_obj, dump_config in generator:
        # logger.debug(response_obj, dump_config)
        if response_obj == done_struct:
            yield done_struct
            continue
        chunk = response_obj.model_dump_json(**dump_config)
        # logger.debug(chunk)
        yield f"data: {chunk}\n\n"
    return

from vllm.entrypoints.openai.protocol import ChatCompletionStreamResponse, DeltaMessage, ChatCompletionResponseStreamChoice

async def adapt_completion_to_chat(generator: AsyncGenerator[Tuple[CompletionStreamResponse, Dict[str, Any]], None]):
    async for response_obj, dump_config in generator:
        # logger.debug("in completion adaptor:", response_obj, dump_config)
        if response_obj == done_struct:
            yield done_struct
            continue
        if isinstance(response_obj, ErrorResponse):
            yield response_obj.model_dump_json(**dump_config)
            continue
        
        assert isinstance(response_obj, CompletionStreamResponse)
        
        chat_response_obj = ChatCompletionStreamResponse(
            id=response_obj.id,
            created=response_obj.created,
            model=response_obj.model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        # TODO(GindaChen): assigned by `OpenAIServingChat.chat_completion_stream_generator.get_chat_request_role(request)`
                        role="assistant",
                        content=choice.text
                    ),
                    logprobs=choice.logprobs,
                    finish_reason=choice.finish_reason,
                    stop_reason=choice.stop_reason
                )
                for choice in response_obj.choices
            ],
            usage=response_obj.usage
        )
        chunk = chat_response_obj.model_dump_json(**dump_config)
        # logger.debug(chunk)
        yield f"data: {chunk}\n\n"

@with_cancellation
async def adaptive_create_completion(request: CompletionRequest, raw_request: Request):
    handler = completion(raw_request)

    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )

    json_obj = await raw_request.json()
    adaptive_compute = json_obj.get("adaptive_compute", None)
    print(f"adaptive_compute: {adaptive_compute}")

    if adaptive_compute is not None:
        generator = handle_adaptive_compute(request, raw_request)
        return StreamingResponse(content=generator, media_type="text/event-stream")

    # logger.debug("Handling normal completion")
    generator = await handler.create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@with_cancellation
async def adaptive_create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    handler = adaptive_chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )

    json_obj = await raw_request.json()
    adaptive_compute = json_obj.get("adaptive_compute", None)
    print(f"adaptive_compute: {adaptive_compute}")

    if adaptive_compute is not None:
        newprompt = await handler.get_completion_prompt(request, raw_request)
        request = CompletionRequest(prompt=newprompt,**json_obj)
        generator = _handle_adaptive_compute(request, raw_request)
        generator = adapt_completion_to_chat(generator)
        return StreamingResponse(content=generator, media_type="text/event-stream")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


# Overriding the api_server routers for adaptive decoding
for path in ("/v1/completions", "/v1/chat/completions"):
    api_server.router.routes[:] = [
        r for r in api_server.router.routes
        if not (hasattr(r, "path") and r.path == path)
    ]
api_server.router.post("/v1/completions")(adaptive_create_completion)
api_server.router.post("/v1/chat/completions")(adaptive_create_chat_completion)

def main():
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(api_server.run_server(args))

if __name__ == "__main__":
    main()