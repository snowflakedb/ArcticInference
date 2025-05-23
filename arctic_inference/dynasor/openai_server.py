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
OpenAI-compatible proxy server that injects Dynasor probes.

Known issues:
- Prompt formatting is currently hardcoded - limitation to get system prompt to format a proper prefix. Potential accuracy degrade (unknown prompt) and/or performance issue (kv reuse)
- API key is currently used as "EMPTY" placeholder.
"""

import argparse
import asyncio
import httpx
import json
import logging
import os
import time
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from typing import Optional, Dict, Any, List, Union, AsyncGenerator

from arctic_inference.dynasor.cot import (
    obtain_answer, formalize_final_response,
    uncertain_words, default_probing_suffix, format_prompt_for_completions,
)
from arctic_inference.dynasor.evaluator import count_not_empty, equal_group


def init_logger() -> logging.Logger:
    """Initialize and configure the logger for the OpenAI proxy server.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = init_logger()


@dataclass
class ProxyConfig:
    """Configuration for the OpenAI proxy server.
    
    Attributes:
        host: Host address to bind the server to
        port: Port number to run the server on
        target_base_url: Base URL of the target OpenAI API server
        api_key: API key for authentication (defaults to "EMPTY")
    """
    host: str
    port: int
    target_base_url: str
    api_key: str = "EMPTY"


def make_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the server.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="OpenAI API Proxy Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8001)",
    )
    parser.add_argument(
        "--target-base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the target OpenAI API server (default: http://localhost:8000)",
    )
    return parser


def parse_args(args_: Optional[List[str]] = None) -> ProxyConfig:
    """Parse command line arguments and create a ProxyConfig instance.
    
    Args:
        args_: Optional list of command line arguments
        
    Returns:
        ProxyConfig: Configuration object with parsed arguments
    """
    parser = make_parser()
    args = parser.parse_args(args=args_)
    return ProxyConfig(
        host=args.host,
        port=args.port,
        target_base_url=args.target_base_url,
    )


app = FastAPI()

# Initialize with None, will be set during startup
config: Optional[ProxyConfig] = None


def set_config(c: ProxyConfig) -> None:
    """Set the global configuration for the server.
    
    Args:
        c: ProxyConfig instance to set as global config
    """
    global config
    config = c


async def execute_single_probe(
    client: AsyncOpenAI,
    model_id: str,
    prompt: str,
    generated: str,
    probe_in_progress_event: asyncio.Event,
    max_tokens: int = 32,
) -> str:
    """Execute a single probe request to check model's certainty.
    
    Args:
        client: AsyncOpenAI client instance
        model_id: ID of the model to use
        prompt: Original prompt text
        generated: Generated text so far
        probe_in_progress_event: Event to track probe status
        max_tokens: Maximum tokens for probe response
        
    Returns:
        str: Probe response text
    """
    try:
        # TODO(GindaChen)(Refactor): Prompt formatting is currently highly hardcoded. 
        # Main issue is that we have to control the `</think>` token, and 
        # except for the `/v1/completions` endpoint, we don't have a 
        # proper way to control.
        # Case: 
        # - If the template is unknown, then we can only submit something reasonoable.
        # - If the template is know-able, then the proxy server has to know about it. 
        #   Either the server provide an endpoint, or the user override this function.
        text = format_prompt_for_completions(prompt, generated)
        probe_response = await client.completions.create(
            model=model_id,
            prompt=text,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
        )

        if probe_response.choices and probe_response.choices[0].text:
            response_text_probe = probe_response.choices[0].text
        else:
            response_text_probe = ""
    finally:
        probe_in_progress_event.clear()
    return response_text_probe


async def handle_chat_completion_request(
    request: Request,
    path: str
) -> AsyncGenerator[bytes, None]:
    """Handle chat completion requests with Dynasor probing.
    Keep the states of the probed results, 
    and stream back the decoding results to the user.
    
    Args:
        request: FastAPI request object
        path: API endpoint path
        
    Yields:
        bytes: Chunks of the streaming response
        
    Raises:
        HTTPException: If the endpoint is not found
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header.split(" ")[1]
    else:
        api_key = config.api_key  # Fallback to default

    body = await request.body()
    body_json = json.loads(body) if body else {}

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=f"{config.target_base_url}/v1",
        max_retries=1
    )

    logger.debug("Handle chat completion request: %s", body_json)

    model_id = body_json.get("model")
    max_tokens = body_json.get("max_tokens", 1024)

    # By default disable dynasor, unless client specifies it.
    dynasor_body = body_json.get("dynasor", {})
    probe_interval = dynasor_body.get("probe_interval", 1e9)
    certainty_window = dynasor_body.get("certainty_window", 3)

    if path == "/v1/chat/completions":
        messages = body_json.get("messages")
        prompt = messages[-1].get("content")

        _response_stream = client.chat.completions.create(
            messages=messages,
            model=model_id,
            max_tokens=max_tokens,
            stream=True,
        )
        response_stream = await _response_stream

    elif path == "/v1/completions":
        prompt = body_json.get("prompt")
        _response_stream = client.completions.create(
            model=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
        )
        response_stream = await _response_stream
    else:
        raise HTTPException(status_code=404)

    probe_task: Optional[asyncio.Task] = None
    probe_in_progress_event = asyncio.Event()
    probe_in_progress_event.clear()

    probe_answers: List[str] = []
    probe_responses: List[str] = []
    adaptive_end = False

    should_launch_next_probe = False
    generated_text = ""
    chunks_processed = 0

    async for chunk in response_stream:
        _chunk = chunk.to_json(indent=None, )
        reconstructed_chunk = f"data: {_chunk}\n\n"
        yield reconstructed_chunk.encode("utf-8")

        # TODO: Properly set the exit condition.
        if (
            chunk.choices[0].finish_reason is not None
            and chunk.choices[0].finish_reason != "length"
        ):
            break

        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
            text = chunk.choices[0].delta.content
            generated_text += text
            chunks_processed += 1

        if chunks_processed > 0 and chunks_processed % probe_interval == 0:
            should_launch_next_probe = True

        if probe_task is not None and probe_task.done():
            # Obtain the result from the probe task.
            probe_text = probe_task.result()
            answer = obtain_answer(probe_text)
            probe_task = None

            # Now check the certaindex for exiting condition.
            probe_answers.append(answer)
            probe_responses.append(probe_text)

            probe_certain_count = [
                not any(word in res.lower() for word in uncertain_words)
                for res in probe_responses[-certainty_window:]
            ]
            is_group_equal = equal_group(probe_answers[-certainty_window:])
            count_not_empty_count = count_not_empty(probe_answers[-certainty_window:])

            if (
                not adaptive_end
                and is_group_equal
                and count_not_empty_count == certainty_window
                and sum(probe_certain_count) == certainty_window
            ):
                adaptive_end = True

            if adaptive_end:
                should_launch_next_probe = False

                # TODO: Make the probe customizable
                output_text = formalize_final_response(generated_text, probe_answers[-1])

                # Make a new chunk with the output text.
                new_chunk = chunk.model_copy()
                new_chunk.choices[0].delta.content = output_text
                new_chunk_bytes = new_chunk.to_json(indent=None)
                reconstructed_chunk = f"data: {new_chunk_bytes}\n\n"
                yield reconstructed_chunk.encode("utf-8")

                new_chunk.choices[0].delta.content = ""
                new_chunk.choices[0].finish_reason = "stop"
                reconstructed_chunk = f"data: {new_chunk.to_json(indent=None)}\n\n"
                yield reconstructed_chunk.encode("utf-8")
                break

        if should_launch_next_probe:
            if not probe_in_progress_event.is_set():
                should_launch_next_probe = False
                probe_in_progress_event.set()
                probe_task = asyncio.create_task(
                    execute_single_probe(
                        client,
                        model_id,
                        prompt,
                        generated_text,
                        probe_in_progress_event,
                        max_tokens=32,
                    )
                )

    await response_stream.close()
    yield "data: [DONE]\n\n".encode("utf-8")


@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: Request) -> StreamingResponse:
    """Handle chat completions endpoint requests.
    
    Args:
        request: FastAPI request object
        
    Returns:
        StreamingResponse: Streaming response with chat completion results
    """
    gen = handle_chat_completion_request(request, "/v1/chat/completions")
    return StreamingResponse(
        gen, media_type="text/event-stream",
    )


@app.post("/v1/completions")
async def completions_endpoint(request: Request) -> StreamingResponse:
    """Handle completions endpoint requests.
    
    Args:
        request: FastAPI request object
        
    Returns:
        StreamingResponse: Streaming response with completion results
    """
    gen = handle_chat_completion_request(request, "/v1/completions")
    return StreamingResponse(
        gen, media_type="text/event-stream",
    )


async def proxy_request(request: Request, path: str) -> StreamingResponse:
    """Proxy requests to the target OpenAI API server.
    
    Args:
        request: FastAPI request object
        path: API endpoint path
        
    Returns:
        StreamingResponse: Streaming response from the target server
        
    Raises:
        HTTPException: If the endpoint is not found
    """
    # Skip chat/completions endpoints since they are handled separately
    if request.method == "POST" and request.url.path in ["/v1/chat/completions", "/v1/completions"]:
        raise HTTPException(status_code=404, detail="Not Found")

    # Get the raw request body
    body = await request.body()
    body_json = json.loads(body) if body else {}

    # Forward headers but exclude host
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    # Construct target URL
    target_url = config.target_base_url.rstrip('/') + '/' + path.lstrip('/')

    # Check if streaming is requested
    is_stream = body_json.get("stream", False)

    async with httpx.AsyncClient() as client:
        # Forward the request with same method, headers, and body
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )

        if is_stream:
            # For streaming responses, stream each chunk
            async def stream_generator():
                buffer = b""
                async for chunk in response.aiter_bytes():
                    yield chunk
                if buffer:  # Yield any remaining data
                    yield buffer

            proxy_headers = {
                k: v for k, v in response.headers.items()
                if k.lower() not in {"content-length", "transfer-encoding", "content-encoding"}
            }
            return StreamingResponse(
                stream_generator(),
                status_code=response.status_code,
                headers=proxy_headers,
                media_type="text/event-stream"
            )
        else:
            # For non-streaming responses, return the full response
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers)
            )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(request: Request, path: str) -> StreamingResponse:
    """Generic proxy endpoint for all other API routes.
    
    Args:
        request: FastAPI request object
        path: API endpoint path
        
    Returns:
        StreamingResponse: Streaming response from the target server
    """
    return await proxy_request(request, "/" + path)


def start_server(config: ProxyConfig) -> None:
    """Start the FastAPI server with the given configuration.
    
    Args:
        config: ProxyConfig instance with server settings
    """
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    args = parse_args()
    config = ProxyConfig(
        host=args.host,
        port=args.port,
        target_base_url=args.target_base_url,
    )
    logger.info(f"Starting server with config: {config}")
    start_server(config)
