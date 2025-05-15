import re
import time
import argparse
import json
from typing import List, Optional

import uvicorn
import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from arctic_inference.dynasor.cot import effort_level
from arctic_inference.dynasor.cot import openai_chat_completion_stream
from arctic_inference.dynasor.util import with_cancellation

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger

logger = init_logger()

DEFAULT_PROBE = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"

class DynasorOpenAIClient:
    # The Dynasor OpenAI Client is a wrapper that applys the chat template
    # and the reasoning stuff to the OpenAI API of vLLM.
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        probe: str = DEFAULT_PROBE,
        token_interval: int = 32,
        certainty_window: int = 2,
    ):
        """
        Initialize the OpenAI Chat Client.

        Args:
            api_key: OpenAI API key.
            model: The model name to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).
            token_interval: Number of tokens before probing (default: 32)
            certainty_window: Number of consistent answers needed (default: 2)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.probe = probe
        self.token_interval = token_interval
        self.certainty_window = certainty_window

        # models = self.client.models.list()
        
        # FIXME: Need a reliable way to get the max tokens (without using transformers)
        self.max_tokens = 131072        
        

app = FastAPI()
client: Optional[DynasorOpenAIClient] = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for request validation
class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


@app.get("/v1/models")
async def models():
    logger.debug("Reaching models models fetching")
    global client
    models = client.client.models.list()
    return models

# TODO(GindaChen): Adapt to different models
def format_history(messages: List[ChatMessage]) -> str:
    """
    Convert chat conversation history into a prompt string.
    """
    formatted = ""
    for message in messages:
        role = message.role
        content = message.content
        if role == "system":
            formatted += "" + content + "\n"
        elif role == "user":
            formatted += "<｜User｜>" + content + "\n"
        else:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            formatted += "<｜Assistant｜>" + content + "\n"
    result = formatted + "<｜Assistant｜>"
    return result


@app.post("/v1/completions")
@with_cancellation
async def completions(request: CompletionRequest, raw_request: Request):
    """Handle chat completion requests."""
    global client
    openai_client = client.client
    prompt = request.prompt
    stream = request.stream

    # Check for extra body parameters
    token_interval = client.token_interval
    certainty_window = client.certainty_window
    
    if hasattr(request, "extra_body") and request.extra_body:
        if "dynasor" in request.extra_body:
            dynasor_config = request.extra_body["dynasor"]
            if "token_interval" in dynasor_config:
                token_interval = dynasor_config["token_interval"]
            if "certainty_window" in dynasor_config:
                certainty_window = dynasor_config["certainty_window"]

    max_tokens = request.max_tokens
    if max_tokens is None:
        max_tokens = client.max_tokens

    generator = openai_chat_completion_stream(
        client=openai_client,
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        max_tokens=max_tokens,
        dynasor_saving_effort=(certainty_window, token_interval),
        probeing_suffix=client.probe,
        # token_interval=token_interval,
        # certainty_window=certainty_window,
    )
    if not stream:
        result = ""
        for i in generator:
            result += i
        return JSONResponse(content={"choices": [{"message": {"content": result}}]})

    

    async def stream_response():
        request_id = f"chatcmpl-{int(time.time())}"
        created_time = int(time.time())

        # Send the role first
        first_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        logger.debug("yielding first chunk")
        yield f"data: {json.dumps(first_chunk)}\n\n"

        # Stream the content
        for content in generator:
            logger.debug("yielding content", content)
            # print("yielding content", content)
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send the final [DONE] message
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# TODO: asyncio cancellation is not working properly.
@app.post("/v1/chat/completions")
@with_cancellation
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests."""
    global client
    openai_client = client.client
    messages = request.messages
    prompt = format_history(messages)
    stream = request.stream

    # Check for extra body parameters
    token_interval = client.token_interval
    certainty_window = client.certainty_window
    
    if hasattr(request, "extra_body") and request.extra_body:
        if "dynasor" in request.extra_body:
            dynasor_config = request.extra_body["dynasor"]
            if "token_interval" in dynasor_config:
                token_interval = dynasor_config["token_interval"]
            if "certainty_window" in dynasor_config:
                certainty_window = dynasor_config["certainty_window"]

    max_tokens = request.max_tokens
    if max_tokens is None:
        max_tokens = client.max_tokens

    generator = openai_chat_completion_stream(
        client=openai_client,
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        max_tokens=max_tokens,
        dynasor_saving_effort=(certainty_window, token_interval),
        probeing_suffix=client.probe,
        # token_interval=token_interval,
        # certainty_window=certainty_window,
    )

    if not stream:
        result = ""
        for i in generator:
            result += i
            logger.debug(result)
        return JSONResponse(content={"choices": [{"message": {"content": result}}]})



    async def stream_response():
        request_id = f"chatcmpl-{int(time.time())}"
        created_time = int(time.time())

        # Send the role first
        first_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": request.model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        # logger.debug("yielding first chunk")
        yield f"data: {json.dumps(first_chunk)}\n\n"

        # Stream the content
        for content in generator:
            # print("yielding content", content)
            logger.debug("yielding content", content)
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send the final [DONE] message
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


def main():
    parser = argparse.ArgumentParser(description="OpenAI Chat Client")
    parser.add_argument("--api-key", type=str, default="token-abc123", help="API key")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port (default: 8001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--probe",
        type=str,
        default=DEFAULT_PROBE,
        help=f"Probe (default: {repr(DEFAULT_PROBE)})",
    )
    # parser.add_argument(
    #     "--dynasor-saving-effort",
    #     type=str,
    #     default="2,32",
    #     help="Dynasor saving effort. It is a tuple of two integers. The first integer is the number of consistent answer to get before early exit, and the second integer is the number of tokens before probing. (default: 2,32)",
    # )
    parser.add_argument(
        "--token-interval",
        type=int,
        default=32,
        help="Token interval for adaptive compute (default: 32)",
    )
    parser.add_argument(
        "--certainty-window",
        type=int,
        default=2,
        help="Certainty window for adaptive compute (default: 2)",
    )
    args = parser.parse_args()
    global client

    # dynasor_saving_effort = tuple(map(int, args.dynasor_saving_effort.split(",")))
    # assert len(dynasor_saving_effort) == 2
    # assert dynasor_saving_effort[0] >= 0
    # assert dynasor_saving_effort[1] >= 0
    client = DynasorOpenAIClient(
        base_url=args.base_url,
        api_key=args.api_key,
        probe=args.probe,
        token_interval=args.token_interval,
        certainty_window=args.certainty_window,
    )
    print(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()