import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
from openai import AsyncOpenAI
import uvicorn
from typing import Optional
import json
import os
from fastapi import HTTPException
import asyncio
from openai import AsyncOpenAI
from arctic_inference.dynasor.evaluator import count_not_empty, eqaul_group
from arctic_inference.dynasor.cot import (
    obtain_answer, formalize_final_response, 
    uncertain_words, default_probeing_suffix, format_prompt_for_completions,
)

app = FastAPI()

# Configure the target OpenAI base URL
TARGET_BASE_URL = "http://localhost:8000"



async def execute_single_probe(
    client: AsyncOpenAI,
    model_id: str, 
    prompt: str, 
    generated: str,
    probe_in_progress_event: asyncio.Event,
    max_tokens: int = 32,
):
    
    try:
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
) -> StreamingResponse:
    body = await request.body()
    body_json = json.loads(body) if body else {}

    client = AsyncOpenAI(  # Changed to AsyncOpenAI
        api_key="EMPTY",  # Using default empty key for local server
        # base_url="http://localhost:8080/v1", 
        base_url="http://localhost:8000/v1", 
        max_retries=1
    )

    print("Handle chat completion request: ", body_json)

    if path == "/v1/chat/completions":
        model_id = body_json.get("model")
        messages = body_json.get("messages")
        max_tokens = body_json.get("max_tokens", 1024)
        
        dynasor_body = body_json.get("dynasor", {})
        probe_interval = dynasor_body.get("probe_interval", 1e9)
        certainty_window = dynasor_body.get("certainty_window", 3)
        

        prompt = messages[-1].get("content")

        _response_stream = client.chat.completions.create(
            messages=messages,
            model=model_id,
            max_tokens=max_tokens,
            stream=True,
        )
        response_stream = await _response_stream
        

        probe_task: Optional[asyncio.Task] = None
        probe_in_progress_event = asyncio.Event()
        probe_in_progress_event.clear()
        
        probe_answers = []
        probe_responses = []
        adaptive_end = False

        should_launch_next_probe = False
        generated_text = ""
        chunks_processed = 0

        async for chunk in response_stream:
            _chunk = chunk.to_json(indent=None,)
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
                print(text, end="", flush=True)
                chunks_processed += 1
            
            if chunks_processed > 0 and chunks_processed % probe_interval == 0:
                should_launch_next_probe = True
                pass

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
                is_group_equal = eqaul_group(probe_answers[-certainty_window:])                
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
                    # new_chunk.choices[0].finish_reason = "stop"
                    new_chunk_bytes = new_chunk.to_json(indent=None)
                    reconstructed_chunk = f"data: {new_chunk_bytes}\n\n"
                    yield reconstructed_chunk.encode("utf-8")

                    new_chunk.choices[0].delta.content = ""
                    new_chunk.choices[0].finish_reason = "stop"
                    reconstructed_chunk = f"data: {new_chunk.to_json(indent=None)}\n\n"
                    yield reconstructed_chunk.encode("utf-8")
                    break
                pass


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

        pass
    elif path == "/v1/completions":
        raise NotImplementedError("Completions are not supported yet")
    else:
        raise HTTPException(status_code=404, detail="Not Found")
    


async def proxy_request(request: Request, path: str) -> StreamingResponse:
    if request.method == "POST" and request.url.path in ["/v1/chat/completions", "/v1/completions"]:
        gen = handle_chat_completion_request(
            request, 
            request.url.path,
        )
        return StreamingResponse(
            gen,
            media_type="text/event-stream",
        )
    
    # Get the raw request body
    body = await request.body()
    body_json = json.loads(body) if body else {}
    
    # Forward headers but exclude host
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    
    # Construct target URL
    target_url = f"{TARGET_BASE_URL}{path}"

    # Check if streaming is requested
    is_stream = body_json.get("stream", False)
    print(f"Streaming: {is_stream}")
    
    async with httpx.AsyncClient() as client:
        # Forward the request with same method, headers, and body
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            # timeout=60.0,
            # stream=is_stream
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
async def proxy(request: Request, path: str):
    return await proxy_request(request, "/" + path)

def start_server(host: str = "0.0.0.0", port: int = 8001):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server(
        host="0.0.0.0",
        port=8001,
    )
