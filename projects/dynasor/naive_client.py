from typing import Optional
import asyncio
from openai import AsyncOpenAI 

def log(message: str):
    print(f"\033[93m{message}\033[0m", flush=True)


sys = f"You are a helpful assistant."

# probe_prompt = "</think>\nOh I suddently got an answer: \\box{"
# probe_prompt = "Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
probe_prompt = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"

def format_prompt(prompt: str, generated: str) -> str:
    text = f"<｜begin▁of▁sentence｜>{sys}<｜User｜>{prompt}<｜Assistant｜><think>\n{generated} {probe_prompt}"
    return text

async def execute_single_probe(
    client: AsyncOpenAI,
    model_id: str, 
    prompt: str, 
    generated: str,
    probe_in_progress_event: asyncio.Event,
    max_tokens: int = 32,
):
    
    try:
        text = format_prompt(prompt, generated)
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


async def main():
    client = AsyncOpenAI(  # Changed to AsyncOpenAI
        api_key="EMPTY",  # Using default empty key for local server
        base_url="http://localhost:8080/v1", 
        max_retries=1
    )

    models = await client.models.list()
    model_id = models.data[0].id

    # prompt = "Explain the theory of relativity in simple terms, covering special and general relativity, key postulates, and some of its observational evidence. Be thorough."
    prompt = "Solve the equation: x^2 + 1 = 0"

    user_messages = [
        {"role": "system", "content": sys},
        # {"role": "user", "content": "What is 2+2?"} # Shorter query for testing without probes
        {"role": "user", "content": prompt} # Longer query
    ]
    _response_stream = client.chat.completions.create(
        messages=user_messages,
        model=model_id,
        max_tokens=1024, # Reduced for testing
        stream=True,
        extra_body=dict(
            dynasor=dict(
                probe_interval=32,
            )
        )
    )
    response_stream = await _response_stream
    

    probe_task: Optional[asyncio.Task] = None
    probe_in_progress_event = asyncio.Event()
    probe_in_progress_event.clear()

    states = []
    should_launch_next_probe = False
    generated_text = ""
    chunks_processed = 0

    async for chunk in response_stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
            text = chunk.choices[0].delta.content
            generated_text += text
            print(text, end="", flush=True)
            chunks_processed += 1
        
        if chunks_processed > 0 and chunks_processed % 32 == 0:
            should_launch_next_probe = True
            pass

        if probe_task is not None and probe_task.done():
            states.append(probe_task.result())
            # TODO: Here add certaindex invoke
            probe_task = None

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

    pass


if __name__ == "__main__":
    asyncio.run(main())