"""
Dynasor naive client implementation for testing and development purposes.

This module provides a basic implementation of the Dynasor client that can be used
for testing and development. It demonstrates the core functionality of the Dynasor
system using async/await patterns.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROBE_PROMPT = (
    "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
)
DEFAULT_PROBE_INTERVAL = 32
DEFAULT_MAX_TOKENS = 1024

def format_prompt(prompt: str, generated: str) -> str:
    """
    Format the prompt for probing the model.
    
    Args:
        prompt: The original user prompt
        generated: The text generated so far
        
    Returns:
        Formatted prompt string
    """
    text = f"<｜begin▁of▁sentence｜>{DEFAULT_SYSTEM_PROMPT}<｜User｜>{prompt}<｜Assistant｜><think>\n{generated} {DEFAULT_PROBE_PROMPT}"
    return text


async def execute_single_probe(
    client: AsyncOpenAI,
    model_id: str,
    prompt: str,
    generated: str,
    probe_in_progress_event: asyncio.Event,
    max_tokens: int = 32,
) -> str:
    """
    Execute a single probe to check model's progress.
    
    Args:
        client: AsyncOpenAI client instance
        model_id: Model identifier
        prompt: Original user prompt
        generated: Text generated so far
        probe_in_progress_event: Event to track probe status
        max_tokens: Maximum tokens for probe response
        
    Returns:
        Probe response text
    """
    try:
        text = format_prompt(prompt, generated)
        probe_response = await client.completions.create(
            model=model_id,
            prompt=text,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
        )
        return probe_response.choices[0].text if probe_response.choices and probe_response.choices[0].text else ""
    except Exception as e:
        logger.error("Error during probe execution: %s", e)
        return ""
    finally:
        probe_in_progress_event.clear()

async def main() -> None:
    """Main execution function."""
    try:
        client = AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8080/v1",
            max_retries=1
        )

        # Get available models
        models = await client.models.list()
        model_id = models.data[0].id
        logger.info("Using model: %s", model_id)

        prompt = "Solve the equation: x^2 + 1 = 0"
        user_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # Initialize streaming response
        response_stream = await client.chat.completions.create(
            messages=user_messages,
            model=model_id,
            max_tokens=DEFAULT_MAX_TOKENS,
            stream=True,
            extra_body=dict(
                dynasor=dict(
                    probe_interval=DEFAULT_PROBE_INTERVAL,
                )
            )
        )

        # Initialize probe tracking
        probe_task: Optional[asyncio.Task] = None
        probe_in_progress_event = asyncio.Event()
        probe_in_progress_event.clear()

        states = []
        should_launch_next_probe = False
        generated_text = ""
        chunks_processed = 0

        # Process response stream
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                generated_text += text
                print(text, end="", flush=True)
                chunks_processed += 1

            if chunks_processed > 0 and chunks_processed % DEFAULT_PROBE_INTERVAL == 0:
                should_launch_next_probe = True

            if probe_task is not None and probe_task.done():
                states.append(probe_task.result())
                probe_task = None

            if should_launch_next_probe and not probe_in_progress_event.is_set():
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
        logger.info("Response stream completed successfully")

    except Exception as e:
        logger.error("Error in main execution: %s", e)
        raise

if __name__ == "__main__":
    asyncio.run(main())
