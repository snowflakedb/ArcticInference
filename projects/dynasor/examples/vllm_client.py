"""dyansor-vLLM client API example"""

import openai
from rich import print
import time

openai_api_key = "dr32r34tnjnfkd"
openai_api_base = "http://localhost:8000/v1"

#model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
client = openai.OpenAI(
    base_url=openai_api_base,
    api_key=openai_api_key,
)

# Query available models from the endpoint


#Print available models
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#print(f"Available models: {[model.id for model in models]}")
print(f"Using model: {model_name}")
print(f"-----")


probe_text = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
probe_text_end = "} \\]"


def format_deepseek_prompt(user_message: str) -> str:
    """Format prompt with DeepSeek template"""
    return f"<｜begin▁of▁sentence｜><｜User｜>{user_message}<｜Assistant｜><think>\n"


def completions_example():
    user_message = "Solve x^4 + x^3 = x^2 + x + 4"
    print(f"Prompt: {user_message}")
    prompt = format_deepseek_prompt(user_message)
    print(f"-----")
    print(f"Streaming response:\n")

    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        stream=True,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        extra_body=dict(
            # TODO: Properly form the names
            adaptive_compute=dict(
                mode="prompting",
                probe_text=probe_text,
                probe_text_end=probe_text_end,
                certainty_window=2,
                token_interval=32,
            )
        ),
    )

    for chunk in response:
        token = chunk.choices[0].text
        print(token, end="", flush=True)


def chat_example():
    user_message = "Solve x^4 + x^3 = x^2 + x + 4"
    print(f"Prompt: {user_message}")
    print(f"-----")
    print(f"Streaming response:\n")

    max_retries = 5
    backoff_factor = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Solve x^4 + x^3 = x^2 + x + 4"}],
                stream=True
            )
            break  # Exit loop if successful
        except openai.InternalServerError as e:
            wait_time = backoff_factor ** attempt
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    else:
        print("All retry attempts failed.")
    
    print(response)

    for chunk in response:
        if chunk.choices and hasattr(chunk.choices[0], 'delta'):
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end="", flush=True)


def main():
    completions_example()


if __name__ == "__main__":
    main()