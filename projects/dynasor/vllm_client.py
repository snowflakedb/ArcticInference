"""dyansor-vLLM client API example"""

import openai
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint
import time
import argparse

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description='vLLM client for model inference')
    parser.add_argument('--model', type=str, 
                      default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help='Model name to use for inference')
    parser.add_argument('--api-base', type=str,
                      default="http://localhost:8000/v1",
                      help='API base URL')
    parser.add_argument('--api-key', type=str,
                      default="dr32r34tnjnfkd",
                      help='API key for authentication')
    parser.add_argument('--temperature', type=float,
                      default=0.7,
                      help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int,
                      default=1024,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--top-p', type=float,
                      default=0.95,
                      help='Top p sampling parameter')
    parser.add_argument('--mode', type=str,
                      choices=['completions', 'chat'],
                      default='completions',
                      help='Inference mode: completions or chat')
    parser.add_argument('--prompt', type=str,
                      default="Solve x^2 + 1 = 0",
                      help='Input prompt for the model')
    parser.add_argument('--disable-adaptive', action='store_true',
                      help='Disable adaptive compute (sets token_interval to 100000)')
    parser.add_argument('--token-interval', type=int,
                      default=32,
                      help='Token interval for adaptive compute')
    parser.add_argument('--certainty-window', type=int,
                      default=2,
                      help='Certainty window for adaptive compute')
    return parser.parse_args()


probe_text = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
probe_text_end = "} \\]"


def format_deepseek_prompt(user_message: str) -> str:
    """Format prompt with DeepSeek template"""
    return f"<｜begin▁of▁sentence｜><｜User｜>{user_message}<｜Assistant｜><think>\n"


def completions_example(client, model_name, prompt, temperature, max_tokens, top_p, disable_adaptive=False, token_interval=32, certainty_window=2):
    console.print(
        Panel(
            f"[bold blue]Prompt:[/bold blue] {prompt}", 
            title="Input", 
            border_style="blue"
        )
    )
    formatted_prompt = format_deepseek_prompt(prompt)
    
    adaptive_compute=dict(
        mode="prompting",
        probe_text=probe_text,
        probe_text_end=probe_text_end,
        certainty_window=certainty_window,
        token_interval=100000 if disable_adaptive else token_interval,
    )

    response = client.completions.create(
        model=model_name,
        prompt=formatted_prompt,
        stream=True,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body=dict(
            adaptive_compute=adaptive_compute,
        ),
    )

    console.print("\n[bold green]Response:[/bold green]")
    response_text = ""
    token_count = 0 
    start_time = time.time()
    for chunk in response:
        # print(chunk)
        token = chunk.choices[0].text

        response_text += token
        print(token, end="", flush=True)
        token_count += 1
    print("\n")
    end_time = time.time()

    console.print(
        Panel(
            (
                f"Tokens: {token_count}\n"
                f"Time: {end_time - start_time:.2f} seconds\n"
                f"Throughput: {token_count / (end_time - start_time):.2f} tps"
            ),
            title="[bold green]Complete Response[/bold green]",
            border_style="green"
        )
    )


def chat_example(client, model_name, prompt, disable_adaptive=False, token_interval=32, certainty_window=2):
    console.print(Panel(f"[bold blue]Prompt:[/bold blue] {prompt}", title="Input", border_style="blue"))

    max_retries = 5
    backoff_factor = 2

    adaptive_compute=dict(
        mode="prompting",
        probe_text=probe_text,
        probe_text_end=probe_text_end,
        certainty_window=certainty_window,
        token_interval=100000 if disable_adaptive else token_interval,
    )
    

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:        
        for attempt in range(max_retries):
            try:
                if not disable_adaptive:
                    extra_body = dict(
                        adaptive_compute=adaptive_compute,
                    )
                else:
                    extra_body = None
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    extra_body=extra_body,
                )
                break
            except openai.InternalServerError as e:
                wait_time = backoff_factor ** attempt
                console.print(f"[yellow]Attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...[/yellow]")
                time.sleep(wait_time)
        else:
            console.print("[bold red]All retry attempts failed.[/bold red]")
            return

        console.print("\n[bold green]Response:[/bold green]")
        response_text = ""
        token_count = 0 
        start_time = time.time()
        for chunk in response:
            # print(chunk)
            if chunk.choices and hasattr(chunk.choices[0], 'delta'):
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    response_text += delta.content
                    print(delta.content, end="")
                    token_count += 1
        end_time = time.time()
        
        console.print("\n")
        console.print(
            Panel(
                f"Tokens: {token_count}\n"
                f"Time: {end_time - start_time:.2f} seconds\n"
                f"Throughput: {token_count / (end_time - start_time):.2f} tps"
            )
        )


def main():
    args = parse_args()
    
    client = openai.OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )
    
    console.print(Panel(
        f"[bold blue]Model:[/bold blue] {args.model}\n"
        f"[bold blue]Mode:[/bold blue] {args.mode}\n"
        f"[bold blue]Temperature:[/bold blue] {args.temperature}\n"
        f"[bold blue]Max Tokens:[/bold blue] {args.max_tokens}\n"
        f"[bold blue]Top P:[/bold blue] {args.top_p}\n"
        f"[bold blue]Adaptive Compute:[/bold blue] {'Disabled' if args.disable_adaptive else 'Enabled'}\n"
        f"[bold blue]Token Interval:[/bold blue] {100000 if args.disable_adaptive else args.token_interval}\n"
        f"[bold blue]Certainty Window:[/bold blue] {args.certainty_window}",
        title="[bold blue]Configuration[/bold blue]",
        border_style="blue"
    ))
    
    if args.mode == 'completions':
        completions_example(
            client=client,
            model_name=args.model,
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            disable_adaptive=args.disable_adaptive,
            token_interval=args.token_interval,
            certainty_window=args.certainty_window
        )
    else:
        chat_example(
            client=client,
            model_name=args.model,
            prompt=args.prompt,
            disable_adaptive=args.disable_adaptive,
            token_interval=args.token_interval,
            certainty_window=args.certainty_window
        )


if __name__ == "__main__":
    main()