import argparse
import re
import sys
from typing import List, Dict

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

# TODO: Properly import with package prefix
from arctic_inference.common.dynasor.cot import effort_level
from arctic_inference.common.dynasor.cot import openai_chat_completion_stream


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        base_url: str = "http://localhost:8000/v1",
    ):
        """
        Initialize the OpenAI Chat Client.

        Args:
            api_key: OpenAI API key.
            model: The model name to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).
        """
        # self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.console = Console()
        self.conversation_history: List[Dict] = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        self.model = self.ensure_model_initialized()

    def ensure_model_initialized(self):
        # query the client to check if the model is available
        try:
            models = self.client.models.list()
            model_ids = [model.id for model in models]
            if not model_ids:
                raise ValueError(
                    f"No models found in the server side: {self.base_url}"
                )
            if self.model is None:
                return model_ids[0]
            if self.model not in model_ids:
                raise ValueError(
                    f"Model {self.model} not found in the server side: {self.base_url}"
                )
        except Exception as e:
            print(f"[bold red]Error: {str(e)}[/bold red]")
            sys.exit(1)

    def format_history(self, messages: List[Dict]) -> str:
        """
        Convert chat conversation history into a prompt string.
        """
        formatted = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                formatted += "" + content + "\n"
            elif role == "user":
                formatted += "<｜User｜>" + content + "\n"
            else:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                formatted += "<｜Assistant｜>" + content + "\n"
        result = formatted + "<｜Assistant｜>"
        return result

    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def display_message(self, content: str, role: str = "assistant"):
        """Display a message with proper formatting."""
        if role == "assistant":
            self.console.print(Markdown(content))
        else:
            self.console.print(f"\n[bold blue]You:[/bold blue] {content}\n")

    def chat(self, dynasor_saving_effort: str = "mid"):
        """Main chat loop."""
        console_output = (
            "[bold green]"
            "Welcome to Dynasor Chat!\n"
            f"• Model: {self.model}\n"
            f"• Endpoint URL: {self.base_url}\n"
            f"• Dynasor Saving Effort: {dynasor_saving_effort}\n"
            "Type 'help' to show help, and 'exit' to end the conversation.\n"
            "Type anything to start chatting."
            "[/bold green]\n"
        )
        self.console.print(console_output)
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("> ")
                if user_input.lower() in ["exit", "quit"]:
                    self.console.print("[bold green]Goodbye![/bold green]")
                    break
                if user_input.lower() in ["help"]:
                    self.console.print("[bold green]Dynasor Chat[/bold green]")
                    self.console.print(
                        "[bold green]Type 'exit' to end the conversation.[/bold green]"
                    )
                    self.console.print(
                        "[bold green]Type 'help' to show this message.[/bold green]"
                    )
                    self.console.print(
                        "[bold green]Type 'dynasor-saving-effort' to show the Dynasor saving effort level.[/bold green]"
                    )
                    continue

                if user_input.lower().startswith("dynasor-saving-effort"):
                    # Parse the effort level from input
                    parts = user_input.lower().split()
                    new_effort = parts[1] if len(parts) > 1 else ""
                    if len(parts) == 1:
                        self.console.print(
                            "[bold red]No effort level provided. Must be one of: 'mild', 'low', 'mid', 'high', 'crazy'[/bold red]"
                        )
                        continue

                    # Validate the new effort level
                    if new_effort and new_effort not in [
                        "mild",
                        "low",
                        "mid",
                        "high",
                        "crazy",
                    ]:
                        self.console.print(
                            "[bold red]Invalid effort level. Must be one of: 'mild', 'low', 'mid', 'high', 'crazy'[/bold red]"
                        )
                    else:
                        # Update the effort level
                        dynasor_saving_effort = new_effort
                        status = "disabled" if not new_effort else new_effort
                        self.console.print(
                            f"[bold green]Dynasor saving effort set to: {status}[/bold green]"
                        )
                    continue

                # Display and store the user message
                self.display_message(user_input, "user")
                self.add_to_history("user", user_input)

                # Convert conversation history to a prompt and get streaming response using the OpenAI client
                prompt = self.format_history(self.conversation_history)

                if dynasor_saving_effort not in ["", "none"]:
                    full_response = ""
                    for chunk in openai_chat_completion_stream(
                        self.client,
                        self.model,
                        prompt,
                        dynasor_saving_effort=effort_level(dynasor_saving_effort),
                    ):
                        full_response += chunk
                        self.console.print(chunk, end="")
                else:
                    full_response = ""
                    for chunk in openai_chat_completion_stream(
                        self.client, self.model, prompt
                    ):
                        full_response += chunk
                        self.console.print(chunk, end="")

                self.console.print()  # New line after response
                self.add_to_history("assistant", full_response)

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Interrupted by user[/bold red]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                self.console.print(
                    "[yellow]Please check if your OpenAI API key is correct and the service is accessible.[/yellow]"
                )


def main():
    parser = argparse.ArgumentParser(description="Dynasor Chat Client")
    parser.add_argument("--api-key", type=str, default="token-abc123", help="API key")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: None)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--dynasor-saving-effort",
        type=str,
        default="mid",
        help="Specify the Dynasor saving effort level. If non-empty, must be one of: 'mild', 'low', 'mid', 'high', or 'crazy'. (Default: mid)",
    )
    args = parser.parse_args()

    dynasor_saving_effort_levels = ["none", "mild", "low", "mid", "high", "crazy", ""]
    if args.dynasor_saving_effort not in dynasor_saving_effort_levels:
        print(
            f"Error: Invalid Dynasor saving effort level. "
            f"Must be one of: {dynasor_saving_effort_levels}"
        )
        sys.exit(1)

    try:
        client = OpenAIChatClient(
            api_key=args.api_key, model=args.model, base_url=args.base_url
        )
        client.chat(dynasor_saving_effort=args.dynasor_saving_effort)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()