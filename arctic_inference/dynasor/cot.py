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

from typing import Optional

from arctic_inference.dynasor.evaluator import count_not_empty, equal_group

uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]
sys = f"You are a helpful assistant."
default_probing_suffix = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"


# TODO: Generalize this to other models.
# The problem is that only the model with known template can properly use this function.
def format_prompt_for_completions(prompt: str, generated: str) -> str:
    text = f"<｜begin▁of▁sentence｜>{sys}<｜User｜>{prompt}<｜Assistant｜><think>\n{generated} {default_probing_suffix}"
    return text


def formalize_final_response(generated_text: str, answer: str) -> str:
    if "</think>" in generated_text:
        output_text = "\n\n... Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{" + answer + "}\n\\]"
    else:
        output_text = "\n\n...</think>\n Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{" + answer + "}\n\\]"
    return output_text


def obtain_answer(s):
    """# Find first unpaired } by counting { and }"""
    stack = []
    for i, c in enumerate(s):
        if c == "{":
            stack.append(c)
        elif c == "}":
            if not stack:  # No matching { found
                return s[:i]
            stack.pop()
    return ""


# TODO: Test stopping condition
def openai_chat_completion_stream(
    client,
    model,
    prompt,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 2048,
    dynasor_saving_effort: tuple = None,
    probing_suffix: str = default_probing_suffix,
):
    assert max_tokens is not None, "max_tokens must be provided"

    if dynasor_saving_effort is not None:
        threshold, chunk_size = dynasor_saving_effort
        accumulated_response = ""
        adaptive_end = False
        append_answer = False
        current_prompt = prompt
        probe_answers = []
        probe_responses = []

        for iter_id in range(0, max_tokens, chunk_size):

            if not adaptive_end:
                probe = client.completions.create(
                    model=model,
                    temperature=0.6,
                    prompt=current_prompt + probing_suffix,
                    stream=True,
                    max_tokens=20,
                    top_p=0.95,
                )

            result = ""
            buffer = ""
            api_response = client.completions.create(
                model=model,
                prompt=current_prompt,
                temperature=temperature,
                max_tokens=chunk_size,
                stream=True,
            )

            for chunk in api_response:
                if (
                    hasattr(chunk.choices[0], "text")
                    and chunk.choices[0].text is not None
                ):
                    content = chunk.choices[0].text
                    yield content
                    accumulated_response += content
                    result += content

            current_prompt += (
                result  # Update the prompt with the new text for subsequent iterations
            )

            if (
                chunk.choices[0].finish_reason is not None
                and chunk.choices[0].finish_reason != "length"
            ):
                break

            if not result:
                break

            if not adaptive_end:
                probe_text = ""
                for probe_chunk in probe:
                    probe_text += probe_chunk.choices[0].text

                answer = obtain_answer(probe_text)
                probe_answers.append(answer)
                probe_responses.append(probe_text)

                probe_certain_count = [
                    not any(word in res.lower() for word in uncertain_words)
                    for res in probe_responses[-threshold:]
                ]

                # print("=" * 100)
                # print(probe_text, answer, certain_count)
                # print("=" * 100)

            if (
                not adaptive_end
                and equal_group(probe_answers[-threshold:])
                and count_not_empty(probe_answers[-threshold:]) == threshold
                and sum(probe_certain_count) == threshold
            ):
                adaptive_end = True

            if adaptive_end and not append_answer:
                # print('Adaptive Ending')
                append_answer = True
                # TODO: Make the probe customizable
                if "</think>" in accumulated_response:
                    yield (
                        "\n\n... Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
                        + probe_answers[-1]
                        + "}\n\\]"
                    )
                else:
                    yield (
                        "\n\n...</think>\n Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
                        + probe_answers[-1]
                        + "}\n\\]"
                    )
                break

    else:

        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_response = ""

        buffer = ""
        # Process the streaming response
        for chunk in response:
            if hasattr(chunk.choices[0], "text") and chunk.choices[0].text is not None:
                content = chunk.choices[0].text
                buffer += content
                # Print when we have a complete word/sentence
                if " " in buffer or "\n" in buffer:
                    yield buffer
                    full_response += buffer
                    buffer = ""
        if buffer:
            yield buffer
            full_response += buffer

        return full_response
