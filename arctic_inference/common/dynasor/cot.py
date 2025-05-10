from typing import Optional

# TODO: Properly import with package prefix
from arctic_inference.common.dynasor.evaluator import count_not_empty, eqaul_group

uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]


def effort_level(effort_level: str) -> int:
    if effort_level == "mild":
        return (8, 64)
    elif effort_level == "low":
        return (5, 64)
    elif effort_level == "mid":
        return (3, 64)
    elif effort_level == "high":
        return (2, 64)
    elif effort_level == "crazy":
        return (2, 32)
    else:
        raise ValueError(f"Invalid effort level: {effort_level}")


def obtain_answer(s):
    # Find first unpaired } by counting { and }
    stack = []
    for i, c in enumerate(s):
        if c == "{":
            stack.append(c)
        elif c == "}":
            if not stack:  # No matching { found
                return s[:i]
            stack.pop()
    return ""


def openai_chat_completion_stream(
    client,
    model,
    prompt,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 2048,
    dynasor_saving_effort: tuple = None,
    probeing_suffix: str = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{",
):
    print("dynasor_saving_effort:", dynasor_saving_effort)

    assert max_tokens is not None, "max_tokens must be provided"
    
    if dynasor_saving_effort is not None:
        threshold, chunk_size = dynasor_saving_effort
        accumulated_response = ""
        adaptive_end = False
        append_answer = False
        current_prompt = prompt
        probe_answers = []
        probe_responses = []

        for _ in range(0, max_tokens, chunk_size):

            result = ""
            buffer = ""
            api_response = client.completions.create(
                model=model,
                prompt=current_prompt,
                temperature=temperature,
                max_tokens=chunk_size,
                stream=True,
            )
            if not adaptive_end:
                probe = client.completions.create(
                    model=model,
                    temperature=0.6,
                    prompt=current_prompt + probeing_suffix,
                    stream=True,
                    max_tokens=20,
                    top_p=0.95,
                )

            for chunk in api_response:
                if (
                    hasattr(chunk.choices[0], "text")
                    and chunk.choices[0].text is not None
                ):
                    content = chunk.choices[0].text
                    buffer += content
                    if " " in buffer or "\n" in buffer:
                        # if console:
                        #    console.print(buffer, end='')
                        yield buffer
                        accumulated_response += buffer
                        result += buffer
                        buffer = ""
            if buffer:
                yield buffer
                accumulated_response += buffer
                # console.print(buffer, end='')
                result += buffer

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
                and eqaul_group(probe_answers[-threshold:])
                and count_not_empty(probe_answers[-threshold:]) == threshold
                and sum(probe_certain_count) == threshold
            ):
                adaptive_end = True
                aanswer = probe_answers[-1]

            if adaptive_end and not append_answer:
                # print('Adaptive Ending')
                append_answer = True
                if "</think>" in accumulated_response:
                    yield "\n\n... Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{" + probe_answers[
                        -1
                    ] + "}\n\\]"
                else:
                    yield "\n\n...</think>\n Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{" + probe_answers[
                        -1
                    ] + "}\n\\]"
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