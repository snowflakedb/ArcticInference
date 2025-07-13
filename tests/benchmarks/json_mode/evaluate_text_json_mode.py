import argparse
import asyncio
import datetime
import json
import os
import sys

from .task_description import TASK_DESCRIPTIONS, TASK_NAME_TO_TASK_SCHEMA
from .utils import call_vllm_complete, compute_sentence_similarity

import pydantic

DATASET_WIKIQUESTIONS = "datasets/WikiQuestions.json"


def load_dataset_wikiquestions(
    filepath: str = DATASET_WIKIQUESTIONS, ) -> list[dict[str, str]]:
    """Load the WikiQuestions dataset from filepath. Return a list of samples."""
    try:
        with open(filepath) as dataset_file:
            data = json.load(dataset_file)
        return data
    except json.JSONDecodeError as e:
        sys.exit(f"ERROR: Can't parse {filepath} ({e})")
    except FileNotFoundError as e:
        sys.exit(f"ERROR: Can't open {filepath} ({e})")


def generate_system_prompt(instructions: str, output_schema: str) -> str:
    """Return a system prompt with given instructions and expected output schema."""
    system_prompt = (
        "You are a helpful assistant. \n"
        f"Instructions: {instructions} \n"
        "Output the result as a JSON string with the "
        f"following format: {output_schema}. \n"
        "IMPORTANT!! Do not start the JSON with ```json or end it with ```.")
    return system_prompt


def generate_user_prompt(task_name: str, sample_data: dict[str, str]) -> str:
    """Generate a LLM query based on the information from the sample data."""
    if task_name == "ParaphraseQuestions":
        user_prompt = f"Question: {sample_data.get('question', 'empty')}"
    elif task_name == "RAGAS":
        user_prompt = (f"Context: {sample_data.get('context', 'empty')} \n"
                       f"Question: {sample_data.get('question', 'empty')} \n"
                       f"Answer: {sample_data.get('answer', 'empty')}")
    else:
        user_prompt = (f"Context: {sample_data.get('context', 'empty')} \n"
                       f"Question: {sample_data.get('question', 'empty')}")

    return user_prompt


def process_batch(
    task_name: str,
    sample_data: list[dict[str, str]],
    llm_name: str,
    options: dict[str, float | dict],
) -> list[dict | None]:
    """Generate the outputs for the given task on the batch of sample data."""
    task_instructions = TASK_DESCRIPTIONS[task_name]["task_instructions"]
    task_output_schema = TASK_DESCRIPTIONS[task_name]["response_format"]
    system_prompt = generate_system_prompt(instructions=task_instructions,
                                           output_schema=task_output_schema)

    # Create a batch of prompts (1 prompt per sample)
    prompts = []
    for sample in sample_data:
        user_prompt = generate_user_prompt(task_name=task_name,
                                           sample_data=sample)
        prompts.append([
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ])

    responses = asyncio.run(
        call_vllm_complete(prompts=prompts, llm_name=llm_name,
                           options=options))

    all_rows = responses
    results: list[None | dict] = []
    for row in all_rows:
        if row is None:
            results.append(None)
            continue

        llm_json_output = row.choices[0].message.content
        # The best way to verify if the LLM output respects the expected schema
        # is to try to create an instance of it.
        try:
            instance = TASK_NAME_TO_TASK_SCHEMA[task_name].model_validate_json(
                str(llm_json_output))
            results.append(instance.model_dump())
        except (pydantic.ValidationError, TypeError):
            results.append(None)

    return results


def compute_answer_score(
    task_name: str,
    sample: dict,
    generated_answer: dict | None,
) -> float:
    """Return the score of the generated answer for the given task."""

    if generated_answer is None:
        return 0.0

    # Read TASK_DESCRIPTION to understand why we evaluate the generated answers
    # that way.
    if task_name == "GenerateAnswer":
        if sample["answerable"]:
            return compute_sentence_similarity(
                sentence_a=sample["answer"],
                sentence_b=generated_answer["answer"],
            )
        return float(
            generated_answer["answer"].upper() == "NOT ENOUGH CONTEXT")

    if task_name == "RateContext":
        # If the question can be answered with the context, we expect the context
        # score to be above 3 (on a scale of 0-5).
        if sample["answerable"]:
            return float(generated_answer["context_score"] >= 3)
        return float(generated_answer["context_score"] < 3)

    if task_name == "AssessAnswerability":
        return float(
            sample["answerable"] == generated_answer["answerable_question"])

    if task_name == "ParaphraseQuestions":
        similarities = [
            compute_sentence_similarity(sentence_a=sample["question"],
                                        sentence_b=generated_question)
            for generated_question in generated_answer["paraphrased_questions"]
        ]
        return sum(similarities) / len(similarities)

    if task_name == "RAGAS":
        # Answer relevance  and Faithfulness are between 0-5. We expects scores
        # above 3 to be considered 'good'. For Context relevance, we expect it
        # to be above 3 only if the question can be answered with the context.
        answer_relevance_ok = generated_answer["answer_relevance_score"] >= 3
        faithfulness_ok = generated_answer["faithfulness_score"] >= 3
        if sample["answerable"]:
            context_ok = generated_answer["context_relevance_score"] >= 3
        else:
            context_ok = generated_answer["context_relevance_score"] < 3

        return float(answer_relevance_ok and faithfulness_ok and context_ok)

    if task_name == "GenerateAnswerWithConfidence":
        if not 0 <= generated_answer["confidence"] <= 5:
            return 0.0

        return compute_sentence_similarity(
            sentence_a=sample["answer"], sentence_b=generated_answer["answer"])

    if task_name == "GenerateAnswersWithConfidence":
        # Check if all confidence scores are correct, and select the one with
        # the highest confidence.
        highest_confidence = -1
        highest_generated_answer = ""
        for el in generated_answer["answers"]:
            confidence = el["confidence"]
            if not 0 <= confidence <= 5:
                return 0.0
            if confidence > highest_confidence:
                highest_confidence = confidence
                highest_generated_answer = el["answer"]

        return compute_sentence_similarity(sentence_a=sample["answer"],
                                           sentence_b=highest_generated_answer)

    return 0.0


def evaluate_task_outputs(
    task_name: str,
    sample_data: list[dict[str, str]],
    task_outputs: list[dict | None],
) -> list[float]:
    """Return the scores of all task outputs for the given task."""
    # Each output is evaluated against its corresponding sample data.
    assert len(sample_data) == len(task_outputs)

    scores = [
        compute_answer_score(task_name=task_name,
                             sample=sample,
                             generated_answer=output)
        for sample, output in zip(sample_data, task_outputs)
    ]
    return scores


def save_results(results: dict[str, float], output_path: str):
    """Write the results in a JSON file in output_folder."""
    results_to_save = {
        "results": {
            task_name: {
                "score": score
            }
            for task_name, score in results.items()
        }
    }

    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=4)


def main(args: argparse.Namespace):
    """Run the evaluation task(s) and aggregate results."""

    evaluation_task = args.task
    llm_name = args.model
    dataset_filepath = args.input if args.input else DATASET_WIKIQUESTIONS
    output_path = args.output
    n_samples_per_task = args.n_samples

    wiki_questions = load_dataset_wikiquestions(filepath=dataset_filepath)
    # Full dataset is made of 112 examples. Only use the N first samples
    # for faster eval and reduced eval cost.
    wiki_questions = wiki_questions[:n_samples_per_task]

    results = {}
    for task_name in TASK_DESCRIPTIONS:
        # We do not do the task if it was not given as an argument, or if
        # not "all" tasks need to be done.
        if (task_name != evaluation_task) and (evaluation_task
                                               != "json-mode-all"):
            continue

        expected_schema = TASK_NAME_TO_TASK_SCHEMA[task_name]
        expected_schema_json: dict[str, str | dict | list] = (
            expected_schema.model_json_schema())

        # Prepare the option object for COMPLETE() based on the expected output
        # schema.
        options: dict[str, float | dict] = {
            "temperature": 0,
            "response_format": {
                "type": "json",
                "schema": expected_schema_json
            },
        }

        # Process samples.
        llm_outputs = process_batch(
            task_name=task_name,
            sample_data=wiki_questions,
            llm_name=llm_name,
            options=options,
        )

        # Get scores.
        scores = evaluate_task_outputs(
            task_name=task_name,
            sample_data=wiki_questions,
            task_outputs=llm_outputs,
        )
        avg_score = sum(scores) / len(scores)
        results[task_name] = float(avg_score)  # to avoid having np.float()

    global_average = sum(results.values()) / len(results)
    results["json-mode-all"] = global_average

    if output_path is not None:
        save_results(results=results, output_path=output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the quality of JSON-mode output in Complete().")

    VALID_TASKS = list(TASK_DESCRIPTIONS.keys()) + ["json-mode-all"]

    parser.add_argument(
        "--task",
        type=str,
        required=False,
        choices=VALID_TASKS,
        help=f'Task name. Must be one of: {", ".join(VALID_TASKS)}',
        default="json-mode-all",
    )

    parser.add_argument("--model",
                        type=str,
                        required=False,
                        help="Name of the LLM to use.",
                        default="Snowflake/Llama-3.1-SwiftKV-8B-Instruct")

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the WikiQuestions dataset file.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Folder path to save the evaluation results.",
    )

    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples to use from the dataset.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    main(args=args)
