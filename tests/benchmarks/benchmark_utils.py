from dataclasses import dataclass
from typing import Any, Callable, Dict

import pandas as pd


@dataclass
class BenchmarkTask:
    # Configuration for the benchmark task.
    config: Dict[str, Any]

    # Metrics to collect for the benchmark task. Maps name -> key where name is
    # the name of the metric that appears in the summary and key is the key for
    # extracting the metric from each benchmark result.
    metrics: Dict[str, str | Callable]

VLLM_CONFIGS = {
    "llama_8b": {
        "model": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        "tensor_parallel_size": 4,
        "enable_prefix_caching": False,
    },
    "llama_8b_shift": {
        "model": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        "tensor_parallel_size": 2,
        "ulysses_sequence_parallel_size": 2,
        "enable_shift_parallel": True,
        "shift_parallel_threshold": 256,
        "enable_prefix_caching": False,
    },
    "llama_8b_swiftkv": {
        "model": "Snowflake/Llama-3.1-SwiftKV-8B-Instruct-FP8",
        "tensor_parallel_size": 4,
        "enable_prefix_caching": False,
    },
    "llama_8b_suffix": {
        "model": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        "tensor_parallel_size": 4,
        "speculative_config": {
            "method": "suffix",
            "disable_by_batch_size": 64,
        },
        "enable_prefix_caching": False,
    },
    "llama_8b_spec": {
        "model": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        "tensor_parallel_size": 4,
        "speculative_config": {
            "method": "arctic",
            "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
            "num_speculative_tokens": 3,
            "disable_by_batch_size": 64,
        },
        "enable_prefix_caching": False,
    },
    "llama_8b_all": {
        "model": "Snowflake/Llama-3.1-SwiftKV-8B-Instruct-FP8",
        "tensor_parallel_size": 2,
        "ulysses_sequence_parallel_size": 2,
        "enable_shift_parallel": True,
        "speculative_config": {
            "method": "arctic",
            "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
            "num_speculative_tokens": 3,
            "enable_suffix_decoding": True,
            "disable_by_batch_size": 64,
        },
        "enable_prefix_caching": False,
    },
}

PERFORMANCE_TASKS = {
    "batch": BenchmarkTask(
        config={
            "dataset_name": "random",
            "random_input_len": 2000,
            "random_output_len": 250,
            "num_prompts": 2000,
        },
        metrics={
            "throughput": "total_token_throughput",
        },
    ),
    "single": BenchmarkTask(
        config={
            "dataset_name": "random",
            "random_input_len": 2000,
            "random_output_len": 250,
            "num_prompts": 20,
            "max_concurrency": 1,
        },
        metrics={
            "ttft_ms": "mean_ttft_ms",
            "tpot_ms": "mean_tpot_ms",
        }
    ),
}

ACCURACY_TASKS = {
    "arc_challenge": BenchmarkTask(
        config={
            "tasks": ["arc_challenge_chat"],
            "apply_chat_template": True,
            "num_fewshot": 5,
        },
        metrics={
            "acc": "exact_match,remove_whitespace",
        },
    ),
    # GPQA is broken in lm_eval==0.4.8, but can be good to add once fixed:
    # https://github.com/EleutherAI/lm-evaluation-harness/issues/2907.
    #
    # "gpqa": BenchmarkTask(
    #     config={
    #         "tasks": ["gpqa_diamond_cot_n_shot"],
    #         "apply_chat_template": True,
    #         "num_fewshot": 5,
    #     },
    #     metrics={
    #         "acc": "exact_match,flexible-extract",
    #     },
    # ),
    "gsm8k": BenchmarkTask(
        config={
            "tasks": ["gsm8k_cot"],
            "apply_chat_template": True,
        },
        metrics={
            "acc": "exact_match,flexible-extract",
        },
    ),
    "ifeval": BenchmarkTask(
        config={
            "tasks": ["ifeval"],
            "apply_chat_template": True,
        },
        metrics={
            "score": lambda result: (result["prompt_level_strict_acc,none"] +
                                     result["inst_level_strict_acc,none"]) / 2,
        },
    ),
    "mmlu_pro": BenchmarkTask(
        config={
            "tasks": ["mmlu_pro"],
            "apply_chat_template": True,
        },
        metrics={
            "acc": "exact_match,custom-extract",
        },
    ),
}

JSON_MODE_TASKS = {
    "json_mode_score": BenchmarkTask(
        config={
            "task" : "json-mode-all",
            "input": "json_mode/datasets/WikiQuestions.json",
            "n_samples": 25,
        },
        metrics={
            "score": "json-mode-all",
        },
    ),
}


def init_benchmark_summary():
    tuples = []
    for name, task in {**PERFORMANCE_TASKS, **ACCURACY_TASKS, **JSON_MODE_TASKS}.items():
        for metric in task.metrics:
            tuples.append((name, metric))
    columns = pd.MultiIndex.from_tuples(tuples, names=['task', 'metric'])
    return pd.DataFrame(index=list(VLLM_CONFIGS.keys()), columns=columns)


def update_benchmark_summary(config: str, task_name: str,
                             result: Dict[str, Any]) -> None:
    # Update the DataFrame with metrics for this config and task
    for metric_name, value in result.items():
        _SUMMARY.loc[config, (task_name, metric_name)] = value


def get_benchmark_summary():
    # Round numeric values and drop rows and columns that are all empty
    return _SUMMARY.round(3).dropna(how='all').dropna(axis=1, how='all')


_SUMMARY = init_benchmark_summary()
