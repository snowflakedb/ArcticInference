import argparse
import json
import multiprocessing
import time

import pytest
import requests
import uvloop
from vllm.entrypoints.openai.api_server import (
    make_arg_parser, run_server, validate_parsed_serve_args)
from vllm.utils import FlexibleArgumentParser

CONFIGS = {
    "llama-8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
    },
    "llama-8b-shift": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "ulysses_sequence_parallel_size": 2,
        "enable_shift_parallel": True,
    },
    "llama-8b-swiftkv": {
        "model": "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        "tensor_parallel_size": 2,
    },
    "llama-8b-spec": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
        "speculative_config": {
            "method": "arctic",
            "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
            "num_speculative_tokens": 3,
            "enable_suffix_decoding": True,
            "disable_by_batch_size": 64,
        },
    },
    "llama-8b-all": {
        "model": "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
        "ulysses_sequence_parallel_size": 2,
        "enable_shift_parallel": True,
        "speculative_config": {
            "method": "arctic",
            "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
            "num_speculative_tokens": 3,
            "enable_suffix_decoding": True,
            "disable_by_batch_size": 64,
        },
    },
}

PERFORMANCE_TASKS = {
    "throughput": {
        "dataset_name": "random",
        "random_input_len": 2000,
        "random_output_len": 250,
        "num_prompts": 2000,
    },
    "latency": {
        "dataset_name": "random",
        "random_input_len": 2000,
        "random_output_len": 250,
        "num_prompts": 20,
        "max_concurrency": 1,
    },
}

ACCURACY_TASKS = {
    "gsm8k": "gsm8k",
    "arc_challenge_chat": "arc_challenge_chat",
}


@pytest.fixture(scope="module", params=list(CONFIGS.keys()))
def openai_server(request):
    """
    Fixture to start the OpenAI API server for testing.
    """
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    args = parser.parse_args([])
    args.disable_log_requests = True
    args.disable_uvicorn_access_log = True

    for key, value in CONFIGS[request.param].items():
        setattr(args, key, value)

    validate_parsed_serve_args(args)

    def _run_process():
        uvloop.run(run_server(args))

    # Start server process
    process = multiprocessing.Process(target=_run_process)
    process.start()

    print("Waiting for server to start...")
    timeout = 1800
    interval = 5
    start = time.time()
    while True:
        try:
            r = requests.get("http://localhost:8000/v1/models")
            if r.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        if not process.is_alive():
            raise RuntimeError("Server process terminated unexpectedly")
        if time.time() - start > timeout:
            raise TimeoutError(f"Server didn't start after {timeout} seconds")
        time.sleep(interval)
    print("Server process started")

    yield args

    # Stop server process
    print("Terminating server process")
    if process.is_alive():
        process.terminate()
        process.join()
    print("Server process terminated")


@pytest.mark.parametrize("task", list(PERFORMANCE_TASKS.keys()))
def test_performance(openai_server, task):
    from vllm.benchmarks.serve import add_cli_args, main

    parser = argparse.ArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(["--model", openai_server.model])

    task_config = PERFORMANCE_TASKS[task]
    for key, value in task_config.items():
        setattr(args, key, value)

    main(args)


@pytest.mark.parametrize("task", list(ACCURACY_TASKS.keys()))
def test_accuracy(openai_server, task):

    task_config = ACCURACY_TASKS[task]

    def _run_process():
        from lm_eval import evaluator
        from lm_eval.utils import make_table

        results = evaluator.simple_evaluate(
            model="local-completions",
            model_args={
                "model": openai_server.model,
                "base_url": "http://localhost:8000/v1/completions",
                "num_concurrent": 64,
            },
            tasks=[task_config],
        )
        print(results["results"][task].keys())
        print(make_table(results))

    # Run lm_eval in a separate process because it imports torch and
    # initializes CUDA, which breaks process forking in later tests.
    process = multiprocessing.Process(target=_run_process)
    process.start()
    process.join()
