import argparse
import multiprocessing
import time

import pytest
import requests
import uvloop
from vllm.entrypoints.openai.api_server import (
    make_arg_parser, run_server, validate_parsed_serve_args)
from vllm.utils import FlexibleArgumentParser


def _run_server_process(args):
    uvloop.run(run_server(args))


@pytest.fixture(scope="module")
def openai_server():
    """
    Fixture to start the OpenAI API server for testing.
    """
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    args = parser.parse_args([])
    args.disable_log_requests = True
    args.model = "meta-llama/Llama-3.1-8B-Instruct"

    validate_parsed_serve_args(args)

    # Start server process
    process = multiprocessing.Process(target=_run_server_process, args=(args,))
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

    yield

    # Stop server process
    print("Terminating server process")
    if process.is_alive():
        process.terminate()
        process.join()
    print("Server process terminated")


def test_throughput(openai_server):
    from vllm.benchmarks.serve import add_cli_args, main
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(["--model", "meta-llama/Llama-3.1-8B-Instruct"])
    args.dataset_name = "random"
    args.random_input_len = 2000
    args.random_output_len = 250
    args.num_prompts = 2000
    main(args)


def test_latency(openai_server):
    from vllm.benchmarks.serve import add_cli_args, main
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(["--model", "meta-llama/Llama-3.1-8B-Instruct"])
    args.dataset_name = "random"
    args.random_input_len = 2000
    args.random_output_len = 250
    args.num_prompts = 20
    args.max_concurrency = 1
    main(args)


def test_gsm8k(openai_server):
    from lm_eval import evaluator
    from lm_eval.utils import make_table

    results = evaluator.simple_evaluate(
        model="local-completions",
        model_args={
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "base_url": "http://localhost:8000/v1/completions",
            "num_concurrent": 64,
        },
        tasks=["gsm8k"],
    )

    print(make_table(results))
