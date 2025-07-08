# test_evaluation.py

import json
import multiprocessing
import time
import pytest
import requests
import uvloop
import argparse
from pathlib import Path

# Imports from vLLM to start the server
from vllm.entrypoints.openai.api_server import make_arg_parser, run_server
from vllm.utils import FlexibleArgumentParser

# Define a minimal configuration for the vLLM server
# NOTE: Change the model name to one you have access to.
# Using a small, fast-loading model is ideal for testing.
VLLM_CONFIGS = {
    "llama_8b_spec": {
        "model": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        "tensor_parallel_size": 2,
        "speculative_config": {
            "method": "arctic",
            "model": "Snowflake/Arctic-LSTM-Speculator-Llama-3.1-8B-Instruct",
            "num_speculative_tokens": 3,
            "disable_by_batch_size": 64,
        },
        "enable_prefix_caching": False,
        "distributed_executor_backend": "mp"
    },
}

@pytest.fixture(scope="module", params=list(VLLM_CONFIGS.keys()))
def vllm_server(request):
    """
    Fixture to start the OpenAI API server for testing.
    """
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    args = parser.parse_args([])
    args.disable_log_requests = True
    args.disable_uvicorn_access_log = True

    for key, value in VLLM_CONFIGS[request.param].items():
        setattr(args, key, value)

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

    yield request.param, args

    # Stop server process
    print("Terminating server process")
    if process.is_alive():
        process.terminate()
        process.join()
    print("Server process terminated")


def test_evaluation_run(vllm_server, tmp_path: Path):
    """
    Test the evaluation script with a vLLM server running.
    """
    model_name, args = vllm_server

    eval_args = [
        "--model", "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        #"--output", str(tmp_path),
    ]

    command = [
        "python", "evaluate_text_json_mode.py",
        *eval_args
    ]

    # run command
    process = multiprocessing.Process(target=lambda: uvloop.run(
        lambda: subprocess.run(command, check=True)
    ))

    process.start()
    print("Waiting for evaluation to complete...")
    

