import argparse
import json
import multiprocessing
import tempfile
import time

import pytest
import requests
import uvloop
from vllm.entrypoints.openai.api_server import (
    make_arg_parser, run_server, validate_parsed_serve_args)
from vllm.utils import FlexibleArgumentParser

from .benchmark_utils import (ACCURACY_TASKS, PERFORMANCE_TASKS, VLLM_CONFIGS,
                              JSON_MODE_TASKS, update_benchmark_summary)                  


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

    yield request.param, args

    # Stop server process
    print("Terminating server process")
    if process.is_alive():
        process.terminate()
        process.join()
    print("Server process terminated")


@pytest.mark.parametrize("task_name", list(PERFORMANCE_TASKS.keys()))
def test_performance(request, vllm_server, task_name):
    from vllm.benchmarks.serve import add_cli_args, main

    config_name, vllm_args = vllm_server
    task = PERFORMANCE_TASKS[task_name]

    parser = argparse.ArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(["--model", vllm_args.model])

    with tempfile.TemporaryDirectory() as tmpdir:
        args.save_result = True
        args.result_dir = str(tmpdir)
        args.result_filename = "result.json"

        for key, value in task.config.items():
            setattr(args, key, value)

        main(args)

        with open(f"{tmpdir}/result.json", "r") as f:
            result = json.load(f)

    benchmark_result_dir = request.config.option.benchmark_result_dir
    if benchmark_result_dir is not None:
        result_path = (benchmark_result_dir / "performance" /
                       f"{config_name}-{task_name}.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

    metrics = {name: key(result) if callable(key) else result[key]
               for name, key in task.metrics.items()}
    update_benchmark_summary(config_name, task_name, metrics)


@pytest.mark.parametrize("task_name", list(ACCURACY_TASKS.keys()))
def test_accuracy(request, vllm_server, task_name):

    config_name, vllm_args = vllm_server
    task = ACCURACY_TASKS[task_name]

    assert len(task.config["tasks"]) == 1, \
        "Accuracy benchmarks should only have one task configured"

    q = multiprocessing.Queue()

    def _run_process():
        # Run lm_eval in a separate process because it imports torch and
        # initializes CUDA, which breaks process forking in later tests.
        try:
            from lm_eval import evaluator
            from lm_eval.utils import handle_non_serializable, make_table

            result = evaluator.simple_evaluate(
                model="local-completions",
                model_args={
                    "model": vllm_args.model,
                    "base_url": "http://localhost:8000/v1/completions",
                    "num_concurrent": 256,
                },
                **task.config,
            )
            print(make_table(result))

            tmpfile = f"{tmpdir}/result.json"
            with open(tmpfile, "w") as f:
                json.dump(result, f, indent=4, default=handle_non_serializable)
        except Exception as exc:
            # If an exception occurs, put it in the queue to be raised later
            q.put(exc)
        else:
            # Send back the temporary file path instead of the result object
            # since multiprocessing queue can hang on large objects.
            q.put(tmpfile)

    with tempfile.TemporaryDirectory() as tmpdir:
        process = multiprocessing.Process(target=_run_process)
        process.start()
        r = q.get()
        process.join()
        if isinstance(r, Exception):
            raise r
        tmpfile = r
        with open(tmpfile, "r") as f:
            result = json.load(f)

    benchmark_result_dir = request.config.option.benchmark_result_dir
    if benchmark_result_dir is not None:
        result_path = (benchmark_result_dir / "accuracy" /
                       f"{config_name}-{task_name}.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

    result = result["results"][task.config["tasks"][0]]
    metrics = {name: key(result) if callable(key) else result[key]
               for name, key in task.metrics.items()}
    update_benchmark_summary(config_name, task_name, metrics)


@pytest.mark.parametrize("task_name", list(JSON_MODE_TASKS.keys()))
def test_json_mode(request, vllm_server, task_name):
    """
    Test JSON mode using the evaluate_text_json_mode script.
    """
    from .json_mode.evaluate_text_json_mode import main as evaluate_json

    config_name, vllm_args = vllm_server
    task = JSON_MODE_TASKS[task_name]

    if (vllm_args.speculative_config and
            vllm_args.speculative_config.get('enable_suffix_decoding', False)):
        pytest.skip("Skipping JSON mode test for spec + suffix decoding enabled")

    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = f"{tmpdir}/result.json"

        args = FlexibleArgumentParser()
        args.model = vllm_args.model
        args.output = result_path
        args.task = task.config["task"]
        args.input = task.config["input"]
        args.n_samples = task.config["n_samples"]

        evaluate_json(args)

        with open(result_path, "r") as f:
            result = json.load(f)

    result_data = result.get("results", {})
    
    metrics = {
        name: key(result_data) if callable(key) else result_data.get(key, {}).get('score')
        for name, key in task.metrics.items()
    }

    update_benchmark_summary(config_name, task_name, metrics)