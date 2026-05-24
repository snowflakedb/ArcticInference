import argparse
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest
import requests
from vllm.entrypoints.openai.api_server import (
    make_arg_parser, validate_parsed_serve_args)
from vllm.utils.argparse_utils import FlexibleArgumentParser

from .benchmark_utils import (ACCURACY_TASKS, PERFORMANCE_TASKS, VLLM_CONFIGS,
                              JSON_MODE_TASKS, update_benchmark_summary)

CUSTOM_PORT = 8080


def _build_server_cli(config: dict, port: int) -> list[str]:
    """Build the CLI arguments for vllm.entrypoints.openai.api_server."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--port", str(port),
        "--no-enable-log-requests",
    ]
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            cmd.append(flag if value else f"--no-{key.replace('_', '-')}")
        elif isinstance(value, dict):
            cmd.extend([flag, json.dumps(value)])
        elif hasattr(value, 'model_dump'):
            cmd.extend([flag, json.dumps(value.model_dump(mode='json'))])
        else:
            cmd.extend([flag, str(value)])
    return cmd


@pytest.fixture(scope="module", params=list(VLLM_CONFIGS.keys()))
def vllm_server(request):
    """Start the OpenAI API server as an isolated subprocess."""
    config = VLLM_CONFIGS[request.param]

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args([])
    args.enable_log_requests = False
    setattr(args, 'port', CUSTOM_PORT)
    for key, value in config.items():
        setattr(args, key, value)
    validate_parsed_serve_args(args)

    cmd = _build_server_cli(config, CUSTOM_PORT)
    env = {**os.environ, "ARCTIC_INFERENCE_ENABLED": "1"}
    server_log = tempfile.NamedTemporaryFile(
        mode='w', prefix=f"vllm_{request.param}_", suffix=".log", delete=False)
    process = subprocess.Popen(
        cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    print(f"Waiting for server to start (pid={process.pid})...")
    print(f"Server cmd: {' '.join(cmd)}")
    timeout = 3600
    interval = 5
    start = time.time()

    health_check_url = f"http://localhost:{CUSTOM_PORT}/v1/models"
    expected_model = config["model"]

    while True:
        try:
            r = requests.get(health_check_url)
            if r.status_code == 200:
                models = r.json().get("data", [])
                served = [m["id"] for m in models]
                if expected_model in served:
                    break
                print(f"Server up but model not ready yet (served: {served})")
        except requests.exceptions.ConnectionError:
            pass
        if process.poll() is not None:
            server_log.flush()
            with open(server_log.name) as f:
                stdout = f.read()
            raise RuntimeError(
                f"Server process terminated unexpectedly "
                f"(exit={process.returncode}):\n{stdout[-4000:]}")
        if time.time() - start > timeout:
            raise TimeoutError(f"Server didn't start after {timeout} seconds")
        time.sleep(interval)
    print("Server process started")

    yield request.param, args

    print("Terminating server process")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
    server_log.close()
    print(f"Server log: {server_log.name}")

    for _ in range(30):
        try:
            r = requests.get(f"http://localhost:{CUSTOM_PORT}/v1/models")
            time.sleep(1)
        except requests.exceptions.ConnectionError:
            break
    print("Server process terminated")


@pytest.mark.parametrize("task_name", list(PERFORMANCE_TASKS.keys()))
def test_performance(request, vllm_server, task_name):
    from vllm.benchmarks.serve import add_cli_args, main

    config_name, vllm_args = vllm_server
    task = PERFORMANCE_TASKS[task_name]

    parser = argparse.ArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(["--model", vllm_args.model])

    setattr(args, 'port', CUSTOM_PORT)

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
        try:
            from lm_eval import evaluator
            from lm_eval.utils import handle_non_serializable, make_table

            base_url = f"http://localhost:{CUSTOM_PORT}/v1/completions"

            result = evaluator.simple_evaluate(
                model="local-completions",
                model_args={
                    "model": vllm_args.model,
                    "base_url": base_url,
                    "num_concurrent": 256,
                    "timeout": 3600,
                },
                **task.config,
            )
            print(make_table(result))

            tmpfile = f"{tmpdir}/result.json"
            with open(tmpfile, "w") as f:
                json.dump(result, f, indent=4, default=handle_non_serializable)
        except Exception as exc:
            q.put(exc)
        else:
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

        args.port = CUSTOM_PORT

        evaluate_json(args)

        with open(result_path, "r") as f:
            result = json.load(f)

    result_data = result.get("results", {})

    metrics = {
        name: key(result_data) if callable(key) else result_data.get(key, {}).get('score')
        for name, key in task.metrics.items()
    }

    update_benchmark_summary(config_name, task_name, metrics)
