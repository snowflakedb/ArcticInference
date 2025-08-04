import argparse
from httpx import URL
import json
import multiprocessing
import pytest
import tempfile

from .benchmark_utils import VLLM_CONFIGS, update_benchmark_summary
from .json_mode import utils as json_mode_utils


def test_performance(benchmark_spec, request):
    """Tests vLLM performance (throughput and latency)."""
    config_name = benchmark_spec["config_name"]
    task_name = benchmark_spec["task_name"]
    task = benchmark_spec["task_obj"]
    port = benchmark_spec["port"]

    vllm_config = VLLM_CONFIGS[config_name]

    from vllm.benchmarks.serve import add_cli_args, main
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(["--model", vllm_config["model"]])
    setattr(args, "port", port)

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
        config_result_dir = benchmark_result_dir / config_name
        config_result_dir.mkdir(parents=True, exist_ok=True)
        result_path = config_result_dir / f"performance-{task_name}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

    metrics = {
        name: key(result) if callable(key) else result[key]
        for name, key in task.metrics.items()
    }
    update_benchmark_summary(config_name, task_name, metrics)


def test_accuracy(benchmark_spec, request):
    """Tests model accuracy on lm-evaluation-harness tasks."""
    config_name = benchmark_spec["config_name"]
    task_name = benchmark_spec["task_name"]
    task = benchmark_spec["task_obj"]
    port = benchmark_spec["port"]

    vllm_config = VLLM_CONFIGS[config_name]
    assert len(
        task.config["tasks"]) == 1, "Accuracy benchmarks must have one task."

    q = multiprocessing.Queue()

    def _run_eval_process():
        try:
            from lm_eval import evaluator
            from lm_eval.utils import handle_non_serializable, make_table

            result = evaluator.simple_evaluate(
                model="local-completions",
                model_args={
                    "model": vllm_config["model"],
                    "base_url": f"http://localhost:{port}/v1/completions",
                    "num_concurrent": 256,
                    "timeout": 3600,
                },
                **task.config,
            )
            print(make_table(result))
            tmpfile = f"{tmpdir}/result.json"
            with open(tmpfile, "w") as f:
                json.dump(result, f, indent=4, default=handle_non_serializable)
            q.put(tmpfile)
        except Exception as exc:
            q.put(exc)

    with tempfile.TemporaryDirectory() as tmpdir:
        p = multiprocessing.Process(target=_run_eval_process)
        p.start()
        result_or_exc = q.get()
        p.join()
        if isinstance(result_or_exc, Exception):
            raise result_or_exc
        with open(result_or_exc, "r") as f:
            result = json.load(f)

    benchmark_result_dir = request.config.option.benchmark_result_dir
    if benchmark_result_dir is not None:
        config_result_dir = benchmark_result_dir / config_name
        config_result_dir.mkdir(parents=True, exist_ok=True)
        result_path = config_result_dir / f"accuracy-{task_name}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4)

    result_data = result["results"][task.config["tasks"][0]]
    metrics = {
        name: key(result_data) if callable(key) else result_data[key]
        for name, key in task.metrics.items()
    }
    update_benchmark_summary(config_name, task_name, metrics)


def test_json_mode(benchmark_spec, request):
    """Tests the server's structured (JSON) output capability."""
    config_name = benchmark_spec["config_name"]
    task_name = benchmark_spec["task_name"]
    task = benchmark_spec["task_obj"]
    port = benchmark_spec["port"]

    vllm_config = VLLM_CONFIGS[config_name]

    if vllm_config.get("speculative_config", {}).get("enable_suffix_decoding"):
        pytest.skip("Skipping JSON mode test for spec + suffix decoding.")

    from .json_mode.evaluate_text_json_mode import main as evaluate_json

    original_base_url = json_mode_utils.client.base_url
    json_mode_utils.client.base_url = URL(f"http://localhost:{port}/v1")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = f"{tmpdir}/result.json"
            final_result_path = None
            benchmark_result_dir = request.config.option.benchmark_result_dir
            if benchmark_result_dir is not None:
                config_result_dir = benchmark_result_dir / config_name
                config_result_dir.mkdir(parents=True, exist_ok=True)
                final_result_path = config_result_dir / f"json_mode-{task_name}.json"
                result_path = str(final_result_path)

            parser = argparse.ArgumentParser()
            parser.add_argument("--model",
                                type=str,
                                default=vllm_config["model"])
            parser.add_argument("--output", type=str, default=result_path)
            parser.add_argument("--task",
                                type=str,
                                default=task.config.get("task"))
            parser.add_argument("--input",
                                type=str,
                                default=task.config.get("input"))
            parser.add_argument("--n-samples",
                                type=int,
                                default=task.config.get("n_samples"))

            args = parser.parse_args([])
            evaluate_json(args)

            with open(result_path, "r") as f:
                result = json.load(f)
    finally:
        json_mode_utils.client.base_url = original_base_url

    result_data = result.get("results", {})
    metrics = {
        name:
        key(result_data)
        if callable(key) else result_data.get(key, {}).get("score")
        for name, key in task.metrics.items()
    }
    update_benchmark_summary(config_name, task_name, metrics)
