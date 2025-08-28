import argparse
import json
import multiprocessing
import pathlib
import tempfile
import traceback
from typing import Any, Dict

import pytest

from .benchmark_utils import VLLM_CONFIGS, update_benchmark_summary


def test_performance(benchmark_spec, request):
    """Tests vLLM performance (throughput/latency) in serial."""
    config_name = benchmark_spec["config_name"]
    task_name = benchmark_spec["task_name"]
    task = benchmark_spec["task_obj"]
    port = benchmark_spec["port"]
    vllm_config = VLLM_CONFIGS[config_name]
    
    from vllm.benchmarks.serve import add_cli_args, main as benchmark_serve_main
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(["--model", vllm_config["model"], "--port", str(port)])

    result_path = (request.config.option.benchmark_result_dir / config_name /
                   f"performance-{task_name}.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    for key, value in task.config.items():
        setattr(args, key, value)
    args.save_result = True
    args.result_dir = str(result_path.parent)
    args.result_filename = str(result_path.name)
    
    benchmark_serve_main(args)

    with open(result_path, "r") as f:
        result = json.load(f)
    metrics = {name: key(result) if callable(key) else result[key]
               for name, key in task.metrics.items()}
    update_benchmark_summary(config_name, task_name, metrics)


def test_json_mode(benchmark_spec, request):
    config_name = benchmark_spec["config_name"]
    task_name = benchmark_spec["task_name"]
    task = benchmark_spec["task_obj"]
    port = benchmark_spec["port"]
    vllm_config = VLLM_CONFIGS[config_name]

    if vllm_config.get("speculative_config", {}).get("enable_suffix_decoding"):
        pytest.skip("Skipping JSON mode test for spec + suffix decoding.")

    from .json_mode.evaluate_text_json_mode import main as evaluate_json

    result_path = (request.config.option.benchmark_result_dir / config_name /
                   f"json_mode-{task_name}.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=vllm_config["model"])
    parser.add_argument("--output", type=str, default=str(result_path))
    parser.add_argument("--port", type=int, default=port)
    for key, value in task.config.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

    evaluate_json(parser.parse_args([]))
    
    with open(result_path, "r") as f:
        result = json.load(f)
    result_data = result.get("results", {})
    metrics = {name: key(result_data) if callable(key) else result_data.get(key, {}).get("score")
               for name, key in task.metrics.items()}
    update_benchmark_summary(config_name, task_name, metrics)


def _run_lm_eval_harness(queue, lm_eval_config, model_name, port):
    try:
        from lm_eval import evaluator
        result = evaluator.simple_evaluate(
            model="local-completions",
            model_args={"model": model_name, "base_url": f"http://localhost:{port}/v1/completions"},
            **lm_eval_config)
        queue.put(result)
    except Exception as exc:
        queue.put(exc)

def _run_accuracy_worker(config_name, port, task_name, task_config,
                         benchmark_result_dir, results_queue):
    try:
        from lm_eval.utils import handle_non_serializable, make_table
        vllm_config = VLLM_CONFIGS[config_name]
        queue = multiprocessing.Queue()
        eval_process = multiprocessing.Process(
            target=_run_lm_eval_harness,
            args=(queue, task_config, vllm_config["model"], port))
        eval_process.start()
        result_or_exc = queue.get()
        eval_process.join()
        if isinstance(result_or_exc, Exception): raise result_or_exc
        
        result = result_or_exc
        print(f"Accuracy results for '{config_name}':\n{make_table(result)}")
        
        result_path = benchmark_result_dir / config_name / f"accuracy-{task_name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4, default=handle_non_serializable)
        results_queue.put({"config_name": config_name, "result_path": result_path})
    except Exception as e:
        results_queue.put({"config_name": config_name, "error": str(e), "traceback": traceback.format_exc()})

def test_batch_accuracy(batch_spec, request):
    """Tests model accuracy for a whole batch in parallel."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
        
    task_name = batch_spec["task_name"]
    task_obj = batch_spec["task_obj"]
    configs_in_batch = batch_spec["configs"]
    port_map = batch_spec["port_map"]
    benchmark_result_dir = request.config.option.benchmark_result_dir or pathlib.Path(tempfile.mkdtemp())

    processes, results_queue = [], multiprocessing.Queue()
    for config_name in configs_in_batch:
        p = multiprocessing.Process(
            target=_run_accuracy_worker,
            args=(config_name, port_map[config_name], task_name,
                  task_obj.config, benchmark_result_dir, results_queue))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
        
    while not results_queue.empty():
        result = results_queue.get()
        config_name = result["config_name"]
        if "error" in result:
            pytest.fail(f"Worker for '{config_name}' failed:\n{result['error']}\n{result['traceback']}")
        
        with open(result["result_path"], "r") as f:
            raw_result = json.load(f)
        lm_eval_task_name = task_obj.config["tasks"][0]
        result_data = raw_result["results"][lm_eval_task_name]
        metrics = {name: key(result_data) if callable(key) else result_data[key]
                   for name, key in task_obj.metrics.items()}
        update_benchmark_summary(config_name, task_name, metrics)