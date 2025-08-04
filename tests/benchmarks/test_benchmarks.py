import argparse
import json
import multiprocessing
import pathlib
import tempfile
import traceback
from typing import Any, Dict

import pytest

from .benchmark_utils import VLLM_CONFIGS, update_benchmark_summary

# =============================================================================
# Worker Functions & Helpers
# =============================================================================

# This function is now at the top level to make it picklable.
def _run_lm_eval_harness(queue: multiprocessing.Queue, lm_eval_config: Dict,
                        model_name: str, port: int):
    """Target function to run lm-eval in a clean process."""
    try:
        from lm_eval import evaluator
        result = evaluator.simple_evaluate(
            model="local-completions",
            model_args={
                "model": model_name,
                "base_url": f"http://localhost:{port}/v1/completions",
                "num_concurrent": 256,
                "timeout": 3600,
            },
            **lm_eval_config,
        )
        queue.put(result)
    except Exception as exc:
        queue.put(exc)


def _run_performance_worker(config_name: str, port: int, task_name: str,
                            task_config: Dict,
                            benchmark_result_dir: pathlib.Path,
                            results_queue: multiprocessing.Queue):
    """Worker for performance benchmarks. Returns the path to the result file."""
    try:
        from vllm.benchmarks.serve import add_cli_args, main as benchmark_serve_main
        vllm_config = VLLM_CONFIGS[config_name]
        parser = argparse.ArgumentParser()
        add_cli_args(parser)
        args = parser.parse_args(["--model", vllm_config["model"], "--port", str(port)])
        
        for key, value in task_config.items():
            setattr(args, key, value)
        
        result_path = benchmark_result_dir / config_name / f"performance-{task_name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        args.save_result = True
        args.result_dir = str(result_path.parent)
        args.result_filename = str(result_path.name)
        
        benchmark_serve_main(args)
        results_queue.put({"config_name": config_name, "result_path": result_path})
    except Exception as e:
        results_queue.put({"config_name": config_name, "error": str(e), "traceback": traceback.format_exc()})


def _run_accuracy_worker(config_name: str, port: int, task_name: str,
                         task_config: Dict, benchmark_result_dir: pathlib.Path,
                         results_queue: multiprocessing.Queue):
    """Worker for accuracy benchmarks. Returns the path to the result file."""
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

        if isinstance(result_or_exc, Exception):
            raise result_or_exc
        
        result = result_or_exc
        print(f"Accuracy results for '{config_name}':\n{make_table(result)}")
        
        result_path = benchmark_result_dir / config_name / f"accuracy-{task_name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=4, default=handle_non_serializable)
        results_queue.put({"config_name": config_name, "result_path": result_path})
    except Exception as e:
        results_queue.put({"config_name": config_name, "error": str(e), "traceback": traceback.format_exc()})


def _run_json_eval_worker(config_name: str, port: int, task_name: str,
                            task_config: Dict,
                            benchmark_result_dir: pathlib.Path,
                            results_queue: multiprocessing.Queue):
    """Worker for JSON mode benchmarks. Returns the path to the result file."""
    try:
        from .json_mode.evaluate_text_json_mode import main as evaluate_json
        vllm_config = VLLM_CONFIGS[config_name]
        if vllm_config.get("speculative_config", {}).get("enable_suffix_decoding"):
            results_queue.put({"config_name": config_name, "status": "skipped"})
            return
            
        result_path = benchmark_result_dir / config_name / f"json_mode-{task_name}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default=vllm_config["model"])
        parser.add_argument("--output", type=str, default=str(result_path))
        parser.add_argument("--port", type=int, default=port)
        for key, value in task_config.items():
            parser.add_argument(f"--{key}", type=type(value), default=value)
        
        args = parser.parse_args([])
        evaluate_json(args)
        results_queue.put({"config_name": config_name, "result_path": result_path})
    except Exception as e:
        results_queue.put({"config_name": config_name, "error": str(e), "traceback": traceback.format_exc()})


# =============================================================================
# Pytest Test Functions & Orchestrator
# =============================================================================

def _run_batch_test(batch_spec: Dict, request: Any, worker_func: callable,
                    test_type_str: str, metric_extractor: callable):
    """Generic orchestrator for running a batch test in parallel."""
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    task_name = batch_spec["task_name"]
    task_obj = batch_spec["task_obj"]
    configs_in_batch = batch_spec["configs"]
    port_map = batch_spec["port_map"]
    benchmark_result_dir = request.config.option.benchmark_result_dir or pathlib.Path(tempfile.mkdtemp())

    processes = []
    results_queue = multiprocessing.Queue()
    
    print(f"\nLaunching parallel {test_type_str} benchmark for Batch {batch_spec['batch_idx']}...")
    for config_name in configs_in_batch:
        p = multiprocessing.Process(
            target=worker_func,
            args=(config_name, port_map[config_name], task_name, task_obj.config,
                  benchmark_result_dir, results_queue))
        processes.append(p)
        p.start()
        print(f"  -> Started {test_type_str} worker for '{config_name}' on port {port_map[config_name]}")

    for p in processes:
        p.join()
        
    print(f"Parallel {test_type_str} benchmark for Batch {batch_spec['batch_idx']} complete. Aggregating results...")
    
    while not results_queue.empty():
        result = results_queue.get()
        config_name = result["config_name"]
        
        if result.get("status") == "skipped":
            print(f"  -> '{config_name}' was skipped.")
            continue
        if "error" in result:
            pytest.fail(f"Worker for '{config_name}' failed with:\n{result['error']}\n{result['traceback']}")

        metrics = metric_extractor(result["result_path"], task_obj)
        update_benchmark_summary(config_name, task_name, metrics)
        print(f"  -> Logged {test_type_str} results for '{config_name}'.")


def test_batch_performance(batch_spec, request):
    """Tests vLLM performance for a whole batch in parallel."""
    def extractor(result_path, task_obj):
        with open(result_path, "r") as f:
            result = json.load(f)
        return {name: key(result) if callable(key) else result[key] for name, key in task_obj.metrics.items()}
    _run_batch_test(batch_spec, request, _run_performance_worker, "performance", extractor)


def test_batch_accuracy(batch_spec, request):
    """Tests model accuracy for a whole batch in parallel."""
    def extractor(result_path, task_obj):
        with open(result_path, "r") as f:
            result = json.load(f)
        lm_eval_task_name = task_obj.config["tasks"][0]
        result_data = result["results"][lm_eval_task_name]
        return {name: key(result_data) if callable(key) else result_data[key] for name, key in task_obj.metrics.items()}
    _run_batch_test(batch_spec, request, _run_accuracy_worker, "accuracy", extractor)


# def test_batch_json_mode(batch_spec, request):
#     """Tests JSON mode for a whole batch in parallel."""
#     def extractor(result_path, task_obj):
#         with open(result_path, "r") as f:
#             result = json.load(f)
#         result_data = result.get("results", {})
#         return {name: key(result_data) if callable(key) else result_data.get(key, {}).get("score") for name, key in task_obj.metrics.items()}
#     _run_batch_test(batch_spec, request, _run_json_eval_worker, "JSON mode", extractor)