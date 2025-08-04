import argparse
import json
import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
import torch
from typing import Any, Dict, List

import pytest
import requests

import vllm

from .benchmark_utils import (ACCURACY_TASKS, JSON_MODE_TASKS,
                              PERFORMANCE_TASKS, VLLM_CONFIGS,
                              get_benchmark_summary)

MAX_GPUS = torch.cuda.device_count()
BASE_PORT = 8080


def pytest_addoption(parser):
    parser.addoption("--benchmark-result-dir", type=pathlib.Path)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summary = get_benchmark_summary()
    if summary.empty:
        return

    terminalreporter.write_sep("=", "Final Benchmark Summary")
    terminalreporter.write_line(summary.to_string())

    benchmark_result_dir = config.option.benchmark_result_dir
    if benchmark_result_dir is not None:
        benchmark_result_dir.mkdir(parents=True, exist_ok=True)
        summary_dict = {}
        for (task, metric), value in summary.items():
            summary_dict.setdefault(task, {})
            summary_dict[task].setdefault(metric, {})
            for config_name, config_value in value.items():
                summary_dict[task][metric][config_name] = config_value
        with open(benchmark_result_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=4)


def _schedule_configs() -> List[List[str]]:
    sorted_configs = sorted(
        VLLM_CONFIGS.items(),
        key=lambda item: item[1].get("tensor_parallel_size", 1) * item[1].get(
            "ulysses_sequence_parallel_size", 1),
        reverse=True)
    batches: List[List[str]] = []
    current_batch: List[str] = []
    gpus_used_in_batch = 0
    for name, config in sorted_configs:
        gpus_needed = config.get("tensor_parallel_size", 1) * config.get(
            "ulysses_sequence_parallel_size", 1)
        if gpus_used_in_batch + gpus_needed <= MAX_GPUS:
            current_batch.append(name)
            gpus_used_in_batch += gpus_needed
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [name]
            gpus_used_in_batch = gpus_needed
    if current_batch:
        batches.append(current_batch)
    return batches


class BatchServerManager:

    def __init__(self):
        self.current_batch_idx = -1
        self.processes: Dict[str, subprocess.Popen] = {}
        self.port_map: Dict[str, int] = {}

    def start_batch(self, batch_idx: int, batch_configs: List[str]):
        if self.current_batch_idx == batch_idx:
            return
        self.teardown_current_batch()
        self.current_batch_idx = batch_idx
        print(f"\nStarting Batch {batch_idx}: {batch_configs} ---")
        gpu_pool = list(range(MAX_GPUS))
        gpus_assigned = 0
        for i, config_name in enumerate(batch_configs):
            port = BASE_PORT + i
            gpus_needed = VLLM_CONFIGS[config_name].get(
                "tensor_parallel_size", 1) * VLLM_CONFIGS[config_name].get(
                    "ulysses_sequence_parallel_size", 1)
            gpu_ids = gpu_pool[gpus_assigned:gpus_assigned + gpus_needed]
            gpus_assigned += gpus_needed
            self.port_map[config_name] = port

            command = self._build_server_command(config_name, port)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

            print(
                f"  -> Launching '{config_name}' on port {port} with GPUs {gpu_ids}..."
            )

            p = subprocess.Popen(command, env=env)
            self.processes[config_name] = p

        for config_name, process in self.processes.items():
            self._wait_for_server_ready(process, self.port_map[config_name])

    def teardown_current_batch(self):
        if not self.processes:
            return
        print(
            f"\n---Terminating servers for Batch {self.current_batch_idx} ---")
        for name, p in self.processes.items():
            if p.poll() is None:
                print(
                    f"  -> Terminating '{name}' on port {self.port_map.get(name)}"
                )
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
        self.processes.clear()
        self.port_map.clear()

    @staticmethod
    def _build_server_command(config_name: str, port: int) -> List[str]:
        """Builds the command-line argument list to launch the server."""
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        config = VLLM_CONFIGS[config_name]

        for key, value in config.items():
            arg_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(arg_name)
            elif isinstance(value, dict):
                command.extend([arg_name, json.dumps(value)])
            else:
                command.extend([arg_name, str(value)])

        command.extend(["--port", str(port)])
        command.append("--disable-log-requests")
        return command

    @staticmethod
    def _wait_for_server_ready(process: subprocess.Popen, port: int):
        url = f"http://localhost:{port}/health"
        start_time = time.time()
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server on port {port} terminated unexpectedly. "
                    f"Return code: {process.returncode}")

            try:
                if requests.get(url, timeout=5).status_code == 200:
                    print(f"Server on port {port} is ready.")
                    return
            except requests.exceptions.RequestException:
                pass

            if time.time() - start_time > 3600:
                raise TimeoutError(f"Server on port {port} failed to start.")
            time.sleep(5)


batch_manager = BatchServerManager()


def pytest_sessionstart(session):
    session.config.vllm_batches = _schedule_configs()


def pytest_generate_tests(metafunc):
    """
    Generates one test per BATCH, not per configuration.
    This allows a single test function to run all configs in a batch in parallel.
    """
    # This will match the new test functions, e.g., test_batch_json_mode
    if "batch_spec" in metafunc.fixturenames:
        batches = metafunc.config.vllm_batches
        all_specs = []
        all_ids = []

        task_map = {
            "test_batch_performance": PERFORMANCE_TASKS,
            "test_batch_accuracy": ACCURACY_TASKS,
            "test_batch_json_mode": JSON_MODE_TASKS,
        }
        test_func_name = metafunc.function.__name__
        
        # Only generate tests if the function name is one we're refactoring
        if test_func_name in task_map:
            tasks_to_run = task_map[test_func_name]
            for batch_idx, configs in enumerate(batches):
                for task_name, task_obj in tasks_to_run.items():
                    all_specs.append({
                        "batch_idx": batch_idx,
                        "configs": configs,
                        "task_name": task_name,
                        "task_obj": task_obj,
                    })
                    all_ids.append(f"b{batch_idx}-{task_name}")

        metafunc.parametrize("batch_spec",
                             all_specs,
                             ids=all_ids,
                             indirect=True)


def pytest_collection_modifyitems(session, config, items):
    """Sorts tests by their batch index to ensure sequential batch execution."""
    # This logic is updated to look for 'batch_spec'
    batch_items = [i for i in items if "batch_spec" in getattr(i, "callspec", {}).params]
    
    if not batch_items:
        return

    batch_items.sort(
        key=lambda item: item.callspec.params["batch_spec"]["batch_idx"])
    
    non_batch_items = [i for i in items if i not in batch_items]
    items[:] = batch_items + non_batch_items


@pytest.fixture(scope="module")
def batch_spec(request):
    """
    A fixture that starts all servers for a batch and provides the test
    function with the configuration and port mapping for that entire batch.
    """
    spec = request.param
    batch_idx = spec["batch_idx"]
    configs_in_batch = spec["configs"]
    
    # Start all servers for the given batch
    batch_manager.start_batch(batch_idx, configs_in_batch)
    
    # Add the complete port map to the spec object
    spec["port_map"] = batch_manager.port_map.copy()
    yield spec


def pytest_sessionfinish(session, exitstatus):
    batch_manager.teardown_current_batch()
