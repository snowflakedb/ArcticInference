import json
import pathlib
import yaml

import benchmark_utils
from .benchmark_utils import get_benchmark_summary



def pytest_addoption(parser):
    parser.addoption("--benchmark-result-dir", type=pathlib.Path)
    parser.addoption("--benchmark-config", type=pathlib.Path)

def pytest_configure(config):
    path = config.getoption("--benchmark-config")
    if not path:
        return

    with open(path, "r") as f:
        overrides = yaml.safe_load(f)

    for key, value in overrides.items():
        config_dict_name = key.upper()
        if config_dict_name in benchmark_utils.__dict__:
            if config_dict_name.endswith('_TASKS'):
                task_dict = {}
                for task_name, task_data in value.items():
                    task_dict[task_name] = benchmark_utils.BenchmarkTask(
                        config=task_data['config'],
                        metrics=task_data['metrics']
                    )
                benchmark_utils.__dict__[config_dict_name] = task_dict
            else:
                benchmark_utils.__dict__[config_dict_name] = value

def pytest_generate_tests(metafunc):
    """Dynamic parametrization that runs after pytest_configure."""
    if "vllm_server" in metafunc.fixturenames:
        metafunc.parametrize("vllm_server", list(benchmark_utils.VLLM_CONFIGS.keys()), indirect=True)
    
    if "task_name" in metafunc.fixturenames:
        if "test_performance" in metafunc.function.__name__:
            metafunc.parametrize("task_name", list(benchmark_utils.PERFORMANCE_TASKS.keys()))
        elif "test_accuracy" in metafunc.function.__name__:
            metafunc.parametrize("task_name", list(benchmark_utils.ACCURACY_TASKS.keys()))
        elif "test_json_mode" in metafunc.function.__name__:
            metafunc.parametrize("task_name", list(benchmark_utils.JSON_MODE_TASKS.keys()))

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Add benchmark summary to pytest's terminal summary, and save it to a file
    if a benchmark result directory is specified.
    """
    summary = get_benchmark_summary()

    if summary.empty:
        return

    # Print the summary to the terminal
    terminalreporter.write_sep("=", "Final Benchmark Summary")
    terminalreporter.write_line(summary.to_string())

    # Save the summary to a file if a benchmark result directory is specified
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
