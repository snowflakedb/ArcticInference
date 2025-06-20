import json
import pathlib

from .benchmark_utils import get_benchmark_summary


def pytest_addoption(parser):
    parser.addoption("--benchmark-result-dir", type=pathlib.Path)


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
