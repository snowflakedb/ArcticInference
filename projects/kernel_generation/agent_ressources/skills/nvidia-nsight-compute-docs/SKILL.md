---
name: nvidia-nsight-compute-docs
description: Use local NVIDIA Nsight Compute documentation for CUDA kernel profiling tasks. Trigger when Codex needs to choose or explain ncu/Nsight Compute CLI commands, collect profile reports, select metrics or sections, reduce profiling overhead, interpret occupancy/memory/SM/Tensor Core metrics, compare baselines, troubleshoot profiling errors, or guide use of the Nsight Compute UI or profiling workflow.
---

# NVIDIA Nsight Compute Docs

Use this skill to ground CUDA kernel profiling advice in the local Nsight Compute docs bundled under `references/`.

## Workflow

1. Identify whether the task is about CLI usage, profiling methodology, UI/report inspection, or metric interpretation.
2. Search the relevant reference file before giving exact flags, metric names, section names, replay behavior, permissions guidance, or UI steps.
3. Prefer minimal, reproducible `ncu` commands that save a report with `-o` and scope collection to the needed kernels or launches.
4. For kernel optimization work, connect findings back to CUDA code changes: occupancy, memory throughput, instruction mix, Tensor Core use, warp stalls, launch configuration, and replay/pass overhead.
5. Call out profiling caveats that can invalidate results, such as first-run warmup, replay requirements, kernel filtering mistakes, missing permissions, unsupported metrics, or profiling too many sections at once.

## References

- `references/nsight_compute_cli.md`: Read for `ncu` command-line workflows, launch/attach modes, output files, filters, sections, metric collection, imports/exports, response files, rules, and CLI troubleshooting.
- `references/nsight_compute_profiling_guide.md`: Read for profiling concepts, overhead, replay modes, metric groups, hardware model, PM sampling, occupancy, source correlation, roofline, baselines, rules, and interpreting performance data.
- `references/nsight_compute_ui.md`: Read for interactive profiling, opening reports, report pages, metric details, source view, baselines, projects, remote connections, and UI-specific workflows.

## Search Cues

Use `rg` inside `references/` instead of loading whole files when possible:

```bash
rg -n "kernel.*filter|launch-skip|launch-count|section|set|metrics|replay|baseline|roofline|occupancy|sampling|permission|ERR_NVGPUCTRPERM" references/
```

For exact metric or section availability, search the local docs first, then verify with the installed tool when available:

```bash
ncu --query-metrics
ncu --list-sets
ncu --list-sections
ncu --help
```

## Command Patterns

Use these as starting points, adapting filters and sections to the task:

```bash
ncu -o profile --set basic ./app args
ncu -o profile --kernel-name regex:my_kernel --launch-count 1 --set full ./app args
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis -o profile ./app args
ncu --import profile.ncu-rep --page details
ncu --csv --import profile.ncu-rep --page raw
```

Keep profiling commands narrow during optimization loops. Start with cheaper sets or targeted sections, then collect deeper metrics only after identifying a bottleneck hypothesis.
