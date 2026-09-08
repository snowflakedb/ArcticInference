---
name: nvidia-cuda
description: Use local NVIDIA CUDA Programming Guide documentation for CUDA C++ programming, kernel correctness, memory hierarchy, streams/events, unified memory, NVCC compilation, occupancy, cooperative groups, dynamic parallelism, CUDA graphs, inline PTX, tensor core programming, asynchronous copy/TMA, memory model, atomics, and architecture-specific feature questions. Trigger when Codex needs to write, debug, optimize, or explain CUDA kernels or CUDA host/runtime code using official local CUDA guide details.
---

# NVIDIA CUDA Programming Guide

Use this skill to ground CUDA programming, correctness, and optimization work in the local CUDA Programming Guide bundled under `references/`.

## Workflow

1. Identify the CUDA topic: kernel launch semantics, thread/block indexing, memory hierarchy, synchronization, streams/events, compiler options, occupancy, cooperative groups, graphs, unified memory, atomics, inline PTX, tensor cores, asynchronous copy, or architecture-specific features.
2. Search `references/cuda.md` before giving exact API names, semantic guarantees, limits, feature availability, compiler flags, or memory-ordering claims.
3. Prefer focused `rg` searches over loading the full guide; open only the nearby section needed for the task.
4. Distinguish CUDA programming semantics from profiling or binary-inspection tasks. Use `nvidia-nsight-compute-docs` for `ncu` metric collection/interpretation and `nvidia-binary-utilities` for PTX/SASS/cubin inspection.
5. When writing kernels, validate assumptions with the guide first, then inspect generated code or profile only if the user asks or performance/correctness depends on it.

## Reference

- `references/cuda.md`: Local reconstructed NVIDIA CUDA Programming Guide. Read for CUDA C++ syntax, runtime APIs, execution model, memory model, compiler workflow, advanced CUDA features, and technical appendices.

## Search Cues

Use these from inside this skill directory, or adjust the path if working from the repository root:

```bash
rg -n "__global__|__device__|threadIdx|blockIdx|gridDim|blockDim|shared memory|constant memory|local memory|coalesc|bank conflict|occupancy" references/cuda.md
rg -n "cudaStream|cudaEvent|default stream|synchronization|cudaDeviceSynchronize|implicit synchronization|CUDA Graph|stream capture" references/cuda.md
rg -n "__syncthreads|__syncwarp|cooperative groups|memory fence|__threadfence|atomic|memory ordering|volatile" references/cuda.md
rg -n "nvcc|--generate-code|arch=compute_|code=sm_|separate compilation|relocatable device code|ptx|cubin" references/cuda.md
rg -n "tensor core|wmma|mma|wgmma|asynchronous copy|cp.async|TMA|barrier|mbarrier|cluster|distributed shared memory" references/cuda.md
rg -n "unified memory|managed memory|cudaMallocManaged|prefetch|memory advise|page-locked|pinned|mapped memory|UVA" references/cuda.md
```

For broad navigation, inspect headings first:

```bash
rg -n "^#{1,4} " references/cuda.md
```

## Command Patterns

Verify exact flags in `references/cuda.md` before presenting final commands:

```bash
nvcc -O3 -arch=sm_90 kernel.cu -o kernel
nvcc -O3 --generate-code arch=compute_90,code=sm_90 kernel.cu -o kernel
nvcc -ptx -arch=sm_90 kernel.cu -o kernel.ptx
nvcc -cubin -arch=sm_90 kernel.cu -o kernel.cubin
nvcc -lineinfo -O3 -arch=sm_90 kernel.cu -o kernel
```

## CUDA Guidance Defaults

- Treat correctness before speed: bounds checks, launch error checks, synchronization scopes, and lifetime/ownership of host/device memory matter.
- Prefer explicit stream semantics; call out legacy default stream versus per-thread default stream when ordering depends on it.
- Be precise about synchronization scope: warp, block, cluster, device, system, stream, or host.
- Tie optimization advice to the bottleneck hypothesis: memory coalescing, shared-memory bank conflicts, occupancy/register pressure, instruction mix, latency hiding, launch overhead, or tensor-core utilization.
- Use architecture-specific features only after confirming availability and required compile targets.
