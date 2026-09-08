---
name: nvidia-torch-cuda-load-inline
description: Use PyTorch cpp_extension load/load_inline for raw NVIDIA CUDA kernels inside Kernelguy solutions.
---

# NVIDIA Torch CUDA `load_inline` Workflow

Use this skill on NVIDIA/CUDA Kernelguy runs when implementing a PyTorch-integrated custom CUDA kernel from `solution/solution.py`.

## Preferred integration path

For serious NVIDIA implementations, write raw CUDA/C++ source under `solution/` and compile it from Python with PyTorch's C++ extension loader:

```text
solution/
  solution.py              # builds/loads extension and exposes class Solution
  kernels/
    kernel.cu              # raw CUDA kernels and launch wrappers
    kernel.cuh             # optional helpers/layout definitions
```

Use one of these from `solution/solution.py`:

- `torch.utils.cpp_extension.load_inline` for compact one-file experiments or generated source strings.
- `torch.utils.cpp_extension.load` for multi-file source trees such as `solution/kernels/*.cu` and `*.cuh`.

The harness validates the CUDA toolchain at startup on Linux NVIDIA runs, including a minimal `load_inline` compile/run check. If AVO starts successfully on NVIDIA, the agent sandbox should be able to build a PyTorch CUDA extension.

## Custom-kernel contract

The benchmarked `Solution.forward` must dispatch a custom kernel implementation, not a vendor/template/DSL implementation.

Allowed for the submitted implementation:

- Hand-written CUDA kernels.
- C++ launch wrappers and PyTorch tensor binding glue.
- Inline PTX inside the hand-written kernel when CUDA C++ cannot express the needed instruction shape or schedule.

Allowed only for throwaway measurement/profiling code, not for the benchmarked operation:

- cuBLAS, cuDNN, CUTLASS, Triton, torch library ops, or other prebuilt/template/DSL kernels.
- Vendor/framework source-tree spelunking as a substitute for the mounted architecture docs.

If a non-raw framework attempt underperforms, stop using that framework as the implementation path and manually implement the corresponding mechanism in CUDA/PTX.

## Minimal `load_inline` shape

Keep extension names stable but source-sensitive enough to avoid stale builds while iterating. Prefer passing contiguous tensors and validating dtypes/shapes in Python before dispatch.

```python
from torch.utils.cpp_extension import load_inline

_CPP_SRC = r"""
#include <torch/extension.h>

void launch_kernel(torch::Tensor out, torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_kernel", &launch_kernel, "custom CUDA kernel launch");
}
"""

_CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void kernel(float* out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];
}

void launch_kernel(torch::Tensor out, torch::Tensor x) {
    int n = out.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel<<<blocks, threads>>>(out.data_ptr<float>(), x.data_ptr<float>(), n);
}
"""

_ext = load_inline(
    name="kg_custom_kernel_ext",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=None,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
```

## When to use `load` instead

Prefer `torch.utils.cpp_extension.load` over `load_inline` when code becomes multi-file, includes headers, or needs more maintainable kernel organization. Place files under `solution/kernels/` so they are versioned with the candidate and available inside the sandbox.

## NVRTC caveat

`torch.cuda._compile_kernel()` uses NVRTC and is not equivalent to `torch.utils.cpp_extension.load_inline`/`load`. NVRTC often lacks normal host C/C++ standard headers, so failures on includes like `<stdint.h>` are expected NVRTC behavior, not proof that CUDA is broken.

Use `load_inline`, `load`, or standalone `nvcc` whenever you need CUDA headers, standard headers, C++ host glue, or multi-file code.

## Inspect generated code

Before concluding the compiler emitted the intended instructions, inspect PTX/SASS with the available CUDA tools:

- `cuobjdump`
- `nvdisasm`
- `ncu` for profiling

Escalate to inline PTX when generated code does not express the instruction shape or schedule needed, especially for tensor-core MMA/WGMMA/tcgen05, TMA, or barrier primitives.
