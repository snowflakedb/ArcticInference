//! Preflight that the sandboxed NVIDIA toolchain is actually usable.
//!
//! Lives in `exec` because it drives a [`Sandbox`]. The pure host-discovery half is
//! [`crate::env::cuda`], which stays free of `exec`.
//!
//! Every check runs inside the jail the agent will use, so what it validates is
//! what the agent gets: tools on `PATH`, versions, `torch` reaching CUDA, `nvcc`
//! compiling and running, `load_inline` building an extension, the disassemblers,
//! and `ncu` profiling a kernel (typically `ERR_NVGPUCTRPERM` when it cannot).

use std::time::Duration;

use crate::exec::sandbox::Sandbox;

const REQUIRED_TOOLS: &[&str] = &["nvidia-smi", "nvcc", "ncu", "cuobjdump", "nvdisasm", "python3"];

#[derive(Debug, Clone)]
pub struct CudaValidationSummary {
    pub lines: Vec<String>,
}

impl CudaValidationSummary {
    #[must_use]
    pub fn format_for_startup(&self) -> String {
        self.lines.join("\n")
    }
}

/// # Errors
///
/// Returns the first failing preflight as a human-readable report (command +
/// truncated output), covering: a required tool from `REQUIRED_TOOLS` not visible
/// inside the sandbox; `nvidia-smi`/`nvcc`/`ncu`/`cuobjdump`/`nvdisasm` version
/// queries failing; `python3`/`torch` unable to reach CUDA; `nvcc` unable to compile and run
/// a minimal program; `torch.utils.cpp_extension.load_inline` unable to build an
/// extension; `cuobjdump`/`nvdisasm` unusable; or `ncu` unable to profile a tiny
/// kernel (typically `ERR_NVGPUCTRPERM`). Also propagates a sandbox launch failure
/// from [`Sandbox::shell`].
pub async fn validate_nvidia_agent_tooling(sb: &Sandbox) -> Result<CudaValidationSummary, String> {
    let mut lines = Vec::new();
    run_required_tools_check(sb, &mut lines).await?;
    run_versions_check(sb, &mut lines).await?;
    run_torch_cuda_check(sb, &mut lines).await?;
    run_nvcc_compile_check(sb, &mut lines).await?;
    run_load_inline_check(sb, &mut lines).await?;
    run_disassembly_check(sb, &mut lines).await?;
    run_ncu_check(sb, &mut lines).await?;
    Ok(CudaValidationSummary { lines })
}

async fn run_required_tools_check(sb: &Sandbox, lines: &mut Vec<String>) -> Result<(), String> {
    let cmd = format!(
        "set -eu\nfor tool in {}; do command -v \"$tool\"; done",
        REQUIRED_TOOLS.join(" ")
    );
    let out = sb
        .shell(&cmd, Some(Duration::from_secs(30)))
        .await
        .map_err(|e| e.to_string())?;
    if out.returncode != 0 {
        return Err(format!(
            "NVIDIA startup validation failed: required tools are not all visible inside the agent sandbox.\n\
             Required tools: {}\n\
             Command: {cmd}\n\
             Output:\n{}\n\
             Fix the host CUDA/NVIDIA installation or PATH before starting a run.",
            REQUIRED_TOOLS.join(", "),
            tail(&out.stdout, 4000)
        ));
    }
    let found = out
        .stdout
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    lines.push(format!("NVIDIA tools visible in sandbox: {}", found.join(", ")));
    Ok(())
}

async fn run_versions_check(sb: &Sandbox, lines: &mut Vec<String>) -> Result<(), String> {
    let cmd = "set -eu\nnvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader\nnvcc --version | tail -n 1\nncu --version | head -n 1\ncuobjdump --version | head -n 1\nnvdisasm --version | head -n 1";
    let out = sb
        .shell(cmd, Some(Duration::from_secs(45)))
        .await
        .map_err(|e| e.to_string())?;
    if out.returncode != 0 {
        return Err(format!(
            "NVIDIA startup validation failed while querying tool versions.\nCommand: {cmd}\nOutput:\n{}",
            tail(&out.stdout, 4000)
        ));
    }
    lines.push(format!(
        "NVIDIA tool versions:\n{}",
        indent(tail(&out.stdout, 2000).trim())
    ));
    Ok(())
}

async fn run_torch_cuda_check(sb: &Sandbox, lines: &mut Vec<String>) -> Result<(), String> {
    let cmd = r"python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('torch.cuda.is_available() is false')
print('device', torch.cuda.get_device_name(0))
PY";
    let out = sb
        .shell(cmd, Some(Duration::from_mins(1)))
        .await
        .map_err(|e| e.to_string())?;
    if out.returncode != 0 {
        return Err(format!(
            "NVIDIA startup validation failed: Python/PyTorch cannot use CUDA inside the agent sandbox.\nCommand: python3 torch CUDA probe\nOutput:\n{}",
            tail(&out.stdout, 4000)
        ));
    }
    lines.push(format!(
        "PyTorch CUDA probe:\n{}",
        indent(tail(&out.stdout, 1200).trim())
    ));
    Ok(())
}

async fn run_nvcc_compile_check(sb: &Sandbox, lines: &mut Vec<String>) -> Result<(), String> {
    let cmd = r#"set -eu
cat > /tmp/kg_cuda_preflight.cu <<'CU'
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
__global__ void kg_preflight(int *out) {
    __nv_bfloat16 x = __float2bfloat16(1.0f);
    out[threadIdx.x] = (int)__bfloat162float(x) + (int)((uint32_t)threadIdx.x);
}
int main() {
    int *d = nullptr;
    int h[1] = {-1};
    cudaError_t err = cudaMalloc(&d, sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err)); return 2; }
    kg_preflight<<<1, 1>>>(d);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel: %s\n", cudaGetErrorString(err)); return 3; }
    err = cudaMemcpy(h, d, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(err)); return 4; }
    cudaFree(d);
    if (h[0] != 1) { fprintf(stderr, "bad result %d\n", h[0]); return 5; }
    puts("nvcc compile/run ok");
    return 0;
}
CU
nvcc -std=c++17 /tmp/kg_cuda_preflight.cu -o /tmp/kg_cuda_preflight
/tmp/kg_cuda_preflight"#;
    let out = sb
        .shell(cmd, Some(Duration::from_mins(2)))
        .await
        .map_err(|e| e.to_string())?;
    if out.returncode != 0 {
        return Err(format!(
            "NVIDIA startup validation failed: nvcc could not compile and run a minimal CUDA program inside the agent sandbox.\n\
             This normal nvcc path must find stdint.h, cuda_runtime.h, and cuda_bf16.h.\n\
             Command: create /tmp/kg_cuda_preflight.cu; nvcc -std=c++17 ...; run binary\n\
             Output:\n{}",
            tail(&out.stdout, 6000)
        ));
    }
    lines.push(tail(&out.stdout, 1000).trim().to_string());
    Ok(())
}

async fn run_load_inline_check(sb: &Sandbox, lines: &mut Vec<String>) -> Result<(), String> {
    let cmd = r"python3 - <<'PY'
import torch
from torch.utils.cpp_extension import load_inline
cpp = 'torch::Tensor kg_launch(torch::Tensor x);'
cuda = r'''
#include <stdint.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
__global__ void kg_inline_kernel(float *out) { out[0] = 7.0f + (float)((uint32_t)0); }
torch::Tensor kg_launch(torch::Tensor x) {
    kg_inline_kernel<<<1, 1>>>(x.data_ptr<float>());
    return x;
}
'''
mod = load_inline(name='kg_cuda_preflight_inline', cpp_sources=cpp, cuda_sources=cuda, functions=['kg_launch'], verbose=False)
out = torch.zeros(1, device='cuda')
mod.kg_launch(out)
torch.cuda.synchronize()
assert out.item() == 7.0, out
print('torch cpp_extension.load_inline ok')
PY";
    let out = sb
        .shell(cmd, Some(Duration::from_mins(3)))
        .await
        .map_err(|e| e.to_string())?;
    if out.returncode != 0 {
        return Err(format!(
            "NVIDIA startup validation failed: torch.utils.cpp_extension.load_inline could not build/run a minimal CUDA extension.\n\
             Agents need this normal workflow for custom kernels that use CUDA/system headers.\n\
             Output:\n{}",
            tail(&out.stdout, 6000)
        ));
    }
    lines.push(tail(&out.stdout, 1000).trim().to_string());
    Ok(())
}

async fn run_disassembly_check(sb: &Sandbox, lines: &mut Vec<String>) -> Result<(), String> {
    let cmd = r#"set -eu
cuobjdump --dump-ptx /tmp/kg_cuda_preflight >/tmp/kg_cuda_preflight.ptx.txt
cat > /tmp/kg_cuda_preflight_device.cu <<'CU'
extern "C" __global__ void kg_disasm_kernel(float *out) {
    out[threadIdx.x] = (float)threadIdx.x;
}
CU
nvcc -cubin /tmp/kg_cuda_preflight_device.cu -o /tmp/kg_cuda_preflight.cubin
nvdisasm /tmp/kg_cuda_preflight.cubin >/tmp/kg_cuda_preflight.sass.txt
test -s /tmp/kg_cuda_preflight.ptx.txt
test -s /tmp/kg_cuda_preflight.sass.txt
echo 'cuobjdump/nvdisasm ok'
"#;
    let out = sb
        .shell(cmd, Some(Duration::from_secs(90)))
        .await
        .map_err(|e| e.to_string())?;
    if out.returncode != 0 {
        return Err(format!(
            "NVIDIA startup validation failed: cuobjdump/nvdisasm are not usable inside the agent sandbox.\nCommand: {cmd}\nOutput:\n{}",
            tail(&out.stdout, 4000)
        ));
    }
    lines.push(tail(&out.stdout, 1000).trim().to_string());
    Ok(())
}

async fn run_ncu_check(sb: &Sandbox, lines: &mut Vec<String>) -> Result<(), String> {
    // ncu reports permission/driver problems (notably ERR_NVGPUCTRPERM) on
    // *stdout*, mixed into the report it is asked to write — so both streams have
    // to be echoed on failure. Reporting only stderr yields a blank error, and an
    // `ncu` that exits 0 while writing an empty report yields no output at all.
    let cmd = "set -eu
if ! ncu --target-processes all --set default --metrics sm__cycles_elapsed.avg --csv /tmp/kg_cuda_preflight >/tmp/kg_ncu.csv 2>/tmp/kg_ncu.err; then
  echo '--- ncu exited non-zero; stdout (report) ---'
  cat /tmp/kg_ncu.csv
  echo '--- stderr ---'
  cat /tmp/kg_ncu.err
  exit 1
fi
if ! test -s /tmp/kg_ncu.csv; then
  echo '--- ncu exited 0 but wrote an empty report; stderr ---'
  cat /tmp/kg_ncu.err
  exit 1
fi
echo 'ncu profile ok'";
    let out = sb
        .shell(cmd, Some(Duration::from_mins(3)))
        .await
        .map_err(|e| e.to_string())?;
    if out.returncode != 0 {
        return Err(format!(
            "NVIDIA startup validation failed: ncu is present but cannot profile a tiny kernel inside the agent sandbox.\n\
             Profiling is required for NVIDIA optimization runs; fix host NVIDIA profiling permissions/driver setup before starting.\n\
             ERR_NVGPUCTRPERM specifically means perf counters are admin-only: set\n\
             `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in /etc/modprobe.d/, then reload the driver.\n\
             Command: {cmd}\n\
             Output:\n{}",
            tail(&out.stdout, 6000)
        ));
    }
    lines.push(tail(&out.stdout, 1000).trim().to_string());
    Ok(())
}

fn tail(s: &str, max: usize) -> String {
    let count = s.chars().count();
    // `checked_sub(..).filter(> 0)` is exactly the old `count <= max` early
    // return, but it binds the overhang so the two uses below can't disagree
    // with the guard.
    let Some(cut) = count.checked_sub(max).filter(|&n| n > 0) else {
        return s.to_string();
    };
    let tail: String = s.chars().skip(cut).collect();
    format!("[...truncated {cut} chars...]\n{tail}")
}

fn indent(s: &str) -> String {
    s.lines().map(|line| format!("  {line}")).collect::<Vec<_>>().join("\n")
}
