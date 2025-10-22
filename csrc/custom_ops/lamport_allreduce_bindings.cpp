// lamport_allreduce_bindings.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>       // at::cuda::getCurrentCUDAStream
#include <c10/cuda/CUDAStream.h>         // c10::cuda::CUDAStream::stream()
#include <cuda.h>
#include <cuda_runtime.h>
#include "lamport_allreduce_kernels.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

using tensorrt_llm::kernels::ar_fusion::AllReduceFusionParams;
using tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern;

static inline nvinfer1::DataType to_nv_dtype(c10::ScalarType dtype) {
    if (dtype == at::kHalf)      return nvinfer1::DataType::kHALF;
    if (dtype == at::kBFloat16)  return nvinfer1::DataType::kBF16;
    if (dtype == at::kFloat)     return nvinfer1::DataType::kFLOAT;
    throw std::runtime_error("Unsupported dtype (use fp16/bf16/fp32)");
}

// --- CUDA IPC helpers ---
static std::pair<uint64_t, py::bytes> allocate_shared_buffer_and_handle(size_t nbytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, nbytes);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    cudaIpcMemHandle_t handle{};
    err = cudaIpcGetMemHandle(&handle, ptr);
    if (err != cudaSuccess) {
        cudaFree(ptr);
        throw std::runtime_error("cudaIpcGetMemHandle failed");
    }
    return { reinterpret_cast<uint64_t>(ptr), py::bytes(reinterpret_cast<char*>(&handle), sizeof(handle)) };
}

static uint64_t open_mem_handle(py::bytes handle_bytes) {
    std::string s = handle_bytes; // copies
    if (s.size() != sizeof(cudaIpcMemHandle_t)) throw std::runtime_error("Bad handle size");
    cudaIpcMemHandle_t handle{};
    memcpy(&handle, s.data(), sizeof(handle));
    void* remote_ptr = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&remote_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) throw std::runtime_error("cudaIpcOpenMemHandle failed");
    return reinterpret_cast<uint64_t>(remote_ptr);
}

static void close_mem_handle(uint64_t ptr) {
    cudaError_t err = cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr));
    if (err != cudaSuccess) throw std::runtime_error("cudaIpcCloseMemHandle failed");
}

static void device_free(uint64_t ptr) {
    cudaError_t err = cudaFree(reinterpret_cast<void*>(ptr));
    if (err != cudaSuccess) throw std::runtime_error("cudaFree failed");
}

static void enable_peer_access_to(int remote_dev) {
    int cur = 0; cudaGetDevice(&cur);
    if (cur == remote_dev) return;
    int can = 0; cudaDeviceCanAccessPeer(&can, cur, remote_dev);
    if (can) { cudaDeviceEnablePeerAccess(remote_dev, 0); /* ignore EALREADY */ (void)cudaGetLastError(); }
}

// --- Kernel wrapper ---
// workspace_ptrs: torch.uint64 on the same CUDA device, length >= 3*world + 1
static at::Tensor lamport_allreduce(at::Tensor input,
                                    at::Tensor workspace_ptrs,
                                    int64_t world_size,
                                    int64_t rank,
                                    bool   trigger_completion_at_end)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(workspace_ptrs.is_cuda(), "workspace_ptrs must be CUDA");
    TORCH_CHECK(workspace_ptrs.scalar_type() == at::kLong, "workspace_ptrs must be torch.long(uint64)");
    TORCH_CHECK(workspace_ptrs.is_contiguous(), "workspace_ptrs must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "Input shape must be [tokens, hidden, ...], hidden is the last dim");

    // Prepare params
    AllReduceFusionParams p{};
    p.nranks     = static_cast<int>(world_size);
    p.rank       = static_cast<int>(rank);
    p.dtype      = to_nv_dtype(input.scalar_type());
    p.size       = static_cast<int>(input.numel());
    p.hidden_dim = static_cast<int>(input.size(-1));
    p.workspace  = reinterpret_cast<void**>(workspace_ptrs.data_ptr<int64_t>());
    p.allreduce_in  = input.data_ptr();
    at::Tensor out = at::empty_like(input);
    p.allreduce_out = out.data_ptr();
    p.use_oneshot   = true;
    p.pattern       = AllReduceFusionPattern::kAllReduce;
    {
        auto s = at::cuda::getCurrentCUDAStream(input.get_device());
        p.stream = s.stream();
    }
    p.trigger_completion_at_end = trigger_completion_at_end;

    tensorrt_llm::kernels::ar_fusion::allreduce_fusion_op(p);
    return out;
}

void init_lamport_allreduce_bindings(py::module_ &m) {
    m.def("lamport_allreduce", &lamport_allreduce,
          "Lamport oneshot allreduce (kAllReduce only)",
          py::arg("input"), py::arg("workspace_ptrs"),
          py::arg("world_size"), py::arg("rank"),
          py::arg("trigger_completion_at_end") = true);

    m.def("allocate_shared_buffer_and_handle", &allocate_shared_buffer_and_handle,
          "cudaMalloc + cudaIpcGetMemHandle(size_bytes) -> (ptr_uint64, handle_bytes)");
    m.def("open_mem_handle",  &open_mem_handle,  "cudaIpcOpenMemHandle(handle_bytes) -> ptr_uint64");
    m.def("close_mem_handle", &close_mem_handle, "cudaIpcCloseMemHandle(ptr_uint64)");
    m.def("device_free",      &device_free,      "cudaFree(ptr_uint64)");
    m.def("enable_peer_access_to", &enable_peer_access_to,
          "cudaDeviceEnablePeerAccess(current->remote)");
}

