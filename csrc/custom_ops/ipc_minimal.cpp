#include "ipc_minimal.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>

extern "C" void lamport_fill_neg_zero(void* ptr, size_t nbytes, cudaStream_t stream); // from fill_neg_zero.cu

static inline void enable_peer_access_all_visible(int my_dev) {
  int ndev = 0;
  TORCH_CUDA_CHECK(cudaGetDeviceCount(&ndev));
  for (int d = 0; d < ndev; ++d) {
    if (d == my_dev) continue;
    int can = 0;
    cudaDeviceCanAccessPeer(&can, my_dev, d);
    if (can) {
      auto st = cudaDeviceEnablePeerAccess(d, 0);
      if (st != cudaSuccess && st != cudaErrorPeerAccessAlreadyEnabled) {
        TORCH_CUDA_CHECK(st);
      }
    }
  }
}

IpcTripleWorkspace build_ipc_triple_workspace(
    c10::intrusive_ptr<c10d::ProcessGroup> pg, int64_t capacity_elems)
{
  TORCH_CHECK(capacity_elems > 0, "capacity_elems must be > 0");
  IpcTripleWorkspace ws{};
  ws.world_size = pg->getSize();
  ws.rank = pg->getRank();

  int dev = -1;
  TORCH_CUDA_CHECK(cudaGetDevice(&dev));
  enable_peer_access_all_visible(dev);

  // Worst-case granularity (float4)
  size_t max_tot_access = (static_cast<size_t>(capacity_elems) + 4 - 1) / 4;
  ws.comm_size_bytes = static_cast<size_t>(ws.world_size) * max_tot_access * sizeof(float4);

  // Local allocations
  TORCH_CUDA_CHECK(cudaMalloc(&ws.data_triple, 3 * ws.comm_size_bytes));
  TORCH_CUDA_CHECK(cudaMalloc(&ws.ctrl_dev,    8 * sizeof(int)));
  int ctrl_host[8] = {};
  ctrl_host[0] = 0; // counter
  ctrl_host[2] = 0; // flag (phase)
  ctrl_host[3] = static_cast<int>(ws.comm_size_bytes); // comm_size_bytes per phase
  ctrl_host[4] = 0; // clear_size
  TORCH_CUDA_CHECK(cudaMemcpy(ws.ctrl_dev, ctrl_host, sizeof(ctrl_host), cudaMemcpyHostToDevice));

  // Prefill sentinels (-0.0f) for all 3 phases
  lamport_fill_neg_zero(ws.data_triple, 3 * ws.comm_size_bytes, at::cuda::getCurrentCUDAStream());

  // Exchange CUDA IPC handles
  cudaIpcMemHandle_t local_handle{};
  TORCH_CUDA_CHECK(cudaIpcGetMemHandle(&local_handle, ws.data_triple));

  auto cpu_b = torch::TensorOptions().dtype(torch::kByte).device(torch::kCPU);
  at::Tensor send = torch::empty((long)sizeof(cudaIpcMemHandle_t), cpu_b);
  std::memcpy(send.data_ptr(), &local_handle, sizeof(cudaIpcMemHandle_t));

  std::vector<at::Tensor> recvs(ws.world_size);
  for (int i = 0; i < ws.world_size; ++i) {
    recvs[i] = torch::empty((long)sizeof(cudaIpcMemHandle_t), cpu_b);
  }
  c10d::AllgatherOptions ag;
  pg->allgather(recvs, send, ag)->wait();

  // Open remote buffers
  ws.opened_ptrs.resize(ws.world_size, nullptr);
  ws.is_self.resize(ws.world_size, 0);
  for (int r = 0; r < ws.world_size; ++r) {
    cudaIpcMemHandle_t h{};
    std::memcpy(&h, recvs[r].data_ptr(), sizeof(cudaIpcMemHandle_t));
    if (r == ws.rank) {
      ws.opened_ptrs[r] = ws.data_triple; // self
      ws.is_self[r] = 1;
    } else {
      void* p = nullptr;
      TORCH_CUDA_CHECK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
      ws.opened_ptrs[r] = p;
      ws.is_self[r] = 0;
    }
  }

  // Build device pointer table (void**)
  const int nptrs = 3 * ws.world_size + 1;
  std::vector<void*> host_ptrs(nptrs, nullptr);
  for (int r = 0; r < ws.world_size; ++r) {
    host_ptrs[2 * ws.world_size + r] = ws.opened_ptrs[r]; // triple-buffer base
  }
  host_ptrs[3 * ws.world_size] = ws.ctrl_dev;

  auto dev_b = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA, dev);
  ws.device_ptr_table = torch::empty((long)(nptrs * sizeof(void*)), dev_b);

  TORCH_CUDA_CHECK(cudaMemcpyAsync(
      ws.device_ptr_table.data_ptr(), host_ptrs.data(),
      nptrs * sizeof(void*),
      cudaMemcpyHostToDevice, at::cuda::getCurrentCUDAStream()));
  TORCH_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

  return ws;
}

