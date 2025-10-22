#pragma once
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <cuda_runtime.h>
#include <vector>

// Holds everything needed by the Lamport kernel.
// Lifetime: the owner (our Torch Workspace class) must close remote handles and free device memory.
struct IpcTripleWorkspace {
  void* data_triple = nullptr;    // device memory: triple-buffer base for *this* rank
  void* ctrl_dev    = nullptr;    // device memory: control ints
  std::vector<void*> opened_ptrs; // per-rank pointers (self + remotes)
  std::vector<uint8_t> is_self;   // 1 if the entry is our own buffer
  at::Tensor device_ptr_table;    // device buffer that stores void** pointer table expected by kernel
  size_t comm_size_bytes = 0;     // per-phase bytes
  int world_size = 0;
  int rank = -1;
};

// Build triple buffers + ctrl block, exchange IPC handles via ProcessGroup (NCCL),
// open remote buffers, and build the device pointer table.
// capacity_elems: maximum number of elements passed to all_reduce in any call on this process.
IpcTripleWorkspace build_ipc_triple_workspace(
    c10::intrusive_ptr<c10d::ProcessGroup> pg, int64_t capacity_elems);

// Prefill helper implemented in fill_neg_zero(s).cu – declare with C linkage to match the definition.
extern "C" void lamport_fill_neg_zero(void* ptr, size_t nbytes, cudaStream_t stream);
