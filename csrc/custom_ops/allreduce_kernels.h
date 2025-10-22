#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace lamport_ar {

using half = __half;

enum class DType : int { kHalf = 0, kBFloat16 = 1, kFloat = 2 };

struct AllReduceParams {
    int nranks;         // world size
    int rank;           // my rank [0..nranks)
    DType dtype;        // element type
    int size;           // total elements (tokens * hidden_dim)
    int hidden_dim;     // last dimension in elements
    void** workspace;   // device ptr to pointer table (see layout below)
    void* allreduce_in; // input tensor ptr
    void* allreduce_out;// output tensor ptr
    cudaStream_t stream;
    bool trigger_completion_at_end; // kept for parity, not used
};

/*
Workspace pointer array layout (device memory, built by ipc_minimal.cpp):
  - For oneshot Lamport, we use a triple-buffer per rank and a shared control block.
  - workspace[2*nranks + r] : base pointer to rank r triple-buffer (uint8_t* of size 3*comm_size_bytes)
  - workspace[3*nranks]     : pointer to control ints (int*), where:
        ctrl[0] : counter
        ctrl[2] : flag (phase 0..2)
        ctrl[3] : comm_size_bytes (per-phase size)
        ctrl[4] : clear_size (elements) from last run
*/

void lamport_allreduce_run(AllReduceParams const& params);

} // namespace lamport_ar

