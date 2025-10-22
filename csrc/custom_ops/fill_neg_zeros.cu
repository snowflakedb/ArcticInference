#include <cuda_runtime.h>
#include <stdint.h>

namespace lamport_ar_detail {

__global__ void fill_neg_zero_u32(uint32_t* base, size_t nwords)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < nwords; i += (size_t)gridDim.x * blockDim.x) {
        base[i] = 0x80000000u; // -0.0f sentinel
    }
}

inline void launch_fill_neg_zero(void* ptr, size_t nbytes, cudaStream_t stream)
{
    size_t nwords = nbytes / sizeof(uint32_t);
    int threads = 256;
    int blocks  = (int)((nwords + threads - 1) / threads);
    blocks = blocks < 1 ? 1 : blocks;
    fill_neg_zero_u32<<<blocks, threads, 0, stream>>>(reinterpret_cast<uint32_t*>(ptr), nwords);
}

} // namespace lamport_ar_detail

extern "C" void lamport_fill_neg_zero(void* ptr, size_t nbytes, cudaStream_t stream)
{
    lamport_ar_detail::launch_fill_neg_zero(ptr, nbytes, stream);
}

