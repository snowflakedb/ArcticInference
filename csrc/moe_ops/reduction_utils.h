#pragma once

#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "kernel_utils.h"

namespace cg = cooperative_groups;

namespace reduce {

enum class ROpType {
    // Addition
    Add,

    // Maximum reduction
    Max,

    // Minimum reduction
    Min,
};

constexpr int max_threads = 1024;
constexpr int max_warps = max_threads / hw_warp_size;

template <ROpType Op, int warp_bound = max_warps>
__device__ __forceinline__ void block(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp, float& val);

template <ROpType Op1, ROpType Op2, int warp_bound = max_warps>
__device__ __forceinline__ void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2);

template <ROpType Op1, ROpType Op2, ROpType Op3, int warp_bound = max_warps>
__device__ __forceinline__ void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3);

template <ROpType Op1, ROpType Op2, ROpType Op3, ROpType Op4, int warp_bound = max_warps>
__device__ __forceinline__ void block(cg::thread_block& tb,
                       cg::thread_block_tile<hw_warp_size>& warp,
                       float& val1,
                       float& val2,
                       float& val3,
                       float& val4);


template <ROpType Op, typename T>
__device__ __forceinline__ T element(const T lhs, const T rhs);

template <ROpType OType, typename T = float>
__device__ __forceinline__ T init();


__device__ __forceinline__ int _warp_rank()
{
    const int thread_rank =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    return thread_rank / hw_warp_size;
}

/* Float element reduce implementations */
template <>
__device__ __forceinline__ float element<ROpType::Add>(const float lhs, const float rhs)
{
    return lhs + rhs;
}

template <>
__device__ __forceinline__ double element<ROpType::Add>(const double lhs, const double rhs)
{
    return lhs + rhs;
}

template <>
__device__ __forceinline__ float element<ROpType::Max>(const float lhs, const float rhs)
{
    return fmaxf(lhs, rhs);
}

template <>
__device__ __forceinline__ float element<ROpType::Min>(const float lhs, const float rhs)
{
    return fminf(lhs, rhs);
}

/* __half element reduce implementation */
template <>
__device__ __forceinline__ __half element<ROpType::Add>(const __half lhs, const __half rhs)
{
    return lhs + rhs;
}

template <>
__device__ __forceinline__ __half element<ROpType::Max>(const __half lhs, const __half rhs)
{
#if __CUDA_ARCH__ >= 800
    // Intrinsic limited to Ampere + newer
    return __hmax(lhs, rhs);
#else
    return (lhs > rhs) ? lhs : rhs;
#endif
}

#ifdef BF16_AVAILABLE
template <>
__device__ __forceinline__ __nv_bfloat16 element<ROpType::Max>(const __nv_bfloat16 lhs, const __nv_bfloat16 rhs)
{
#if __CUDA_ARCH__ >= 800
    // Intrinsic limited to Ampere + newer
    return __hmax(lhs, rhs);
#else
    return (lhs > rhs) ? lhs : rhs;
#endif
}
#endif

template <>
__device__ __forceinline__ __half element<ROpType::Min>(const __half lhs, const __half rhs)
{
#if __CUDA_ARCH__ >= 800
    // Intrinsic limited to Ampere + newer
    return __hmin(lhs, rhs);
#else
    return (lhs < rhs) ? lhs : rhs;
#endif
}

/* __half2 element reduce implementation */
template <>
__device__ __forceinline__ __half2 element<ROpType::Add>(const __half2 lhs, const __half2 rhs)
{
    return lhs + rhs;
}

template <>
__device__ __forceinline__ __half2 element<ROpType::Max>(const __half2 lhs, const __half2 rhs)
{
#if __CUDA_ARCH__ >= 800
    return __hmax2(lhs, rhs);
#else
    __half2 ret_val;
    ret_val.x = (lhs.x > rhs.x) ? lhs.x : rhs.x;
    ret_val.y = (lhs.y > rhs.y) ? lhs.y : rhs.y;
    return ret_val;
#endif
}

#ifdef BF16_AVAILABLE
template <>
__device__ __forceinline__ __nv_bfloat162 element<ROpType::Max>(const __nv_bfloat162 lhs, const __nv_bfloat162 rhs)
{
#if __CUDA_ARCH__ >= 800
    return __hmax2(lhs, rhs);
#else
    __nv_bfloat162 ret_val;
    ret_val.x = (lhs.x > rhs.x) ? lhs.x : rhs.x;
    ret_val.y = (lhs.y > rhs.y) ? lhs.y : rhs.y;
    return ret_val;
#endif
}
#endif

template <>
__device__ __forceinline__ __half2 element<ROpType::Min>(const __half2 lhs, const __half2 rhs)
{
#if __CUDA_ARCH__ >= 800
    return __hmin2(lhs, rhs);
#else
    __half2 ret_val;
    ret_val.x = (lhs.x < rhs.x) ? lhs.x : rhs.x;
    ret_val.y = (lhs.y < rhs.y) ? lhs.y : rhs.y;
    return ret_val;
#endif
}

template <>
__device__ __forceinline__ int32_t element<ROpType::Add>(const int32_t lhs, const int32_t rhs)
{
    return lhs + rhs;
}

template <>
__device__ __forceinline__ int32_t element<ROpType::Max>(const int32_t lhs, const int32_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
__device__ __forceinline__ int32_t element<ROpType::Min>(const int32_t lhs, const int32_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

template <>
__device__ __forceinline__ uint32_t element<ROpType::Add>(const uint32_t lhs, const uint32_t rhs)
{
    return lhs + rhs;
}

template <>
__device__ __forceinline__ uint32_t element<ROpType::Max>(const uint32_t lhs, const uint32_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
__device__ __forceinline__ uint32_t element<ROpType::Min>(const uint32_t lhs, const uint32_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

template <>
__device__ __forceinline__ int64_t element<ROpType::Add>(const int64_t lhs, const int64_t rhs)
{
    return lhs + rhs;
}

template <>
__device__ __forceinline__ int64_t element<ROpType::Max>(const int64_t lhs, const int64_t rhs)
{
    return (lhs > rhs) ? lhs : rhs;
}

template <>
__device__ __forceinline__ int64_t element<ROpType::Min>(const int64_t lhs, const int64_t rhs)
{
    return (lhs < rhs) ? lhs : rhs;
}

/*
Reduction initialization primitives
*/
template <>
__device__ __forceinline__ float init<ROpType::Add>()
{
    return 0.0f;
}
template <>
__device__ __forceinline__ double init<ROpType::Add>()
{
    return (double)0.0f;
}

template <>
__device__ __forceinline__ float init<ROpType::Min>()
{
    // Positive infinity
    return INFINITY;
}

template <>
__device__ __forceinline__ float init<ROpType::Max>()
{
    // Negative infinity
    return -INFINITY;
}

template <>
__device__ __forceinline__ __half init<ROpType::Add>()
{
    constexpr __half_raw zero = {0x0000};
    return __half(zero);
}

template <>
__device__ __forceinline__ __half init<ROpType::Min>()
{
    constexpr __half_raw inf = {0x7C00};
    return __half(inf);
}

template <>
__device__ __forceinline__ __half init<ROpType::Max>()
{
    constexpr __half_raw neg_inf = {0xFC00};
    return __half(neg_inf);
}

#ifdef BF16_AVAILABLE
template <>
__device__ __forceinline__ __nv_bfloat16 init<ROpType::Max>()
{
    constexpr __nv_bfloat16_raw neg_inf = {0xFF80};
    return __nv_bfloat16(neg_inf);
}
#endif

template <>
__device__ __forceinline__ __half2 init<ROpType::Add>()
{
#ifdef __HIP_PLATFORM_AMD__
    return __half2{_Float16_2{0x0000, 0x0000}};
#else
    constexpr __half2_raw zero = {0x0000, 0x0000};
    return __half2(zero);
#endif
}

template <>
__device__ __forceinline__ __half2 init<ROpType::Min>()
{
#ifdef __HIP_PLATFORM_AMD__
    return __half2{_Float16_2{0x7C00, 0x7C00}};
#else
    constexpr __half2_raw inf = {0x7C00, 0x7C00};
    return __half2(inf);
#endif
}

template <>
__device__ __forceinline__ __half2 init<ROpType::Max>()
{
#ifdef __HIP_PLATFORM_AMD__
    return __half2{_Float16_2{0xFC00, 0xFC00}};
#else
    constexpr __half2_raw neg_inf = {0xFC00, 0xFC00};
    return __half2(neg_inf);
#endif
}

template <>
__device__ __forceinline__ int32_t init<ROpType::Add>()
{
    return 0;
}

template <>
__device__ __forceinline__ int32_t init<ROpType::Min>()
{
    return 0x7FFFFFFF;
}

template <>
__device__ __forceinline__ int32_t init<ROpType::Max>()
{
    return 0x80000000;
}

template <>
__device__ __forceinline__ uint32_t init<ROpType::Add>()
{
    return 0;
}

template <>
__device__ __forceinline__ uint32_t init<ROpType::Min>()
{
    return 0xFFFFFFFF;
}

template <>
__device__ __forceinline__ uint32_t init<ROpType::Max>()
{
    return 0;
}

template <>
__device__ __forceinline__ int64_t init<ROpType::Add>()
{
    return 0;
}

template <>
__device__ __forceinline__ int64_t init<ROpType::Min>()
{
    return 0x7FFFFFFFFFFFFFFF;
}

template <>
__device__ __forceinline__ int64_t init<ROpType::Max>()
{
    return 0x8000000000000000;
}

template <>
__device__ __forceinline__ uint64_t init<ROpType::Add>()
{
    return 0;
}

template <>
__device__ __forceinline__ uint64_t init<ROpType::Min>()
{
    return 0xFFFFFFFFFFFFFFFF;
}

template <>
__device__ __forceinline__ uint64_t init<ROpType::Max>()
{
    return 0;
}

template <ROpType Op, typename T>
__device__ __forceinline__ void init(T* data)
{
    data[0] = init<Op, T>();
}


template <typename T, ROpType Op, int reduce_width = hw_warp_size>
__device__ __forceinline__ void _warp(cg::thread_block_tile<hw_warp_size>& warp, T* data)
{
#pragma unroll
    for (int i = 1; i < reduce_width; i *= 2) {
        data[0] = element<Op>(data[0], warp.shfl_xor(data[0], i));
    }
}

template <typename T, int total_warps, ROpType Op>
__device__ __forceinline__ void _block(cg::thread_block& tb,
                        cg::thread_block_tile<hw_warp_size>& warp_arg,
                        T* data)
{
    constexpr int bytes = sizeof(T);
    __shared__ T reduce_buffer[max_warps];

#ifdef __HIP_PLATFORM_AMD__
    const int total_threads = blockDim.x * blockDim.y * blockDim.z;
    const int running_warps = total_threads / hw_warp_size;
#else
    const int running_warps = warp_arg.meta_group_size();
#endif

    // Always perform warp-scope reduction
    _warp<T, Op>(warp_arg, data);

    // If max_warps == 1 let's skip the runtime check
    if (total_warps != 1) {
        if (warp_arg.thread_rank() == 0) {
            mem_access::store_shared<bytes>(reduce_buffer + _warp_rank(), data);
        }

        // Synchronization inside block-uniform conditional is safe
        tb.sync();

        if (_warp_rank() == 0) {
            if (warp_arg.thread_rank() < running_warps) {
                mem_access::load_shared<bytes>(
                        data, reduce_buffer + warp_arg.thread_rank());
            } else {
                init<Op>(data);
            }

            _warp<T, Op, total_warps>(warp_arg, data);

            mem_access::store_shared<bytes>(reduce_buffer + warp_arg.thread_rank(),
                                                data);
        }

        // Synchronization inside block-uniform conditional is safe
        tb.sync();

        mem_access::load_shared<bytes>(data, reduce_buffer + _warp_rank());
    }
}

template <ROpType Op, int warp_bound>
__device__ __forceinline__ void block(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp, float& val)
{
    _block<float, warp_bound, Op>(tb, warp, &val);
}


__align__(8) struct IdxReduceResult {
    /*
    NOTE: ORDERING MATTERS HERE! The idx is the least significant set of bits
    and the val is the most significant. Changing the order of this declaration
    will break the code.
    */
    int idx;
    float val;
};

template <ROpType Op, int warpBound>
__device__ __forceinline__ IdxReduceResult
idx_reduce(cg::thread_block& tb, cg::thread_block_tile<hw_warp_size>& warp, float val, int idx)
{
    IdxReduceResult res = {idx, val};

    // Clear out the nan. This shouldn't be an issue for our initial applications
    if (isnan(val)) res.val = init<Op>();

    // Can do float compares as integers. By packing the index into the lower bits
    // we can just do a single int64 rather than a branch, compare, and select.
    // One side benefit of this is that it is by nature a stable algorithm and
    // will always bias ties to the higher index.
    int64_t* res_as_int = reinterpret_cast<int64_t*>(&res);

    // The way floating point compare works is normally to perform a sign comparison
    // and if they match, then do a comparison of the rest of the bits as unsigned
    // integers. Since we are bundling these, that means for negative values we need
    // to reverse the sort order, which we can do with an XOR.
    if (val < 0) { *res_as_int ^= 0x7fffffff00000000; }

    _block<int64_t, warpBound, Op>(tb, warp, res_as_int);

    // Sign bit is preserved, so we can check if we need to invert the mantissa back
    if (res.val < 0) { *res_as_int ^= 0x7fffffff00000000; }

    return res;
}

}  // namespace reduce
