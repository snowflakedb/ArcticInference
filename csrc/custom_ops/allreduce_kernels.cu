#include "allreduce_kernels.h"
#include <cuda.h>
#include <type_traits>

namespace lamport_ar {

template <typename T> struct ElemsPerAccess;
template <> struct ElemsPerAccess<half>         { static constexpr int value = 8; };
template <> struct ElemsPerAccess<__nv_bfloat16>{ static constexpr int value = 8; };
template <> struct ElemsPerAccess<float>        { static constexpr int value = 4; };

template <typename T> __device__ __forceinline__ int kElemsPerAccess() { return ElemsPerAccess<T>::value; }

__device__ __forceinline__ bool is_neg_zero(float v) {
    return __float_as_uint(v) == 0x80000000u;
}
__device__ __forceinline__ bool is_neg_zero(float4 v) {
    return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) || is_neg_zero(v.w);
}
__device__ __forceinline__ float4 neg_zero_vec() {
    float4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i) reinterpret_cast<unsigned*>(&v)[i] = 0x80000000u;
    return v;
}
__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
    float4 val;
    asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(addr));
    return val;
}

template <typename DType, typename Packed>
__device__ __forceinline__ Packed add128(Packed const& a, Packed const& b) {
    static constexpr int kN = sizeof(Packed) / sizeof(DType);
    Packed c;
#pragma unroll
    for (int i = 0; i < kN; ++i) {
        reinterpret_cast<DType*>(&c)[i] =
            reinterpret_cast<DType const*>(&a)[i] + reinterpret_cast<DType const*>(&b)[i];
    }
    return c;
}

template <typename DType, int NRanks>
__device__ __forceinline__ float4 allreduce_sum(float4* vals) {
    float4 acc = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
        acc = add128<DType>(acc, vals[r]);
    }
    return acc;
}

template <int NRanks>
struct LamportComm {
    __device__ __forceinline__ LamportComm(void** workspace, int rank) {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr    = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
        clear_ptr   = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
        int comm_sz = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
        flag_value  = *flag_ptr;

        int data_phase   = flag_value % 3;
        int clear_phase  = (flag_value + 2) % 3;

        for (int r = 0; r < NRanks; ++r) {
            data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_phase  * comm_sz;
        }
        clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_phase * comm_sz;
        clear_size = *clear_ptr;

        __syncthreads();
        if (threadIdx.x == 0) { atomicAdd(counter_ptr, 1); }
    }

    __device__ __forceinline__ void update(int new_clear_size) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (*reinterpret_cast<volatile int*>(counter_ptr) != gridDim.x) { }
            *flag_ptr  = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int* counter_ptr;
    int* flag_ptr;
    int* clear_ptr;
    uint8_t* data_bufs[NRanks];
    uint8_t* clear_buf;
    int clear_size;
    int flag_value;
};

template <typename DType>
struct IndexHelper {
    __device__ __forceinline__ IndexHelper(AllReduceParams const& p) {
        token_id            = blockIdx.x;
        access_id_in_token  = threadIdx.x;
        token_stride        = gridDim.x;
        access_stride       = gridDim.x * (p.hidden_dim / kElemsPerAccess<DType>());
        access_id           = token_id * (p.hidden_dim / kElemsPerAccess<DType>()) + access_id_in_token;
        tot_access          = p.size / kElemsPerAccess<DType>();
    }
    int token_id, access_id_in_token, token_stride;
    int access_id, access_stride, tot_access;
};

template <typename DType, int NRanks>
__global__ void lamport_oneshot_kernel(AllReduceParams params)
{
    IndexHelper<DType> ix(params);
    int access_id     = ix.access_id;
    int access_stride = ix.access_stride;
    int tot_access    = ix.tot_access;

    float4 clear_vec = neg_zero_vec();

    LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess<DType>();

    // 1) push my chunks to all ranks' current phase buffers
    for (int idx = access_id; idx < tot_access; idx += access_stride) {
        float4 v = reinterpret_cast<float4*>(params.allreduce_in)[idx];
        // sanitize -0.0 sentinels in user input
        float* f = reinterpret_cast<float*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) if (is_neg_zero(f[i])) f[i] = 0.0f;
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx] = v;
        }
    }

    // 2) clear previous phase of my buffer
    for (int idx = access_id; idx < clear_access; idx += access_stride) {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    // 3) read/accumulate/write out
    for (int idx = access_id; idx < tot_access; idx += access_stride) {
        float4 vals[NRanks];
        bool ready = false;
        while (!ready) {
            ready = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r) {
                vals[r] = ld_global_volatile(
                    &reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx]);
                ready &= !is_neg_zero(vals[r]);
            }
        }
        float4 sumv = allreduce_sum<DType, NRanks>(vals);
        reinterpret_cast<float4*>(params.allreduce_out)[idx] = sumv;
    }

    // 4) advance phase
    comm.update(params.size * params.nranks);
}

static inline int next_pow2_le(int v, int cap) {
    int t = 1;
    while ((t << 1) <= v && (t << 1) <= cap) t <<= 1;
    return t;
}

template <typename DType, int NRanks>
static void launch(AllReduceParams const& p)
{
    int tpt = p.hidden_dim / kElemsPerAccess<DType>();
    tpt = tpt > 0 ? tpt : 1;
    int block = next_pow2_le(tpt, 1024);
    int grid  = p.size / p.hidden_dim;
    grid = grid > 0 ? grid : 1;

    void* args[] = { const_cast<AllReduceParams*>(&p) };
    auto fn = lamport_oneshot_kernel<DType, NRanks>;
    cudaLaunchKernel((void*)fn, dim3(grid), dim3(block), args, 0, p.stream);
}

static void dispatch(AllReduceParams const& p)
{
    auto ranks = p.nranks;
    auto dt    = p.dtype;

#define CASE_DTYPE(DT) \
    if (ranks == 2)      { launch<DT, 2>(p);  return; } \
    else if (ranks == 4) { launch<DT, 4>(p);  return; } \
    else if (ranks == 8) { launch<DT, 8>(p);  return; } \
    else if (ranks == 16){ launch<DT,16>(p);  return; }

    switch (dt) {
    case DType::kHalf:      { CASE_DTYPE(half); break; }
    case DType::kBFloat16:  { CASE_DTYPE(__nv_bfloat16); break; }
    case DType::kFloat:     { CASE_DTYPE(float); break; }
    default: break;
    }
#undef CASE_DTYPE
}

void lamport_allreduce_run(AllReduceParams const& params)
{
    if (params.size % params.hidden_dim != 0) return;
    switch (params.dtype) {
    case DType::kHalf:
    case DType::kBFloat16:
        if (params.hidden_dim % 8) return;
        break;
    case DType::kFloat:
        if (params.hidden_dim % 4) return;
        break;
    default: return;
    }
    if (!(params.nranks == 2 || params.nranks == 4 || params.nranks == 8 || params.nranks == 16)) return;

    dispatch(params);
}

} // namespace lamport_ar

