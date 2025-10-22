// kernels/lamport_allreduce_kernels.cu
#include "lamport_allreduce_kernels.h"
#include <cooperative_groups.h>
#include <array>

namespace cg = cooperative_groups;
namespace tensorrt_llm::common {
// Minimal substitutes
static inline int getSMVersion() {
    int dev = 0; cudaDeviceProp p{}; TLLM_CUDA_CHECK(cudaGetDevice(&dev));
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    return p.major * 10 + p.minor;
}
static inline int getEnvEnablePDL() {
#ifdef IGNORE_PDL_IF_UNAVAILABLE
    return 0;
#else
    const char* v = std::getenv("ENABLE_PROGRAMMATIC_STREAM_SERIALIZATION");
    return v ? 1 : 0;
#endif
}
} // namespace tensorrt_llm::common

namespace tensorrt_llm::kernels::ar_fusion {

// ----------------- Shared comm structs (unchanged) -----------------
template <int NRanks>
struct LamportComm {
    __device__ __forceinline__ LamportComm(void** workspace, int rank) {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr    = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
        clear_ptr   = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
        flag_value  = *flag_ptr;
        int comm_size  = reinterpret_cast<int*>(workspace[NRanks * 3])[3];  // bytes per slot
        clear_size = *clear_ptr;                                            // elements (DType)
        int data_offset  = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r) {
            data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0) { atomicAdd(counter_ptr, 1); }
    }
    __device__ __forceinline__ void update(int new_clear_size) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {}
            *flag_ptr  = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }
    int* counter_ptr; int* flag_ptr; int* clear_ptr;
    uint8_t* data_bufs[NRanks]; uint8_t* clear_buf;
    int clear_size; int flag_value;
};

// --------------- math helpers -----------------
template <typename DType, typename PackedType>
__device__ __forceinline__ PackedType add128(PackedType const& a, PackedType const& b) {
    constexpr int kMathCount = sizeof(PackedType) / sizeof(DType);
    PackedType c;
#pragma unroll
    for (int i = 0; i < kMathCount; ++i) {
        reinterpret_cast<DType*>(&c)[i] = reinterpret_cast<DType const*>(&a)[i]
                                        + reinterpret_cast<DType const*>(&b)[i];
    }
    return c;
}

__device__ __forceinline__ bool is_neg_zero(float v) { return *reinterpret_cast<uint32_t*>(&v) == 0x80000000; }
__device__ __forceinline__ bool is_neg_zero(float4 v) {
    return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) || is_neg_zero(v.w);
}
__device__ __forceinline__ float4 get_neg_zero() {
    float4 vec;
#pragma unroll
    for (int i = 0; i < 4; ++i) { reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000; }
    return vec;
}
__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
    float4 val;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) : "l"(addr));
    return val;
}

template <typename DType, int NRanks, bool Fp32Acc>
__device__ __forceinline__ float4 allreduce_sum(float4* vals) {
    if constexpr (Fp32Acc) {
        static_assert(!std::is_same_v<DType, float>);
        float acc_f32[ElemsPerAccess<DType>::value];
#pragma unroll
        for (int i = 0; i < ElemsPerAccess<DType>::value; ++i)
            acc_f32[i] = static_cast<float>(reinterpret_cast<DType*>(&vals[0])[i]);
#pragma unroll
        for (int r = 1; r < NRanks; ++r) {
#pragma unroll
            for (int i = 0; i < ElemsPerAccess<DType>::value; ++i) {
                acc_f32[i] += static_cast<float>(reinterpret_cast<DType*>(&vals[r])[i]);
            }
        }
        float4 acc;
#pragma unroll
        for (int i = 0; i < ElemsPerAccess<DType>::value; ++i)
            reinterpret_cast<DType*>(&acc)[i] = static_cast<DType>(acc_f32[i]);
        return acc;
    } else {
        float4 acc = vals[0];
#pragma unroll
        for (int r = 1; r < NRanks; ++r) acc = add128<DType>(acc, vals[r]);
        return acc;
    }
}

// Index helper
template <typename DType>
class IndexHelper {
public:
    __device__ __forceinline__ IndexHelper(AllReduceFusionParams const& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cg::cluster_group cluster = cg::this_cluster();
        cg::grid_group grid = cg::this_grid();
        token_id = grid.cluster_rank();
        access_id_in_token = cluster.thread_rank();
        token_stride = grid.num_clusters();
#else
        token_id = blockIdx.x;
        access_id_in_token = threadIdx.x;
        token_stride = gridDim.x;
#endif
        access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> + access_id_in_token;
        access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
        tot_access = params.size / kElemsPerAccess<DType>;
    }
    int token_id, access_id_in_token, token_stride, access_id, access_stride, tot_access;
};

// Fused op for kAllReduce only: write AR result
template <AllReduceFusionPattern Pattern, typename DType>
class FusedOp {
public:
    __device__ __forceinline__ FusedOp(AllReduceFusionParams const& params, int access_id, int /*in_token*/)
        : m_params(params), m_access_id(access_id) {}
    __device__ __forceinline__ void update(int access_id) { m_access_id = access_id; }
    __device__ __forceinline__ void operator()(float4 val, int /*token_id*/) {
        if constexpr (HasAllReduceOut<Pattern>) {
            reinterpret_cast<float4*>(m_params.allreduce_out)[m_access_id] = val;
        }
    }
private:
    AllReduceFusionParams const& m_params; int m_access_id;
};

// ---------------- kernel (oneshot lamport) ----------------
template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024)
allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams params)
{
    IndexHelper<DType> index_helper(params);
    int token_id           = index_helper.token_id;
    int access_id_in_token = index_helper.access_id_in_token;
    int token_stride       = index_helper.token_stride;
    int access_id          = index_helper.access_id;
    int access_stride      = index_helper.access_stride;
    int tot_access         = index_helper.tot_access;

    float4 clear_vec = get_neg_zero();
    FusedOp<Pattern, DType> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd) { cudaTriggerProgrammaticLaunchCompletion(); }
#endif

    LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess<DType>;

    // Push local data into all peers’ buffers (current slot)
    for (int idx = access_id; idx < tot_access; idx += access_stride) {
        alignas(16) float val[4];
        *reinterpret_cast<float4*>(val) = reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
        for (int i = 0; i < 4; ++i) if (is_neg_zero(val[i])) val[i] = 0.f;
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx]
                = *reinterpret_cast<float4*>(val);
        }
    }
    // Clear the previous slot region for this rank
    for (int idx = access_id; idx < clear_access; idx += access_stride) {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    // Wait until all peers have filled current slot, then reduce
    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride) {
        fused_op.update(idx);
        float4 vals[NRanks];
        bool done = false;
        while (!done) {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r) {
                vals[r] = ld_global_volatile(&reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx]);
                done &= !is_neg_zero(vals[r]);
            }
        }
        float4 sum_val = allreduce_sum<DType, NRanks, Fp32Acc>(vals);
        fused_op(sum_val, tidx);
    }

    comm.update(params.size * NRanks);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (TriggerCompletionAtEnd) { cudaTriggerProgrammaticLaunchCompletion(); }
#endif
}

// ----------------- host launcher -----------------
static inline int get_sm_count() {
    static int sm_count = 0;
    if (sm_count == 0) {
        int dev = 0; TLLM_CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{}; TLLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        sm_count = prop.multiProcessorCount;
    }
    return sm_count;
}

template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
static void launch_oneshot_lamport(AllReduceFusionParams const& params, cudaLaunchConfig_t& cfg) {
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
        allreduce_fusion_kernel_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc, TriggerCompletionAtEnd>, params));
}

template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc>
static void allreduce_fusion_kernel_launcher(AllReduceFusionParams const& params) {
    TLLM_CHECK(params.size % params.hidden_dim == 0);
    TLLM_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);

    const int SM = tensorrt_llm::common::getSMVersion();
    const int token_num = params.size / params.hidden_dim;
    TLLM_CHECK(params.use_oneshot); // only oneshot implemented

    int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
    int cluster_size = (SM >= 90) ? 8 : 1;
    while (threads_per_token % cluster_size != 0 && cluster_size > 1) cluster_size /= 2;
    int threads_per_block = threads_per_token / cluster_size;
    while (threads_per_block < 128 && cluster_size >= 2) { threads_per_block *= 2; cluster_size /= 2; }

    int sm_count = get_sm_count();
    while (token_num * cluster_size > sm_count && cluster_size > 1 && threads_per_block <= 512) {
        threads_per_block *= 2;
        cluster_size /= 2;
    }

    int block_size = threads_per_block;
    TLLM_CHECK(block_size <= 1024 && cluster_size > 0);

    int grid_size = (std::min(sm_count, token_num * cluster_size) / cluster_size) * cluster_size;

    cudaLaunchConfig_t cfg{}; cudaLaunchAttribute attr[2]{};
    cfg.gridDim = grid_size;
    cfg.blockDim = block_size;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;

    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = cluster_size; attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = (SM >= 90) ? 2 : 0;

    // Accumulate policy: default is fp16/bf16 accumulation, unless env is set
    const bool fp32_acc = (std::getenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32") != nullptr);
    if constexpr (std::is_same_v<DType,float>) {
        launch_oneshot_lamport<Pattern, DType, NRanks, false, true>(params, cfg);
    } else {
        if (fp32_acc)
            launch_oneshot_lamport<Pattern, DType, NRanks, true,  true>(params, cfg);
        else
            launch_oneshot_lamport<Pattern, DType, NRanks, false, true>(params, cfg);
    }
}

void allreduce_fusion_op(AllReduceFusionParams const& params) {
#define DISPATCH_DTYPE(NR) \
    if (params.dtype == nvinfer1::DataType::kHALF)      { allreduce_fusion_kernel_launcher<AllReduceFusionPattern::kAllReduce, half,        NR, false>(params); return; } \
    else if (params.dtype == nvinfer1::DataType::kBF16) { allreduce_fusion_kernel_launcher<AllReduceFusionPattern::kAllReduce, nv_bfloat16, NR, false>(params); return; } \
    else if (params.dtype == nvinfer1::DataType::kFLOAT){ allreduce_fusion_kernel_launcher<AllReduceFusionPattern::kAllReduce, float,       NR, false>(params); return; } \
    else { TLLM_CHECK_WITH_INFO(false, "Unsupported dtype"); }

    switch (params.nranks) {
        case 2:  DISPATCH_DTYPE(2);
        case 4:  DISPATCH_DTYPE(4);
        case 8:  DISPATCH_DTYPE(8);
        default: TLLM_CHECK_WITH_INFO(false, "Unsupported nranks (only 2/4/8).");
    }
#undef DISPATCH_DTYPE
}

} // namespace tensorrt_llm::kernels::ar_fusion

