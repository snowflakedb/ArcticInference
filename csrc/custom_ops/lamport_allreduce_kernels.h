// kernels/lamport_allreduce_kernels.h
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

#ifndef TLLM_CUDA_CHECK
#define TLLM_CUDA_CHECK(call)                                       \
    do {                                                            \
        cudaError_t _err = (call);                                  \
        if (_err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(_err));  \
            abort();                                                \
        }                                                           \
    } while (0)
#endif

#ifndef TLLM_CHECK
#include <stdexcept>
#define TLLM_CHECK(cond) \
    do { if (!(cond)) { throw std::runtime_error("TLLM_CHECK failed: " #cond); } } while(0)
#endif

#ifndef TLLM_CHECK_WITH_INFO
#define TLLM_CHECK_WITH_INFO(cond, msg) \
    do { if (!(cond)) { throw std::runtime_error(msg); } } while(0)
#endif

// A tiny substitute for TensorRT types we use
namespace nvinfer1 {
enum class DataType : int {
    kFLOAT = 0,
    kHALF  = 1,
    kBF16  = 2,
};
}

// ---------------- Consts & helpers ----------------
namespace tensorrt_llm::kernels::ar_fusion {

template <typename DType> struct ElemsPerAccess;
template <> struct ElemsPerAccess<half>       { static constexpr int value = 8; };
template <> struct ElemsPerAccess<nv_bfloat16>{ static constexpr int value = 8; };
template <> struct ElemsPerAccess<float>      { static constexpr int value = 4; };

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

static constexpr int kOneShotMaxToken = 128;   // keep same constant
static constexpr int kBarrierFlagCount = 256;  // unused in oneshot path

// Only the pattern we implement
enum class AllReduceFusionPattern : int { kAllReduce = 0 };

// Only the quant type we implement
enum class QuantType : int { kNone = 0 };

template <AllReduceFusionPattern Pattern> struct FusionPatternTraits;

template <> struct FusionPatternTraits<AllReduceFusionPattern::kAllReduce> {
    static constexpr bool kHasAllReduceOut = true;
    static constexpr bool kHasResidual     = false;
    static constexpr bool kHasResidualOut  = false;
    static constexpr bool kHasRMSNorm      = false;
    static constexpr bool kHasNormOut      = false;
    static constexpr QuantType kQuantType  = QuantType::kNone;
};

template <AllReduceFusionPattern Pattern> constexpr bool HasResidual      = FusionPatternTraits<Pattern>::kHasResidual;
template <AllReduceFusionPattern Pattern> constexpr bool HasRMSNorm       = FusionPatternTraits<Pattern>::kHasRMSNorm;
template <AllReduceFusionPattern Pattern> constexpr bool HasAllReduceOut  = FusionPatternTraits<Pattern>::kHasAllReduceOut;
template <AllReduceFusionPattern Pattern> constexpr bool HasResidualOut   = FusionPatternTraits<Pattern>::kHasResidualOut;
template <AllReduceFusionPattern Pattern> constexpr bool HasNormOut       = FusionPatternTraits<Pattern>::kHasNormOut;
template <AllReduceFusionPattern Pattern> constexpr QuantType GetQuantType= FusionPatternTraits<Pattern>::kQuantType;

// Parameters (compatible layout with your reference, unused fields allowed)
struct AllReduceFusionParams {
    int nranks;
    int rank;
    nvinfer1::DataType dtype;
    int size;           // number of elements
    int hidden_dim;     // last-dim (per token)
    void** workspace;   // array of pointers (device memory), see Python wrapper
    void* allreduce_in;
    void* residual_in;  // unused in kAllReduce
    void* allreduce_out;
    void* residual_out; // unused
    void* norm_out;     // unused
    void* quant_out;    // unused
    void* scale_out;    // unused
    void* rms_gamma;    // unused
    float rms_eps;      // unused
    float* scale_factor;// unused
    bool use_oneshot;   // must be true
    // quant layout (unused)
    int layout_pad_ = 0;
    cudaStream_t stream;
    AllReduceFusionPattern pattern;
    bool trigger_completion_at_end = true;
};

// Entry
void allreduce_fusion_op(AllReduceFusionParams const& params);

} // namespace tensorrt_llm::kernels::ar_fusion

