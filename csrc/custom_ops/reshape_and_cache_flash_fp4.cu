#include "custom_ops.h"
#include "dispatch_utils.h"
#include "quant_utils.cuh"
#include "vectorization_utils.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>

#include <cuda_fp8.h>

#include <vector>

namespace vllm {

template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  __device__ __forceinline__ void operator()(OutT &dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};

namespace nvfp4 {

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

template <typename T> struct TypePair {
  using V2 = half2;
};
template <> struct TypePair<half> {
  using V2 = half2;
};
template <> struct TypePair<__nv_bfloat16> {
  using V2 = __nv_bfloat162;
};

template <class T> struct PackedVec {
  typename TypePair<T>::V2 elts[4];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
__device__ inline uint8_t float_to_e2m1_rn(float val) {
  if (isnan(val))
    return 0x0;
  if (isinf(val))
    val = val < 0.f ? -6.f : 6.f;
  uint32_t sign_bit = (reinterpret_cast<uint32_t &>(val) & 0x80000000) >> 28;
  float x = fabsf(val);
  uint8_t mag;
  if (x > 5.0f)
    mag = 0x7;
  else if (x > 3.5f)
    mag = 0x6;
  else if (x > 2.5f)
    mag = 0x5;
  else if (x > 1.75f)
    mag = 0x4;
  else if (x > 1.25f)
    mag = 0x3;
  else if (x > 0.75f)
    mag = 0x2;
  else if (x > 0.25f)
    mag = 0x1;
  else
    mag = 0x0;
  return sign_bit | mag;
}
#endif

inline __device__ float rcp_approx_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

inline __device__ uint32_t fp32_8_to_e2m1x2_packed(float (&a)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile("{\n"
               ".reg .b8 byte0;\n"
               ".reg .b8 byte1;\n"
               ".reg .b8 byte2;\n"
               ".reg .b8 byte3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
               "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
               "}"
               : "=r"(val)
               : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "f"(a[4]),
                 "f"(a[5]), "f"(a[6]), "f"(a[7]));
  return val;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t r = 0;
  uint8_t *rb = reinterpret_cast<uint8_t *>(&r);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint8_t v1 = float_to_e2m1_rn(a[2 * i]);
    uint8_t v2 = float_to_e2m1_rn(a[2 * i + 1]);
    rb[i] = (v2 << 4) | (v1 & 0x0F);
  }
  return r;
#else
  return 0u;
#endif
}

inline __device__ uint32_t fp32_8_to_e2m1x2_packed(float2 (&a)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile("{\n"
               ".reg .b8 byte0;\n"
               ".reg .b8 byte1;\n"
               ".reg .b8 byte2;\n"
               ".reg .b8 byte3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
               "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
               "}"
               : "=r"(val)
               : "f"(a[0].x), "f"(a[0].y), "f"(a[1].x), "f"(a[1].y),
                 "f"(a[2].x), "f"(a[2].y), "f"(a[3].x), "f"(a[3].y));
  return val;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t r = 0;
  uint8_t *rb = reinterpret_cast<uint8_t *>(&r);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint8_t v1 = float_to_e2m1_rn(a[i].x);
    uint8_t v2 = float_to_e2m1_rn(a[i].y);
    rb[i] = (v2 << 4) | (v1 & 0x0F);
  }
  return r;
#else
  return 0u;
#endif
}

template <class T, bool UE8M0_SF = false>
__device__ uint32_t warp_quant_8_to_nvfp4(PackedVec<T> &vec8,
                                          float sf_scale_val,
                                          uint8_t *sf_out_byte_if_leader) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  auto localMax = __habs2(vec8.elts[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    localMax = __hmax2(localMax, __habs2(vec8.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1), localMax);
  float vmax = float(__hmax(localMax.x, localMax.y));

  float sf_val = sf_scale_val * (vmax * rcp_approx_ftz(6.0f));
  uint8_t sf_byte;
  if constexpr (UE8M0_SF) {
    uint32_t tmp = reinterpret_cast<uint32_t &>(sf_val) >> 23;
    sf_byte = tmp & 0xff;
    reinterpret_cast<uint32_t &>(sf_val) = tmp << 23;
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_val);
    reinterpret_cast<__nv_fp8_e4m3 &>(sf_byte) = tmp;
    sf_val = float(tmp);
  }

  float out_scale = (sf_val != 0.f)
                        ? rcp_approx_ftz(sf_val * rcp_approx_ftz(sf_scale_val))
                        : 0.0f;

  if (sf_out_byte_if_leader) {
    *sf_out_byte_if_leader = sf_byte;
  }

  float2 f2[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (std::is_same_v<T, half>) {
      f2[i] = __half22float2(vec8.elts[i]);
    } else {
      f2[i] = __bfloat1622float2(vec8.elts[i]);
    }
    f2[i].x *= out_scale;
    f2[i].y *= out_scale;
  }
  return fp32_8_to_e2m1x2_packed(f2);
#else
  return 0u;
#endif
}

__host__ __device__ inline int q_words_per_head(int head_size) {
  return head_size / 8;
}
__host__ __device__ inline int sf_bytes_per_head(int head_size) {
  int scales = head_size / 16;
  int pad4 = ((scales + 3) / 4) * 4;
  return pad4;
}
__host__ __device__ inline int head_words_total(int head_size) {
  return q_words_per_head(head_size) + sf_bytes_per_head(head_size) / 4;
}

} // namespace nvfp4

// NVFP4 cache writer: converts FP16/BF16 key/value to NVFP4 and writes
// packed data + per-16 scale bytes into [key_cache|value_cache].
template <typename scalar_t, bool UE8M0_SF = false>
__global__ void reshape_and_cache_flash_nvfp4_kernel(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    uint32_t *__restrict__ key_cache,   // int32 view; row=[Q words][SF bytes/4]
    uint32_t *__restrict__ value_cache,
    const int64_t *__restrict__ slot_mapping, // [num_tokens or actual]
    const int64_t block_stride_words, const int64_t page_stride_words,
    const int64_t head_stride_words, const int64_t key_stride_elems,
    const int64_t value_stride_elems, const int num_heads, const int head_size,
    const int block_size, const float *k_sf_scale_ptr,
    const float *v_sf_scale_ptr) {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0)
    return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_off = slot_idx % block_size;

  const scalar_t *__restrict__ key_src = key + token_idx * key_stride_elems;
  const scalar_t *__restrict__ value_src =
      value + token_idx * value_stride_elems;

  uint32_t *__restrict__ key_dst_words = key_cache +
                                         block_idx * block_stride_words +
                                         block_off * page_stride_words;
  uint32_t *__restrict__ value_dst_words = value_cache +
                                           block_idx * block_stride_words +
                                           block_off * page_stride_words;

  const float k_sf_scale = (k_sf_scale_ptr ? *k_sf_scale_ptr : 1.0f);
  const float v_sf_scale = (v_sf_scale_ptr ? *v_sf_scale_ptr : 1.0f);

  const int q_words = nvfp4::q_words_per_head(head_size);
  const int head_words = nvfp4::head_words_total(head_size);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;

  for (int head = warp_id; head < num_heads; head += warps_per_block) {
    const scalar_t *__restrict__ k_src_h = key_src + head * head_size;
    const scalar_t *__restrict__ v_src_h = value_src + head * head_size;

    uint32_t *__restrict__ k_head_base =
        key_dst_words + static_cast<int64_t>(head) * head_stride_words;
    uint32_t *__restrict__ v_head_base =
        value_dst_words + static_cast<int64_t>(head) * head_stride_words;

    uint32_t *__restrict__ k_q_out = k_head_base;
    uint8_t *__restrict__ k_sf_out =
        reinterpret_cast<uint8_t *>(k_head_base + q_words);

    uint32_t *__restrict__ v_q_out = v_head_base;
    uint8_t *__restrict__ v_sf_out =
        reinterpret_cast<uint8_t *>(v_head_base + q_words);

    const int n_groups8 = head_size / nvfp4::CVT_FP4_ELTS_PER_THREAD;

    for (int col = lane; col < n_groups8; col += 32) {
      using PV = nvfp4::PackedVec<scalar_t>;
      const PV *__restrict__ k_src8 = reinterpret_cast<const PV *>(k_src_h);
      const PV *__restrict__ v_src8 = reinterpret_cast<const PV *>(v_src_h);
      PV k_vec8 = k_src8[col];
      PV v_vec8 = v_src8[col];

      uint8_t *k_sf_ptr = ((lane & 1) == 0) ? (k_sf_out + (col >> 1)) : nullptr;
      uint8_t *v_sf_ptr = ((lane & 1) == 0) ? (v_sf_out + (col >> 1)) : nullptr;

      uint32_t k_packed = nvfp4::warp_quant_8_to_nvfp4<scalar_t, UE8M0_SF>(
          k_vec8, k_sf_scale, k_sf_ptr);
      uint32_t v_packed = nvfp4::warp_quant_8_to_nvfp4<scalar_t, UE8M0_SF>(
          v_vec8, v_sf_scale, v_sf_ptr);

      k_q_out[col] = k_packed;
      v_q_out[col] = v_packed;
    }
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) { /* no-op on older arch */
  }
#endif
}

} // namespace vllm

void reshape_and_cache_flash_fp4(
    torch::Tensor &key,       // [num_tokens, num_heads, head_size]
    torch::Tensor &value,     // [num_tokens, num_heads, head_size]
    torch::Tensor &key_cache, // NVFP4: int32 view with packed [Q|S]
    torch::Tensor &value_cache, torch::Tensor &slot_mapping,
    const std::string &kv_cache_dtype, torch::Tensor &k_scale,
    torch::Tensor &v_scale) {

  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);

  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(head_size % 16 == 0,
              "nvfp4: head_size must be a multiple of 16.");
  TORCH_CHECK(key_cache.scalar_type() == torch::kInt32 &&
                  value_cache.scalar_type() == torch::kInt32,
              "nvfp4: key_cache/value_cache must be int32 tensors storing "
              "packed FP4 data followed by scale bytes.");

  dim3 block(std::min(512, std::max(32, num_heads * 32)));

  switch (key.scalar_type()) {
  case torch::kHalf: {
    vllm::reshape_and_cache_flash_nvfp4_kernel<half, false>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<half const *>(key.data_ptr()),
            reinterpret_cast<half const *>(value.data_ptr()),
            reinterpret_cast<uint32_t *>(key_cache.data_ptr()),
            reinterpret_cast<uint32_t *>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,
            head_stride, key_stride, value_stride, num_heads, head_size,
            block_size, reinterpret_cast<const float *>(k_scale.data_ptr()),
            reinterpret_cast<const float *>(v_scale.data_ptr()));
    break;
  }
  case torch::kBFloat16: {
    vllm::reshape_and_cache_flash_nvfp4_kernel<__nv_bfloat16, false>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 const *>(key.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const *>(value.data_ptr()),
            reinterpret_cast<uint32_t *>(key_cache.data_ptr()),
            reinterpret_cast<uint32_t *>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,
            head_stride, key_stride, value_stride, num_heads, head_size,
            block_size, reinterpret_cast<const float *>(k_scale.data_ptr()),
            reinterpret_cast<const float *>(v_scale.data_ptr()));
    break;
  }
  default:
    TORCH_CHECK(false,
                "nvfp4: unsupported key/value dtype (expected FP16/BF16).");
  }
}
