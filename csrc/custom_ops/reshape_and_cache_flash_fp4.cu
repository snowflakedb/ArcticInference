#include "custom_ops.h"
#include "dispatch_utils.h"
#include "quant_utils.cuh"
#include "vectorization_utils.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <torch/cuda.h>

#include <vector>

namespace vllm {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <typename T> struct TypeConverter {
  using Type = half2;
};
template <> struct TypeConverter<half2> {
  using Type = half;
};
template <> struct TypeConverter<half> {
  using Type = half2;
};
template <> struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};
template <> struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

template <class Type> struct PackedVec8 {
  typename TypeConverter<Type>::Type elts[4];
};

#if __CUDA_ARCH__ < 1000
__device__ uint8_t float_to_e2m1_rn(float val) {
  if (isnan(val)) {
    return 0x0;
  }
  if (isinf(val)) {
    val = val < 0.f ? -6.f : 6.f;
  }
  uint32_t sign_bit = (reinterpret_cast<uint32_t &>(val) & 0x80000000) >> 28;
  float x = fabsf(val);
  uint8_t magnitude_bits;
  if (x > 5.0f)
    magnitude_bits = 0x7; // 6.0
  else if (x > 3.5f)
    magnitude_bits = 0x6; // 4.0
  else if (x > 2.5f)
    magnitude_bits = 0x5; // 3.0
  else if (x > 1.75f)
    magnitude_bits = 0x4; // 2.0
  else if (x > 1.25f)
    magnitude_bits = 0x3; // 1.5
  else if (x > 0.75f)
    magnitude_bits = 0x2; // 1.0
  else if (x > 0.25f)
    magnitude_bits = 0x1; // 0.5
  else
    magnitude_bits = 0x0; // 0.0
  return sign_bit | magnitude_bits;
}
#endif

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile("{\n"
               ".reg .b8 byte0, byte1, byte2, byte3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
               "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
               "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
               "}"
               : "=r"(val)
               : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x),
                 "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
                 "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  uint32_t result = 0;
  uint8_t *result_bytes = reinterpret_cast<uint8_t *>(&result);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint8_t val1 = float_to_e2m1_rn(array[i].x);
    uint8_t val2 = float_to_e2m1_rn(array[i].y);
    result_bytes[i] = (val2 << 4) | (val1 & 0x0F);
  }
  return result;
#endif
}

inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template <typename scalar_t>
__device__ __forceinline__ void
quantize_16_to_fp4(const scalar_t *__restrict__ src_base,
                   uint32_t *__restrict__ dst_base,
                   uint8_t *__restrict__ scale_dst, const int tid_in_block) {

  const int lane_in_pair = tid_in_block % 2;

  using PackedVec = PackedVec8<scalar_t>;
  PackedVec vec =
      *reinterpret_cast<const PackedVec *>(src_base + lane_in_pair * 8);

  auto localMax = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < 4; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  localMax = __hmax2(__shfl_xor_sync(0xffffffff, localMax, 1), localMax);
  float vecMax = float(__hmax(localMax.x, localMax.y));
  float SFValue = vecMax * reciprocal_approximate_ftz(6.0f);

  if (lane_in_pair == 0) {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    *scale_dst = reinterpret_cast<uint8_t &>(tmp);
    SFValue = float(tmp);
  }

  SFValue = __shfl_sync(0xffffffff, SFValue, tid_in_block & ~1);
  float outputScale =
      (SFValue != 0.0f) ? reciprocal_approximate_ftz(SFValue) : 0.0f;

  float2 fp2Vals[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    if constexpr (std::is_same_v<scalar_t, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  dst_base[lane_in_pair] = fp32_vec_to_e2m1(fp2Vals);
}
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

template <typename scalar_t, typename cache_t>
__global__ void reshape_and_cache_flash_kernel_fp4(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    cache_t *__restrict__ key_cache,    // NHD or HND
    cache_t *__restrict__ value_cache,
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float *k_scale, const float *v_scale,
    uint8_t *__restrict__ key_scale_cache,
    uint8_t *__restrict__ value_scale_cache, const bool is_nhd) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n_elems = num_heads * head_size;

  const scalar_t *__restrict__ key_src = key + token_idx * key_stride;
  const scalar_t *__restrict__ value_src = value + token_idx * value_stride;

  cache_t *__restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t *__restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t *__restrict__ key_dst_fp4 = reinterpret_cast<uint32_t *>(key_dst);
  uint32_t *__restrict__ value_dst_fp4 =
      reinterpret_cast<uint32_t *>(value_dst);

  const int64_t scale_block_stride = block_stride / 8;
  const int64_t scale_page_stride = page_stride / 8;
  const int64_t scale_head_stride = head_stride / 8;

  uint8_t *__restrict__ key_scale_dst = key_scale_cache +
                                        block_idx * scale_block_stride +
                                        block_offset * scale_page_stride;
  uint8_t *__restrict__ value_scale_dst = value_scale_cache +
                                          block_idx * scale_block_stride +
                                          block_offset * scale_page_stride;

  constexpr int CHUNK_SIZE = 16;
  constexpr int THREADS_PER_CHUNK = 2;

  const int lane = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;
  const int warps_in_block = blockDim.x / 32;
  const int chunks_per_warp = 32 / THREADS_PER_CHUNK;

  if (is_nhd) {
    // NHD layout
    const int num_chunks = n_elems / CHUNK_SIZE;
    for (int chunk_base_idx = warp_id * chunks_per_warp;
         chunk_base_idx < num_chunks;
         chunk_base_idx += warps_in_block * chunks_per_warp) {
      const int chunk_idx = chunk_base_idx + (lane / THREADS_PER_CHUNK);
      if (chunk_idx < num_chunks) {
        quantize_16_to_fp4(key_src + chunk_idx * CHUNK_SIZE,
                           key_dst_fp4 + chunk_idx * (CHUNK_SIZE / 8),
                           key_scale_dst + chunk_idx, threadIdx.x);
        quantize_16_to_fp4(value_src + chunk_idx * CHUNK_SIZE,
                           value_dst_fp4 + chunk_idx * (CHUNK_SIZE / 8),
                           value_scale_dst + chunk_idx, threadIdx.x);
      }
    }
  } else {
    // HND layout
    const int num_chunks_per_head = head_size / CHUNK_SIZE;
    for (int head = warp_id; head < num_heads; head += warps_in_block) {
      const scalar_t *__restrict__ k_src_h = key_src + head * head_size;
      const scalar_t *__restrict__ v_src_h = value_src + head * head_size;

      cache_t *__restrict__ k_dst_head_u8 =
          key_dst + static_cast<int64_t>(head) * head_stride;
      cache_t *__restrict__ v_dst_head_u8 =
          value_dst + static_cast<int64_t>(head) * head_stride;

      uint32_t *__restrict__ k_dst_h =
          reinterpret_cast<uint32_t *>(k_dst_head_u8);
      uint32_t *__restrict__ v_dst_h =
          reinterpret_cast<uint32_t *>(v_dst_head_u8);

      uint8_t *__restrict__ k_scale_dst_h =
          key_scale_dst + static_cast<int64_t>(head) * scale_head_stride;
      uint8_t *__restrict__ v_scale_dst_h =
          value_scale_dst + static_cast<int64_t>(head) * scale_head_stride;

      for (int chunk_idx = lane / THREADS_PER_CHUNK;
           chunk_idx < num_chunks_per_head; chunk_idx += chunks_per_warp) {
        quantize_16_to_fp4(k_src_h + chunk_idx * CHUNK_SIZE,
                           k_dst_h + chunk_idx * (CHUNK_SIZE / 8),
                           k_scale_dst_h + chunk_idx, threadIdx.x);
        quantize_16_to_fp4(v_src_h + chunk_idx * CHUNK_SIZE,
                           v_dst_h + chunk_idx * (CHUNK_SIZE / 8),
                           v_scale_dst_h + chunk_idx, threadIdx.x);
      }
    }
  }
#endif // __CUDA_ARCH__ >= 900
}

} // namespace vllm

void reshape_and_cache_flash_fp4(
    torch::Tensor &key,   // [num_tokens, num_heads, head_size]
    torch::Tensor &value, // [num_tokens, num_heads, head_size]
    torch::Tensor &key_cache, torch::Tensor &value_cache,
    torch::Tensor &slot_mapping, const std::string &kv_cache_dtype,
    torch::Tensor &k_scale, torch::Tensor &v_scale,
    torch::Tensor &key_scale_cache, torch::Tensor &value_scale_cache) {
  const int64_t num_tokens = slot_mapping.size(0);
  const int64_t num_heads = key.size(1);
  const int64_t head_size = key.size(2);

  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "KV cache must be rank-4");

  const bool is_nhd = (key_cache.size(2) == num_heads);

  const int64_t block_stride = key_cache.stride(0);
  const int64_t page_stride =
      is_nhd ? key_cache.stride(1) : key_cache.stride(2);
  const int64_t head_stride =
      is_nhd ? key_cache.stride(2) : key_cache.stride(1);
  const int block_size = is_nhd ? key_cache.size(1) : key_cache.size(2);

  TORCH_CHECK(value_cache.stride(0) == block_stride, "block_stride mismatch");
  TORCH_CHECK((is_nhd ? value_cache.stride(1) : value_cache.stride(2)) ==
                  page_stride,
              "page_stride mismatch");
  TORCH_CHECK((is_nhd ? value_cache.stride(2) : value_cache.stride(1)) ==
                  head_stride,
              "head_stride mismatch");
  TORCH_CHECK((is_nhd ? value_cache.size(1) : value_cache.size(2)) ==
                  block_size,
              "block_size mismatch between key_cache and value_cache");

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  int threads = std::min<int>(num_heads * head_size, 512);
  threads = std::max(32, ((threads + 31) / 32) * 32);
  dim3 block(threads);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (key.scalar_type()) {
  case torch::kHalf: {
    vllm::reshape_and_cache_flash_kernel_fp4<half, uint8_t>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<half *>(key.data_ptr()),
            reinterpret_cast<half *>(value.data_ptr()),
            reinterpret_cast<uint8_t *>(key_cache.data_ptr()),
            reinterpret_cast<uint8_t *>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,
            head_stride, key_stride, value_stride, num_heads, head_size,
            block_size, k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
            key_scale_cache.data_ptr<uint8_t>(),
            value_scale_cache.data_ptr<uint8_t>(), is_nhd);
    break;
  }
  case torch::kBFloat16: {
    vllm::reshape_and_cache_flash_kernel_fp4<__nv_bfloat16, uint8_t>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(key.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(value.data_ptr()),
            reinterpret_cast<uint8_t *>(key_cache.data_ptr()),
            reinterpret_cast<uint8_t *>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,
            head_stride, key_stride, value_stride, num_heads, head_size,
            block_size, k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
            key_scale_cache.data_ptr<uint8_t>(),
            value_scale_cache.data_ptr<uint8_t>(), is_nhd);
    break;
  }
  default: {
    TORCH_CHECK(false, "Unsupported input dtype for reshape_and_cache_fp4. "
                       "Must be half or bfloat16.");
  }
  }
}