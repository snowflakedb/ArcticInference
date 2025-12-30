#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>
#include <type_traits>

template <typename T> struct IsSupported : std::false_type {};
template <> struct IsSupported<at::Half> : std::true_type {};
template <> struct IsSupported<at::BFloat16> : std::true_type {};

__device__ __forceinline__ float to_float_device(__half h) {
  return __half2float(h);
}

__device__ __forceinline__ __half from_float_device(float f) {
  return __float2half(f);
}

__device__ __forceinline__ float to_float_device(__nv_bfloat16 h) {
#if !defined(__CUDA_NO_BF16__)
  return __bfloat162float(h);
#else
  uint16_t raw = *reinterpret_cast<const uint16_t *>(&h);
  uint32_t u32 = (uint32_t)raw << 16;
  float out = __uint_as_float(u32);
  return out;
#endif
}

__device__ __forceinline__ __nv_bfloat16 from_float_device_bf16(float f) {
#if !defined(__CUDA_NO_BF16__)
  return __float2bfloat16(f);
#else
  uint32_t u = __float_as_uint(f);
  uint16_t hi = (uint16_t)(u >> 16);
  __nv_bfloat16 h;
  *reinterpret_cast<uint16_t *>(&h) = hi;
  return h;
#endif
}

template <typename T> struct DevHalf;

template <> struct DevHalf<at::Half> {
  using type = __half;
  static __device__ __forceinline__ float to_float(__half h) {
    return to_float_device(h);
  }
  static __device__ __forceinline__ __half from_float(float f) {
    return from_float_device(f);
  }
};

template <> struct DevHalf<at::BFloat16> {
  using type = __nv_bfloat16;
  static __device__ __forceinline__ float to_float(__nv_bfloat16 h) {
    return to_float_device(h);
  }
  static __device__ __forceinline__ __nv_bfloat16 from_float(float f) {
    return from_float_device_bf16(f);
  }
};

template <typename T, int N> struct alignas(sizeof(T) * N) Pack {
  T v[N];
};

static inline bool is_aligned(const void *p, size_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) % bytes) == 0;
}
