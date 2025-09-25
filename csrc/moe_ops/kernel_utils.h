#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

#ifdef BF16_AVAILABLE
#include <cuda_bf16.h>
#endif

// constexpr variant of warpSize for templating
constexpr int hw_warp_size = 32;

#if __CUDA_ARCH__ >= 530
#define HALF_PRECISION_AVAILABLE = 1
#define PTX_AVAILABLE
#endif  // __CUDA_ARCH__ >= 530

#if __CUDA_ARCH__ >= 800
#define ASYNC_COPY_AVAILABLE
#endif  // __CUDA_ARCH__ >= 800

#include <cooperative_groups.h>