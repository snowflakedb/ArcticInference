// speculator_ln_kernel.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>

#include <stdint.h>
#include <type_traits>

// ---- Utilities ----

template <typename T>
struct IsSupported : std::false_type {};
template <>
struct IsSupported<at::Half> : std::true_type {};
template <>
struct IsSupported<at::BFloat16> : std::true_type {};

// Conversion helpers (device)

__device__ __forceinline__ float to_float_device(__half h) {
  return __half2float(h);
}

__device__ __forceinline__ __half from_float_device(float f) {
  return __float2half(f);
}

__device__ __forceinline__ float to_float_device(__nv_bfloat16 h) {
#if !defined(__CUDA_NO_BF16__)
  // Intrinsic is available on Ampere+ toolchains
  return __bfloat162float(h);
#else
  // Fallback bit-hack (host builds without BF16 support)
  uint16_t raw = *reinterpret_cast<const uint16_t*>(&h);
  uint32_t u32 = (uint32_t)raw << 16;
  float out = __uint_as_float(u32);
  return out;
#endif
}

__device__ __forceinline__ __nv_bfloat16 from_float_device_bf16(float f) {
#if !defined(__CUDA_NO_BF16__)
  return __float2bfloat16(f);
#else
  // Round-to-nearest-even truncate float32 to bf16
  uint32_t u = __float_as_uint(f);
  // simple truncation (you can improve with proper rounding if desired)
  uint16_t hi = (uint16_t)(u >> 16);
  __nv_bfloat16 h;
  *reinterpret_cast<uint16_t*>(&h) = hi;
  return h;
#endif
}

// TDispatch: map ATen scalar types to device half types
template <typename T>
struct DevHalf;

template <>
struct DevHalf<at::Half> {
  using type = __half;
  static __device__ __forceinline__ float to_float(__half h) { return to_float_device(h); }
  static __device__ __forceinline__ __half from_float(float f) { return from_float_device(f); }
};

template <>
struct DevHalf<at::BFloat16> {
  using type = __nv_bfloat16;
  static __device__ __forceinline__ float to_float(__nv_bfloat16 h) { return to_float_device(h); }
  static __device__ __forceinline__ __nv_bfloat16 from_float(float f) { return from_float_device_bf16(f); }
};

// Vector pack with alignment
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  T v[N];
};

// Sum op for CUB
struct SumOp {
  __device__ __forceinline__ float operator()(const float& a, const float& b) const {
    return a + b;
  }
};

// Compute per-row inverse RMS in two passes (vectorized), then write affine result.
// input: [num_rows, hidden] with row stride = row_stride (in elements).
template <typename ATenScalarT, int VEC>
__global__ void spec_ln_vec_kernel(
    typename DevHalf<ATenScalarT>::type* __restrict__ out,
    const typename DevHalf<ATenScalarT>::type* __restrict__ in,
    const typename DevHalf<ATenScalarT>::type* __restrict__ weight, // may be nullptr
    const typename DevHalf<ATenScalarT>::type* __restrict__ bias,   // may be nullptr
    int64_t row_stride, int hidden, float eps) {

  using DH = DevHalf<ATenScalarT>;
  using H = typename DH::type;
  using VPack = Pack<H, VEC>;

  extern __shared__ char smem[];
  __shared__ float s_inv_rms;

  const int row = blockIdx.x;
  const H* row_in  = in  + row * row_stride;
  H*       row_out = out + row * hidden;

  const int vec_len = hidden / VEC;
  const int tail = hidden - vec_len * VEC;

  // Pass 1: sum of squares
  float local_ss = 0.f;

  const VPack* in_vec = reinterpret_cast<const VPack*>(row_in);

  for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
    VPack p = in_vec[i];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float x = DH::to_float(p.v[k]);
      local_ss += x * x;
    }
  }

  // tail (scalar)
  for (int j = threadIdx.x + vec_len * VEC; j < hidden; j += blockDim.x) {
    float x = DH::to_float(row_in[j]);
    local_ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sumsq = BlockReduce(temp_storage).Reduce(local_ss, SumOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    float mean = sumsq / static_cast<float>(hidden);
    s_inv_rms = rsqrtf(mean + eps);
  }
  __syncthreads();

  // Pass 2: write normalized (optionally affine) output
  const VPack* w_vec = reinterpret_cast<const VPack*>(weight);
  const VPack* b_vec = reinterpret_cast<const VPack*>(bias);
  VPack* out_vec = reinterpret_cast<VPack*>(row_out);

  for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
    VPack px = reinterpret_cast<const VPack*>(row_in)[i];
    VPack py;
    VPack pw;
    VPack pb;
    bool use_w = weight != nullptr;
    bool use_b = bias   != nullptr;
    if (use_w) pw = w_vec[i];
    if (use_b) pb = b_vec[i];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float y = DH::to_float(px.v[k]);
      y = y * s_inv_rms;
      if (use_w) y *= DH::to_float(pw.v[k]);
      if (use_b) y += DH::to_float(pb.v[k]);
      py.v[k] = DH::from_float(y);
    }
    out_vec[i] = py;
  }

  // tail (scalar)
  for (int j = threadIdx.x + vec_len * VEC; j < hidden; j += blockDim.x) {
    float y = DH::to_float(row_in[j]) * s_inv_rms;
    if (weight) y *= DH::to_float(weight[j]);
    if (bias)   y += DH::to_float(bias[j]);
    row_out[j] = DH::from_float(y);
  }
}

// Scalar fallback (no vectorization)
template <typename ATenScalarT>
__global__ void spec_ln_scalar_kernel(
    typename DevHalf<ATenScalarT>::type* __restrict__ out,
    const typename DevHalf<ATenScalarT>::type* __restrict__ in,
    const typename DevHalf<ATenScalarT>::type* __restrict__ weight, // may be nullptr
    const typename DevHalf<ATenScalarT>::type* __restrict__ bias,   // may be nullptr
    int64_t row_stride, int hidden, float eps) {

  using DH = DevHalf<ATenScalarT>;
  using H = typename DH::type;

  __shared__ float s_inv_rms;

  const int row = blockIdx.x;
  const H* row_in  = in  + row * row_stride;
  H*       row_out = out + row * hidden;

  float local_ss = 0.f;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    float x = DH::to_float(row_in[j]);
    local_ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sumsq = BlockReduce(temp_storage).Reduce(local_ss, SumOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    float mean = sumsq / static_cast<float>(hidden);
    s_inv_rms = rsqrtf(mean + eps);
  }
  __syncthreads();

  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    float y = DH::to_float(row_in[j]) * s_inv_rms;
    if (weight) y *= DH::to_float(weight[j]);
    if (bias)   y += DH::to_float(bias[j]);
    row_out[j] = DH::from_float(y);
  }
}

// ---- Host launcher ----

static inline bool is_aligned(const void* p, size_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) % bytes) == 0;
}

template <typename ATenScalarT>
torch::Tensor speculator_ln_cuda_impl(
    const torch::Tensor& input,             // [..., hidden]
    const c10::optional<torch::Tensor>& w,   // [hidden] or None
    const c10::optional<torch::Tensor>& b,   // [hidden] or None
    double eps_d) {

  TORCH_CHECK(input.is_cuda(), "speculator_ln: input must be CUDA");
  TORCH_CHECK(IsSupported<ATenScalarT>::value, "speculator_ln: dtype must be fp16 or bf16");

  const auto dtype = input.scalar_type();
  const auto device = input.device();
  const float eps = static_cast<float>(eps_d);

  // Last dimension is hidden
  const int64_t hidden = input.size(-1);
  TORCH_CHECK(hidden > 0, "speculator_ln: hidden size must be > 0");
  TORCH_CHECK(input.stride(-1) == 1, "speculator_ln: last dimension must be contiguous (stride -1 == 1)");

  // Optional weight/bias
  const typename DevHalf<ATenScalarT>::type* w_ptr = nullptr;
  const typename DevHalf<ATenScalarT>::type* b_ptr = nullptr;

  if (w.has_value() && w->defined()) {
    TORCH_CHECK(w->is_cuda(), "weight must be CUDA");
    TORCH_CHECK(w->scalar_type() == dtype, "weight dtype must match input dtype");
    TORCH_CHECK(w->dim() == 1 && w->numel() == hidden, "weight must be 1D of size hidden");
    TORCH_CHECK(w->is_contiguous(), "weight must be contiguous");
    w_ptr = reinterpret_cast<const typename DevHalf<ATenScalarT>::type*>(w->data_ptr());
  }
  if (b.has_value() && b->defined()) {
    TORCH_CHECK(b->is_cuda(), "bias must be CUDA");
    TORCH_CHECK(b->scalar_type() == dtype, "bias dtype must match input dtype");
    TORCH_CHECK(b->dim() == 1 && b->numel() == hidden, "bias must be 1D of size hidden");
    TORCH_CHECK(b->is_contiguous(), "bias must be contiguous");
    b_ptr = reinterpret_cast<const typename DevHalf<ATenScalarT>::type*>(b->data_ptr());
  }

  // Flatten to 2D [num_rows, hidden] to get clean row stride
  auto in_2d = input.view({-1, hidden});
  const int64_t num_rows = in_2d.size(0);
  const int64_t row_stride = in_2d.stride(0); // elements (not bytes)

  // Allocate output (contiguous last dim)
  auto out = at::empty_like(input);
  auto out_2d = out.view({-1, hidden});

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int BLOCK = (num_rows < 256) ? 1024 : 256;
  dim3 grid(num_rows);
  dim3 block(std::min<int64_t>(hidden, BLOCK));

  using H = typename DevHalf<ATenScalarT>::type;
  const H* in_ptr  = reinterpret_cast<const H*>(in_2d.data_ptr());
  H* out_ptr       = reinterpret_cast<H*>(out_2d.data_ptr());

  // Try VEC=8 (128-bit), then VEC=4 (64-bit), else scalar
  const bool can_vec8 =
      (hidden % 8 == 0) &&
      (row_stride % 8 == 0) &&
      is_aligned(in_ptr, 16) && is_aligned(out_ptr, 16) &&
      (!w_ptr || is_aligned(w_ptr, 16)) &&
      (!b_ptr || is_aligned(b_ptr, 16));

  const bool can_vec4 =
      (hidden % 4 == 0) &&
      (row_stride % 4 == 0) &&
      is_aligned(in_ptr, 8) && is_aligned(out_ptr, 8) &&
      (!w_ptr || is_aligned(w_ptr, 8)) &&
      (!b_ptr || is_aligned(b_ptr, 8));

  if (can_vec8) {
    spec_ln_vec_kernel<ATenScalarT, 8><<<grid, block, 0, stream>>>(
        out_ptr, in_ptr, w_ptr, b_ptr, row_stride, (int)hidden, eps);
  } else if (can_vec4) {
    spec_ln_vec_kernel<ATenScalarT, 4><<<grid, block, 0, stream>>>(
        out_ptr, in_ptr, w_ptr, b_ptr, row_stride, (int)hidden, eps);
  } else {
    spec_ln_scalar_kernel<ATenScalarT><<<grid, block, 0, stream>>>(
        out_ptr, in_ptr, w_ptr, b_ptr, row_stride, (int)hidden, eps);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


torch::Tensor speculator_ln_cuda(
    const torch::Tensor& input,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias,
    double eps) {

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  switch (input.scalar_type()) {
    case at::kHalf:
      return speculator_ln_cuda_impl<at::Half>(input, weight, bias, eps);
    case at::kBFloat16:
      return speculator_ln_cuda_impl<at::BFloat16>(input, weight, bias, eps);
    default:
      TORCH_CHECK(false, "speculator_ln: only fp16 and bf16 are supported.");
  }
}

