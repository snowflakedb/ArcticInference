#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <type_traits>

#include "dtype_common.cuh"

struct SumOp {
  __device__ __forceinline__ float operator()(const float &a,
                                              const float &b) const {
    return a + b;
  }
};

template <typename ATenScalarT, int VEC>
__global__ void spec_ln_vec_kernel(
    typename DevHalf<ATenScalarT>::type *__restrict__ out,
    const typename DevHalf<ATenScalarT>::type *__restrict__ in,
    const typename DevHalf<ATenScalarT>::type *__restrict__ weight,
    const typename DevHalf<ATenScalarT>::type *__restrict__ bias,
    int64_t row_stride, int hidden, float eps) {

  using DH = DevHalf<ATenScalarT>;
  using H = typename DH::type;
  using VPack = Pack<H, VEC>;

  extern __shared__ char smem[];
  __shared__ float s_inv_rms;

  const int row = blockIdx.x;
  const H *row_in = in + row * row_stride;
  H *row_out = out + row * hidden;

  const int vec_len = hidden / VEC;
  const int tail = hidden - vec_len * VEC;

  float local_ss = 0.f;

  const VPack *in_vec = reinterpret_cast<const VPack *>(row_in);

  for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
    VPack p = in_vec[i];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float x = DH::to_float(p.v[k]);
      local_ss += x * x;
    }
  }

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

  const VPack *w_vec = reinterpret_cast<const VPack *>(weight);
  const VPack *b_vec = reinterpret_cast<const VPack *>(bias);
  VPack *out_vec = reinterpret_cast<VPack *>(row_out);

  for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
    VPack px = reinterpret_cast<const VPack *>(row_in)[i];
    VPack py;
    VPack pw;
    VPack pb;
    bool use_w = weight != nullptr;
    bool use_b = bias != nullptr;
    if (use_w)
      pw = w_vec[i];
    if (use_b)
      pb = b_vec[i];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float y = DH::to_float(px.v[k]);
      y = y * s_inv_rms;
      if (use_w)
        y *= DH::to_float(pw.v[k]);
      if (use_b)
        y += DH::to_float(pb.v[k]);
      py.v[k] = DH::from_float(y);
    }
    out_vec[i] = py;
  }

  for (int j = threadIdx.x + vec_len * VEC; j < hidden; j += blockDim.x) {
    float y = DH::to_float(row_in[j]) * s_inv_rms;
    if (weight)
      y *= DH::to_float(weight[j]);
    if (bias)
      y += DH::to_float(bias[j]);
    row_out[j] = DH::from_float(y);
  }
}

template <typename ATenScalarT>
__global__ void spec_ln_scalar_kernel(
    typename DevHalf<ATenScalarT>::type *__restrict__ out,
    const typename DevHalf<ATenScalarT>::type *__restrict__ in,
    const typename DevHalf<ATenScalarT>::type *__restrict__ weight,
    const typename DevHalf<ATenScalarT>::type *__restrict__ bias,
    int64_t row_stride, int hidden, float eps) {

  using DH = DevHalf<ATenScalarT>;
  using H = typename DH::type;

  __shared__ float s_inv_rms;

  const int row = blockIdx.x;
  const H *row_in = in + row * row_stride;
  H *row_out = out + row * hidden;

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
    if (weight)
      y *= DH::to_float(weight[j]);
    if (bias)
      y += DH::to_float(bias[j]);
    row_out[j] = DH::from_float(y);
  }
}

template <typename ATenScalarT>
torch::Tensor speculator_ln_cuda_impl(
    const torch::Tensor &input,            // [..., hidden]
    const c10::optional<torch::Tensor> &w, // [hidden] or None
    const c10::optional<torch::Tensor> &b, // [hidden] or None
    double eps_d) {

  TORCH_CHECK(input.is_cuda(), "speculator_ln: input must be CUDA");
  TORCH_CHECK(IsSupported<ATenScalarT>::value,
              "speculator_ln: dtype must be fp16 or bf16");

  const auto dtype = input.scalar_type();
  const auto device = input.device();
  const float eps = static_cast<float>(eps_d);

  const int64_t hidden = input.size(-1);
  TORCH_CHECK(hidden > 0, "speculator_ln: hidden size must be > 0");
  TORCH_CHECK(
      input.stride(-1) == 1,
      "speculator_ln: last dimension must be contiguous (stride -1 == 1)");

  const typename DevHalf<ATenScalarT>::type *w_ptr = nullptr;
  const typename DevHalf<ATenScalarT>::type *b_ptr = nullptr;

  if (w.has_value() && w->defined()) {
    TORCH_CHECK(w->is_cuda(), "weight must be CUDA");
    TORCH_CHECK(w->scalar_type() == dtype,
                "weight dtype must match input dtype");
    TORCH_CHECK(w->dim() == 1 && w->numel() == hidden,
                "weight must be 1D of size hidden");
    TORCH_CHECK(w->is_contiguous(), "weight must be contiguous");
    w_ptr = reinterpret_cast<const typename DevHalf<ATenScalarT>::type *>(
        w->data_ptr());
  }
  if (b.has_value() && b->defined()) {
    TORCH_CHECK(b->is_cuda(), "bias must be CUDA");
    TORCH_CHECK(b->scalar_type() == dtype, "bias dtype must match input dtype");
    TORCH_CHECK(b->dim() == 1 && b->numel() == hidden,
                "bias must be 1D of size hidden");
    TORCH_CHECK(b->is_contiguous(), "bias must be contiguous");
    b_ptr = reinterpret_cast<const typename DevHalf<ATenScalarT>::type *>(
        b->data_ptr());
  }

  auto in_2d = input.view({-1, hidden});
  const int64_t num_rows = in_2d.size(0);
  const int64_t row_stride = in_2d.stride(0);

  auto out = at::empty_like(input);
  auto out_2d = out.view({-1, hidden});

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int BLOCK = (num_rows < 256) ? 1024 : 256;
  dim3 grid(num_rows);
  dim3 block(std::min<int64_t>(hidden, BLOCK));

  using H = typename DevHalf<ATenScalarT>::type;
  const H *in_ptr = reinterpret_cast<const H *>(in_2d.data_ptr());
  H *out_ptr = reinterpret_cast<H *>(out_2d.data_ptr());

  const bool can_vec8 = (hidden % 8 == 0) && (row_stride % 8 == 0) &&
                        is_aligned(in_ptr, 16) && is_aligned(out_ptr, 16) &&
                        (!w_ptr || is_aligned(w_ptr, 16)) &&
                        (!b_ptr || is_aligned(b_ptr, 16));

  const bool can_vec4 = (hidden % 4 == 0) && (row_stride % 4 == 0) &&
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

torch::Tensor speculator_ln_cuda(const torch::Tensor &input,
                                 const c10::optional<torch::Tensor> &weight,
                                 const c10::optional<torch::Tensor> &bias,
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
