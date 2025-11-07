#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <type_traits>

#include "dtype_common.cuh"

__device__ __forceinline__ float sigmoid_f(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_erf(float x) {
  const float kInvSqrt2 = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float kSqrt2OverPi = 0.79788456080286535588f;
  return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

template <typename ATenScalarT, int VEC, int BLOCK_THREADS>
__global__ void sum_lstm_vec_kernel(
    typename DevHalf<ATenScalarT>::type *__restrict__ out_state,
    typename DevHalf<ATenScalarT>::type *__restrict__ out_cell,

    const typename DevHalf<ATenScalarT>::type *__restrict__ states_4d,
    const typename DevHalf<ATenScalarT>::type *__restrict__ z4_4d,
    const typename DevHalf<ATenScalarT>::type *__restrict__ prev_cell,

    const typename DevHalf<ATenScalarT>::type *__restrict__ w_cell,
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_cell,
    const typename DevHalf<ATenScalarT>::type *__restrict__ w_state,
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_state,

    int64_t states_row_stride, int64_t z4_row_stride, int64_t cell_row_stride,
    int64_t out_row_stride, int D_eff, int D_gate,

    float alpha, float eps_cell, float eps_state, int use_fast_gelu) {
  using DH = DevHalf<ATenScalarT>;
  using H = typename DH::type;
  using VPack = Pack<H, VEC>;
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;

  const int row = blockIdx.x;
  const H *row_s = states_4d + row * states_row_stride;
  const H *row_z4 = z4_4d + row * z4_row_stride;
  const H *row_pc = prev_cell + row * cell_row_stride;

  H *row_out_cell = out_cell + row * out_row_stride;
  H *row_out_state = out_state + row * out_row_stride;

  const int vec_len = D_eff / VEC;

  float local_ss_cpre = 0.f;
  const VPack *s3_vec = reinterpret_cast<const VPack *>(row_s + 3 * D_gate);
  const VPack *z3_vec = reinterpret_cast<const VPack *>(row_z4 + 3 * D_gate);

  for (int i = threadIdx.x; i < vec_len; i += BLOCK_THREADS) {
    VPack s3 = s3_vec[i];
    VPack z3 = z3_vec[i];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float c = DH::to_float(s3.v[k]) + alpha * DH::to_float(z3.v[k]);
      local_ss_cpre += c * c;
    }
  }

  __shared__ typename BlockReduce::TempStorage temp0;
  float sumsq_cpre = BlockReduce(temp0).Sum(local_ss_cpre);

  __shared__ float inv_rms_cpre;
  if (threadIdx.x == 0) {
    float mean = sumsq_cpre / static_cast<float>(D_eff);
    inv_rms_cpre = rsqrtf(mean + eps_cell);
  }
  __syncthreads();

  float local_ss_cnew = 0.f;

  const VPack *s0_vec = reinterpret_cast<const VPack *>(row_s + 0 * D_gate);
  const VPack *s1_vec = reinterpret_cast<const VPack *>(row_s + 1 * D_gate);
  const VPack *s2_vec = reinterpret_cast<const VPack *>(row_s + 2 * D_gate);

  const VPack *z0_vec = reinterpret_cast<const VPack *>(row_z4 + 0 * D_gate);
  const VPack *z1_vec = reinterpret_cast<const VPack *>(row_z4 + 1 * D_gate);
  const VPack *z2_vec = reinterpret_cast<const VPack *>(row_z4 + 2 * D_gate);

  const VPack *pc_vec = reinterpret_cast<const VPack *>(row_pc);
  VPack *outc_vec = reinterpret_cast<VPack *>(row_out_cell);

  const VPack *wcell_vec = reinterpret_cast<const VPack *>(w_cell);
  const VPack *bcell_vec = reinterpret_cast<const VPack *>(b_cell);

  const bool use_w_cell = (w_cell != nullptr);
  const bool use_b_cell = (b_cell != nullptr);

  for (int i = threadIdx.x; i < vec_len; i += BLOCK_THREADS) {
    VPack s0 = s0_vec[i], s1 = s1_vec[i], s2 = s2_vec[i], s3 = s3_vec[i];
    VPack z0 = z0_vec[i], z1 = z1_vec[i], z2 = z2_vec[i], z3 = z3_vec[i];
    VPack pc = pc_vec[i];

    VPack oc;
    VPack wcell, bcell;
    if (use_w_cell)
      wcell = wcell_vec[i];
    if (use_b_cell)
      bcell = bcell_vec[i];

#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float pre_f = DH::to_float(s0.v[k]) + alpha * DH::to_float(z0.v[k]);
      float pre_i = DH::to_float(s1.v[k]) + alpha * DH::to_float(z1.v[k]);
      float cpre = DH::to_float(s3.v[k]) + alpha * DH::to_float(z3.v[k]);

      float fgate = sigmoid_f(pre_f);
      float igate = sigmoid_f(pre_i);

      float cn = cpre * inv_rms_cpre;
      if (use_w_cell)
        cn *= DH::to_float(wcell.v[k]);
      if (use_b_cell)
        cn += DH::to_float(bcell.v[k]);

      float cact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));
      float pcv = DH::to_float(pc.v[k]);

      float cnew = pcv * fgate + cact * igate;

      local_ss_cnew += cnew * cnew;
      oc.v[k] = DH::from_float(cnew);
    }
    outc_vec[i] = oc;
  }

  __shared__ typename BlockReduce::TempStorage temp1;
  float sumsq_cnew = BlockReduce(temp1).Sum(local_ss_cnew);

  __shared__ float inv_rms_cnew;
  if (threadIdx.x == 0) {
    float mean = sumsq_cnew / static_cast<float>(D_eff);
    inv_rms_cnew = rsqrtf(mean + eps_state);
  }
  __syncthreads();

  const VPack *outc_read = reinterpret_cast<const VPack *>(row_out_cell);
  VPack *outs_vec = reinterpret_cast<VPack *>(row_out_state);

  const VPack *wstate_vec = reinterpret_cast<const VPack *>(w_state);
  const VPack *bstate_vec = reinterpret_cast<const VPack *>(b_state);
  const bool use_w_st = (w_state != nullptr);
  const bool use_b_st = (b_state != nullptr);

  for (int i = threadIdx.x; i < vec_len; i += BLOCK_THREADS) {
    VPack s2 = s2_vec[i];
    VPack z2 = z2_vec[i];
    VPack oc = outc_read[i];

    VPack wst, bst;
    if (use_w_st)
      wst = wstate_vec[i];
    if (use_b_st)
      bst = bstate_vec[i];

    VPack os;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float cnew = DH::to_float(oc.v[k]);
      float cn = cnew * inv_rms_cnew;
      if (use_w_st)
        cn *= DH::to_float(wst.v[k]);
      if (use_b_st)
        cn += DH::to_float(bst.v[k]);

      float sact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));

      float pre_o = DH::to_float(s2.v[k]) + alpha * DH::to_float(z2.v[k]);
      float ogate = sigmoid_f(pre_o);

      float st = sact * ogate;
      os.v[k] = DH::from_float(st);
    }
    outs_vec[i] = os;
  }
}

template <typename ATenScalarT, int BLOCK_THREADS>
__global__ void sum_lstm_scalar_kernel(

    typename DevHalf<ATenScalarT>::type *__restrict__ out_state,
    typename DevHalf<ATenScalarT>::type *__restrict__ out_cell,

    const typename DevHalf<ATenScalarT>::type *__restrict__ states_4d,
    const typename DevHalf<ATenScalarT>::type *__restrict__ z4_4d,
    const typename DevHalf<ATenScalarT>::type *__restrict__ prev_cell,

    const typename DevHalf<ATenScalarT>::type *__restrict__ w_cell,
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_cell,
    const typename DevHalf<ATenScalarT>::type *__restrict__ w_state,
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_state,

    int64_t states_row_stride, int64_t z4_row_stride, int64_t cell_row_stride,
    int64_t out_row_stride, int D_eff, int D_gate,

    float alpha, float eps_cell, float eps_state, int use_fast_gelu) {
  using DH = DevHalf<ATenScalarT>;
  using H = typename DH::type;
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;

  const int row = blockIdx.x;
  const H *s = states_4d + row * states_row_stride;
  const H *z4 = z4_4d + row * z4_row_stride;
  const H *pc = prev_cell + row * cell_row_stride;

  H *oc = out_cell + row * out_row_stride;
  H *os = out_state + row * out_row_stride;

  float local_ss_cpre = 0.f;
  for (int j = threadIdx.x; j < D_eff; j += BLOCK_THREADS) {
    float c = DH::to_float(s[3 * D_gate + j]) +
              alpha * DH::to_float(z4[3 * D_gate + j]);
    local_ss_cpre += c * c;
  }

  __shared__ typename BlockReduce::TempStorage temp0;
  float sumsq_cpre = BlockReduce(temp0).Sum(local_ss_cpre);

  __shared__ float inv_rms_cpre;
  if (threadIdx.x == 0) {
    float mean = sumsq_cpre / static_cast<float>(D_eff);
    inv_rms_cpre = rsqrtf(mean + eps_cell);
  }
  __syncthreads();

  float local_ss_cnew = 0.f;
  for (int j = threadIdx.x; j < D_eff; j += BLOCK_THREADS) {
    float pre_f = DH::to_float(s[0 * D_gate + j]) +
                  alpha * DH::to_float(z4[0 * D_gate + j]);
    float pre_i = DH::to_float(s[1 * D_gate + j]) +
                  alpha * DH::to_float(z4[1 * D_gate + j]);
    float cpre = DH::to_float(s[3 * D_gate + j]) +
                 alpha * DH::to_float(z4[3 * D_gate + j]);

    float fgate = sigmoid_f(pre_f);
    float igate = sigmoid_f(pre_i);

    float cn = cpre * inv_rms_cpre;
    if (w_cell)
      cn *= DH::to_float(w_cell[j]);
    if (b_cell)
      cn += DH::to_float(b_cell[j]);

    float cact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));
    float cnew = DH::to_float(pc[j]) * fgate + cact * igate;

    oc[j] = DH::from_float(cnew);
    local_ss_cnew += cnew * cnew;
  }

  __shared__ typename BlockReduce::TempStorage temp1;
  float sumsq_cnew = BlockReduce(temp1).Sum(local_ss_cnew);

  __shared__ float inv_rms_cnew;
  if (threadIdx.x == 0) {
    float mean = sumsq_cnew / static_cast<float>(D_eff);
    inv_rms_cnew = rsqrtf(mean + eps_state);
  }
  __syncthreads();

  for (int j = threadIdx.x; j < D_eff; j += BLOCK_THREADS) {
    float cn = DH::to_float(oc[j]) * inv_rms_cnew;
    if (w_state)
      cn *= DH::to_float(w_state[j]);
    if (b_state)
      cn += DH::to_float(b_state[j]);

    float sact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));
    float pre_o = DH::to_float(s[2 * D_gate + j]) +
                  alpha * DH::to_float(z4[2 * D_gate + j]);
    float ogate = sigmoid_f(pre_o);

    float st = sact * ogate;
    os[j] = DH::from_float(st);
  }
}

template <typename ATenScalarT>
static std::tuple<torch::Tensor, torch::Tensor>
sum_lstm_cuda_impl(const torch::Tensor &states_4d, const torch::Tensor &z4_4d,
                   const torch::Tensor &prev_cell_d,
                   const c10::optional<torch::Tensor> &w_cell,
                   const c10::optional<torch::Tensor> &b_cell,
                   const c10::optional<torch::Tensor> &w_state,
                   const c10::optional<torch::Tensor> &b_state, double alpha_d,
                   double eps_cell_d, double eps_state_d, bool use_fast_gelu) {
  TORCH_CHECK(states_4d.is_cuda() && z4_4d.is_cuda() && prev_cell_d.is_cuda(),
              "sum_lstm: inputs must be CUDA tensors");
  TORCH_CHECK(IsSupported<ATenScalarT>::value,
              "sum_lstm: dtype must be fp16 or bf16");

  const auto dtype = states_4d.scalar_type();
  TORCH_CHECK(z4_4d.scalar_type() == dtype &&
                  prev_cell_d.scalar_type() == dtype,
              "sum_lstm: all input dtypes must match");

  const int64_t hidden4 = states_4d.size(-1);
  TORCH_CHECK(hidden4 > 0 && hidden4 % 4 == 0,
              "sum_lstm: last dim of states must be 4*D_gate");
  const int64_t D_gate = hidden4 / 4;

  TORCH_CHECK(z4_4d.size(-1) == hidden4,
              "sum_lstm: z4 must have last dim 4*D_gate");

  const int64_t D_cell = prev_cell_d.size(-1);
  TORCH_CHECK(D_cell == D_gate,
              "sum_lstm: prev_cell last dim must equal D_gate. Got ", D_cell,
              " vs expected ", D_gate, ".");
  const int64_t D_eff = D_gate;

  TORCH_CHECK(states_4d.stride(-1) == 1 && z4_4d.stride(-1) == 1 &&
                  prev_cell_d.stride(-1) == 1,
              "sum_lstm: last dimension must be contiguous (stride -1 == 1)");

  auto check_opt_len = [&](const c10::optional<torch::Tensor> &t,
                           const char *name) {
    if (t.has_value()) {
      TORCH_CHECK(t->is_cuda(), "sum_lstm: ", name, " must be CUDA");
      TORCH_CHECK(t->scalar_type() == dtype, "sum_lstm: ", name,
                  " dtype mismatch");
      TORCH_CHECK(t->numel() >= D_eff, "sum_lstm: ", name,
                  " must have length >= D_eff");
      TORCH_CHECK(t->is_contiguous(), "sum_lstm: ", name,
                  " must be contiguous");
    }
  };
  check_opt_len(w_cell, "w_cell");
  check_opt_len(b_cell, "b_cell");
  check_opt_len(w_state, "w_state");
  check_opt_len(b_state, "b_state");

  auto s2 = states_4d.view({-1, hidden4});
  auto z2 = z4_4d.view({-1, hidden4});
  auto p2 = prev_cell_d.view({-1, D_cell});

  const int64_t rows = s2.size(0);
  const int64_t s_stride = s2.stride(0);
  const int64_t z_stride = z2.stride(0);
  const int64_t p_stride = p2.stride(0);

  auto out_cell = at::empty_strided(p2.sizes(), p2.strides(), p2.options());
  auto out_state = at::empty_strided(p2.sizes(), p2.strides(), p2.options());

  const int64_t out_stride_cell = out_cell.stride(0);
  const int64_t out_stride_state = out_state.stride(0);
  TORCH_CHECK(out_stride_cell == out_stride_state,
              "sum_lstm: internal - output strides mismatch");

  auto out_cell_orig = out_cell.view(prev_cell_d.sizes());
  auto out_state_orig = out_state.view(prev_cell_d.sizes());

  using H = typename DevHalf<ATenScalarT>::type;
  const auto stream = at::cuda::getCurrentCUDAStream();

  H *out_state_ptr = reinterpret_cast<H *>(out_state.data_ptr());
  H *out_cell_ptr = reinterpret_cast<H *>(out_cell.data_ptr());
  const H *s_ptr = reinterpret_cast<const H *>(s2.data_ptr());
  const H *z_ptr = reinterpret_cast<const H *>(z2.data_ptr());
  const H *p_ptr = reinterpret_cast<const H *>(p2.data_ptr());

  const bool row_aligned_8 =
      (s_stride % 8 == 0) && (z_stride % 8 == 0) && (out_stride_cell % 8 == 0);
  const bool row_aligned_4 =
      (s_stride % 4 == 0) && (z_stride % 4 == 0) && (out_stride_cell % 4 == 0);

  const bool base_aligned_16 =
      is_aligned(s_ptr, 16) && is_aligned(z_ptr, 16) && is_aligned(p_ptr, 16) &&
      is_aligned(out_state_ptr, 16) && is_aligned(out_cell_ptr, 16);

  const bool base_aligned_8 =
      is_aligned(s_ptr, 8) && is_aligned(z_ptr, 8) && is_aligned(p_ptr, 8) &&
      is_aligned(out_state_ptr, 8) && is_aligned(out_cell_ptr, 8);

  const bool gates_div_8 = (D_gate % 8 == 0);
  const bool gates_div_4 = (D_gate % 4 == 0);

  const bool can_vec8 =
      (D_eff % 8 == 0) && gates_div_8 && row_aligned_8 && base_aligned_16;

  const bool can_vec4 =
      (D_eff % 4 == 0) && gates_div_4 && row_aligned_4 && base_aligned_8;

  float alpha = static_cast<float>(alpha_d);
  float eps_cell = static_cast<float>(eps_cell_d);
  float eps_state = static_cast<float>(eps_state_d);
  int fast = use_fast_gelu ? 1 : 0;

  constexpr int BLK_128 = 128;
  constexpr int BLK_256 = 256;

  dim3 grid(rows);

  auto wcell_ptr =
      w_cell ? reinterpret_cast<const H *>(w_cell->data_ptr()) : nullptr;
  auto bcell_ptr =
      b_cell ? reinterpret_cast<const H *>(b_cell->data_ptr()) : nullptr;
  auto wstate_ptr =
      w_state ? reinterpret_cast<const H *>(w_state->data_ptr()) : nullptr;
  auto bstate_ptr =
      b_state ? reinterpret_cast<const H *>(b_state->data_ptr()) : nullptr;

#define LAUNCH_VEC(V, BLK)                                                     \
  do {                                                                         \
    dim3 block((BLK));                                                         \
    sum_lstm_vec_kernel<ATenScalarT, (V), (BLK)><<<grid, block, 0, stream>>>(  \
        out_state_ptr, out_cell_ptr, s_ptr, z_ptr, p_ptr, wcell_ptr,           \
        bcell_ptr, wstate_ptr, bstate_ptr, s_stride, z_stride, p_stride,       \
        out_stride_cell, static_cast<int>(D_eff), static_cast<int>(D_gate),    \
        alpha, eps_cell, eps_state, fast);                                     \
  } while (0)

#define LAUNCH_SCALAR(BLK)                                                     \
  do {                                                                         \
    dim3 block((BLK));                                                         \
    sum_lstm_scalar_kernel<ATenScalarT, (BLK)><<<grid, block, 0, stream>>>(    \
        out_state_ptr, out_cell_ptr, s_ptr, z_ptr, p_ptr, wcell_ptr,           \
        bcell_ptr, wstate_ptr, bstate_ptr, s_stride, z_stride, p_stride,       \
        out_stride_cell, static_cast<int>(D_eff), static_cast<int>(D_gate),    \
        alpha, eps_cell, eps_state, fast);                                     \
  } while (0)

  if (can_vec8) {
    LAUNCH_VEC(8, BLK_128);
  } else if (can_vec4) {
    LAUNCH_VEC(4, BLK_256);
  } else {
    LAUNCH_SCALAR(BLK_256);
  }

#undef LAUNCH_VEC
#undef LAUNCH_SCALAR

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_state_orig, out_cell_orig};
}

std::tuple<torch::Tensor, torch::Tensor>
sum_lstm_cuda(const torch::Tensor &states_4d, const torch::Tensor &z4_4d,
              const torch::Tensor &prev_cell_d,
              const c10::optional<torch::Tensor> &w_cell,
              const c10::optional<torch::Tensor> &b_cell,
              const c10::optional<torch::Tensor> &w_state,
              const c10::optional<torch::Tensor> &b_state, double alpha,
              double eps_cell, double eps_state, bool use_fast_gelu) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(states_4d));
  switch (states_4d.scalar_type()) {
  case at::kHalf:
    return sum_lstm_cuda_impl<at::Half>(states_4d, z4_4d, prev_cell_d, w_cell,
                                        b_cell, w_state, b_state, alpha,
                                        eps_cell, eps_state, use_fast_gelu);
  case at::kBFloat16:
    return sum_lstm_cuda_impl<at::BFloat16>(
        states_4d, z4_4d, prev_cell_d, w_cell, b_cell, w_state, b_state, alpha,
        eps_cell, eps_state, use_fast_gelu);
  default:
    TORCH_CHECK(false, "sum_lstm: only fp16 and bf16 are supported.");
  }
}