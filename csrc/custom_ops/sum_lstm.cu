// sum_lstm.cu
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <type_traits>

// ------------------------ dtype helpers (same pattern as speculator_ln) ------------------------
template <typename T> struct IsSupported : std::false_type {};
template <> struct IsSupported<at::Half> : std::true_type {};
template <> struct IsSupported<at::BFloat16> : std::true_type {};

__device__ __forceinline__ float to_float_device(__half h) { return __half2float(h); }
__device__ __forceinline__ __half from_float_device(float f) { return __float2half(f); }

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
  static __device__ __forceinline__ float to_float(__half h) { return to_float_device(h); }
  static __device__ __forceinline__ __half from_float(float f) { return from_float_device(f); }
};
template <> struct DevHalf<at::BFloat16> {
  using type = __nv_bfloat16;
  static __device__ __forceinline__ float to_float(__nv_bfloat16 h) { return to_float_device(h); }
  static __device__ __forceinline__ __nv_bfloat16 from_float(float f) { return from_float_device_bf16(f); }
};

template <typename T, int N> struct alignas(sizeof(T) * N) Pack { T v[N]; };

struct SumOp {
  __device__ __forceinline__ float operator()(const float &a, const float &b) const { return a + b; }
};

static inline bool is_aligned(const void *p, size_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) % bytes) == 0;
}

// ------------------------ math helpers ------------------------
__device__ __forceinline__ float sigmoid_f(float x) {
  // numerically stable sigmoid
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_erf(float x) {
  // exact GELU (matches torch.nn.GELU(default))
  const float kInvSqrt2 = 0.70710678118654752440f; // 1/sqrt(2)
  return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  // fast/approximate GELU (Hendrycks 2016)
  const float kSqrt2OverPi = 0.79788456080286535588f;
  return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

// ------------------------ fused kernels ------------------------
template <typename ATenScalarT, int VEC>
__global__ void sum_lstm_vec_kernel(
    // outputs
    typename DevHalf<ATenScalarT>::type *__restrict__ out_state, // [rows, D]
    typename DevHalf<ATenScalarT>::type *__restrict__ out_cell,  // [rows, D]
    // inputs
    const typename DevHalf<ATenScalarT>::type *__restrict__ states_4d, // [rows, 4D]
    const typename DevHalf<ATenScalarT>::type *__restrict__ z4_4d,     // [rows, 4D] (the repeated emb)
    const typename DevHalf<ATenScalarT>::type *__restrict__ prev_cell, // [rows, D]
    // layernorm params
    const typename DevHalf<ATenScalarT>::type *__restrict__ w_cell, // [D] or null
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_cell, // [D] or null
    const typename DevHalf<ATenScalarT>::type *__restrict__ w_state,// [D] or null
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_state,// [D] or null
    // strides / sizes
    int64_t states_row_stride, int64_t z4_row_stride, int64_t cell_row_stride,
    int64_t out_row_stride, int D,
    // scalars
    float alpha, float eps_cell, float eps_state,
    int use_fast_gelu)
{
  using DH = DevHalf<ATenScalarT>;
  using H  = typename DH::type;
  using VPack = Pack<H, VEC>;

  const int row = blockIdx.x;
  const H *row_s  = states_4d + row * states_row_stride;
  const H *row_z4 = z4_4d     + row * z4_row_stride;
  const H *row_pc = prev_cell + row * cell_row_stride;

  H *row_out_cell  = out_cell  + row * out_row_stride;
  H *row_out_state = out_state + row * out_row_stride;

  const int vec_len = D / VEC; // vector packs along D

  // ---- Stage 0: RMS of cell_candidate pre-act (c_pre = s3 + alpha*z3) ----
  float local_ss_cpre = 0.f;
  const VPack* s3_vec  = reinterpret_cast<const VPack*>(row_s  + 3*D);
  const VPack* z3_vec  = reinterpret_cast<const VPack*>(row_z4 + 3*D);

  for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
    VPack s3 = s3_vec[i];
    VPack z3 = z3_vec[i];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float c = DH::to_float(s3.v[k]) + alpha * DH::to_float(z3.v[k]);
      local_ss_cpre += c * c;
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage temp0;
  float sumsq_cpre = BlockReduce(temp0).Reduce(local_ss_cpre, SumOp{}, blockDim.x);

  __shared__ float inv_rms_cpre;
  if (threadIdx.x == 0) {
    float mean = sumsq_cpre / static_cast<float>(D);
    inv_rms_cpre = rsqrtf(mean + eps_cell);
  }
  __syncthreads();

  // ---- Stage 1: build new cell, accumulate its RMS ----
  float local_ss_cnew = 0.f;

  const VPack* s0_vec = reinterpret_cast<const VPack*>(row_s + 0*D);
  const VPack* s1_vec = reinterpret_cast<const VPack*>(row_s + 1*D);
  const VPack* s2_vec = reinterpret_cast<const VPack*>(row_s + 2*D);
  // s3_vec already defined

  const VPack* z0_vec = reinterpret_cast<const VPack*>(row_z4 + 0*D);
  const VPack* z1_vec = reinterpret_cast<const VPack*>(row_z4 + 1*D);
  const VPack* z2_vec = reinterpret_cast<const VPack*>(row_z4 + 2*D);
  // z3_vec already defined

  const VPack* pc_vec = reinterpret_cast<const VPack*>(row_pc);
  VPack* outc_vec     = reinterpret_cast<VPack*>(row_out_cell);

  const VPack* wcell_vec = reinterpret_cast<const VPack*>(w_cell);
  const VPack* bcell_vec = reinterpret_cast<const VPack*>(b_cell);

  for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
    VPack s0 = s0_vec[i], s1 = s1_vec[i], s2 = s2_vec[i], s3 = s3_vec[i];
    VPack z0 = z0_vec[i], z1 = z1_vec[i], z2 = z2_vec[i], z3 = z3_vec[i];
    VPack pc = pc_vec[i];

    VPack oc; // out cell pack
    VPack wcell, bcell;
    bool use_w_cell = (w_cell != nullptr);
    bool use_b_cell = (b_cell != nullptr);
    if (use_w_cell) wcell = wcell_vec[i];
    if (use_b_cell) bcell = bcell_vec[i];

#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float z0f = DH::to_float(z0.v[k]);
      float z1f = DH::to_float(z1.v[k]);
      float z2f = DH::to_float(z2.v[k]);
      float z3f = DH::to_float(z3.v[k]);

      float pre_f = DH::to_float(s0.v[k]) + alpha * z0f;
      float pre_i = DH::to_float(s1.v[k]) + alpha * z1f;
      float pre_o = DH::to_float(s2.v[k]) + alpha * z2f; // used in stage 3
      (void)pre_o; // silence unused warning (will recompute later anyway)
      float cpre  = DH::to_float(s3.v[k]) + alpha * z3f;

      float fgate = sigmoid_f(pre_f);
      float igate = sigmoid_f(pre_i);

      // cell LN (RMS), then GELU
      float cn = cpre * inv_rms_cpre;
      if (use_w_cell) cn *= DH::to_float(wcell.v[k]);
      if (use_b_cell) cn += DH::to_float(bcell.v[k]);

      float cact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));
      float pcv  = DH::to_float(pc.v[k]);

      float cnew = pcv * fgate + cact * igate;

      local_ss_cnew += cnew * cnew;
      oc.v[k] = DH::from_float(cnew);
    }
    outc_vec[i] = oc; // write new cell (kept for output + stage 3)
  }

  __shared__ typename BlockReduce::TempStorage temp1;
  float sumsq_cnew = BlockReduce(temp1).Reduce(local_ss_cnew, SumOp{}, blockDim.x);

  __shared__ float inv_rms_cnew;
  if (threadIdx.x == 0) {
    float mean = sumsq_cnew / static_cast<float>(D);
    inv_rms_cnew = rsqrtf(mean + eps_state);
  }
  __syncthreads();

  // ---- Stage 3: final state = GELU( LN(cnew) ) * ogate ----
  const VPack* outc_read = reinterpret_cast<const VPack*>(row_out_cell);
  VPack* outs_vec        = reinterpret_cast<VPack*>(row_out_state);

  const VPack* wstate_vec = reinterpret_cast<const VPack*>(w_state);
  const VPack* bstate_vec = reinterpret_cast<const VPack*>(b_state);

  for (int i = threadIdx.x; i < vec_len; i += blockDim.x) {
    VPack s2 = s2_vec[i];
    VPack z2 = z2_vec[i];
    VPack oc = outc_read[i];

    VPack wst, bst;
    bool use_w_st = (w_state != nullptr);
    bool use_b_st = (b_state != nullptr);
    if (use_w_st) wst = wstate_vec[i];
    if (use_b_st) bst = bstate_vec[i];

    VPack os; // out state pack
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float cnew = DH::to_float(oc.v[k]);
      float cn   = cnew * inv_rms_cnew;
      if (use_w_st) cn *= DH::to_float(wst.v[k]);
      if (use_b_st) cn += DH::to_float(bst.v[k]);

      float sact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));

      float pre_o = DH::to_float(s2.v[k]) + alpha * DH::to_float(z2.v[k]);
      float ogate = sigmoid_f(pre_o);

      float st = sact * ogate;
      os.v[k] = DH::from_float(st);
    }
    outs_vec[i] = os;
  }
}

template <typename ATenScalarT>
__global__ void sum_lstm_scalar_kernel(
    // outputs
    typename DevHalf<ATenScalarT>::type *__restrict__ out_state,
    typename DevHalf<ATenScalarT>::type *__restrict__ out_cell,
    // inputs
    const typename DevHalf<ATenScalarT>::type *__restrict__ states_4d,
    const typename DevHalf<ATenScalarT>::type *__restrict__ z4_4d,
    const typename DevHalf<ATenScalarT>::type *__restrict__ prev_cell,
    // layernorm params
    const typename DevHalf<ATenScalarT>::type *__restrict__ w_cell,
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_cell,
    const typename DevHalf<ATenScalarT>::type *__restrict__ w_state,
    const typename DevHalf<ATenScalarT>::type *__restrict__ b_state,
    // strides / sizes
    int64_t states_row_stride, int64_t z4_row_stride, int64_t cell_row_stride,
    int64_t out_row_stride, int D,
    // scalars
    float alpha, float eps_cell, float eps_state,
    int use_fast_gelu)
{
  using DH = DevHalf<ATenScalarT>;
  using H  = typename DH::type;

  const int row = blockIdx.x;
  const H *s  = states_4d + row * states_row_stride;
  const H *z4 = z4_4d     + row * z4_row_stride;
  const H *pc = prev_cell + row * cell_row_stride;

  H *oc = out_cell  + row * out_row_stride;
  H *os = out_state + row * out_row_stride;

  // Stage 0: RMS(c_pre)
  float local_ss_cpre = 0.f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float c = DH::to_float(s[3*D + j]) + alpha * DH::to_float(z4[3*D + j]);
    local_ss_cpre += c * c;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage temp0;
  float sumsq_cpre = BlockReduce(temp0).Reduce(local_ss_cpre, SumOp{}, blockDim.x);

  __shared__ float inv_rms_cpre;
  if (threadIdx.x == 0) {
    float mean = sumsq_cpre / static_cast<float>(D);
    inv_rms_cpre = rsqrtf(mean + eps_cell);
  }
  __syncthreads();

  // Stage 1: new cell + its RMS
  float local_ss_cnew = 0.f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float pre_f = DH::to_float(s[0*D + j]) + alpha * DH::to_float(z4[0*D + j]);
    float pre_i = DH::to_float(s[1*D + j]) + alpha * DH::to_float(z4[1*D + j]);
    float cpre  = DH::to_float(s[3*D + j]) + alpha * DH::to_float(z4[3*D + j]);

    float fgate = sigmoid_f(pre_f);
    float igate = sigmoid_f(pre_i);

    float cn = cpre * inv_rms_cpre;
    if (w_cell) cn *= DH::to_float(w_cell[j]);
    if (b_cell) cn += DH::to_float(b_cell[j]);

    float cact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));
    float cnew = DH::to_float(pc[j]) * fgate + cact * igate;

    oc[j] = DH::from_float(cnew);
    local_ss_cnew += cnew * cnew;
  }

  __shared__ typename BlockReduce::TempStorage temp1;
  float sumsq_cnew = BlockReduce(temp1).Reduce(local_ss_cnew, SumOp{}, blockDim.x);

  __shared__ float inv_rms_cnew;
  if (threadIdx.x == 0) {
    float mean = sumsq_cnew / static_cast<float>(D);
    inv_rms_cnew = rsqrtf(mean + eps_state);
  }
  __syncthreads();

  // Stage 3: final state
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    float cn = DH::to_float(oc[j]) * inv_rms_cnew;
    if (w_state) cn *= DH::to_float(w_state[j]);
    if (b_state) cn += DH::to_float(b_state[j]);

    float sact = (use_fast_gelu ? gelu_tanh(cn) : gelu_erf(cn));
    float pre_o = DH::to_float(s[2*D + j]) + alpha * DH::to_float(z4[2*D + j]);
    float ogate = sigmoid_f(pre_o);

    float st = sact * ogate;
    os[j] = DH::from_float(st);
  }
}

// ------------------------ host dispatch ------------------------
template <typename ATenScalarT>
static std::tuple<torch::Tensor, torch::Tensor> sum_lstm_cuda_impl(
    const torch::Tensor& states_4d,   // [..., 4D]
    const torch::Tensor& z4_4d,       // [..., 4D]
    const torch::Tensor& prev_cell_d, // [..., D]
    const c10::optional<torch::Tensor>& w_cell,
    const c10::optional<torch::Tensor>& b_cell,
    const c10::optional<torch::Tensor>& w_state,
    const c10::optional<torch::Tensor>& b_state,
    double alpha_d, double eps_cell_d, double eps_state_d,
    bool use_fast_gelu)
{
  TORCH_CHECK(states_4d.is_cuda() && z4_4d.is_cuda() && prev_cell_d.is_cuda(),
              "sum_lstm: inputs must be CUDA tensors");
  TORCH_CHECK(IsSupported<ATenScalarT>::value, "sum_lstm: dtype must be fp16 or bf16");

  const auto dtype = states_4d.scalar_type();
  TORCH_CHECK(z4_4d.scalar_type() == dtype && prev_cell_d.scalar_type() == dtype,
              "sum_lstm: all input dtypes must match");

  const int64_t hidden4 = states_4d.size(-1);
  TORCH_CHECK(hidden4 > 0 && hidden4 % 4 == 0, "sum_lstm: last dim of states must be 4*D");
  const int64_t D = hidden4 / 4;

  TORCH_CHECK(z4_4d.size(-1) == hidden4, "sum_lstm: z4 must have last dim 4*D");
  TORCH_CHECK(prev_cell_d.size(-1) == D, "sum_lstm: prev_cell last dim must be D");

  TORCH_CHECK(states_4d.stride(-1) == 1 && z4_4d.stride(-1) == 1 && prev_cell_d.stride(-1) == 1,
              "sum_lstm: last dimension must be contiguous (stride -1 == 1)");

  // Flatten to 2D while keeping row strides
  auto s2 = states_4d.view({-1, hidden4});
  auto z2 = z4_4d.view({-1, hidden4});
  auto p2 = prev_cell_d.view({-1, D});

  const int64_t rows     = s2.size(0);
  const int64_t s_stride = s2.stride(0);   // elements per row in states (== 4*D + pad4)
  const int64_t z_stride = z2.stride(0);   // elements per row in z4 (== 4*D + pad4)
  const int64_t p_stride = p2.stride(0);   // elements per row in prev_cell (== D + padD)

  // --- Allocate outputs with the **same strides** as p2 (robust for pitched inputs)
  auto out_cell  = at::empty_strided(p2.sizes(),  p2.strides(),  p2.options());
  auto out_state = at::empty_strided(p2.sizes(),  p2.strides(),  p2.options());

  // Or (acceptable alternative): at::empty_like(p2) and then use out_cell.stride(0)
  const int64_t out_stride_cell  = out_cell.stride(0);
  const int64_t out_stride_state = out_state.stride(0);
  TORCH_CHECK(out_stride_cell == out_stride_state,
              "sum_lstm: internal - output strides mismatch");

  // Return tensors shaped like prev_cell_d; stride may differ (that’s fine).
  auto out_cell_orig  = out_cell.view(prev_cell_d.sizes());
  auto out_state_orig = out_state.view(prev_cell_d.sizes());

  using H = typename DevHalf<ATenScalarT>::type;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int BLOCK = (rows < 256) ? 1024 : 256;
  dim3 grid(rows);
  dim3 block(std::min<int64_t>(D, BLOCK));

  H* out_state_ptr = reinterpret_cast<H*>(out_state.data_ptr());
  H* out_cell_ptr  = reinterpret_cast<H*>(out_cell.data_ptr());
  const H* s_ptr   = reinterpret_cast<const H*>(s2.data_ptr());
  const H* z_ptr   = reinterpret_cast<const H*>(z2.data_ptr());
  const H* p_ptr   = reinterpret_cast<const H*>(p2.data_ptr());

  // Alignment / vectorization checks (relaxed; no effect on correctness)
  const bool row_aligned_8 =
      (s_stride % 8 == 0) && (z_stride % 8 == 0) && (out_stride_cell % 8 == 0);
  const bool row_aligned_4 =
      (s_stride % 4 == 0) && (z_stride % 4 == 0) && (out_stride_cell % 4 == 0);

  const bool can_vec8 =
      (D % 8 == 0) && row_aligned_8 &&
      is_aligned(s_ptr, 16) && is_aligned(z_ptr, 16) && is_aligned(p_ptr, 16) &&
      is_aligned(out_state_ptr, 16) && is_aligned(out_cell_ptr, 16);

  const bool can_vec4 =
      (D % 4 == 0) && row_aligned_4 &&
      is_aligned(s_ptr, 8) && is_aligned(z_ptr, 8) && is_aligned(p_ptr, 8) &&
      is_aligned(out_state_ptr, 8) && is_aligned(out_cell_ptr, 8);

  float alpha     = static_cast<float>(alpha_d);
  float eps_cell  = static_cast<float>(eps_cell_d);
  float eps_state = static_cast<float>(eps_state_d);
  int fast        = use_fast_gelu ? 1 : 0;

  if (can_vec8) {
    sum_lstm_vec_kernel<ATenScalarT, 8><<<grid, block, 0, stream>>>(
        out_state_ptr, out_cell_ptr,
        s_ptr, z_ptr, p_ptr,
        /*w_cell*/  w_cell  ? reinterpret_cast<const H*>(w_cell->data_ptr())  : nullptr,
        /*b_cell*/  b_cell  ? reinterpret_cast<const H*>(b_cell->data_ptr())  : nullptr,
        /*w_state*/ w_state ? reinterpret_cast<const H*>(w_state->data_ptr()) : nullptr,
        /*b_state*/ b_state ? reinterpret_cast<const H*>(b_state->data_ptr()) : nullptr,
        s_stride, z_stride, p_stride, out_stride_cell, (int)D,
        alpha, eps_cell, eps_state, fast);
  } else if (can_vec4) {
    sum_lstm_vec_kernel<ATenScalarT, 4><<<grid, block, 0, stream>>>(
        out_state_ptr, out_cell_ptr,
        s_ptr, z_ptr, p_ptr,
        /*w_cell*/  w_cell  ? reinterpret_cast<const H*>(w_cell->data_ptr())  : nullptr,
        /*b_cell*/  b_cell  ? reinterpret_cast<const H*>(b_cell->data_ptr())  : nullptr,
        /*w_state*/ w_state ? reinterpret_cast<const H*>(w_state->data_ptr()) : nullptr,
        /*b_state*/ b_state ? reinterpret_cast<const H*>(b_state->data_ptr()) : nullptr,
        s_stride, z_stride, p_stride, out_stride_cell, (int)D,
        alpha, eps_cell, eps_state, fast);
  } else {
    sum_lstm_scalar_kernel<ATenScalarT><<<grid, block, 0, stream>>>(
        out_state_ptr, out_cell_ptr,
        s_ptr, z_ptr, p_ptr,
        /*w_cell*/  w_cell  ? reinterpret_cast<const H*>(w_cell->data_ptr())  : nullptr,
        /*b_cell*/  b_cell  ? reinterpret_cast<const H*>(b_cell->data_ptr())  : nullptr,
        /*w_state*/ w_state ? reinterpret_cast<const H*>(w_state->data_ptr()) : nullptr,
        /*b_state*/ b_state ? reinterpret_cast<const H*>(b_state->data_ptr()) : nullptr,
        s_stride, z_stride, p_stride, out_stride_cell, (int)D,
        alpha, eps_cell, eps_state, fast);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_state_orig, out_cell_orig};
}

std::tuple<torch::Tensor, torch::Tensor> sum_lstm_cuda(
    const torch::Tensor& states_4d,   // [..., 4D]
    const torch::Tensor& z4_4d,       // [..., 4D]  (repeat along last dim)
    const torch::Tensor& prev_cell_d, // [..., D]
    const c10::optional<torch::Tensor>& w_cell,
    const c10::optional<torch::Tensor>& b_cell,
    const c10::optional<torch::Tensor>& w_state,
    const c10::optional<torch::Tensor>& b_state,
    double alpha, double eps_cell, double eps_state,
    bool use_fast_gelu)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(states_4d));
  switch (states_4d.scalar_type()) {
    case at::kHalf:
      return sum_lstm_cuda_impl<at::Half>(
          states_4d, z4_4d, prev_cell_d, w_cell, b_cell, w_state, b_state,
          alpha, eps_cell, eps_state, use_fast_gelu);
    case at::kBFloat16:
      return sum_lstm_cuda_impl<at::BFloat16>(
          states_4d, z4_4d, prev_cell_d, w_cell, b_cell, w_state, b_state,
          alpha, eps_cell, eps_state, use_fast_gelu);
    default:
      TORCH_CHECK(false, "sum_lstm: only fp16 and bf16 are supported.");
  }
}

