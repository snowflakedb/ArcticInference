#include "custom_ops.h"
#include "dispatch_utils.h"
#include "quant_utils.cuh"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

namespace vllm {

template <typename scalar_t, 
          typename cache_t, 
          Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_bulk_kernel(
    const scalar_t* __restrict__ keys,
    const scalar_t* __restrict__ values,
    int64_t* key_cache_ptrs,
    int64_t* value_cache_ptrs,
    const int64_t* __restrict__ slot_mapping,
    const int block_stride, 
    const int key_stride, 
    const int value_stride,
    const int num_heads, 
    const int head_size, 
    const int block_size,
    int64_t* k_scale_ptrs, 
    int64_t* v_scale_ptrs) {
  const int64_t layer_idx = blockIdx.x;
  const int64_t token_idx = blockIdx.y;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;

  cache_t* __restrict__ key_cache =
      reinterpret_cast<cache_t*>(key_cache_ptrs[layer_idx]);
  cache_t* __restrict__ value_cache =
      reinterpret_cast<cache_t*>(value_cache_ptrs[layer_idx]);
  const float* __restrict__ k_scale =
      reinterpret_cast<const float*>(k_scale_ptrs[layer_idx]);
  const float* __restrict__ v_scale =
      reinterpret_cast<const float*>(v_scale_ptrs[layer_idx]);

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + layer_idx * n + i;
    const int64_t src_value_idx = token_idx * value_stride + layer_idx * n + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int64_t tgt_key_value_idx = block_idx * block_stride +
                                      block_offset * num_heads * head_size +
                                      head_idx * head_size + head_offset;
    scalar_t tgt_key = keys[src_key_idx];
    scalar_t tgt_value = values[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_value_idx] = tgt_key;
      value_cache[tgt_key_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *k_scale);
      value_cache[tgt_key_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, *v_scale);
    }
  }
}

} // namespace vllm

#define CALL_RESHAPE_AND_CACHE_FLASH_BULK(KV_T, CACHE_T, KV_DTYPE)       \
  vllm::reshape_and_cache_flash_bulk_kernel<KV_T, CACHE_T, KV_DTYPE>     \
      <<<grid, block, 0, stream>>>(                                      \
          reinterpret_cast<KV_T*>(keys.data_ptr()),                      \
          reinterpret_cast<KV_T*>(values.data_ptr()),                    \
          key_cache_ptrs_tensor.data_ptr<int64_t>(),                     \
          value_cache_ptrs_tensor.data_ptr<int64_t>(),                   \
          slot_mapping.data_ptr<int64_t>(), block_stride, key_stride,    \
          value_stride, static_cast<int>(num_heads),                     \
          static_cast<int>(head_size), block_size,                       \
          k_scale_ptrs_tensor.data_ptr<int64_t>(),                       \
          v_scale_ptrs_tensor.data_ptr<int64_t>());

void reshape_and_cache_flash_bulk(
    torch::Tensor& keys, 
    torch::Tensor& values,
    std::vector<torch::Tensor> const& key_caches,
    std::vector<torch::Tensor> const& value_caches,
    torch::Tensor& slot_mapping, 
    const std::string& kv_cache_dtype, 
    std::vector<torch::Tensor> const& k_scales, 
    std::vector<torch::Tensor> const& v_scales,
    int64_t num_heads,
    int64_t head_size) {
  int num_layers = key_caches.size();

  if (num_layers == 0) {
    return;
  }

  TORCH_CHECK(num_layers == key_caches.size());
  TORCH_CHECK(num_layers == value_caches.size());
  TORCH_CHECK(num_layers == k_scales.size());
  TORCH_CHECK(num_layers == v_scales.size());

  int num_tokens = slot_mapping.size(0);
  int block_size = key_caches[0].size(1);

  int key_stride = keys.stride(0);
  int value_stride = values.stride(0);
  int block_stride = key_caches[0].stride(0);
  TORCH_CHECK(block_stride == value_caches[0].stride(0));

  int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  int64_t k_scale_ptrs[num_layers];
  int64_t v_scale_ptrs[num_layers];

  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
    k_scale_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(k_scales[layer_idx].data_ptr());
    v_scale_ptrs[layer_idx] =
        reinterpret_cast<int64_t>(v_scales[layer_idx].data_ptr());
  }

  torch::Device device_of_key = keys.device();
  const at::cuda::OptionalCUDAGuard device_guard(device_of_key);

  torch::Tensor key_cache_ptrs_tensor =
      torch::from_blob(key_cache_ptrs, {num_layers}, torch::kInt64)
          .to(device_of_key);
  torch::Tensor value_cache_ptrs_tensor =
      torch::from_blob(value_cache_ptrs, {num_layers}, torch::kInt64)
          .to(device_of_key);
  torch::Tensor k_scale_ptrs_tensor =
      torch::from_blob(k_scale_ptrs, {num_layers}, torch::kInt64)
          .to(device_of_key);
  torch::Tensor v_scale_ptrs_tensor =
      torch::from_blob(v_scale_ptrs, {num_layers}, torch::kInt64)
          .to(device_of_key);

  dim3 grid(num_layers, num_tokens);
  dim3 block(std::min(static_cast<int>(num_heads) * static_cast<int>(head_size), 512));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(keys.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH_BULK);
}