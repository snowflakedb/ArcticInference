#include "custom_ops.h"

#include <torch/library.h>

TORCH_LIBRARY(arctic_inference, ops) {
  ops.def(
      "copy_caches_with_index(Tensor(a!)[] src_caches, Tensor[](b!) dst_caches, "
      "Tensor shared_indices) -> ()");
  ops.impl("copy_caches_with_index", torch::kCUDA, &copy_caches_with_index);

  ops.def(
      "reshape_and_cache_flash_bulk(Tensor keys,"
      "                             Tensor values,"
      "                             Tensor(c!)[] key_caches," 
      "                             Tensor(d!)[] value_caches,"
      "                             Tensor slot_mapping,"
      "                             str kv_cache_dtype,"
      "                             Tensor(e)[] k_scales,"
      "                             Tensor(f)[] v_scales,"
      "                             int num_heads,"
      "                             int head_size) -> ()");
  ops.impl("reshape_and_cache_flash_bulk", torch::kCUDA,
           &reshape_and_cache_flash_bulk);
}
