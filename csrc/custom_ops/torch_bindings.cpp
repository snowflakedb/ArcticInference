#include "custom_ops.h"

#include <torch/library.h>

TORCH_LIBRARY(arctic_inference, ops) {
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

  ops.def(
      "cutlass_scaled_fp4_mm_sm100a(Tensor! out, Tensor a, Tensor b,"
      "                             Tensor block_scale_a, Tensor block_scale_b,"
      "                             Tensor alpha) -> ()");
  ops.impl("cutlass_scaled_fp4_mm_sm100a", torch::kCUDA, &cutlass_scaled_fp4_mm_sm100a);
}
