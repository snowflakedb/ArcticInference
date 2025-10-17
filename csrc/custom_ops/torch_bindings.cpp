#include "custom_ops.h"

#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(arctic_inference, ops) {
  ops.def("reshape_and_cache_flash_bulk(Tensor keys,"
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

  ops.def("reshape_and_cache_flash_fp4(Tensor key,"
          "                          Tensor value,"
          "                          Tensor(c!) key_cache,"
          "                          Tensor(d!) value_cache,"
          "                          Tensor slot_mapping,"
          "                          str kv_cache_dtype,"
          "                          Tensor k_scale,"
          "                          Tensor v_scale,"
          "                          Tensor(e!) key_scale_cache,"
          "                          Tensor(f!) value_scale_cache) -> ()");
  ops.impl("reshape_and_cache_flash_fp4", torch::kCUDA,
           &reshape_and_cache_flash_fp4);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
