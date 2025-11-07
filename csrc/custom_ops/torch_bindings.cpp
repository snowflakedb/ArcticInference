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
          "                            Tensor value,"
          "                            Tensor(c!) key_cache,"
          "                            Tensor(d!) value_cache,"
          "                            Tensor slot_mapping,"
          "                            str kv_cache_dtype,"
          "                            Tensor k_scale,"
          "                            Tensor v_scale,"
          "                            Tensor(e!) key_scale_cache,"
          "                            Tensor(f!) value_scale_cache) -> ()");
  ops.impl("reshape_and_cache_flash_fp4", torch::kCUDA,
           &reshape_and_cache_flash_fp4);

  ops.def("speculator_ln_cuda(Tensor input,"
          "                   Tensor? weight,"
          "                   Tensor? bias,"
          "                   float eps) -> Tensor");
  ops.impl("speculator_ln_cuda", torch::kCUDA, &speculator_ln_cuda);

  ops.def("sum_lstm_cuda(Tensor states_4d,"
          "               Tensor z4_4d,"
          "               Tensor prev_cell_d,"
          "               Tensor? w_cell,"
          "               Tensor? b_cell,"
          "               Tensor? w_state,"
          "               Tensor? b_state,"
          "               float alpha,"
          "               float eps_cell,"
          "               float eps_state,"
          "               bool use_fast_gelu) -> (Tensor, Tensor)");
  ops.impl("sum_lstm_cuda", torch::kCUDA, &sum_lstm_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
