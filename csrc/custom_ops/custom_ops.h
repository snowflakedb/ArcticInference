#pragma once

#include <optional>
#include <vector>

#include <torch/all.h>
#include <torch/library.h>

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
    int64_t head_size
);