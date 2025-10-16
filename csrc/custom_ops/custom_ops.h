#pragma once

#include <optional>
#include <vector>

#include <torch/all.h>
#include <torch/library.h>

void reshape_and_cache_flash_bulk(
    torch::Tensor &keys, torch::Tensor &values,
    std::vector<torch::Tensor> const &key_caches,
    std::vector<torch::Tensor> const &value_caches, torch::Tensor &slot_mapping,
    const std::string &kv_cache_dtype,
    std::vector<torch::Tensor> const &k_scales,
    std::vector<torch::Tensor> const &v_scales, int64_t num_heads,
    int64_t head_size);

void reshape_and_cache_flash_fp4(
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor
        &key_cache,
    torch::Tensor &value_cache, torch::Tensor &slot_mapping,
    const std::string &kv_cache_dtype, torch::Tensor &k_scale,
    torch::Tensor &v_scale, torch::Tensor &key_scale_cache,
    torch::Tensor &value_scale_cache);