// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "topk_router.cuh"
#include "kernel_utils.h"

/*
Perform softmax plus atomics to get token mapping.
*/
void top_k_gating(torch::Tensor& expert_counts,
                  torch::Tensor& scores,
                  torch::Tensor& assignments,
                  torch::Tensor& offsets,
                  torch::Tensor& logits,
                  torch::Tensor& e_bias);
