
#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"
#include "topk_router.cuh"
#include "topk_utils.h"
#include "kernel_utils.h"


using ROp = reduce::ROpType;


template <typename T, int TOP_K>
__global__ void top_k_gating_kernel3(int32_t* expert_counts,
                                    float* scores,
                                    float* e_bias,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const T* logits,
                                    const int32_t n_tokens,
                                    const int32_t n_groups,
                                    const int32_t group_topk,
                                    const int32_t n_experts,
                                    const bool apply_sigmoid,
                                    const bool norm_topk,
                                    const bool compute_offset)
{
    const int32_t token_idx = blockIdx.x;
    int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;
    const int32_t wid = expert_idx >> 5;

    __shared__ float grouped_scores[32];
    __shared__ int32_t grouped_eids[32];

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= n_tokens) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    if (e_bias != nullptr)
        logit_val = logit_val + e_bias[expert_idx];

    float reduce_val = logit_val;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;

    for (int i = 0; i < group_topk; i++)
    {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        grouped_eids[wid * group_topk + i] = n_experts - res.idx - 1;
        grouped_scores[wid * group_topk + i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val); }
    }

    tb.sync();

    if (threadIdx.x < group_topk * n_groups)
    {
        reduce_val = grouped_scores[threadIdx.x];
        expert_idx = grouped_eids[threadIdx.x];
    }

    tb.sync();

    inverted_expert = n_experts - expert_idx - 1;

    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val); }
    }

    const float max_logit = local_assigned_logits[0];

    float reduce_sum;
    if (!apply_sigmoid)
    {
        reduce_sum = __expf(logit_val - max_logit);
        reduce::block<ROp::Add>(tb, warp, reduce_sum);
    }
    float score;
    if (threadIdx.x < TOP_K)
    {
        if (apply_sigmoid)
            score = 1.0f / (1.0f + __expf(-local_assigned_logits[threadIdx.x]));
        else
            score = __expf(local_assigned_logits[threadIdx.x] - max_logit) / reduce_sum;
    }
    else
        score = 0;

    if (norm_topk){
        reduce_sum = score;
        reduce::block<ROp::Add>(tb, warp, reduce_sum);
        score = score / reduce_sum;
    }

    if (threadIdx.x < TOP_K) {
        scores[token_idx * TOP_K + threadIdx.x] = score;
        assignments[token_idx * TOP_K + threadIdx.x] = local_assigned_experts[threadIdx.x];
    }

    if (compute_offset)
    {
        for (int i = 0; i < TOP_K; ++i) {
            offsets[token_idx * TOP_K + i] =
                    atomicAdd(expert_counts + local_assigned_experts[i], 1);
        }
    }
}

template <typename T, int TOP_K>
__global__ void top_k_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const T* logits,
                                    const float* e_bias,
                                    const int32_t n_tokens,
                                    const int32_t n_experts,
                                    const bool compute_offset)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;
    const int32_t group_topk = 4;
    __shared__ float grouped_scores[36];
    __shared__ int32_t grouped_eids[36];
    const int32_t wid = expert_idx >> 5;
    const int32_t lane = expert_idx & 0x1f;
    const int32_t n_local_experts = 32;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= n_tokens) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val, bias;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
        logit_val = 1.0f / (1.0f + __expf(-logit_val));
        bias = e_bias[expert_idx];
    } else {
        reduce::init<ROp::Max>(&logit_val);
        reduce::init<ROp::Max>(&bias);
    }
    float reduce_val = logit_val + bias;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_local_experts - lane - 1;

    
    for (int i = 0; i < group_topk; i++)
    {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, 1>(tb, warp, reduce_val, inverted_expert);
        if (lane == 0){
            grouped_eids[wid * group_topk + i] = (wid << 5) + (n_local_experts - res.idx - 1);
            grouped_scores[wid * group_topk + i] = res.val;
        }
        // Set the max logit to -inf so that it is not selected again
        if (lane == n_local_experts - res.idx - 1) { 
            reduce::init<ROp::Max>(&reduce_val); 
        }
        tb.sync();
    }
    
    if (threadIdx.x < 32)
        reduce_val = grouped_scores[threadIdx.x];
    else
        reduce::init<ROp::Max>(&reduce_val); 

    tb.sync();
    
    inverted_expert = n_experts - expert_idx - 1;
    // Find the final top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        int32_t tidx = n_experts - res.idx - 1;
        local_assigned_experts[i] = grouped_eids[tidx];
        local_assigned_logits[i] = res.val - e_bias[local_assigned_experts[i]];

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == tidx) { reduce::init<ROp::Max>(&reduce_val); }
    }

    // const float max_logit = local_assigned_logits[0];
    // float softmax_sum = __expf(logit_val - max_logit);
    // reduce::block<ROp::Add>(tb, warp, softmax_sum);
    float score;
    if (threadIdx.x < TOP_K) 
        score = local_assigned_logits[threadIdx.x];
    else 
        score = 0;
    float reduce_sum = score;
    reduce::block<ROp::Add, 1>(tb, warp, reduce_sum);
    score = score / reduce_sum;

    if (threadIdx.x < TOP_K) {
        // const float softmax = __expf(local_assigned_logits[threadIdx.x] - max_logit) / softmax_sum;
        // float score = 1.0f / (1.0f + __expf(-local_assigned_logits[threadIdx.x]));
        scores[token_idx * TOP_K + threadIdx.x] = score; // softmax;
        assignments[token_idx * TOP_K + threadIdx.x] = local_assigned_experts[threadIdx.x];
    }
    if (compute_offset)
    {
        for (int i = 0; i < TOP_K; ++i) {
            if (threadIdx.x == 0) {
                offsets[token_idx * TOP_K + i] =
                    atomicAdd(expert_counts + local_assigned_experts[i], 1);
            }
        }
    }
}

template <typename T, int TOP_K>
__global__ void top_k_gating_kernel1(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    const T* logits,
                                    const int32_t n_tokens,
                                    const int32_t n_experts)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= n_tokens) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < TOP_K; i++) {
                assignments[token_idx * TOP_K + i] = gating::unassigned;
                offsets[token_idx * TOP_K + i] = gating::unassigned;
            }
        }
        return;
    }

    const T* token_logits = logits + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;

    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val); }
    }

    const float max_logit = local_assigned_logits[0];
    float softmax_sum = __expf(logit_val - max_logit);
    reduce::block<ROp::Add>(tb, warp, softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        const float softmax = __expf(local_assigned_logits[i] - max_logit) / softmax_sum;

        if (threadIdx.x == 0) {
            scores[token_idx * TOP_K + i] = softmax;
            assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
            offsets[token_idx * TOP_K + i] =
                atomicAdd(expert_counts + local_assigned_experts[i], 1);
        }
    }
}

template <typename T>
void launch_top_k_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         const T* logits,
                         const float* e_bias,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts + hw_warp_size - 1) / hw_warp_size) * hw_warp_size);
    TOP_K_SWITCH(n_top_k, [&] {
        // top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
        //     expert_counts, scores, nullptr, assignments, offsets, logits, n_tokens, 8, 4, n_experts, true, true, false);
        top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
            expert_counts, scores, assignments, offsets, logits, e_bias, n_tokens, n_experts, false);
        // top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
        //     expert_counts, scores, assignments, offsets, logits, n_tokens, n_experts);
    });
}

#define INSTANTIATE_top_k_KERNEL(T)                                                   \
    template void launch_top_k_gating<T>(int32_t * expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* assignments,                        \
                                         int32_t* offsets,                            \
                                         const T* logits,                             \
                                         const float* e_bias,                         \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_experts,                     \
                                         const int32_t n_top_k,                       \
                                         cudaStream_t stream);

INSTANTIATE_top_k_KERNEL(float) INSTANTIATE_top_k_KERNEL(__half)
#ifdef BF16_AVAILABLE
    INSTANTIATE_top_k_KERNEL(__nv_bfloat16)
#endif
