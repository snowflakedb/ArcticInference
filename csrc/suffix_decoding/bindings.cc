// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include "suffix_tree.h"

namespace nb = nanobind;

using Int32Array1D = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>,
                                 nb::device::cpu, nb::any_contig>;
using BatchVecSingle = std::vector<std::pair<int, std::vector<int32_t>>>;
using BatchVecMulti = std::vector<std::tuple<SuffixTree&, int, std::vector<int32_t>>>;


// batch single-tree
void batch_extend_single(SuffixTree& tree, const BatchVecSingle& batches) {
    // Process all updates in a single pass to minimize overhead
    for (const auto& item : batches) {
        const int seq_id = item.first;
        const auto& vec = item.second;
        if (!vec.empty()) {
            // Use span view directly for zero-copy access
            tree.extend(seq_id, std::span<const int32_t>(vec.data(), vec.size()));
        }
    }
}

// batch across multiple trees
void batch_extend(const BatchVecMulti& batch) {
    // Process all updates in a single pass to minimize nanobind overhead
    // Group by tree for potential future optimizations
    for (const auto& tup : batch) {
        SuffixTree& tree = std::get<0>(tup);
        int seq_id = std::get<1>(tup);
        const auto& vec = std::get<2>(tup);
        if (!vec.empty()) {
            // Use span view directly for zero-copy access
            tree.extend(seq_id, std::span<const int32_t>(vec.data(), vec.size()));
        }
    }
}


void extend_ndarray(SuffixTree& tree,
                    int seq_id,
                    const Int32Array1D& tokens) {
    tree.extend(
        seq_id,
        std::span<const int32_t>(tokens.data(), tokens.size()));
}


void extend_vector(SuffixTree& tree,
                   int seq_id,
                   const std::vector<int32_t>& tokens) {
    tree.extend(seq_id, std::span<const int32_t>(tokens));
}


Draft speculate_ndarray(SuffixTree& tree,
                        const Int32Array1D& context,
                        int max_spec_tokens,
                        float max_spec_factor,
                        float max_spec_offset,
                        float min_token_prob,
                        bool use_tree_spec) {
    return tree.speculate(
        std::span<const int32_t>(context.data(), context.size()),
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


Draft speculate_vector(SuffixTree& tree,
                       const std::vector<int32_t>& context,
                       int max_spec_tokens,
                       float max_spec_factor,
                       float max_spec_offset,
                       float min_token_prob,
                       bool use_tree_spec) {
    return tree.speculate(
        std::span<const int32_t>(context),
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


// Batch speculation across multiple trees and contexts.
// Each tuple contains: (tree, context, max_spec_tokens, max_spec_factor, max_spec_offset, min_token_prob, use_tree_spec)
using SpeculateParams = std::tuple<SuffixTree&, std::vector<int32_t>, int, float, float, float, bool>;

std::vector<Draft> batch_speculate(const std::vector<SpeculateParams>& batch) {
    // Pre-allocate results vector to avoid reallocations
    std::vector<Draft> results;
    results.reserve(batch.size());
    
    // Process all speculations in a single pass to minimize nanobind overhead
    for (const auto& params : batch) {
        const auto& [tree, context, max_spec_tokens, max_spec_factor,
                     max_spec_offset, min_token_prob, use_tree_spec] = params;
        
        // Use span view directly for zero-copy access when context is not empty
        if (!context.empty()) {
            Draft draft = tree.speculate(
                std::span<const int32_t>(context.data(), context.size()),
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec);
            results.push_back(std::move(draft));
        } else {
            // Empty context produces empty draft
            results.emplace_back();
        }
    }
    
    return results;
}


// Dual-tree batch speculation: speculates on both local and global trees,
// selects the best draft in C++ to minimize Python overhead.
// Each tuple contains: (local_tree, global_tree, context, max_spec_tokens, max_spec_factor, max_spec_offset, min_token_prob, use_tree_spec)
using DualSpeculateParams = std::tuple<SuffixTree&, SuffixTree&, std::vector<int32_t>, int, float, float, float, bool>;

std::vector<Draft> batch_speculate_dual(const std::vector<DualSpeculateParams>& batch) {
    // Pre-allocate results vector to avoid reallocations
    std::vector<Draft> results;
    results.reserve(batch.size());
    
    // Process all dual-tree speculations in a single pass
    for (const auto& params : batch) {
        const auto& [local_tree, global_tree, context, max_spec_tokens, max_spec_factor,
                     max_spec_offset, min_token_prob, use_tree_spec] = params;
        
        if (context.empty()) {
            results.emplace_back();
            continue;
        }
        
        // Create span view for zero-copy access
        std::span<const int32_t> ctx_span(context.data(), context.size());
        
        // Speculate on local tree
        Draft local_draft = local_tree.speculate(
            ctx_span, max_spec_tokens, max_spec_factor,
            max_spec_offset, min_token_prob, use_tree_spec);
        
        // Speculate on global tree
        Draft global_draft = global_tree.speculate(
            ctx_span, max_spec_tokens, max_spec_factor,
            max_spec_offset, min_token_prob, use_tree_spec);
        
        // Select best draft based on score (done in C++ for efficiency)
        if (local_draft.score >= global_draft.score) {
            results.push_back(std::move(local_draft));
        } else {
            results.push_back(std::move(global_draft));
        }
    }
    
    return results;
}


// Optimized dual-tree batch speculation with numpy array support.
// Each tuple contains: (local_tree, global_tree, context_ndarray, max_spec_tokens, max_spec_factor, max_spec_offset, min_token_prob, use_tree_spec)
using DualSpeculateNdarrayParams = std::tuple<SuffixTree&, SuffixTree&, Int32Array1D, int, float, float, float, bool>;

std::vector<Draft> batch_speculate_dual_ndarray(const std::vector<DualSpeculateNdarrayParams>& batch) {
    // Pre-allocate results vector to avoid reallocations
    std::vector<Draft> results;
    results.reserve(batch.size());
    
    // Process all dual-tree speculations in a single pass with zero-copy numpy arrays
    for (const auto& params : batch) {
        const auto& [local_tree, global_tree, context, max_spec_tokens, max_spec_factor,
                     max_spec_offset, min_token_prob, use_tree_spec] = params;
        
        if (context.size() == 0) {
            results.emplace_back();
            continue;
        }
        
        // Use numpy array directly with zero-copy span
        std::span<const int32_t> ctx_span(context.data(), context.size());
        
        // Speculate on local tree
        Draft local_draft = local_tree.speculate(
            ctx_span, max_spec_tokens, max_spec_factor,
            max_spec_offset, min_token_prob, use_tree_spec);
        
        // Speculate on global tree
        Draft global_draft = global_tree.speculate(
            ctx_span, max_spec_tokens, max_spec_factor,
            max_spec_offset, min_token_prob, use_tree_spec);
        
        // Select best draft based on score
        if (local_draft.score >= global_draft.score) {
            results.push_back(std::move(local_draft));
        } else {
            results.push_back(std::move(global_draft));
        }
    }
    
    return results;
}




// Packed batch extend for a single tree using zero-copy numpy arrays.
// seq_ids, offsets, lengths specify segments in the tokens array.
void batch_extend_packed_ndarray(
    SuffixTree& tree,
    const Int32Array1D& seq_ids,
    const Int32Array1D& offsets,
    const Int32Array1D& lengths,
    const Int32Array1D& tokens) {
    // Basic validation
    if (seq_ids.size() != offsets.size() || seq_ids.size() != lengths.size()) {
        nb::value_error("seq_ids, offsets, and lengths must have the same size");
    }
    const int32_t* seq_ids_ptr = seq_ids.data();
    const int32_t* offsets_ptr = offsets.data();
    const int32_t* lengths_ptr = lengths.data();
    const int32_t* tokens_ptr = tokens.data();

    size_t num_spans = static_cast<size_t>(seq_ids.size());
    for (size_t i = 0; i < num_spans; ++i) {
        int seq = static_cast<int>(seq_ids_ptr[i]);
        int off = static_cast<int>(offsets_ptr[i]);
        int len = static_cast<int>(lengths_ptr[i]);
        if (len <= 0) {
            continue;
        }
        // Bounds check (defensive)
        if (off < 0 || off + len > tokens.size()) {
            nb::value_error("Invalid offset/length for tokens array");
        }
        tree.extend(seq, std::span<const int32_t>(tokens_ptr + off, static_cast<size_t>(len)));
    }
}

NB_MODULE(_C, m) {
    nb::set_leak_warnings(false);

    nb::class_<Draft>(m, "Draft")
        .def_rw("token_ids", &Draft::token_ids)
        .def_rw("parents", &Draft::parents)
        .def_rw("probs", &Draft::probs)
        .def_rw("score", &Draft::score)
        .def_rw("match_len", &Draft::match_len);

    nb::class_<SuffixTree>(m, "SuffixTree")
        .def(nb::init<int>())
        .def("num_seqs", &SuffixTree::num_seqs)
        .def("remove", &SuffixTree::remove)
        // Overloads for extend method. Use different names to avoid overload
        // resolution overhead at run-time.
        .def("extend", &extend_vector)
        .def("extend_ndarray", &extend_ndarray)
        // Overloads for speculate method.
        .def("speculate", &speculate_vector)
        .def("speculate_ndarray", &speculate_ndarray)
        // Debugging methods, not meant to be used in critical loop.
        .def("check_integrity", &SuffixTree::check_integrity)
        .def("estimate_memory", &SuffixTree::estimate_memory);

    m.def("batch_extend", &batch_extend);
    m.def("batch_extend_single", &batch_extend_single);
    m.def("batch_speculate", &batch_speculate);
    m.def("batch_speculate_dual", &batch_speculate_dual);
    m.def("batch_speculate_dual_ndarray", &batch_speculate_dual_ndarray);
    m.def("batch_extend_packed_ndarray", &batch_extend_packed_ndarray);
}
