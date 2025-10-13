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

#include "suffix_tree.h"

namespace nb = nanobind;

using Int32Array1D = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>,
                                 nb::device::cpu, nb::any_contig>;


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
}
