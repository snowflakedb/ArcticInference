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
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "suffix_tree.h"

namespace nb = nanobind;

NB_MODULE(_C, m) {
    nb::class_<Candidate>(m, "Candidate")
        .def_rw("token_ids", &Candidate::token_ids)
        .def_rw("parents", &Candidate::parents)
        .def_rw("probs", &Candidate::probs)
        .def_rw("score", &Candidate::score)
        .def_rw("match_len", &Candidate::match_len);

    nb::class_<SuffixTree>(m, "SuffixTree")
        .def(nb::init<int>())
        .def("num_seqs", &SuffixTree::num_seqs)
        .def("append", &SuffixTree::append)
        .def("extend", &SuffixTree::extend)
        .def("remove", &SuffixTree::remove)
        .def("speculate", &SuffixTree::speculate)
        .def("check_integrity", &SuffixTree::check_integrity)
        .def("estimate_memory", &SuffixTree::estimate_memory);
}
