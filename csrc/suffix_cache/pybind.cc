#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "suffix_tree.h"

namespace py = pybind11;


PYBIND11_MODULE(_C, m) {
    py::class_<Candidate>(m, "Candidate")
        .def_readwrite("token_ids", &Candidate::token_ids)
        .def_readwrite("parents", &Candidate::parents)
        .def_readwrite("probs", &Candidate::probs)
        .def_readwrite("score", &Candidate::score)
        .def_readwrite("match_len", &Candidate::match_len);

    py::class_<SuffixTree>(m, "SuffixTree")
        .def(py::init<int>())
        .def("num_seqs", &SuffixTree::num_seqs)
        .def("append", &SuffixTree::append)
        .def("extend", &SuffixTree::extend)
        .def("speculate", &SuffixTree::speculate);
}