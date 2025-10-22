// lamport_allreduce_bindings.h
#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Declaration for the registration function
void init_lamport_allreduce_bindings(py::module_ &m);
