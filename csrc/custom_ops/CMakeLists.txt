cmake_minimum_required(VERSION 3.14)

project(CustomOps CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCH_CUDA_ARCH_LIST "7.0 7.5 8.0 8.6 8.9 9.0+PTX")

message(STATUS "TORCH_CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(custom_ops
  kernels.cu
  torch_bindings.cpp
)

target_compile_definitions(custom_ops PRIVATE
  -DTORCH_API_INCLUDE_EXTENSION_H
  -DTORCH_EXTENSION_NAME=custom_ops
  -DTORCH_API_INCLUDE_TYPEDEFS
)

target_include_directories(custom_ops PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(custom_ops PRIVATE
  torch
  ${GPU_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
)
