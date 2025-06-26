# cmake -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.12/dist-packages/torch/share/cmake -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1 ..
cmake_minimum_required(VERSION 3.14)
include(FetchContent)

project(CustomOps CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCH_CUDA_ARCH_LIST "10.0a")

message(STATUS "TORCH_CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
set(CUTLASS_REVISION "v3.9.2" CACHE STRING "CUTLASS revision to use")

FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    # Please keep this in sync with CUTLASS_REVISION line above.
    GIT_TAG ${CUTLASS_REVISION}
    GIT_PROGRESS TRUE

    # Speed up CUTLASS download by retrieving only the specified GIT_TAG instead of the history.
    # Important: If GIT_SHALLOW is enabled then GIT_TAG works only with branch names and tags.
    # So if the GIT_TAG above is updated to a commit hash, GIT_SHALLOW must be set to FALSE
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(cutlass)

pybind11_add_module(custom_ops
  kernels.cu
  nvfp4_scaled_mm_kernels.cu
  torch_bindings.cpp
)

target_compile_definitions(custom_ops PRIVATE
  -DTORCH_API_INCLUDE_EXTENSION_H
  -DTORCH_EXTENSION_NAME=custom_ops
  -DTORCH_API_INCLUDE_TYPEDEFS
)

find_package(Torch REQUIRED)

target_include_directories(custom_ops PRIVATE 
  ${TORCH_INCLUDE_DIRS} 
  ${CUTLASS_INCLUDE_DIR}
  ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
)

target_link_libraries(custom_ops PRIVATE
  torch
  ${GPU_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
)
