cmake_minimum_required(VERSION 3.14)

project(CustomOps CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCH_CUDA_ARCH_LIST "9.0+PTX 10.0a")

message(STATUS "TORCH_CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

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

target_include_directories(custom_ops PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(custom_ops PRIVATE
  torch
  ${GPU_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
)
