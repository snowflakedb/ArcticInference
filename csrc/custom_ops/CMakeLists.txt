cmake_minimum_required (VERSION 3.14)

project(CustomOps CXX CUDA)

find_package(CUDA REQUIRED)

set(TORCH_CUDA_ARCH_LIST "7.0 7.5 8.0 8.6 8.9 9.0+PTX")

message(STATUS "TORCH_CMAKE_PREFIX_PATH: ${TORCH_CMAKE_PREFIX_PATH}")

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

add_library(CustomOps SHARED
  kernels.cu
  torch_bindings.cpp)

set_property(TARGET CustomOps PROPERTY CXX_STANDARD 17)

target_compile_definitions(CustomOps PRIVATE
  -DTORCH_API_INCLUDE_EXTENSION_H
  -DTORCH_EXTENSION_NAME=CustomOps
  -DTORCH_API_INCLUDE_TYPEDEFS)

target_include_directories(CustomOps PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(CustomOps PRIVATE torch ${GPU_LIBRARIES})
target_link_libraries(CustomOps PRIVATE CUDA::cudart CUDA::cuda_driver)