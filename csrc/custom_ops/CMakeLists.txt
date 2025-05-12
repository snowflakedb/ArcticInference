cmake_minimum_required (VERSION 3.14)

project(CustomOps CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES 90)
# endif()

add_library(CustomOps SHARED
  kernels.cu
  torch_bindings.cpp)

set(CMAKE_CUDA_ARCHITECTURES 90)

set_property(TARGET CustomOps PROPERTY CXX_STANDARD 17)

target_compile_definitions(CustomOps PRIVATE
  -DTORCH_API_INCLUDE_EXTENSION_H
  -DTORCH_EXTENSION_NAME=CustomOps
  -DTORCH_API_INCLUDE_TYPEDEFS)

target_include_directories(CustomOps PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(CustomOps PRIVATE torch ${GPU_LIBRARIES})
target_link_libraries(CustomOps PRIVATE CUDA::cudart CUDA::cuda_driver)

#install(TARGETS CustomOps LIBRARY arctic_inference custom_ops)

# add_library(CustomOps SHARED
#   sampler_kernels.cu
#   torch_bindings.cpp)

# target_link_libraries(CustomOps PRIVATE ${TORCH_LIBRARIES})