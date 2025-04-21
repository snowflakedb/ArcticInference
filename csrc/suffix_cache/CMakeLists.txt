cmake_minimum_required(VERSION 3.14)
project(SuffixCache CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb")

find_package(pybind11 REQUIRED)

pybind11_add_module(_C pybind.cc suffix_tree.cc)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_C PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
