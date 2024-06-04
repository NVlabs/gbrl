##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
##############################################################################
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# Set project details
project(gbrl LANGUAGES CXX VERSION 1.0.0)
# Set C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Default to release build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if (POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()
#-- Options
include(CMakeDependentOption)
option(USE_CUDA  "Build with GPU acceleration" OFF)

if (USE_CUDA AND NOT APPLE)
    # set(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)
    # Ensure CUDA architectures are set before enabling CUDA language
    message(STATUS "Compiling CUDA")
    set(CUDA_ARCHS "60" "61" "62" "70" "75")
    if(CUDA_VERSION VERSION_GREATER_EQUAL "110")
        list(APPEND CUDA_ARCHS "80")
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "111")
        list(APPEND CUDA_ARCHS "86")
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "115")
        list(APPEND CUDA_ARCHS "87")
    endif()
    if(CUDA_VERSION VERSION_GREATER_EQUAL "118")
        list(APPEND CUDA_ARCHS "89")
        list(APPEND CUDA_ARCHS "90")
    endif()
    list(POP_BACK CUDA_ARCHS CUDA_LAST_SUPPORTED_ARCH)
    list(APPEND CUDA_ARCHS "${CUDA_LAST_SUPPORTED_ARCH}-virtual")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})

    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH
            "The compiler executable to use when compiling host code for CUDA")
        mark_as_advanced(CMAKE_CUDA_HOST_COMPILER)
        message(STATUS "Configured CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    enable_language(CUDA)
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 11.0)
        message(FATAL_ERROR "CUDA version must be at least 11.0!")
    endif()
    if(DEFINED GPU_COMPUTE_VER)
        compute_cmake_cuda_archs("${GPU_COMPUTE_VER}")
    endif()
    # Check for CUDA availability
    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "Compiling for CPU only")
endif()

# Find required packages
find_package(pybind11 CONFIG REQUIRED)
find_program(GRAPHVIZ_DOT_EXECUTABLE dot)

if(APPLE)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
      # Try again with extra path info; required for libomp 15+ from Homebrew
      execute_process(COMMAND brew --prefix libomp
                      OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(OpenMP_C_FLAGS
        "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_FLAGS
        "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_C_LIB_NAMES omp)
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
      find_package(OpenMP REQUIRED)
    endif()
else()
    find_package(OpenMP REQUIRED)
    set(OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
    set(OpenMP_CXX_LIB_NAMES ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_omp_LIBRARY ${OpenMP_omp_LIBRARY})
endif()

# Include directories
include_directories(${pybind11_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/gbrl/src/cpp)

# Graphviz
if (GRAPHVIZ_DOT_EXECUTABLE)
    add_definitions(-DUSE_GRAPHVIZ)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        include_directories(/usr/include/graphviz)
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        execute_process(COMMAND brew --prefix graphviz OUTPUT_VARIABLE GRAPHVIZ_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        include_directories(${GRAPHVIZ_PREFIX}/include/graphviz)
    endif()
endif()

# Add subdirectories for cpp and cuda
add_subdirectory(gbrl/src/cpp)

if (USE_CUDA AND NOT APPLE)
    include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/gbrl/src/cuda)
    add_definitions(-DUSE_CUDA)
    add_subdirectory(gbrl/src/cuda)
    set(CUDA_SOURCES $<TARGET_OBJECTS:cuda_gbrl_src>)

else()
    set(CUDA_SOURCES "")
endif()

# Combine object files into a single library
set(CPP_SOURCES
    gbrl/src/cpp/gbrl_binding.cpp
    gbrl/src/cpp/gbrl.cpp
    gbrl/src/cpp/types.cpp
    gbrl/src/cpp/optimizer.cpp
    gbrl/src/cpp/scheduler.cpp
    gbrl/src/cpp/node.cpp
    gbrl/src/cpp/utils.cpp
    gbrl/src/cpp/fitter.cpp
    gbrl/src/cpp/predictor.cpp
    gbrl/src/cpp/split_candidate_generator.cpp
    gbrl/src/cpp/loss.cpp
    gbrl/src/cpp/math_ops.cpp
)

pybind11_add_module(gbrl_cpp MODULE ${CPP_SOURCES} ${CUDA_SOURCES})
target_compile_definitions(gbrl_cpp PRIVATE MODULE_NAME="gbrl_cpp")

# Platform-specific settings and linking
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(gbrl_cpp PRIVATE OpenMP::OpenMP_CXX)
    if (USE_CUDA)
        target_link_libraries(gbrl_cpp PRIVATE CUDA::cudart)
    endif()
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    find_package(LLVM REQUIRED)
    include_directories(${LLVM_INCLUDE_DIRS})
    link_directories(${LLVM_LIBRARY_DIRS})
    target_link_libraries(gbrl_cpp PRIVATE ${LLVM_LIBRARIES})
elseif (WIN32)
    if (USE_CUDA)
        set(cuda_lib_path "${CUDAToolkit_ROOT_DIR}/lib/x64")
        target_link_libraries(gbrl_cpp PRIVATE ${cuda_lib_path}/cudart.lib)
    endif()
endif()

# Link Graphviz libraries if available
if (GRAPHVIZ_DOT_EXECUTABLE)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        target_link_libraries(gbrl_cpp PRIVATE gvc cgraph)
    elseif (WIN32)
        target_link_libraries(gbrl_cpp PRIVATE gvc.lib cgraph.lib)
    endif()
endif()
