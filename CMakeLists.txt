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
project(gbrl LANGUAGES C CXX VERSION 1.0.0)
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

# Allow overriding Python paths
if (DEFINED PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Path to the Python executable")
endif()

if (DEFINED PYTHON_INCLUDE_DIR)
    set(Python3_INCLUDE_DIR ${PYTHON_INCLUDE_DIR} CACHE FILEPATH "Path to the Python include directory")
endif()

if (DEFINED PYTHON_LIBRARY)
    set(Python3_LIBRARY ${PYTHON_LIBRARY} CACHE FILEPATH "Path to the Python library")
endif()

# Find Python 3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

#-- Options
include(CMakeDependentOption)
option(USE_CUDA  "Build with GPU acceleration" OFF)
option(COVERAGE  "Run coverage report" OFF)

if(COVERAGE)
  message(STATUS "Coverage build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

# Check for CUDA availability
if (USE_CUDA AND NOT APPLE)
    find_package(CUDAToolkit QUIET HINTS $ENV{CONDA_PREFIX}) 
    # 2. Fallback to system-wide CUDA if not found in Anaconda
    if (NOT CUDAToolkit_FOUND)
        find_package(CUDAToolkit QUIET)
    endif()
    if (CUDAToolkit_FOUND)
        set(CUDA_VERSION "${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}")
        # Ensure CUDA architectures are set before enabling CUDA language
        set(CUDA_ARCHS "60" "61" "62" "70" "75")
        
        if(${CUDA_VERSION} VERSION_GREATER_EQUAL "110")
            list(APPEND CUDA_ARCHS "80")
        endif()
        if(${CUDA_VERSION} VERSION_GREATER_EQUAL "111")
            list(APPEND CUDA_ARCHS "86")
        endif()
        if(${CUDA_VERSION} VERSION_GREATER_EQUAL "115")
            list(APPEND CUDA_ARCHS "87")
        endif()
        if(${CUDA_VERSION} VERSION_GREATER_EQUAL "118")
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
        if(NOT DEFINED CMAKE_CUDA_COMPILER)
            if(DEFINED ENV{CUDACXX})
                set(CMAKE_CUDA_COMPILER $ENV{CUDACXX} CACHE FILEPATH "CUDA Compiler")
                message(STATUS "CMAKE_CUDA_COMPILER set to $ENV{CUDACXX} from environment.")
            else()
                message(STATUS "Setting CMAKE_CUDA_COMPILER to ${CUDAToolkit_NVCC_EXECUTABLE}")
                set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
            endif()
        else()
            message(STATUS "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
        endif()
        # Validate the compiler before enabling CUDA
        if (NOT EXISTS ${CMAKE_CUDA_COMPILER})
            message(WARNING "CUDA toolkit found, but could not locate nvcc at ${CMAKE_CUDA_COMPILER}. Disabling GPU acceleration.")
            set(USE_CUDA OFF)
        endif()
        if (USE_CUDA)
            enable_language(CUDA)
            if(${CUDA_VERSION} VERSION_LESS 11.0)
                message(FATAL_ERROR "CUDA version must be at least 11.0!")
            endif()
            if(DEFINED GPU_COMPUTE_VER)
                compute_cmake_cuda_archs("${GPU_COMPUTE_VER}")
            endif()
        endif()
    else()
        message(STATUS "CUDA not found, compiling for CPU only")
        set(USE_CUDA OFF)
    endif()
endif()

if(NOT USE_CUDA)
    message(STATUS "Compiling for CPU only")
else()
    message(STATUS "NVCC ${CUDAToolkit_VERSION} found, compiling for CPU and GPU")
endif()
# Find required packages
# Attempt to find pybind11 automatically with Anaconda
find_package(pybind11 CONFIG QUIET HINTS $ENV{CONDA_PREFIX}) 
# 2. Fallback to system-wide CUDA if not found in Anaconda
if (NOT pybind11)
    find_package(pybind11 CONFIG QUIET)
endif()
# If pybind11 is not found, use Python to find it
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found automatically, attempting to locate with Python")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT EXISTS ${pybind11_DIR})
        message(FATAL_ERROR "pybind11 not found, please install it using 'pip install pybind11'")
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
    find_package(pybind11 CONFIG REQUIRED)
endif()

find_path(GRAPHVIZ_INCLUDE_DIR         NAMES graphviz/cgraph.h
          HINTS ${_GRAPHVIZ_INCLUDE_DIR})
find_library(GRAPHVIZ_GVC_LIBRARY      NAMES gvc 
             HINTS ${_GRAPHVIZ_LIBRARY_DIR})
find_library(GRAPHVIZ_CGRAPH_LIBRARY   NAMES cgraph
             HINTS ${_GRAPHVIZ_LIBRARY_DIR})
if(GRAPHVIZ_INCLUDE_DIR AND GRAPHVIZ_GVC_LIBRARY AND GRAPHVIZ_CGRAPH_LIBRARY)
    set(GRAPHVIZ_FOUND TRUE)
else()
    set(GRAPHVIZ_FOUND FALSE)
endif()
         

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

# Ensure OpenMP flags and directories are applied
if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add subdirectories for cpp and cuda
add_subdirectory(gbrl/src/cpp)

if (USE_CUDA AND NOT APPLE)
    set(CMAKE_CUDA_STANDARD 14)  
    include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/gbrl/src/cuda)
    add_definitions(-DUSE_CUDA)
    add_subdirectory(gbrl/src/cuda)
    set(CUDA_SOURCES $<TARGET_OBJECTS:cuda_gbrl_src>)

else()
    set(CUDA_SOURCES "")
endif()

# Only include pybind11 in the binding file
set(PYBIND_SOURCES
    gbrl/src/cpp/gbrl_binding.cpp
)

pybind11_add_module(gbrl_cpp MODULE ${PYBIND_SOURCES})
target_include_directories(gbrl_cpp PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/gbrl/src/cpp)
target_compile_definitions(gbrl_cpp PRIVATE MODULE_NAME="gbrl_cpp")
target_link_libraries(gbrl_cpp PRIVATE ${Python3_LIBRARIES})
target_link_libraries(gbrl_cpp PRIVATE gbrl_cpp_src)

if (USE_CUDA)
    target_link_libraries(gbrl_cpp PRIVATE cuda_gbrl_src)
endif()
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
    target_link_libraries(gbrl_cpp PRIVATE ${LLVM_LIBRARIES} OpenMP::OpenMP_CXX)
elseif (WIN32)
    target_link_libraries(gbrl_cpp PRIVATE OpenMP::OpenMP_CXX)
    if (USE_CUDA)
        set(cuda_lib_path "${CUDAToolkit_ROOT_DIR}/lib/x64")
        message(STATUS "cuda_lib_path: ${cuda_lib_path}")
        target_link_libraries(gbrl_cpp PRIVATE CUDA::cudart)
    endif()
endif()

# Link Graphviz libraries if available
if (GRAPHVIZ_FOUND)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        # target_link_libraries(gbrl_cpp PRIVATE gvc cgraph)
        target_link_libraries(gbrl_cpp PRIVATE ${GRAPHVIZ_GVC_LIBRARY} ${GRAPHVIZ_CGRAPH_LIBRARY})
    elseif (WIN32)
        target_link_libraries(gbrl_cpp PRIVATE gvc.lib cgraph.lib)
    endif()
endif()
