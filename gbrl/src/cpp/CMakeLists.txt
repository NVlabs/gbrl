##############################################################################
# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
##############################################################################

set(CPP_SOURCES
    gbrl.cpp
    types.cpp
    optimizer.cpp
    scheduler.cpp
    node.cpp
    utils.cpp
    fitter.cpp
    shap.cpp
    data_structs.cpp
    predictor.cpp
    split_candidate_generator.cpp
    loss.cpp
    math_ops.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gbrl/include)

if (CUDAToolkit_FOUND AND USE_CUDA AND NOT APPLE)
    include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/gbrl/src/cuda)
    add_definitions(-DUSE_CUDA)
endif()

if (GRAPHVIZ_FOUND)
message(STATUS "Found Graphviz")
    set(GRAPHVIZ_INCLUDE_DIRS ${GRAPHVIZ_INCLUDE_DIR} ${GRAPHVIZ_INCLUDE_DIR}/graphviz)
    add_definitions(-DUSE_GRAPHVIZ)
    include_directories(${GRAPHVIZ_INCLUDE_DIRS})
    if (MSVC)
        add_definitions(-DGVDLL)
    endif()
else()
    message(STATUS "Did not Find Graphviz")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(BASE_CXX_FLAGS "-Wall -Wpedantic -Wextra -std=c++14 -fPIC")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(BASE_CXX_FLAGS "-Wall -Wpedantic -Wextra -std=c++14 -fPIC")
elseif (WIN32)
    set(BASE_CXX_FLAGS "/std:c++14 /W3")
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP") # Enable warnings and multi-processor compilation
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
endif()

# Debug-specific flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring for Debug build.")
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(DEBUG_CXX_FLAGS "-g")
        if (ASAN)
            message(STATUS "Configuring for ASAN.")
            set(DEBUG_CXX_FLAGS "-g -fsanitize=address,undefined -fno-omit-frame-pointer")
            set(DEBUG_LINK_FLAGS "-fsanitize=address,undefined")
        endif()
    elseif (MSVC)
        set(DEBUG_CXX_FLAGS "/Zi")  # Debug symbols
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG") # Generate .pdb
    endif()
else()
    message(STATUS "Configuring for Release build.")
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(RELEASE_CXX_FLAGS "-O3 -march=native")
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(RELEASE_CXX_FLAGS "-O3")
    elseif (WIN32)
        if (MSVC)
            set(RELEASE_CXX_FLAGS "/O2 /Ob2 /Oi /Ot /Oy")
        else()
            set(RELEASE_CXX_FLAGS "/O2")
        endif()
    endif()
endif()

# Combine base flags with build-specific flags
set(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${CMAKE_CXX_FLAGS} ${DEBUG_CXX_FLAGS} ${RELEASE_CXX_FLAGS}")


add_library(gbrl_cpp_src OBJECT ${CPP_SOURCES})

