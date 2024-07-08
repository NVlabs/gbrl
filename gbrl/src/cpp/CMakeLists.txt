##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl/license.html
#
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (CUDAToolkit_FOUND AND USE_CUDA AND NOT APPLE)
    include_directories(${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/gbrl/src/cuda)
    add_definitions(-DUSE_CUDA)
endif()

if (GRAPHVIZ_FOUND)
message(STATUS "Found Graphviz")
    set(GRAPHVIZ_INCLUDE_DIRS ${GRAPHVIZ_INCLUDE_DIR} ${GRAPHVIZ_INCLUDE_DIR}/graphviz)
    add_definitions(-DUSE_GRAPHVIZ)
    include_directories(${GRAPHVIZ_INCLUDE_DIRS})
else()
    message(STATUS "Did not Find Graphviz")
endif()


if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -Wall -Wpedantic -Wextra -march=native -fPIC")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -Wall -Wpedantic -Wextra -fPIC")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /std:c++14 /W3")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()


add_library(gbrl_cpp_src OBJECT ${CPP_SOURCES})
# target_compile_definitions(gbrl_cpp_src PUBLIC MODULE_NAME="gbrl_cpp")
