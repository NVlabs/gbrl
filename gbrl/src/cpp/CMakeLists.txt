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

