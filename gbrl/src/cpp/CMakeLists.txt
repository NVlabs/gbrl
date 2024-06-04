# gbrl/src/cpp/CMakeLists.txt

set(CPP_SOURCES
    gbrl_binding.cpp
    gbrl.cpp
    types.cpp
    optimizer.cpp
    scheduler.cpp
    node.cpp
    utils.cpp
    fitter.cpp
    predictor.cpp
    split_candidate_generator.cpp
    loss.cpp
    math_ops.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (CUDAToolkit_FOUND AND NOT APPLE)
    include_directories(${CMAKE_SOURCE_DIR}/gbrl/src/cuda)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -Wall -Wpedantic -Wextra -march=native")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -Wall -Wpedantic -Wextra -Xpreprocessor -fopenmp")
elseif (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++14 /O2 /W3")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

add_library(gbrl_cpp_src OBJECT ${CPP_SOURCES})
target_compile_definitions(gbrl_cpp_src PRIVATE MODULE_NAME="gbrl_cpp")
