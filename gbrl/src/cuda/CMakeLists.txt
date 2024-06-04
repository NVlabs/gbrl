# gbrl/src/cuda/CMakeLists.txt

set(CUDA_SOURCES
    cuda_predictor.cu
    cuda_fitter.cu
    cuda_loss.cu
    cuda_types.cu
    cuda_utils.cu
    cuda_preprocess.cu
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gbrl/src/cpp)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# -I/usr/include/boost
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -O3 -Xcompiler -fPIC")
elseif (WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/MD")
endif()

# Condition to add -g flag
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
endif()

add_library(cuda_gbrl_src OBJECT ${CUDA_SOURCES})
