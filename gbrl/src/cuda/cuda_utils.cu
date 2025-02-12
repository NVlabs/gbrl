//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.h"
#include "cuda_types.h"

cudaError_t allocateCudaMemory(void** device_ptr, size_t size, const std::string& error_message) {
    cudaError_t error = cudaMalloc(device_ptr, size);
    if (error != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        std::cerr << "CUDA Allocation Error: " << error_message << " - " 
                  << cudaGetErrorString(error) << " when trying to allocate "
                  << (size / (1024.0 * 1024.0)) << " MB." << std::endl;

        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB, "
                  << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB." << std::endl;
    }
    return error;
}

void get_grid_dimensions(int n_elements, int& blocks, int& threads_per_block) {
    const int max_threads_per_block = THREADS_PER_BLOCK; // You can adjust this based on your GPU

    if (n_elements < WARP_SIZE) {
        threads_per_block = WARP_SIZE;
        blocks = 1;
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    blocks = n_elements / max_threads_per_block + 1;

    threads_per_block = 1 << static_cast<int>(ceilf(log2f(static_cast<float>(n_elements) / blocks)));
    threads_per_block = (threads_per_block > deviceProp.maxThreadsPerBlock) ? deviceProp.maxThreadsPerBlock : threads_per_block;
}

__global__ void selective_copyi(const int num_indices, const int* indices, int* dest, const int* src, const int elements_dim){
    // elements_dim is typically small (< 10)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        int start_idx = indices[idx];
        // printf("copying for idx %d from start_idx %d\n", idx, start_idx);
        for (int j = 0; j < elements_dim; ++j) {
            dest[idx*elements_dim + j] = src[start_idx*elements_dim + j];
        }
    }
}

__global__ void selective_copyb(const int num_indices, const int* indices, bool* dest, const bool* src, const int elements_dim){
    // elements_dim is typically small (< 10)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        int start_idx = indices[idx];
        for (int j = 0; j < elements_dim; ++j) {
            dest[idx*elements_dim + j] = src[start_idx*elements_dim + j];
        }
    }
}

__global__ void selective_copyf(const int num_indices, const int* indices, float* dest, const float* src, const int elements_dim){
    // elements_dim is typically small (< 10)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        int start_idx = indices[idx];
        for (int j = 0; j < elements_dim; ++j) {
            dest[idx*elements_dim + j] = src[start_idx*elements_dim + j];
        }
    }
}

__global__ void selective_copyc(const int num_indices, const int* indices, char* dest, const char* src, const int elements_dim){
    // Categorical values are stored as a 3D tensor - n_leaf/n_trees (oblivious or greedy tree) x max depth x MAX_CHAR_SIZE 
    if (blockIdx.x  < num_indices) {
        int start_idx = indices[blockIdx.x];
        int dest_idx = (blockIdx.x * elements_dim) * MAX_CHAR_SIZE;
        int src_idx = (start_idx * elements_dim) * MAX_CHAR_SIZE;
        for (int char_idx = threadIdx.x; char_idx < MAX_CHAR_SIZE; char_idx += blockDim.x)
            dest[dest_idx + char_idx] = src[src_idx + char_idx];
    }
}