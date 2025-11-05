//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda_utils.h
 * @brief CUDA utility functions for memory management and kernel configuration
 * 
 * Provides helper functions for GPU memory allocation and optimal kernel
 * launch parameter calculation.
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <string>
#include <cuda_runtime.h>

/**
 * @brief Allocate CUDA device memory with error handling
 * 
 * @param device_ptr Pointer to store allocated device address
 * @param size Number of bytes to allocate
 * @param error_message Custom error message on failure
 * @return cudaError_t status code
 */
cudaError_t allocateCudaMemory(
    void** device_ptr,
    size_t size,
    const std::string& error_message
);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate optimal grid dimensions for kernel launch
 * 
 * Computes blocks and threads per block for given element count.
 * 
 * @param elements Total number of elements to process
 * @param blocks Output: number of blocks
 * @param threads_per_block Output: threads per block
 */
void get_grid_dimensions(
    int elements,
    int& blocks,
    int& threads_per_block
);

/**
 * @brief Calculate threads per block for fixed block count
 * 
 * @param n_elements Total number of elements
 * @param blocks Number of blocks (fixed)
 * @param threads_per_block Output: threads per block
 */
void get_tpb_dimensions(
    int n_elements,
    int blocks,
    int& threads_per_block
);

#ifdef __cplusplus
}
#endif

#endif 



