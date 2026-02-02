//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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



