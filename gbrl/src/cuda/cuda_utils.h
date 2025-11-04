//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <string>

#include <cuda_runtime.h>

cudaError_t allocateCudaMemory(void** device_ptr, size_t size, const std::string& error_message);

#ifdef __cplusplus
extern "C" {
#endif
void get_grid_dimensions(int elements, int& blocks, int& threads_per_block);
void get_tpb_dimensions(int n_elements, int blocks, int& threads_per_block);
#ifdef __cplusplus
}
#endif

#endif 



