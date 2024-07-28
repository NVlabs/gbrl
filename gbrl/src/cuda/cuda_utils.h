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

#ifdef __cplusplus
extern "C" {
#endif
void get_grid_dimensions(int elements, int& blocks, int& threads_per_block);
#ifdef __CUDACC__  
__global__ void selective_copyi(const int num_indices, const int* indices, int* dest, const int* src, const int elements_dim);
__global__ void selective_copyf(const int num_indices, const int* indices, float* dest, const float* src, const int elements_dim);
__global__ void selective_copyb(const int num_indices, const int* indices, bool* dest, const bool* src, const int elements_dim);
__global__ void selective_copyc(const int num_indices, const int* indices, char* dest, const char* src, const int elements_dim);
#endif 
#ifdef __cplusplus
}
#endif

#endif 



