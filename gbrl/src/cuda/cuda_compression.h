//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_COMPRESSION_H
#define CUDA_COMPRESSION_H

#include "types.h"
#include "cuda_types.h"


#ifdef __cplusplus
extern "C" {
#endif
void get_matrix_representation_cuda(dataSet *dataset, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, matrixRepresentation *matrix);
ensembleData* compress_ensemble_cuda(ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W);
#ifdef __CUDACC__  // This macro is defined by NVCC
__global__ void get_representation_oblivious_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features,  const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const int* __restrict__ tree_indices, const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);
__global__ void get_representation_oblivious_kernel_numerical_only(const float* __restrict__ obs, const int n_samples, const int n_num_features,  const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const int* __restrict__ tree_indices, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);
__global__ void get_representation_kernel_numerical_only(const float* __restrict__ obs, const int n_samples, const int n_num_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);
__global__ void get_representation_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values,  const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);
__global__ void add_W_matrix_to_values_kernel(const float * __restrict__ W, float* __restrict__ leaf_values, float* __restrict__ bias, SGDOptimizerGPU** opts, const int n_opts, const int n_leaves, const int output_dim);
__global__ void get_V_kernel(float* __restrict__ V, const float* __restrict__ leaf_values, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int n_leaves);
#endif 


#ifdef __cplusplus
}
#endif // extern C
#endif 