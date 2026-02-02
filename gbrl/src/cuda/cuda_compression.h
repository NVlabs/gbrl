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
 * @file cuda_compression.h
 * @brief CUDA declarations for GPU-accelerated tree ensemble compression
 * 
 * @warning EXPERIMENTAL - UNDER ACTIVE RESEARCH
 * This module implements tree ensemble compression algorithms that are currently
 * under active research and development. The API, behavior, and results may change
 * without notice. This feature has not been fully validated and is intended for
 * internal research use only.
 * 
 * Provides function declarations for CUDA-based compression operations and
 * kernel function signatures. Enables significant performance improvements
 * for large tree ensembles through GPU parallelization.
 */

#ifndef CUDA_COMPRESSION_H
#define CUDA_COMPRESSION_H

#include "types.h"
#include "cuda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate matrix representation on GPU
 * 
 * CUDA implementation of matrix representation generation for tree ensembles.
 * 
 * @param dataset Input dataset (observations on GPU)
 * @param metadata Ensemble metadata
 * @param edata Ensemble data on GPU
 * @param opts GPU optimizer array
 * @param n_opts Number of optimizers
 * @param matrix Output matrix (allocated on CPU)
 */
void get_matrix_representation_cuda(dataSet *dataset, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, matrixRepresentation *matrix);

/**
 * @brief Compress ensemble on GPU
 * 
 * Applies correction matrix and creates compressed ensemble using GPU kernels.
 * 
 * @param metadata Ensemble metadata
 * @param edata Ensemble data on GPU
 * @param opts GPU optimizer array
 * @param n_opts Number of optimizers
 * @param n_compressed_leaves Number of leaves after compression
 * @param n_compressed_trees Number of trees after compression
 * @param leaf_indices Indices of leaves to retain
 * @param tree_indices Indices of trees to retain
 * @param new_tree_indices New tree starting indices
 * @param W Correction matrix
 * @return Compressed ensemble data on GPU
 */
ensembleData* compress_ensemble_cuda(ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W);

#ifdef __CUDACC__  // Kernel declarations only visible to NVCC

/**
 * @brief CUDA kernel for oblivious tree representation with mixed features
 * 
 * Each block processes one tree, threads handle samples in parallel.
 * Uses binary path indexing for efficient leaf lookup.
 */
__global__ void get_representation_oblivious_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features,  const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const int* __restrict__ tree_indices, const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);

/**
 * @brief CUDA kernel for oblivious tree representation with numerical features only
 * 
 * Optimized version skipping categorical feature checks.
 * Each block processes one tree, threads handle samples in parallel.
 */
__global__ void get_representation_oblivious_kernel_numerical_only(const float* __restrict__ obs, const int n_samples, const int n_num_features,  const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const int* __restrict__ tree_indices, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);

/**
 * @brief CUDA kernel for greedy tree representation with numerical features only
 * 
 * Each block processes one leaf, checking conditions from leaf to root.
 * Optimized for numerical-only feature sets.
 */
__global__ void get_representation_kernel_numerical_only(const float* __restrict__ obs, const int n_samples, const int n_num_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);

/**
 * @brief CUDA kernel for greedy tree representation with mixed features
 * 
 * Each block processes one leaf, checking all split conditions.
 * Handles both numerical and categorical feature comparisons.
 */
__global__ void get_representation_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values,  const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, const int output_dim, const int max_depth, const int n_leaves, bool* __restrict__ A);

/**
 * @brief CUDA kernel to apply correction matrix W to leaf values
 * 
 * Inverse-scales W values and subtracts from leaf values for compression.
 * Optimized with loop unrolling for 1-2 optimizer common case.
 */
__global__ void add_W_matrix_to_values_kernel(const float * __restrict__ W, float* __restrict__ leaf_values, float* __restrict__ bias, SGDOptimizerGPU** opts, const int n_opts, const int n_leaves, const int output_dim);

/**
 * @brief CUDA kernel to extract and scale leaf values into matrix V
 * 
 * Copies leaf values scaled by negative learning rate.
 * Optimized with loop unrolling for 1-2 optimizer common case.
 */
__global__ void get_V_kernel(float* __restrict__ V, const float* __restrict__ leaf_values, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int n_leaves);

#endif 


#ifdef __cplusplus
}
#endif // extern C
#endif 