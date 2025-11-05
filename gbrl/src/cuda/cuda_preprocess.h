//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda_preprocess.h
 * @brief GPU preprocessing functions for gradient boosting
 * 
 * Provides CUDA kernels for data preprocessing, candidate generation,
 * matrix operations, and shuffling on NVIDIA GPUs.
 */

#ifndef CUDA_PREPROCESSING_H
#define CUDA_PREPROCESSING_H

#include "types.h"
#include "cuda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sort observation indices by feature values on GPU
 * 
 * @param obs Numerical observations (n_samples x n_features)
 * @param n_samples Number of samples
 * @param n_features Number of features
 * @return Sorted indices (allocated on GPU)
 */
int* sort_indices_cuda(
    const float* __restrict__ obs,
    const int n_samples,
    const int n_features
);

/**
 * @brief Preprocess gradient matrices on GPU
 * 
 * Centers and normalizes gradients based on scoring function.
 * 
 * @param grads Gradient values (modified in place)
 * @param grads_norm Gradient norms
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 * @param split_score_func Split scoring function
 */
void preprocess_matrices(
    float* __restrict__ grads,
    float* __restrict__ grads_norm,
    const int n_rows,
    const int n_cols,
    const scoreFunc split_score_func
);

/**
 * @brief Generate quantile-based split candidates on GPU
 * 
 * @param obs Numerical observations
 * @param sorted_indices Sorted observation indices
 * @param candidate_indices Output candidate feature indices
 * @param candidate_values Output candidate threshold values
 * @param n_samples Number of samples
 * @param n_features Number of features
 * @param n_bins Number of quantile bins
 * @return Number of candidates generated
 */
int quantile_candidates_cuda(
    const float* __restrict__ obs,
    const int* __restrict__ sorted_indices,
    int* candidate_indices,
    float* candidate_values,
    const int n_samples,
    const int n_features,
    const int n_bins
);

/**
 * @brief Generate uniform-spaced split candidates on GPU
 * 
 * @param obs Numerical observations
 * @param candidate_indices Output candidate feature indices
 * @param candidate_values Output candidate threshold values
 * @param n_samples Number of samples
 * @param n_features Number of features
 * @param n_bins Number of uniform bins
 * @return Number of candidates generated
 */
int uniform_candidates_cuda(
    const float* __restrict__ obs,
    int* candidate_indices,
    float* candidate_values,
    const int n_samples,
    const int n_features,
    const int n_bins
);

/**
 * @brief Process and generate all split candidates on GPU
 * 
 * Handles both numerical and categorical features.
 * 
 * @param gpu_obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param grad_norms Gradient norms for feature weighting
 * @param candidate_indices Output candidate feature indices
 * @param candidate_values Output candidate threshold values
 * @param candidate_categories Output categorical values
 * @param candidate_numerics Output feature type indicators
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param n_bins Number of bins per feature
 * @param generator_type Candidate generation strategy
 * @return Total number of candidates
 */
int process_candidates_cuda(
    const float* gpu_obs,
    const char* categorical_obs,
    const float *grad_norms,
    int* candidate_indices,
    float *candidate_values,
    char* candidate_categories,
    bool* candidate_numerics,
    const int n_samples,
    const int n_num_features,
    const int n_cat_features,
    const int n_bins,
    const generatorType generator_type
);

/**
 * @brief Transpose matrix on GPU
 * 
 * @param mat Input matrix
 * @param trans_mat Output transposed matrix
 * @param width Matrix width
 * @param height Matrix height
 * @return Pointer to transposed matrix
 */
float* transpose_matrix(
    const float *mat,
    float *trans_mat,
    const int width,
    const int height
);

/**
 * @brief Shuffle and copy training data on GPU
 * 
 * @param d_obs Source observations
 * @param d_targets Source targets
 * @param d_categorical_obs Source categorical observations
 * @param d_indices Shuffle indices
 * @param d_training_obs Destination observations
 * @param d_training_targets Destination targets
 * @param d_training_cat_obs Destination categorical observations
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param output_dim Output dimensionality
 * @param max_char_size Maximum categorical string length
 */
void shuffle_and_copy_cuda(
    const float* d_obs,
    const float* d_targets,
    const char* d_categorical_obs,
    const int* d_indices,
    float* d_training_obs,
    float* d_training_targets,
    char* d_training_cat_obs,
    int n_samples,
    int n_num_features,
    int n_cat_features,
    int output_dim,
    int max_char_size
);

/**
 * @brief Compute column-wise mean on GPU
 * 
 * @param d_in Input matrix
 * @param d_out Output mean vector
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 */
void column_mean_reduce(
    const float* d_in,
    float* d_out,
    size_t n_rows,
    size_t n_cols
);

#ifdef __CUDACC__  // NVCC only

/**
 * @brief CUDA kernel to initialize array with sequential values
 * 
 * @param arr Output array
 * @param size Array size
 */
__global__ void iota_kernel(int *arr, int size);

/**
 * @brief CUDA kernel for bitonic sort of indices
 * 
 * @param input Values to sort by
 * @param indices Index array (sorted in place)
 * @param n_rows Number of rows
 * @param n_features Number of features
 */
__global__ void bitonic_sort_kernel(
    const float* __restrict__ input,
    int* __restrict__ indices,
    int n_rows,
    int n_features
);

/**
 * @brief CUDA kernel to center matrix columns
 * 
 * @param input Matrix (modified in place)
 * @param n_cols Number of columns
 * @param n_rows Number of rows
 */
__global__ void center_matrix(
    float* __restrict__ input,
    const int n_cols,
    const int n_rows
);

/**
 * @brief CUDA kernel for row-wise squared norm
 * 
 * @param input Input matrix
 * @param n_cols Number of columns
 * @param per_row_results Output row norms
 * @param n_rows Number of rows
 */
__global__ void rowwise_squared_norm(
    const float* __restrict__ input,
    const int n_cols,
    float* __restrict__ per_row_results,
    const int n_rows
);

/**
 * @brief CUDA kernel for quantile candidate generation
 * 
 * @param obs Numerical observations
 * @param sorted_indices Sorted observation indices
 * @param n_samples Number of samples
 * @param n_features Number of features
 * @param n_bins Number of quantile bins
 * @param candidate_indices Output candidate feature indices
 * @param candidate_values Output candidate threshold values
 * @param candidate_cateogories Output categorical values (unused)
 * @param candidate_numerics Output feature type indicators
 */
__global__ void quantile_kernel(
    const float* __restrict__ obs,
    const int* __restrict__ sorted_indices,
    const int n_samples,
    const int n_features,
    const int n_bins,
    int* __restrict__ candidate_indices,
    float* __restrict__ candidate_values,
    char* __restrict__ candidate_cateogories,
    bool* __restrict__ candidate_numerics
);

/**
 * @brief CUDA kernel to compact unique elements
 * 
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_cateogories Categorical values
 * @param candidate_numerics Feature type indicators
 * @param size Array size
 * @param pos_idx Output position counter
 */
__global__ void place_unique_elements_infront_kernel(
    int* __restrict__ candidate_indices,
    float* __restrict__ candidate_values,
    char* __restrict__ candidate_cateogories,
    bool* __restrict__ candidate_numerics,
    int size,
    int* pos_idx
);

/**
 * @brief CUDA kernel for column-wise minimum
 * 
 * @param input Input matrix
 * @param n_cols Number of columns
 * @param per_col_max Output minimum values (note: parameter name is misleading)
 * @param n_rows Number of rows
 */
__global__ void get_colwise_min(
    const float* __restrict__ input,
    const int n_cols,
    float* __restrict__ per_col_max,
    const int n_rows
);

/**
 * @brief CUDA kernel for column-wise maximum
 * 
 * @param input Input matrix
 * @param n_cols Number of columns
 * @param per_col_max Output maximum values
 * @param n_rows Number of rows
 */
__global__ void get_colwise_max(
    const float* __restrict__ input,
    const int n_cols,
    float* __restrict__ per_col_max,
    const int n_rows
);

/**
 * @brief CUDA kernel for uniform linspace candidate generation
 * 
 * @param min_vec Minimum values per feature
 * @param max_vec Maximum values per feature
 * @param n_features Number of features
 * @param n_bins Number of bins
 * @param candidate_indices Output candidate feature indices
 * @param candidate_values Output candidate threshold values
 * @param candidate_cateogories Output categorical values (unused)
 * @param candidate_numerics Output feature type indicators
 */
__global__ void linspace_kernel(
    const float* __restrict__ min_vec,
    const float* __restrict__ max_vec,
    int n_features,
    int n_bins,
    int* __restrict__ candidate_indices,
    float* __restrict__ candidate_values,
    char* __restrict__ candidate_cateogories,
    bool* __restrict__ candidate_numerics
);

/**
 * @brief CUDA kernel for 1D matrix transpose
 * 
 * @param out Output transposed matrix
 * @param in Input matrix
 * @param width Matrix width
 * @param height Matrix height
 */
__global__ void transpose1D(
    float *out,
    const float *in,
    int width,
    int height
);

/**
 * @brief CUDA kernel for shuffled data copy
 * 
 * @param obs Source observations
 * @param targets Source targets
 * @param categorical_obs Source categorical observations
 * @param indices Shuffle indices
 * @param training_obs Destination observations
 * @param training_targets Destination targets
 * @param training_cat_obs Destination categorical observations
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param output_dim Output dimensionality
 * @param max_char_size Maximum categorical string length
 */
__global__ void shuffle_and_copy_kernel(
    const float* __restrict__ obs,
    const float* __restrict__ targets,
    const char* __restrict__ categorical_obs,
    const int* __restrict__ indices,
    float* __restrict__ training_obs,
    float* __restrict__ training_targets,
    char* __restrict__ training_cat_obs,
    int n_samples,
    int n_num_features,
    int n_cat_features,
    int output_dim,
    int max_char_size
);

/**
 * @brief CUDA kernel for column mean reduction
 * 
 * @param in Input matrix
 * @param out Output mean vector
 * @param n_rows Number of rows
 * @param n_cols Number of columns
 */
__global__ void column_mean_reduce_kernel(
    const float * __restrict__ in,
    float * __restrict__ out,
    size_t n_rows,
    size_t n_cols
);

#endif

#ifdef __cplusplus
}
#endif

#endif // CUDA_PREPROCESSING_H