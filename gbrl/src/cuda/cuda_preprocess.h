//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
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
#ifndef CUDA_PREPROCESSING_H
#define CUDA_PREPROCESSING_H

#include "types.h"
#include "cuda_types.h"

#ifdef __cplusplus
extern "C" {
#endif
int* sort_indices_cuda(const float* __restrict__ obs, const int n_samples, const int n_features);
void preprocess_matrices(float* __restrict__ grads, float* __restrict__ grads_norm, const int n_rows, const int n_cols, const scoreFunc split_score_func);
int quantile_candidates_cuda(const float* __restrict__ obs, const int* __restrict__ sorted_indices, int* candidate_indices, float* candidate_values,const int n_samples, const int n_features, const int n_bins);
int uniform_candidates_cuda(const float* __restrict__ obs, int* candidate_indices, float* candidate_values,const int n_samples, const int n_features, const int n_bins);
int process_candidates_cuda(const float* gpu_obs, const char* categorical_obs, const float *grad_norms, int* candidate_indices, float *candidate_values, char* candidate_categories, bool* candidate_numerics, const int n_samples, const int n_num_features, const int n_cat_features, const int n_bins, const generatorType generator_type);
float* transpose_matrix(const float *mat, float *trans_mat, const int width, const int height);
#ifdef __CUDACC__  // This macro is defined by NVCC
__global__ void iota_kernel(int *arr, int size);
__global__ void bitonic_sort_kernel(const float* __restrict__ input, int* __restrict__ indices, int n_rows, int n_features);
__global__ void center_matrix(float* __restrict__ input, const int n_cols, const int n_rows);
__global__ void rowwise_squared_norm(const float* __restrict__ input, const int n_cols, float* __restrict__ per_row_results, const int n_rows);
__global__ void quantile_kernel(const float* __restrict__ obs, const int* __restrict__ sorted_indices, const int n_samples, const int n_features, const int n_bins,  int* __restrict__ candidate_indices, float* __restrict__ candidate_values, char* __restrict__ candidate_cateogories, bool* __restrict__ candidate_numerics);
__global__ void place_unique_elements_infront_kernel(int* __restrict__ candidate_indices, float* __restrict__ candidate_values, char* __restrict__ candidate_cateogories, bool* __restrict__ candidate_numerics, int size, int* pos_idx);
__global__ void get_colwise_min(const float* __restrict__ input, const int n_cols, float* __restrict__ per_col_max, const int n_rows);
__global__ void get_colwise_max(const float* __restrict__ input, const int n_cols, float* __restrict__ per_col_max, const int n_rows);
__global__ void linspace_kernel(const float* __restrict__ min_vec, const float* __restrict__ max_vec, int n_features, int n_bins, int* __restrict__ candidate_indices, float* __restrict__ candidate_values, char* __restrict__ candidate_cateogories, bool* __restrict__ candidate_numerics);
__global__ void transpose1D(float *out, const float *in, int width, int height);
#endif 

#ifdef __cplusplus
}
#endif

#endif // end CUDA_MATH_H