//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_PREDICTOR_H
#define CUDA_PREDICTOR_H

#include "scheduler.h"
#include "optimizer.h"
#include "node.h"

#include "types.h"
#include "cuda_types.h"


SGDOptimizerGPU** deepCopySGDOptimizerVectorToGPU(const std::vector<Optimizer*>& host_opts);
#ifdef __cplusplus
extern "C" {
#endif
void predict_cuda(dataSet *dataset, float *host_preds, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, int start_tree_idx, int stop_tree_idx);
void predict_cuda_no_host(dataSet *dataset, float *device_preds, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, int start_tree_idx, int stop_tree_idx, const bool add_bias);
void freeSGDOptimizer(SGDOptimizerGPU **device_ops, const int n_opts);
#ifdef __CUDACC__  // This macro is defined by NVCC
__global__ void add_vec_to_mat_kernel(const float *vec, float *mat, const int n_samples, const int n_cols);
__global__ void predict_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int n_cat_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int leaf_offset);
__global__ void predict_kernel_numerical_only(const float* __restrict__ obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int leaf_offset);
__global__ void predict_sample_wise_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int n_cat_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int leaf_offset, const int n_leaves);
__global__ void predict_oblivious_kernel_numerical_only(const float* __restrict__ obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, const int* __restrict__ tree_indices, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int tree_offset);
__global__ void predict_oblivious_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int n_cat_features, const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values,const int* __restrict__ tree_indices, const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int tree_offset);
#endif 


#ifdef __cplusplus
}
#endif // extern C
#endif 