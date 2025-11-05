//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda_predictor.h
 * @brief GPU prediction functions for gradient boosted ensembles
 * 
 * Provides CUDA kernels and utilities for fast parallel prediction
 * from decision tree ensembles on NVIDIA GPUs.
 */

#ifndef CUDA_PREDICTOR_H
#define CUDA_PREDICTOR_H

#include "scheduler.h"
#include "optimizer.h"
#include "node.h"
#include "types.h"
#include "cuda_types.h"

/**
 * @brief Deep copy optimizer vector to GPU
 * 
 * @param host_opts Vector of CPU optimizers
 * @return Array of GPU optimizers
 */
SGDOptimizerGPU** deepCopySGDOptimizerVectorToGPU(
    const std::vector<Optimizer*>& host_opts
);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate predictions on GPU with host transfer
 * 
 * Runs prediction kernels and copies results to host memory.
 * 
 * @param dataset Input dataset on GPU
 * @param preds Output predictions (allocated on host)
 * @param metadata Ensemble configuration
 * @param edata Ensemble parameters on GPU
 * @param opts GPU optimizer array
 * @param n_opts Number of optimizers
 * @param start_tree_idx Starting tree index
 * @param stop_tree_idx Stopping tree index
 */
void predict_cuda(
    dataSet *dataset,
    float *&preds,
    ensembleMetaData *metadata,
    ensembleData *edata,
    SGDOptimizerGPU** opts,
    const int n_opts,
    int start_tree_idx,
    int stop_tree_idx
);

/**
 * @brief Generate predictions on GPU without host transfer
 * 
 * @param dataset Input dataset on GPU
 * @param device_preds Output predictions on GPU
 * @param metadata Ensemble configuration
 * @param edata Ensemble parameters on GPU
 * @param opts GPU optimizer array
 * @param n_opts Number of optimizers
 * @param start_tree_idx Starting tree index
 * @param stop_tree_idx Stopping tree index
 * @param add_bias Whether to add bias term
 */
void predict_cuda_no_host(
    dataSet *dataset,
    float *device_preds,
    ensembleMetaData *metadata,
    ensembleData *edata,
    SGDOptimizerGPU** opts,
    const int n_opts,
    int start_tree_idx,
    int stop_tree_idx,
    const bool add_bias
);

/**
 * @brief Free GPU optimizer array
 * 
 * @param device_ops GPU optimizer array
 * @param n_opts Number of optimizers
 */
void freeSGDOptimizer(SGDOptimizerGPU **device_ops, const int n_opts);

#ifdef __CUDACC__  // NVCC only

/**
 * @brief CUDA kernel to add vector to matrix rows
 * 
 * @param vec Vector to add
 * @param mat Matrix (modified in place)
 * @param n_samples Number of matrix rows
 * @param n_cols Number of matrix columns
 */
__global__ void add_vec_to_mat_kernel(
    const float *vec,
    float *mat,
    const int n_samples,
    const int n_cols
);

/**
 * @brief CUDA kernel for tree-wise prediction
 * 
 * Each thread processes multiple trees for one sample.
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param preds Output predictions
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param feature_indices Feature indices for all splits
 * @param depths Node depths
 * @param feature_values Split thresholds
 * @param inequality_directions Split directions
 * @param leaf_values Leaf predictions
 * @param categorical_values Categorical split values
 * @param is_numerics Feature type indicators
 * @param opts GPU optimizers
 * @param n_opts Number of optimizers
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param leaf_offset Leaf value offset
 */
__global__ void predict_kernel_tree_wise(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    float* __restrict__ preds,
    const int n_samples,
    const int n_num_features,
    const int n_cat_features,
    const int* __restrict__ feature_indices,
    const int* __restrict__ depths,
    const float* __restrict__ feature_values,
    const bool* __restrict__ inequality_directions,
    const float* __restrict__ leaf_values,
    const char* __restrict__ categorical_values,
    const bool* __restrict__ is_numerics,
    SGDOptimizerGPU** opts,
    const int n_opts,
    const int output_dim,
    const int max_depth,
    const int leaf_offset
);

/**
 * @brief CUDA kernel for numerical-only tree-wise prediction
 * 
 * Optimized variant for datasets without categorical features.
 * 
 * @param obs Numerical observations
 * @param preds Output predictions
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param feature_indices Feature indices for all splits
 * @param depths Node depths
 * @param feature_values Split thresholds
 * @param inequality_directions Split directions
 * @param leaf_values Leaf predictions
 * @param opts GPU optimizers
 * @param n_opts Number of optimizers
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param leaf_offset Leaf value offset
 */
__global__ void predict_kernel_numerical_only(
    const float* __restrict__ obs,
    float* __restrict__ preds,
    const int n_samples,
    const int n_num_features,
    const int* __restrict__ feature_indices,
    const int* __restrict__ depths,
    const float* __restrict__ feature_values,
    const bool* __restrict__ inequality_directions,
    const float* __restrict__ leaf_values,
    SGDOptimizerGPU** opts,
    const int n_opts,
    const int output_dim,
    const int max_depth,
    const int leaf_offset
);

/**
 * @brief CUDA kernel for sample-wise tree-wise prediction
 * 
 * Each thread processes all trees for one sample.
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param preds Output predictions
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param feature_indices Feature indices for all splits
 * @param depths Node depths
 * @param feature_values Split thresholds
 * @param inequality_directions Split directions
 * @param leaf_values Leaf predictions
 * @param categorical_values Categorical split values
 * @param is_numerics Feature type indicators
 * @param opts GPU optimizers
 * @param n_opts Number of optimizers
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param leaf_offset Leaf value offset
 * @param n_leaves Number of leaf nodes
 */
__global__ void predict_sample_wise_kernel_tree_wise(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    float* __restrict__ preds,
    const int n_samples,
    const int n_num_features,
    const int n_cat_features,
    const int* __restrict__ feature_indices,
    const int* __restrict__ depths,
    const float* __restrict__ feature_values,
    const bool* __restrict__ inequality_directions,
    const float* __restrict__ leaf_values,
    const char* __restrict__ categorical_values,
    const bool* __restrict__ is_numerics,
    SGDOptimizerGPU** opts,
    const int n_opts,
    const int output_dim,
    const int max_depth,
    const int leaf_offset,
    const int n_leaves
);

/**
 * @brief CUDA kernel for oblivious tree prediction (numerical only)
 * 
 * @param obs Numerical observations
 * @param preds Output predictions
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param feature_indices Feature indices for all splits
 * @param depths Node depths
 * @param feature_values Split thresholds
 * @param inequality_directions Split directions
 * @param leaf_values Leaf predictions
 * @param tree_indices Tree index array
 * @param opts GPU optimizers
 * @param n_opts Number of optimizers
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param tree_offset Tree index offset
 */
__global__ void predict_oblivious_kernel_numerical_only(
    const float* __restrict__ obs,
    float* __restrict__ preds,
    const int n_samples,
    const int n_num_features,
    const int* __restrict__ feature_indices,
    const int* __restrict__ depths,
    const float* __restrict__ feature_values,
    const bool* __restrict__ inequality_directions,
    const float* __restrict__ leaf_values,
    const int* __restrict__ tree_indices,
    SGDOptimizerGPU** opts,
    const int n_opts,
    const int output_dim,
    const int max_depth,
    const int tree_offset
);

/**
 * @brief CUDA kernel for oblivious tree-wise prediction
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param preds Output predictions
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param feature_indices Feature indices for all splits
 * @param depths Node depths
 * @param feature_values Split thresholds
 * @param inequality_directions Split directions
 * @param leaf_values Leaf predictions
 * @param tree_indices Tree index array
 * @param categorical_values Categorical split values
 * @param is_numerics Feature type indicators
 * @param opts GPU optimizers
 * @param n_opts Number of optimizers
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param tree_offset Tree index offset
 */
__global__ void predict_oblivious_kernel_tree_wise(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    float* __restrict__ preds,
    const int n_samples,
    const int n_num_features,
    const int n_cat_features,
    const int* __restrict__ feature_indices,
    const int* __restrict__ depths,
    const float* __restrict__ feature_values,
    const bool* __restrict__ inequality_directions,
    const float* __restrict__ leaf_values,
    const int* __restrict__ tree_indices,
    const char* __restrict__ categorical_values,
    const bool* __restrict__ is_numerics,
    SGDOptimizerGPU** opts,
    const int n_opts,
    const int output_dim,
    const int max_depth,
    const int tree_offset
);
#endif

#ifdef __cplusplus
}
#endif

#endif 