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
 * @brief Average per-objective leaf densities on GPU
 *
 * For every sample, traverses the ensemble and accumulates the per-objective
 * density vector of each leaf it lands in, then divides by the number of
 * traversed trees. Result is left in GPU memory.
 *
 * @param dataset Input dataset (obs may live on host or device)
 * @param densities_out Output densities on GPU (n_samples x n_objs), assigned by this call
 * @param metadata Ensemble configuration
 * @param edata Ensemble parameters on GPU
 * @param start_tree_idx Starting tree index
 * @param stop_tree_idx Stopping tree index (0 means all trees)
 */
void predict_densities_cuda(
    dataSet *dataset,
    float *&densities_out,
    ensembleMetaData *metadata,
    ensembleData *edata,
    int start_tree_idx,
    int stop_tree_idx
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

/**
 * @brief CUDA kernel accumulating leaf densities for greedy trees
 *
 * One block per leaf; each thread strides over samples. Threads atomically add
 * the leaf's density vector into the row of every sample that reaches it.
 * Categorical arrays are only dereferenced when is_numerics is false, so they
 * may be null when the model has no categorical features.
 *
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations (may be null if n_cat_features == 0)
 * @param densities_out Output densities (n_samples x n_objs), accumulated in place
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param feature_indices Feature indices for all splits
 * @param depths Node depths
 * @param feature_values Split thresholds
 * @param inequality_directions Split directions
 * @param densities Per-leaf density vectors (n_leaves x n_objs)
 * @param categorical_values Categorical split values (may be null if n_cat_features == 0)
 * @param is_numerics Feature type indicators
 * @param n_objs Number of objectives
 * @param max_depth Maximum tree depth
 * @param leaf_offset Leaf index offset
 */
__global__ void predict_densities_kernel_greedy(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    float* __restrict__ densities_out,
    const int n_samples,
    const int n_num_features,
    const int n_cat_features,
    const int* __restrict__ feature_indices,
    const int* __restrict__ depths,
    const float* __restrict__ feature_values,
    const bool* __restrict__ inequality_directions,
    const float* __restrict__ densities,
    const char* __restrict__ categorical_values,
    const bool* __restrict__ is_numerics,
    const int n_objs,
    const int max_depth,
    const int leaf_offset
);

/**
 * @brief CUDA kernel accumulating leaf densities for oblivious trees
 *
 * One block per tree; each thread strides over samples, computing the leaf
 * index bitwise and adding that leaf's density vector to the sample's row.
 *
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations (may be null if n_cat_features == 0)
 * @param densities_out Output densities (n_samples x n_objs), accumulated in place
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param feature_indices Feature indices for all splits
 * @param depths Tree depths
 * @param feature_values Split thresholds
 * @param densities Per-leaf density vectors (n_leaves x n_objs)
 * @param tree_indices Starting leaf index of each tree
 * @param categorical_values Categorical split values (may be null if n_cat_features == 0)
 * @param is_numerics Feature type indicators
 * @param n_objs Number of objectives
 * @param max_depth Maximum tree depth
 * @param tree_offset Tree index offset
 */
__global__ void predict_densities_oblivious_kernel(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    float* __restrict__ densities_out,
    const int n_samples,
    const int n_num_features,
    const int n_cat_features,
    const int* __restrict__ feature_indices,
    const int* __restrict__ depths,
    const float* __restrict__ feature_values,
    const float* __restrict__ densities,
    const int* __restrict__ tree_indices,
    const char* __restrict__ categorical_values,
    const bool* __restrict__ is_numerics,
    const int n_objs,
    const int max_depth,
    const int tree_offset
);

/**
 * @brief CUDA kernel scaling every element of a matrix by a scalar
 *
 * @param mat Matrix modified in place
 * @param scale Multiplier
 * @param size Number of elements
 */
__global__ void scale_mat_kernel(
    float* __restrict__ mat,
    const float scale,
    const int size
);
#endif

#ifdef __cplusplus
}
#endif

#endif 