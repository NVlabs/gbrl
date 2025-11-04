//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda_fitter.h
 * @brief GPU tree fitting functions for gradient boosting
 * 
 * Provides CUDA kernels and utilities for parallel tree construction
 * on NVIDIA GPUs, including greedy and oblivious tree growing strategies.
 */

#ifndef CUDA_FITTER_H
#define CUDA_FITTER_H

#include "types.h"
#include "cuda_types.h"
#include "node.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate root tree node on GPU
 * 
 * @param dataset Training dataset
 * @param metadata Ensemble configuration
 * @return Pointer to GPU tree node
 */
TreeNodeGPU* allocate_root_tree_node(
    dataSet *dataset,
    ensembleMetaData *metadata
);

/**
 * @brief Allocate child tree node on GPU
 * 
 * @param host_parent Host copy of parent node
 * @param device_child Output pointer to GPU child node
 */
void allocate_child_tree_node(
    TreeNodeGPU* host_parent,
    TreeNodeGPU** device_child
);

/**
 * @brief Allocate and partition left/right child nodes on GPU
 * 
 * @param dataset Training dataset
 * @param parent_node GPU parent node
 * @param host_parent Host copy of parent node
 * @param left_child Output left child node
 * @param right_child Output right child node
 * @param candidata Split candidates
 * @param split_data Split evaluation buffers
 * @param metadata Ensemble configuration
 */
void allocate_child_tree_nodes(
    dataSet *dataset,
    TreeNodeGPU* parent_node,
    TreeNodeGPU* host_parent,
    TreeNodeGPU** left_child,
    TreeNodeGPU** right_child,
    candidatesData *candidata,
    splitDataGPU *split_data,
    ensembleMetaData *metadata
);

/**
 * @brief Free GPU tree node
 * 
 * @param node Tree node to deallocate
 */
void free_tree_node(TreeNodeGPU* node);

/**
 * @brief Evaluate split candidates greedily on GPU
 * 
 * @param dataset Training dataset
 * @param edata Ensemble data
 * @param node Current tree node
 * @param candidata Split candidates
 * @param metadata Ensemble configuration
 * @param split_data Split evaluation buffers
 * @param threads_per_block CUDA threads per block
 * @param parent_n_samples Parent node sample count
 */
void evaluate_greedy_splits(
    dataSet *dataset,
    ensembleData *edata,
    const TreeNodeGPU *node,
    candidatesData *candidata,
    ensembleMetaData *metadata,
    splitDataGPU* split_data,
    const int threads_per_block,
    const int parent_n_samples
);

/**
 * @brief Evaluate oblivious split candidates on GPU
 * 
 * @param dataset Training dataset
 * @param edata Ensemble data
 * @param nodes Array of tree nodes at same depth
 * @param depth Current tree depth
 * @param candidata Split candidates
 * @param metadata Ensemble configuration
 * @param split_data Split evaluation buffers
 */
void evaluate_oblivious_splits_cuda(
    dataSet *dataset,
    ensembleData *edata,
    TreeNodeGPU ** nodes,
    const int depth,
    candidatesData *candidata,
    ensembleMetaData *metadata,
    splitDataGPU *split_data
);

/**
 * @brief Calculate parallelism for greedy split evaluation
 * 
 * @param n_candidates Number of split candidates
 * @param output_dim Output dimensionality
 * @param threads_per_block Output threads per block
 * @param split_score_fun Split scoring function
 */
void calc_parallelism(
    const int n_candidates,
    const int output_dim,
    int &threads_per_block,
    const scoreFunc split_score_fun
);

/**
 * @brief Calculate parallelism for oblivious split evaluation
 * 
 * @param n_candidates Number of split candidates
 * @param output_dim Output dimensionality
 * @param threads_per_block Output threads per block
 * @param split_score_func Split scoring function
 * @param depth Current tree depth
 */
void calc_oblivious_parallelism(
    const int n_candidates,
    const int output_dim,
    int &threads_per_block,
    const scoreFunc split_score_func,
    const int depth
);

/**
 * @brief Fit oblivious tree on GPU
 * 
 * @param dataset Training dataset
 * @param edata Ensemble data
 * @param metadata Ensemble configuration
 * @param candidata Split candidates
 * @param split_data Split evaluation buffers
 */
void fit_tree_oblivious_cuda(
    dataSet *dataset,
    ensembleData *edata,
    ensembleMetaData *metadata,
    candidatesData *candidata,
    splitDataGPU *split_data
);

/**
 * @brief Fit greedy tree on GPU
 * 
 * @param dataset Training dataset
 * @param edata Ensemble data
 * @param metadata Ensemble configuration
 * @param candidata Split candidates
 * @param split_data Split evaluation buffers
 */
void fit_tree_greedy_cuda(
    dataSet *dataset,
    ensembleData *edata,
    ensembleMetaData *metadata,
    candidatesData *candidata,
    splitDataGPU *split_data
);

/**
 * @brief Add leaf node to ensemble
 * 
 * @param node Tree node to convert to leaf
 * @param depth Node depth
 * @param metadata Ensemble configuration
 * @param edata Ensemble data
 * @param dataset Training dataset
 */
void add_leaf_node(
    const TreeNodeGPU *node,
    const int depth,
    ensembleMetaData *metadata,
    ensembleData *edata,
    dataSet *dataset
);

#ifdef __CUDACC__

/**
 * @brief CUDA kernel for cosine split scoring
 * 
 * Evaluates split quality using cosine similarity.
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param grads Gradient values
 * @param feature_weights Per-feature importance weights
 * @param node Current tree node
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param min_data_in_leaf Minimum samples per leaf
 * @param split_scores Output split scores
 * @param global_n_samples Total number of samples
 * @param n_num_features Number of numerical features
 */
__global__ void split_score_cosine_cuda(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const float* __restrict__ feature_weights,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int min_data_in_leaf,
    float* __restrict__ split_scores,
    const int global_n_samples,
    const int n_num_features
);

/**
 * @brief CUDA kernel for L2 split scoring
 * 
 * Evaluates split quality using L2 variance reduction.
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param grads Gradient values
 * @param feature_weights Per-feature importance weights
 * @param node Current tree node
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param min_data_in_leaf Minimum samples per leaf
 * @param split_scores Output split scores
 * @param global_n_samples Total number of samples
 * @param n_num_features Number of numerical features
 */
__global__ void split_score_l2_cuda(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const float* __restrict__ feature_weights,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int min_data_in_leaf,
    float* __restrict__ split_scores,
    const int global_n_samples,
    const int n_num_features
);

/**
 * @brief CUDA kernel to find best split candidate
 * 
 * @param split_scores Candidate split scores
 * @param n_candidates Number of candidates
 * @param best_idx Output best candidate index
 * @param best_score Output best score
 * @param node Current tree node
 */
__global__ void update_best_candidate_cuda(
    float* __restrict__ split_scores,
    int n_candidates,
    int* __restrict__ best_idx,
    float* __restrict__ best_score,
    const TreeNodeGPU* __restrict__ node
);

/**
 * @brief CUDA kernel to reduce leaf gradient sum
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param grads Gradient values
 * @param values Output summed values
 * @param node Tree node
 * @param n_samples Number of samples
 * @param global_idx Global sample offset
 * @param policy_dim Output dimensionality
 */
__global__ void reduce_leaf_sum(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    float* __restrict__ values,
    const TreeNodeGPU* __restrict__ node,
    const int n_samples,
    const int global_idx,
    const int policy_dim
);

/**
 * @brief CUDA kernel to partition samples into child nodes
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param parent_node Parent tree node
 * @param left_child Left child node
 * @param right_child Right child node
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param best_idx Best split index
 * @param tree_counters Atomic counters for partitioning
 * @param global_n_samples Total number of samples
 */
__global__ void partition_samples_kernel(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    TreeNodeGPU* __restrict__ parent_node,
    TreeNodeGPU* __restrict__ left_child,
    TreeNodeGPU* __restrict__ right_child,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ best_idx,
    int* __restrict__ tree_counters,
    const int global_n_samples
);

/**
 * @brief CUDA kernel to compute L2 node score
 * 
 * @param node Tree node
 * @param mean Node mean values
 */
__global__ void node_l2_kernel(
    TreeNodeGPU* __restrict__ node,
    const float* __restrict__ mean
);

/**
 * @brief CUDA kernel to compute cosine node score
 * 
 * @param node Tree node
 * @param grads Gradient values
 * @param mean Output mean values
 */
__global__ void node_cosine_kernel(
    TreeNodeGPU* __restrict__ node,
    const float* __restrict__ grads,
    float* __restrict__ mean
);

/**
 * @brief CUDA kernel to update child node metadata
 * 
 * @param parent_node Parent tree node
 * @param left_child Left child node
 * @param right_child Right child node
 * @param tree_counters Partition counters
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_numeric Feature type indicators
 * @param candidate_categories Candidate categorical values
 * @param best_idx Best split index
 * @param best_score Best split score
 */
__global__ void update_child_nodes_kernel(
    const TreeNodeGPU* __restrict__ parent_node,
    TreeNodeGPU* __restrict__ left_child,
    TreeNodeGPU* __restrict__ right_child,
    int* __restrict__ tree_counters,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const bool* __restrict__ candidate_numeric,
    const char* __restrict__ candidate_categories,
    const int* __restrict__ best_idx,
    const float* __restrict__ best_score
);

/**
 * @brief CUDA kernel for column-wise reduction
 * 
 * @param in Input matrix
 * @param out Output column sums
 * @param n_cols Number of columns
 * @param n_rows Number of rows
 */
__global__ void column_sums_reduce(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t n_cols,
    size_t n_rows
);

/**
 * @brief CUDA kernel for node-specific column mean reduction
 * 
 * @param in Input matrix
 * @param out Output column means
 * @param n_cols Number of columns
 * @param node Tree node defining sample subset
 */
__global__ void node_column_mean_reduce(
    const float * __restrict__ in,
    float * __restrict__ out,
    size_t n_cols,
    const TreeNodeGPU* __restrict__ node
);

/**
 * @brief CUDA kernel to copy node to ensemble data
 * 
 * @param node Source tree node
 * @param depths Output depth array
 * @param feature_indices Output feature index array
 * @param feature_values Output threshold array
 * @param edge_weights Output edge weight array
 * @param inequality_directions Output direction array
 * @param is_numerics Output feature type array
 * @param categorical_values Output categorical value array
 * @param global_idx Global tree index
 * @param leaf_idx Leaf index
 * @param max_depth Maximum tree depth
 */
__global__ void copy_node_to_data(
    const TreeNodeGPU* __restrict__ node,
    int* __restrict__ depths,
    int* __restrict__ feature_indices,
    float* __restrict__ feature_values,
    float* __restrict__ edge_weights,
    bool* __restrict__ inequality_directions,
    bool* __restrict__ is_numerics,
    char * __restrict__  categorical_values,
    const int global_idx,
    const int leaf_idx,
    const int max_depth
);

/**
 * @brief CUDA kernel to print tree node (debug)
 * 
 * @param node Tree node to print
 */
__global__ void print_tree_node(
    const TreeNodeGPU* __restrict__ node
);

/**
 * @brief CUDA kernel to print tree indices (debug)
 * 
 * @param tree_indices Tree index array
 * @param size Array size
 */
__global__ void print_tree_indices_kernel(
    const int* __restrict__ tree_indices,
    int size
);

/**
 * @brief CUDA kernel to print vector (debug)
 * 
 * @param vec Vector to print
 * @param size Vector size
 */
__global__ void print_vector_kernel(
    const float* __restrict__ vec,
    const int size
);

/**
 * @brief CUDA device function for string comparison
 * 
 * @param str_a First string
 * @param str_b Second string
 * @return 0 if equal, non-zero otherwise
 */
__device__ int strcmpCuda(
    const char* __restrict__ str_a,
    const char* __restrict__ str_b
);

/**
 * @brief CUDA kernel to print candidate scores (debug)
 * 
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param split_scores Split scores
 * @param n_candidates Number of candidates
 */
__global__ void print_candidate_scores(
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    float* __restrict__ split_scores,
    const int n_candidates
);

/**
 * @brief CUDA kernel for conditional sum computation
 * 
 * Computes sums and counts for left/right child nodes.
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param grads Gradient values
 * @param node Current tree node
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param n_candidates Number of candidates
 * @param global_n_samples Total number of samples
 * @param left_sum Output left child sums
 * @param right_sum Output right child sums
 * @param left_count Output left child counts
 * @param right_count Output right child counts
 */
__global__ void split_conditional_sum_kernel(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int n_candidates,
    const int global_n_samples,
    float* __restrict__ left_sum,
    float* __restrict__ right_sum,
    float* __restrict__ left_count,
    float* __restrict__ right_count
);

/**
 * @brief CUDA kernel for conditional dot product computation
 * 
 * Computes sums, counts, and dot products for cosine scoring.
 * 
 * @param obs Numerical observations
 * @param categorical_obs Categorical observations
 * @param grads Gradient values
 * @param node Current tree node
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param n_candidates Number of candidates
 * @param global_n_samples Total number of samples
 * @param left_sum Output left child sums
 * @param right_sum Output right child sums
 * @param left_count Output left child counts
 * @param right_count Output right child counts
 * @param ldot Output left child dot products
 * @param rdot Output right child dot products
 */
__global__ void split_conditional_dot_kernel(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values, 
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int n_candidates,
    const int global_n_samples,
    float* __restrict__ left_sum,
    float* __restrict__ right_sum,
    float* __restrict__ left_count,
    float* __restrict__ right_count,
    float* __restrict__ ldot,
    float* __restrict__ rdot
);

/**
 * @brief CUDA kernel to compute cosine split scores
 * 
 * @param node Current tree node
 * @param feature_weights Per-feature importance weights
 * @param split_scores Output split scores
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param n_candidates Number of candidates
 * @param lsum Left child sums
 * @param rsum Right child sums
 * @param lcount Left child counts
 * @param rcount Right child counts
 * @param ldot Left child dot products
 * @param rdot Right child dot products
 * @param min_data_in_leaf Minimum samples per leaf
 * @param n_num_features Number of numerical features
 */
__global__ void split_cosine_score_kernel(
    const TreeNodeGPU* __restrict__ node,
    const float* __restrict__ feature_weights,
    float* __restrict__ split_scores,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int n_candidates,
    float* __restrict__ lsum,
    float* __restrict__ rsum,
    float* __restrict__ lcount,
    float* __restrict__ rcount,
    float* __restrict__ ldot,
    float* __restrict__ rdot,
    const int min_data_in_leaf,
    const int n_num_features
);

/**
 * @brief CUDA kernel to compute L2 split scores
 * 
 * @param node Current tree node
 * @param feature_weights Per-feature importance weights
 * @param split_scores Output split scores
 * @param candidate_indices Candidate feature indices
 * @param candidate_values Candidate threshold values
 * @param candidate_categories Candidate categorical values
 * @param candidate_numeric Feature type indicators
 * @param n_candidates Number of candidates
 * @param lsum Left child sums
 * @param rsum Right child sums
 * @param lcount Left child counts
 * @param rcount Right child counts
 * @param min_data_in_leaf Minimum samples per leaf
 * @param n_num_features Number of numerical features
 */
__global__ void split_l2_score_kernel(
    const TreeNodeGPU* __restrict__ node,
    const float* __restrict__ feature_weights,
    float* __restrict__ split_scores,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int n_candidates,
    float* __restrict__ lsum,
    float* __restrict__ rsum,
    float* __restrict__ lcount,
    float* __restrict__ rcount,
    const int min_data_in_leaf,
    const int n_num_features
);

#endif

#ifdef __cplusplus
}
#endif

#endif 