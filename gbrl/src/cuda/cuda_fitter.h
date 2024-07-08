//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_FITTER_H
#define CUDA_FITTER_H

#include "types.h"
#include "cuda_types.h"
#include "node.h"

#ifdef __cplusplus
extern "C" {
#endif
TreeNodeGPU* allocate_root_tree_node(dataSet *dataset, ensembleMetaData *metadata);
void allocate_child_tree_node(TreeNodeGPU* host_parent, TreeNodeGPU** device_child);
void allocate_child_tree_nodes(dataSet *dataset, TreeNodeGPU* parent_node, TreeNodeGPU* host_parent, TreeNodeGPU** left_child, TreeNodeGPU** right_child, candidatesData *candidata, splitDataGPU *split_data);
void free_tree_node(TreeNodeGPU* node);
void evaluate_greedy_splits(dataSet *dataset, const TreeNodeGPU *node, candidatesData *candidata, ensembleMetaData *metadata, splitDataGPU* split_data, const int threads_per_block, const int parent_n_samples);
void evaluate_oblivious_splits_cuda(dataSet *dataset, TreeNodeGPU ** nodes, const int depth, candidatesData *candidata, ensembleMetaData *metadata, splitDataGPU *split_data);
void calc_parallelism(const int n_candidates, const int output_dim, int &threads_per_block, const scoreFunc split_score_fun);
void calc_oblivious_parallelism(const int n_candidates, const int output_dim, int &threads_per_block, const scoreFunc split_score_func, const int depth);
void fit_tree_oblivious_cuda(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, candidatesData *candidata, splitDataGPU *split_data);
void fit_tree_greedy_cuda(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, candidatesData *candidata, splitDataGPU *split_data);

void add_leaf_node(const TreeNodeGPU *node, const int depth, ensembleMetaData *metadata, ensembleData *edata, dataSet *dataset);
#ifdef __CUDACC__  // This macro is defined by NVCC
#ifdef DEBUG
__global__ void average_leaf_value_kernel(float* __restrict__ values, const int output_dim, int* __restrict__ n_samples, const int global_idx, const int leaf_idx, float *count);
#else
__global__ void average_leaf_value_kernel(float* __restrict__ values, const int output_dim, const int global_idx, float *count);
#endif
__global__ void split_score_cosine_cuda(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const float* __restrict__ grads, const float* __restrict__ grads_norm, const float* __restrict__ feature_weights, const TreeNodeGPU* __restrict__ node, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values, const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, const int min_data_in_leaf, float* __restrict__ split_scores, const int global_n_samples, const int n_num_features);
__global__ void split_score_l2_cuda(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const float* __restrict__ grads, const float* __restrict__ feature_weights, const TreeNodeGPU* __restrict__ node, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values,  const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, const int min_data_in_leaf, float* __restrict__ split_scores, const int global_n_samples, const int n_num_features);
__global__ void update_best_candidate_cuda(float *split_scores, int n_candidates, int *best_idx, float *best_score, const TreeNodeGPU* __restrict__ node);
__global__ void reduce_leaf_sum(const float *obs, const char *categorical_obs, const float *grads, float* __restrict__ values, const TreeNodeGPU *node, const int n_samples, const int global_idx, float *count_f);
__global__ void partition_samples_kernel(const float* __restrict__ obs, const char* __restrict__ categorical_obs, TreeNodeGPU* __restrict__ parent_node, TreeNodeGPU* __restrict__ left_child, TreeNodeGPU* __restrict__ right_child, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values, const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, const int* __restrict__ best_idx, int* __restrict__ tree_counters, const int global_n_samples);
__global__ void node_cosine_kernel(TreeNodeGPU* node, const float *grads, const float *grads_norm, float *mean);
__global__ void update_child_nodes_kernel(const TreeNodeGPU* __restrict__ parent_node, TreeNodeGPU* __restrict__ left_child, TreeNodeGPU* __restrict__ right_child, int* __restrict__ tree_counters, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values, const bool* __restrict__ candidate_numeric, const char* __restrict__ candidate_categories, const int* __restrict__ best_idx, const float* __restrict__ best_score);
__global__ void column_sums_reduce(const float * __restrict__ in, float * __restrict__ out, size_t n_cols, size_t n_rows);
__global__ void node_column_mean_reduce(const float * __restrict__ in, float * __restrict__ out, size_t n_cols, const TreeNodeGPU* __restrict__ node);
__global__ void copy_node_to_data(const TreeNodeGPU* __restrict__ node, int* __restrict__ depths, int* __restrict__ feature_indices, float* __restrict__ feature_values, float* __restrict__ edge_weights, bool* __restrict__ inequality_directions, bool* __restrict__ is_numerics, char * __restrict__  categorical_values, const int global_idx, const int leaf_idx, const int max_depth);
__global__ void print_tree_node(const TreeNodeGPU *node);
__global__ void print_tree_indices_kernel(int *tree_indices, int size);
__global__ void print_vector_kernel(const float *vec, const int size);
__device__ int strcmpCuda(const char *str_a, const char *str_b);
__global__ void print_candidate_scores(const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values,  const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, float* __restrict__ split_scores, const int n_candidates);


__global__ void split_conditional_sum_kernel(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const float* __restrict__ grads, const TreeNodeGPU* __restrict__ node, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values,  const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, const int n_candidates, const int global_n_samples, float* __restrict__ left_sum, float* __restrict__ right_sum, float* __restrict__ left_count, float* __restrict__ right_count);
__global__ void split_contidional_dot_kernel(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const float* __restrict__ grads, const float* __restrict__ grad_norms, const TreeNodeGPU* __restrict__ node, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values,  const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, const int n_candidates, const int global_n_samples, float* __restrict__ left_sum, float* __restrict__ right_sum, float* __restrict__ left_count, float* __restrict__ right_count, float* __restrict__ ldot, float* __restrict__ rdot, float* __restrict__ lnorms, float* __restrict__ rnorms);
__global__ void split_cosine_score_kernel(const TreeNodeGPU* __restrict__ node, const float* __restrict__ feature_weights, float* __restrict__ split_scores, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values,  const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, const int n_candidates, float* __restrict__ lsum, float* __restrict__ rsum, float* __restrict__ lcount, float* __restrict__ rcount, float* __restrict__ ldot, float* __restrict__ rdot, float* __restrict__ lnorms, float* __restrict__ rnorms, const int min_data_in_leaf, const int n_num_features);
__global__ void split_l2_score_kernel(const TreeNodeGPU* __restrict__ node, const float* __restrict__ feature_weights, float* __restrict__ split_scores, const int* __restrict__ candidate_indices, const float* __restrict__ candidate_values,  const char* __restrict__ candidate_categories, const bool* __restrict__ candidate_numeric, const int n_candidates, float* __restrict__ lsum, float* __restrict__ rsum, float* __restrict__ lcount, float* __restrict__ rcount, const int min_data_in_leaf, const int n_num_features);
#endif 

#ifdef __cplusplus
}
#endif

#endif 