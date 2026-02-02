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
 * @file compression.cpp
 * @brief Implementation of tree ensemble compression algorithms
 * 
 * @warning EXPERIMENTAL - UNDER ACTIVE RESEARCH
 * This module implements tree ensemble compression algorithms that are currently
 * under active research and development. The API, behavior, and results may change
 * without notice. This feature has not been fully validated and is intended for
 * internal research use only.
 * 
 * Provides CPU-based compression for gradient boosted tree ensembles,
 * including matrix representation generation and tree selection for model
 * compression while maintaining prediction accuracy.
 */

#include <cstring>
#include <omp.h>
#include <iostream>

#include "compression.h"
#include "types.h"
#include "utils.h"
#include "math_ops.h"

/**
 * @brief Generate matrix representation of tree ensemble on CPU
 * 
 * Converts the tree ensemble into a matrix form (A, V) where A is a binary
 * matrix mapping samples to leaf nodes, and V contains the leaf values scaled
 * by optimizer learning rates. Supports both parallel and sequential execution.
 * 
 * @param dataset Input dataset containing observations
 * @param edata Ensemble data (tree structure, leaf values, etc.)
 * @param metadata Ensemble metadata (dimensions, hyperparameters)
 * @param parallel_predict Whether to use parallel execution
 * @param matrix Output matrix representation structure
 * @param opts Vector of optimizers for scaling leaf values
 */
void Compressor::get_matrix_representation_cpu(dataSet *dataset, const ensembleData *edata, const ensembleMetaData *metadata, const bool parallel_predict, matrixRepresentation *matrix, std::vector<Optimizer*> opts){
    const int par_th = metadata->par_th, n_samples = dataset->n_samples;
    // first column is all ones for including bias
    bool *A = new bool[n_samples*(metadata->n_leaves + 1)];
    memset(A, 0, n_samples*(metadata->n_leaves + 1));
    for (int i = 0; i < n_samples; i++)
        A[i*(metadata->n_leaves + 1)] = true;
    matrix->A = A;
    matrix->V = new float[metadata->output_dim*(metadata->n_leaves + 1)];
    memcpy(matrix->V, edata->bias, sizeof(float)*metadata->output_dim);
    matrix->n_leaves = metadata->n_leaves;
    void (*getRepresentationFunc)(const float*, const char*, const int, const ensembleData*, const ensembleMetaData*, const int, const int, matrixRepresentation*) = nullptr;
    getRepresentationFunc = (metadata->grow_policy == OBLIVIOUS) ? &Compressor::get_representation_matrix_over_trees : &Compressor::get_representation_matrix_over_leaves;
    int n_tree_threads = calculate_num_threads(metadata->n_trees, par_th);
    int n_sample_threads = calculate_num_threads(n_samples, par_th);
    // parallellize over trees
    if (n_tree_threads > 1 && parallel_predict && n_tree_threads > n_sample_threads){
        
        int trees_per_thread = metadata->n_trees / n_tree_threads;
        omp_set_num_threads(n_tree_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int thread_start_tree_idx = thread_id * trees_per_thread;
            int thread_stop_tree_idx = (thread_id == n_tree_threads - 1) ? metadata->n_trees : thread_start_tree_idx + trees_per_thread;
            for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx){
                getRepresentationFunc(dataset->obs->data, dataset->categorical_obs->data, sample_idx, edata, metadata, thread_start_tree_idx, thread_stop_tree_idx, matrix);
            }        
        }

    // sample parallelization
    } else if (n_sample_threads > 1) {
        int samples_per_thread = n_samples / n_sample_threads;
        omp_set_num_threads(n_sample_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * samples_per_thread;
            int end_idx = (thread_id == n_sample_threads - 1) ? n_samples : start_idx + samples_per_thread;
            for (int sample_idx = start_idx; sample_idx < end_idx; ++sample_idx) {
                getRepresentationFunc(dataset->obs->data, dataset->categorical_obs->data, sample_idx, edata, metadata, 0, metadata->n_trees, matrix);
            }
        }
    // no parallelization
    } else{ 
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx){
            getRepresentationFunc(dataset->obs->data, dataset->categorical_obs->data, sample_idx, edata, metadata, 0, metadata->n_trees, matrix);
        }
    }
    get_V(matrix, edata, metadata, opts);
    matrix->n_leaves_per_tree = new int[metadata->n_trees];
    for (int i = 0; i < metadata->n_trees - 1; ++i )
        matrix->n_leaves_per_tree[i] = edata->ensemble_info->tree_indices[i+1] - edata->ensemble_info->tree_indices[i];
    matrix->n_leaves_per_tree[metadata->n_trees - 1] = metadata->n_leaves - edata->ensemble_info->tree_indices[metadata->n_trees - 1];
    matrix->n_trees = metadata->n_trees;
}

/**
 * @brief Generate matrix representation for greedy (non-oblivious) trees
 * 
 * For each tree, traverses from root to leaf following split conditions
 * to determine which leaf each sample reaches. Sets corresponding entry
 * in matrix A to true.
 * 
 * @param obs Numerical observations (n_samples x n_num_features)
 * @param categorical_obs Categorical observations (n_samples x n_cat_features)
 * @param sample_idx Index of sample being processed
 * @param edata Ensemble data structure
 * @param metadata Ensemble metadata
 * @param start_tree_idx First tree to process (inclusive)
 * @param stop_tree_idx Last tree to process (exclusive)
 * @param matrix Output matrix representation
 */
void Compressor::get_representation_matrix_over_leaves(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, matrixRepresentation *matrix){
    const int max_depth = metadata->max_depth;
    const int n_num_features = metadata->n_num_features;
    const int n_cat_features = metadata->n_cat_features;
    const int n_leaves = metadata->n_leaves;
    const int n_leaves_plus_one = n_leaves + 1;
    
    const int obs_row = sample_idx * n_num_features;
    const int categorical_obs_row = sample_idx * n_cat_features;

    const bool *numerics = edata->feature_data->is_numerics;
    const float *feature_values = edata->feature_data->feature_values;
    const int* feature_indices = edata->feature_data->feature_indices;
    const int* tree_indices = edata->ensemble_info->tree_indices;
    const bool* inequality_directions = edata->feature_data->inequality_directions;
    const char* categorical_values = edata->feature_data->categorical_values;
    
    int tree_idx = start_tree_idx;
    int leaf_idx = tree_indices[tree_idx];
    const int base_leaf_idx = leaf_idx;
    bool *A_row = matrix->A + sample_idx * n_leaves_plus_one;

    while (leaf_idx < n_leaves && tree_idx < stop_tree_idx)
    {
        const int depth = edata->ensemble_info->depths[leaf_idx];
        const int cond_idx = leaf_idx * max_depth;
        bool passed = false;
        
        // Check conditions from deepest to shallowest
        for (int depth_idx = depth - 1; depth_idx >= 0; --depth_idx){
            const int cond_offset = cond_idx + depth_idx;
            if (numerics[cond_offset]) {
                passed = (obs[obs_row + feature_indices[cond_offset]] > feature_values[cond_offset]) == inequality_directions[cond_offset];
            } else {
                const int cat_offset = (categorical_obs_row + feature_indices[cond_offset]) * MAX_CHAR_SIZE;
                const int cond_cat_offset = cond_offset * MAX_CHAR_SIZE;
                passed = (strcmp(&categorical_obs[cat_offset], &categorical_values[cond_cat_offset]) == 0) == inequality_directions[cond_offset];
            }
            if (!passed)
                break;
        }
        
        if (passed){
            A_row[leaf_idx + 1 - base_leaf_idx] = true;
            ++tree_idx;
            if (tree_idx < stop_tree_idx)
                leaf_idx = tree_indices[tree_idx];
        } else {
            ++leaf_idx;
        }
    }
}

/**
 * @brief Generate matrix representation for oblivious (symmetric) trees
 * 
 * For oblivious trees, all leaves at the same depth use the same split condition.
 * This allows computing the leaf index directly from a binary path representation
 * rather than traversing each node individually.
 * 
 * @param obs Numerical observations (n_samples x n_num_features)
 * @param categorical_obs Categorical observations (n_samples x n_cat_features)
 * @param sample_idx Index of sample being processed
 * @param edata Ensemble data structure
 * @param metadata Ensemble metadata
 * @param start_tree_idx First tree to process (inclusive)
 * @param stop_tree_idx Last tree to process (exclusive)
 * @param matrix Output matrix representation
 */
void Compressor::get_representation_matrix_over_trees(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, matrixRepresentation *matrix){
    const int max_depth = metadata->max_depth;
    const int n_num_features = metadata->n_num_features;
    const int n_cat_features = metadata->n_cat_features;
    const int n_leaves_plus_one = metadata->n_leaves + 1;
    
    const int obs_row = sample_idx * n_num_features;
    const int categorical_obs_row = sample_idx * n_cat_features;

    const bool *numerics = edata->feature_data->is_numerics;
    const int *depths = edata->ensemble_info->depths;
    const float *feature_values = edata->feature_data->feature_values;
    const int* feature_indices = edata->feature_data->feature_indices;
    const int* tree_indices = edata->ensemble_info->tree_indices;
    const char* categorical_values = edata->feature_data->categorical_values;
    const int offset_leaf_idx = tree_indices[start_tree_idx];
    
    bool *A_row = matrix->A + sample_idx * n_leaves_plus_one;

    for (int tree_idx = start_tree_idx; tree_idx < stop_tree_idx; ++tree_idx)
    {
        const int initial_leaf_idx = tree_indices[tree_idx];
        const int tree_depth = depths[tree_idx];
        const int cond_idx = tree_idx * max_depth;
        int leaf_idx = 0;
        
        // Build binary path through tree
        for (int depth_idx = 0; depth_idx < tree_depth; ++depth_idx){ 
            const int cond_offset = cond_idx + depth_idx;
            bool passed;
            
            if (numerics[cond_offset]) {
                passed = obs[obs_row + feature_indices[cond_offset]] > feature_values[cond_offset];
            } else {
                const int cat_offset = (categorical_obs_row + feature_indices[cond_offset]) * MAX_CHAR_SIZE;
                const int cond_cat_offset = cond_offset * MAX_CHAR_SIZE;
                passed = strcmp(&categorical_obs[cat_offset], &categorical_values[cond_cat_offset]) == 0;
            }
            
            // Build leaf index from binary path
            leaf_idx |= (passed << (tree_depth - 1 - depth_idx));
        }
        A_row[initial_leaf_idx + leaf_idx + 1 - offset_leaf_idx] = true;
    }
}

/**
 * @brief Apply correction matrix W to leaf values during compression
 * 
 * Adds the correction matrix values to leaf values (after inverse scaling
 * by optimizer learning rate) to maintain prediction accuracy after compression.
 * Supports parallel execution for efficiency.
 * 
 * @param W Correction matrix (n_leaves+1 x output_dim)
 * @param edata Ensemble data containing leaf values to modify
 * @param metadata Ensemble metadata
 * @param opts Vector of optimizers for inverse scaling
 */
void Compressor::add_W_matrix_to_values(const float *W, const ensembleData *edata, const ensembleMetaData *metadata, std::vector<Optimizer*> opts){
    const int size = metadata->n_leaves;
    const int output_dim = metadata->output_dim;
    const int n_threads = calculate_num_threads(size, metadata->par_th);
    
    if (n_threads > 1){
        const int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            const int start_idx = thread_id * elements_per_thread;
            const int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int leaf_idx = start_idx; leaf_idx < end_idx; ++leaf_idx){
                float *leaf_values = edata->leaf_data->values + leaf_idx * output_dim;
                const float *W_values = W + (leaf_idx + 1) * output_dim;
                for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                    opts[opt_idx]->add_scaled(leaf_values, W_values, 0);
                }
            }     
        }
    } else {
        for (int leaf_idx = 0; leaf_idx < size; ++leaf_idx){
            float *leaf_values = edata->leaf_data->values + leaf_idx * output_dim;
            const float *W_values = W + (leaf_idx + 1) * output_dim;
            for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                opts[opt_idx]->add_scaled(leaf_values, W_values, 0);
            }
        }     
    }
}

/**
 * @brief Extract and scale leaf values into matrix V
 * 
 * Copies leaf values from ensemble data and scales them by negative learning rate
 * (as per boosting convention) to populate the V matrix. Supports parallel execution.
 * 
 * @param matrix Matrix representation to populate (modifies matrix->V)
 * @param edata Ensemble data containing leaf values
 * @param metadata Ensemble metadata
 * @param opts Vector of optimizers for scaling
 */
void Compressor::get_V(matrixRepresentation *matrix, const ensembleData *edata, const ensembleMetaData *metadata, std::vector<Optimizer*> opts){
    const int size = metadata->n_leaves;
    const int output_dim = metadata->output_dim;
    const int n_threads = calculate_num_threads(size, metadata->par_th);
    
    if (n_threads > 1){
        const int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            const int start_idx = thread_id * elements_per_thread;
            const int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int leaf_idx = start_idx; leaf_idx < end_idx; ++leaf_idx){
                float *V_dest = matrix->V + (leaf_idx + 1) * output_dim;
                const float *leaf_values = edata->leaf_data->values + leaf_idx * output_dim;
                for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                    opts[opt_idx]->copy_and_scale(V_dest, leaf_values, 0);
                }
            }
        }
    } else {
        for (int leaf_idx = 0; leaf_idx < size; ++leaf_idx){
            float *V_dest = matrix->V + (leaf_idx + 1) * output_dim;
            const float *leaf_values = edata->leaf_data->values + leaf_idx * output_dim;
            for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                opts[opt_idx]->copy_and_scale(V_dest, leaf_values, 0);
            }
        }     
    }
}

/**
 * @brief Compress tree ensemble by selecting subset and applying correction matrix
 * 
 * Performs the final compression step: creates a new ensemble with only the selected
 * leaves and trees, then applies correction matrix W to the compressed leaf values
 * to maintain prediction accuracy. The original ensemble data is deallocated after
 * compression.
 * 
 * @param metadata Ensemble metadata (modified to reflect compressed dimensions)
 * @param edata Original ensemble data (will be deallocated)
 * @param opts Vector of optimizers
 * @param n_compressed_leaves Number of leaves in compressed ensemble
 * @param n_compressed_trees Number of trees in compressed ensemble
 * @param leaf_indices Indices of leaves to retain
 * @param tree_indices Indices of trees to retain
 * @param new_tree_indices New starting indices for each tree in compressed ensemble
 * @param W Correction matrix to apply after compression (shape: n_compressed_leaves+1 x output_dim)
 * @return Pointer to new compressed ensemble data
 */
ensembleData* Compressor::compress_ensemble(ensembleMetaData *metadata, ensembleData *edata, std::vector<Optimizer*> opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W){
    // First copy selected leaves/trees to compressed ensemble (this also updates metadata)
    ensembleData* compressed_edata = copy_compressed_ensemble_data(edata, metadata, leaf_indices, tree_indices, n_compressed_leaves, n_compressed_trees, new_tree_indices);
    // Deallocate original ensemble
    ensemble_data_dealloc(edata);
    // Now apply W correction to the compressed ensemble (metadata->n_leaves is now n_compressed_leaves)
    Compressor::add_W_matrix_to_values(W, compressed_edata, metadata, opts);
    return compressed_edata;
}