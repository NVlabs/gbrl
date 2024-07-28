//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <cstring>
#include <omp.h>
#include <iostream>

#include "compression.h"
#include "types.h"
#include "utils.h"
#include "math_ops.h"


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
    void (*getRepresentationFunc)(const float*, const char*, const int, const ensembleData*, const ensembleMetaData*, const int, const int, std::vector<Optimizer*>, matrixRepresentation*) = nullptr;
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
                getRepresentationFunc(dataset->obs, dataset->categorical_obs, sample_idx, edata, metadata, thread_start_tree_idx, thread_stop_tree_idx, opts, matrix);
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
                getRepresentationFunc(dataset->obs, dataset->categorical_obs, sample_idx, edata, metadata, 0, metadata->n_trees, opts, matrix);
            }
        }
    // no parallelization
    } else{ 
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx){
            getRepresentationFunc(dataset->obs, dataset->categorical_obs, sample_idx, edata, metadata, 0, metadata->n_trees, opts, matrix);
        }
    }
    matrix->n_leaves_per_tree = new int[metadata->n_trees];
    for (int i = 0; i < metadata->n_trees - 1; ++i )
        matrix->n_leaves_per_tree[i] = edata->tree_indices[i+1] - edata->tree_indices[i];
    matrix->n_leaves_per_tree[metadata->n_trees - 1] = metadata->n_leaves - edata->tree_indices[metadata->n_trees - 1];
    matrix->n_trees = metadata->n_trees;
}

void Compressor::get_representation_matrix_over_leaves(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, std::vector<Optimizer*> opts, matrixRepresentation *matrix){
    /* Return A - a binary matrix mapping inputs to leaves for non-oblivious trees
    */
    int tree_idx = start_tree_idx;
    int max_depth = metadata->max_depth;
    bool passed;
    int obs_row = sample_idx*metadata->n_num_features;
    int categorical_obs_row = sample_idx*metadata->n_cat_features;

    const bool *numerics = edata->is_numerics;
    const float *feature_values = edata->feature_values;
    const float *values = edata->values;
    const int* feature_indices = edata->feature_indices;
    const int* tree_indices = edata->tree_indices;
    const bool* inequality_directions = edata->inequality_directions;
    const char* categorical_values = edata->categorical_values;
    int leaf_idx = tree_indices[tree_idx];
    int base_leaf_idx = leaf_idx;

    while (leaf_idx < metadata->n_leaves && tree_idx < stop_tree_idx)
    {
        int depth = edata->depths[leaf_idx];
        passed = false;
        int cond_idx = leaf_idx*max_depth;
        for (int depth_idx = depth - 1; depth_idx >= 0; --depth_idx){
            passed = (numerics[cond_idx + depth_idx]) ? (obs[obs_row + feature_indices[cond_idx + depth_idx]] >  feature_values[cond_idx + depth_idx]) == inequality_directions[cond_idx + depth_idx] : (strcmp(&categorical_obs[(categorical_obs_row + feature_indices[cond_idx + depth_idx]) * MAX_CHAR_SIZE],  categorical_values + (cond_idx + depth_idx)*MAX_CHAR_SIZE) == 0) ==  inequality_directions[cond_idx + depth_idx];
            if (!passed)
                break;
        }
        if (passed){
            matrix->A[sample_idx*(metadata->n_leaves + 1) + leaf_idx + 1 - base_leaf_idx] = true;
            for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                opts[opt_idx]->copy_and_scale(matrix->V + (leaf_idx + 1)*metadata->output_dim, values + leaf_idx*metadata->output_dim, tree_idx);
            }
            ++tree_idx;
            if (tree_idx < stop_tree_idx)
                leaf_idx = tree_indices[tree_idx];
        } else {
            ++leaf_idx;
        }
    }
}

void Compressor::get_representation_matrix_over_trees(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, std::vector<Optimizer*> opts, matrixRepresentation *matrix){
    /* Return A - a binary matrix mapping inputs to leaves for oblivious trees
    */
    int tree_idx = start_tree_idx;
    int max_depth = metadata->max_depth;
    bool passed;
    int obs_row = sample_idx*metadata->n_num_features;
    int categorical_obs_row = sample_idx*metadata->n_cat_features;

    const bool *numerics = edata->is_numerics;
    const int *depths = edata->depths;
    const float *feature_values = edata->feature_values;
    const float *values = edata->values;
    const int* feature_indices = edata->feature_indices;
    const int* tree_indices = edata->tree_indices;
    const char* categorical_values = edata->categorical_values;
    const int offset_leaf_idx = tree_indices[tree_idx];
    

    while (tree_idx < stop_tree_idx)
    {
        int initial_leaf_idx = tree_indices[tree_idx];
        int cond_idx = tree_idx*max_depth;
        int leaf_idx = 0;
        for (int depth_idx = 0; depth_idx < depths[tree_idx]; depth_idx++){ 
            passed = (numerics[cond_idx + depth_idx]) ? (obs[obs_row + feature_indices[cond_idx + depth_idx]] >  feature_values[cond_idx + depth_idx]) : strcmp(&categorical_obs[(categorical_obs_row + feature_indices[cond_idx + depth_idx]) * MAX_CHAR_SIZE],  categorical_values + (cond_idx + depth_idx)*MAX_CHAR_SIZE) == 0;
            leaf_idx |= (passed <<  (depths[tree_idx] - 1 - depth_idx));
        }
        matrix->A[sample_idx*(metadata->n_leaves + 1) + initial_leaf_idx + leaf_idx + 1 - offset_leaf_idx] = true;
        int value_idx = (initial_leaf_idx + leaf_idx)*metadata->output_dim;
        for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
            opts[opt_idx]->copy_and_scale(matrix->V + (initial_leaf_idx + leaf_idx + 1 - offset_leaf_idx) *metadata->output_dim, values + value_idx, tree_idx);
        }
    
        ++tree_idx;
    }
}


void Compressor::add_W_matrix_to_values(const float *W, const ensembleData *edata, const ensembleMetaData *metadata, std::vector<Optimizer*> opts){
    int size = metadata->n_leaves;
    int n_threads = calculate_num_threads(size, metadata->par_th);
     if (n_threads > 1){
        int elements_per_thread = (size) / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int leaf_idx = start_idx; leaf_idx < end_idx; ++leaf_idx){
                for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                    opts[opt_idx]->add_scaled(edata->values + leaf_idx*metadata->output_dim, W + (leaf_idx + 1) * metadata->output_dim, 0);
                }
            }
                
        }
     } else {
        for (int leaf_idx = 0; leaf_idx < size; ++leaf_idx){
            for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                opts[opt_idx]->add_scaled(edata->values + leaf_idx*metadata->output_dim, W + (leaf_idx + 1) * metadata->output_dim, 0);
            }
        }     
    }
}

ensembleData* Compressor::compress_ensemble(ensembleMetaData *metadata, ensembleData *edata, std::vector<Optimizer*> opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W){
    Compressor::add_W_matrix_to_values(W, edata, metadata, opts);
    ensembleData* compressed_edata = copy_compressed_ensemble_data(edata, metadata, leaf_indices, tree_indices, n_compressed_leaves, n_compressed_trees, new_tree_indices);
    ensemble_data_dealloc(edata);
    return compressed_edata;
}