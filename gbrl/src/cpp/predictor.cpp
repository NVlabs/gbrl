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

#include "predictor.h"
#include "types.h"
#include "utils.h"
#include "math_ops.h"


void Predictor::momentum_over_leaves(const float *obs, const char *categorical_obs, float *momentum, const ensembleData *edata, const ensembleMetaData *metadata, int start_tree_idx, const int stop_tree_idx, const int sample_idx){
    int tree_idx = start_tree_idx;
    int n_leaves = metadata->n_leaves, max_depth = metadata->max_depth;
    bool passed;
    int output_dim = metadata->output_dim;
    int obs_row = sample_idx*metadata->n_num_features;
    int categorical_obs_row = sample_idx*metadata->n_cat_features;
    const float cv_beta = metadata->cv_beta, cv_1_m_beta = 1.0f - metadata->cv_beta;
    
    const bool *numerics = edata->is_numerics;
    const float *feature_values = edata->feature_values;
    const float *values = edata->values;
    const int* feature_indices = edata->feature_indices;
    const int* tree_indices = edata->tree_indices;
    const bool* inequality_directions = edata->inequality_directions;
    const char* categorical_values = edata->categorical_values;

    int leaf_idx = tree_indices[tree_idx];
    while (leaf_idx < n_leaves && tree_idx < stop_tree_idx)
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
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < output_dim; ++d){
                momentum[sample_idx + d] *= cv_beta;
                momentum[sample_idx + d] += cv_1_m_beta * values[leaf_idx * output_dim + d];
            }

            ++tree_idx;
            if (tree_idx < stop_tree_idx)
                leaf_idx = tree_indices[tree_idx];
        } else {
            ++leaf_idx;
        }
    }
}

void Predictor::momentum_over_trees(const float *obs, const char *categorical_obs, float *momentum, const ensembleData *edata, const ensembleMetaData *metadata, int start_tree_idx, const int stop_tree_idx, const int sample_idx){
    int tree_idx = start_tree_idx;
    int max_depth = metadata->max_depth;
    bool passed;
    int obs_row = sample_idx*metadata->n_num_features;
    int categorical_obs_row = sample_idx*metadata->n_cat_features;
    const float cv_beta = metadata->cv_beta, cv_1_m_beta = 1.0f - metadata->cv_beta;

    const bool *numerics = edata->is_numerics;
    const int *depths = edata->depths;
    const float *feature_values = edata->feature_values;
    const float *values = edata->values;
    const int* feature_indices = edata->feature_indices;
    const int* tree_indices = edata->tree_indices;
    const char* categorical_values = edata->categorical_values;
    

    while (tree_idx < stop_tree_idx)
    {
        int initial_leaf_idx = tree_indices[tree_idx];
        int cond_idx = tree_idx*max_depth;
        int leaf_idx = 0;
        for (int depth_idx = 0; depth_idx < depths[tree_idx]; depth_idx++){ 
            passed = (numerics[cond_idx + depth_idx]) ? (obs[obs_row + feature_indices[cond_idx + depth_idx]] >  feature_values[cond_idx + depth_idx]) : strcmp(&categorical_obs[(categorical_obs_row + feature_indices[cond_idx + depth_idx]) * MAX_CHAR_SIZE],  categorical_values + (cond_idx + depth_idx)*MAX_CHAR_SIZE) == 0;
            leaf_idx |= (passed <<  (depths[tree_idx] - 1 - depth_idx));
        }
        
        int value_idx = (initial_leaf_idx + leaf_idx)*metadata->output_dim;
#ifndef _MSC_VER
    #pragma omp simd
#endif
        for (int d = 0; d < metadata->output_dim; ++d){
            momentum[sample_idx + d] *= cv_beta;
            momentum[sample_idx + d] += cv_1_m_beta * values[value_idx + d];
        }

        ++tree_idx;
    }
}


void Predictor::predict_cpu(dataSet *dataset, float *preds, const ensembleData *edata, const ensembleMetaData *metadata, int start_tree_idx, int stop_tree_idx, const bool parallel_predict, std::vector<Optimizer*> opts){
    // Add bias to current predictions
    const int output_dim = metadata->output_dim, par_th = metadata->par_th, n_samples = dataset->n_samples;
    add_vec_to_mat(preds, edata->bias, n_samples, output_dim, par_th);
    
    int n_trees = metadata->n_trees;
    if (n_trees == 0)
        return;
    if (stop_tree_idx > n_trees){
        std::cerr << "Given stop_tree_idx: " << stop_tree_idx << " greater than number of trees in model: " << n_trees << std::endl;
        return;
    } 
    if (n_trees == 0)
        return; 
    
    if (stop_tree_idx == 0)
        stop_tree_idx = n_trees;

    if (opts.size() == 0){
        std::cerr << "No optimizers." << std::endl;
        return;
    }
    n_trees = stop_tree_idx - start_tree_idx;
    void (*predictFunc)(const float*, const char*, float*, const int, const ensembleData*, const ensembleMetaData*, const int, const int, std::vector<Optimizer*>) = nullptr;
    predictFunc = (metadata->grow_policy == OBLIVIOUS) ? &Predictor::predict_over_trees : &Predictor::predict_over_leaves;
    int n_tree_threads = calculate_num_threads(n_trees, par_th);
    int n_sample_threads = calculate_num_threads(n_samples, par_th);
    // parallellize over trees
    if (n_tree_threads > 1 && parallel_predict && n_tree_threads > n_sample_threads){
        std::vector<float *> preds_buffer(n_tree_threads);
        int trees_per_thread = n_trees / n_tree_threads;
        omp_set_num_threads(n_tree_threads);
        for (int i = 0; i < n_tree_threads; ++i)
            preds_buffer[i] = init_zero_mat(n_samples*output_dim);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int thread_start_tree_idx = thread_id * trees_per_thread + start_tree_idx;
            int thread_stop_tree_idx = (thread_id == n_tree_threads - 1) ? stop_tree_idx : thread_start_tree_idx + trees_per_thread;
            for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx){
                predictFunc(dataset->obs, dataset->categorical_obs, preds_buffer[thread_id], sample_idx, edata, metadata, thread_start_tree_idx, thread_stop_tree_idx, opts);
            }        
        }
        for (int thread_id = 0; thread_id < n_tree_threads; ++thread_id){
            _element_wise_addition(preds, preds_buffer[thread_id], n_samples*output_dim, par_th);
            delete[] preds_buffer[thread_id];
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
                predictFunc(dataset->obs, dataset->categorical_obs, preds, sample_idx, edata, metadata, start_tree_idx, stop_tree_idx, opts);
            }
        }
    // no parallelization
    } else{ 
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx){
            predictFunc(dataset->obs, dataset->categorical_obs, preds, sample_idx, edata, metadata, start_tree_idx, stop_tree_idx, opts);
        }
    }
}


void Predictor::predict_over_leaves(const float *obs, const char *categorical_obs, float *theta, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, std::vector<Optimizer*> opts){
    int tree_idx = start_tree_idx;
    int n_leaves = metadata->n_leaves;
    int max_depth = metadata->max_depth;
    bool passed;
    int obs_row = sample_idx*metadata->n_num_features;
    int categorical_obs_row = sample_idx*metadata->n_cat_features;
    int theta_row = sample_idx*metadata->output_dim;

    const bool *numerics = edata->is_numerics;
    const float *feature_values = edata->feature_values;
    const float *values = edata->values;
    const int* feature_indices = edata->feature_indices;
    const int* tree_indices = edata->tree_indices;
    const bool* inequality_directions = edata->inequality_directions;
    const char* categorical_values = edata->categorical_values;
    int leaf_idx = tree_indices[tree_idx];

    while (leaf_idx < n_leaves && tree_idx < stop_tree_idx)
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
            for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
                opts[opt_idx]->step(theta, values + leaf_idx*metadata->output_dim, tree_idx, theta_row);
            }
            ++tree_idx;
            if (tree_idx < stop_tree_idx)
                leaf_idx = tree_indices[tree_idx];
        } else {
            ++leaf_idx;
        }
    }
}

void Predictor::predict_over_trees(const float *obs, const char *categorical_obs, float *theta, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, std::vector<Optimizer*> opts){
    int tree_idx = start_tree_idx;
    int max_depth = metadata->max_depth;
    bool passed;
    int obs_row = sample_idx*metadata->n_num_features;
    int categorical_obs_row = sample_idx*metadata->n_cat_features;
    int theta_row = sample_idx*metadata->output_dim;

    const bool *numerics = edata->is_numerics;
    const int *depths = edata->depths;
    const float *feature_values = edata->feature_values;
    const float *values = edata->values;
    const int* feature_indices = edata->feature_indices;
    const int* tree_indices = edata->tree_indices;
    const char* categorical_values = edata->categorical_values;
    

    while (tree_idx < stop_tree_idx)
    {
        int initial_leaf_idx = tree_indices[tree_idx];
        int cond_idx = tree_idx*max_depth;
        int leaf_idx = 0;
        for (int depth_idx = 0; depth_idx < depths[tree_idx]; depth_idx++){ 
            passed = (numerics[cond_idx + depth_idx]) ? (obs[obs_row + feature_indices[cond_idx + depth_idx]] >  feature_values[cond_idx + depth_idx]) : strcmp(&categorical_obs[(categorical_obs_row + feature_indices[cond_idx + depth_idx]) * MAX_CHAR_SIZE],  categorical_values + (cond_idx + depth_idx)*MAX_CHAR_SIZE) == 0;
            leaf_idx |= (passed <<  (depths[tree_idx] - 1 - depth_idx));
        }
        
        int value_idx = (initial_leaf_idx + leaf_idx)*metadata->output_dim;
        for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
            opts[opt_idx]->step(theta, values + value_idx, tree_idx, theta_row);
        }
        ++tree_idx;
    }
}


