
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
#include <omp.h>

#include "cuda_types.h"
#include "types.h"

ensembleData* ensemble_data_alloc_cuda(ensembleMetaData *metadata){
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        return nullptr;
    }

    char *data;
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t tree_size = metadata->max_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->max_trees : metadata->max_leaves;
    size_t value_sizes = metadata->output_dim * metadata->max_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->max_depth * metadata->max_leaves;
    size_t data_size = bias_size
                     + tree_size
                     + split_sizes * sizeof(int) // depths
                     + value_sizes 
                     + edge_size * (sizeof(bool) + sizeof(float)) // inequality directions + edge weights
                     + cond_sizes * (sizeof(int) + sizeof(float) + sizeof(bool) + sizeof(char)*MAX_CHAR_SIZE);
#ifdef DEBUG 
    size_t sample_size = metadata->max_leaves * sizeof(int);
    data_size += sample_size;
#endif
    cudaError_t alloc_error = cudaMalloc((void**)&data, data_size);
    if (alloc_error != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA error: " << cudaGetErrorString(alloc_error)
                << " when trying to allocate " << (data_size / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return nullptr;
    }
    cudaMemset(data, 0, data_size);
    size_t trace = 0;
    edata->bias = (float*)(data + trace);
    trace += bias_size;
#ifdef DEBUG 
    edata->n_samples = (int *)(data + trace);
    trace += sample_size;
#endif
    edata->tree_indices = (int *)(data + trace);
    trace += tree_size;
    edata->depths = (int *)(data + trace);
    trace += split_sizes*sizeof(int);
    edata->values = (float *)(data + trace);
    trace += value_sizes;
    edata->feature_indices = (int *)(data + trace);
    trace += cond_sizes * sizeof(int);
    edata->feature_values = (float *)(data + trace);
    trace += cond_sizes * sizeof(float);
    edata->edge_weights = (float *)(data + trace);
    trace += edge_size * sizeof(float);
    edata->is_numerics = (bool *)(data + trace);
    trace += cond_sizes * sizeof(bool);
    edata->inequality_directions = (bool *)(data + trace);
    trace += edge_size * sizeof(bool);
    edata->categorical_values = (char *)(data + trace);
    return edata;
}

ensembleData* ensemble_copy_data_alloc_cuda(ensembleMetaData *metadata){
    // Same function as normal alloc just allocates exact amount
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        return nullptr;
    }

    char *data;
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->n_leaves*metadata->max_depth;

    size_t data_size = bias_size
                     + tree_size
                     + split_sizes * sizeof(int) // depths
                     + value_sizes 
                     + edge_size * (sizeof(bool) + sizeof(float)) // inequality directions + edge_weights
                     + cond_sizes * (sizeof(int) + sizeof(float) + sizeof(bool) + sizeof(char)*MAX_CHAR_SIZE); 
#ifdef DEBUG 
    size_t sample_size = metadata->max_leaves * sizeof(int);
    data_size += sample_size;
#endif
    cudaError_t alloc_error = cudaMalloc((void**)&data, data_size);
    if (alloc_error != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA error: " << cudaGetErrorString(alloc_error)
                << " when trying to allocate " << (data_size / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return nullptr;
    }
    cudaMemset(data, 0, data_size);
    size_t trace = 0;
    edata->bias = (float*)(data + trace);
    trace += bias_size;
#ifdef DEBUG 
    edata->n_samples = (int *)(data + trace);
    trace += sample_size;
#endif
    edata->tree_indices = (int *)(data + trace);
    trace += tree_size;
    edata->depths = (int *)(data + trace);
    trace += split_sizes*sizeof(int);
    edata->values = (float *)(data + trace);
    trace += value_sizes;
    edata->feature_indices = (int *)(data + trace);
    trace += cond_sizes * sizeof(int);
    edata->feature_values = (float *)(data + trace);
    trace += cond_sizes * sizeof(float);
    edata->edge_weights = (float *)(data + trace);
    trace += edge_size * sizeof(float);
    edata->is_numerics = (bool *)(data + trace);
    trace += cond_sizes * sizeof(bool);
    edata->inequality_directions = (bool *)(data + trace);
    trace += edge_size * sizeof(bool);
    edata->categorical_values = (char *)(data + trace);

    metadata->max_trees = metadata->n_trees;
    metadata->max_leaves = metadata->n_leaves;
    return edata;
}

ensembleData* ensemble_data_copy_gpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata){
    ensembleData *edata = ensemble_copy_data_alloc_cuda(metadata);
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->n_leaves*metadata->max_depth;

    cudaMemcpy(edata->bias, other_edata->bias, bias_size, cudaMemcpyDeviceToDevice);
#ifdef DEBUG 
    size_t sample_size = metadata->n_leaves * sizeof(int);
    cudaMemcpy(edata->n_samples, other_edata->n_samples, sample_size, cudaMemcpyDeviceToDevice);
#endif
    cudaMemcpy(edata->tree_indices, other_edata->tree_indices, tree_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->depths, other_edata->depths, split_sizes*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->values, other_edata->values, value_sizes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_indices, other_edata->feature_indices, cond_sizes*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_values, other_edata->feature_values, cond_sizes*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->edge_weights, other_edata->edge_weights, edge_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->is_numerics, other_edata->is_numerics, cond_sizes * sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->inequality_directions, other_edata->inequality_directions, edge_size * sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->categorical_values, other_edata->categorical_values, cond_sizes * sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice); 
    return edata;
}

ensembleData* ensemble_data_copy_gpu_cpu(ensembleMetaData *metadata, ensembleData *other_edata){
    ensembleData *edata = ensemble_copy_data_alloc(metadata);
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->n_leaves*metadata->max_depth;
    
    cudaMemcpy(edata->bias, other_edata->bias, bias_size, cudaMemcpyDeviceToHost);
#ifdef DEBUG 
    size_t sample_size = metadata->n_leaves * sizeof(int);
    cudaMemcpy(edata->n_samples, other_edata->n_samples, sample_size, cudaMemcpyDeviceToHost);
#endif
    cudaMemcpy(edata->tree_indices, other_edata->tree_indices, tree_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->depths, other_edata->depths, split_sizes*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->values, other_edata->values, value_sizes, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_indices, other_edata->feature_indices, cond_sizes*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_values, other_edata->feature_values, cond_sizes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->edge_weights, other_edata->edge_weights,edge_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->is_numerics, other_edata->is_numerics, cond_sizes * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->inequality_directions, other_edata->inequality_directions, edge_size * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->categorical_values, other_edata->categorical_values, cond_sizes * sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyDeviceToHost); 
    return edata;
}

ensembleData* ensemble_data_copy_cpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata){
    ensembleData *edata = ensemble_copy_data_alloc_cuda(metadata);
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->n_leaves*metadata->max_depth;

    cudaMemcpy(edata->bias, other_edata->bias, bias_size, cudaMemcpyHostToDevice);
#ifdef DEBUG 
    size_t sample_size = metadata->n_leaves * sizeof(int);
    cudaMemcpy(edata->n_samples, other_edata->n_samples, sample_size, cudaMemcpyHostToDevice);
#endif
    cudaMemcpy(edata->tree_indices, other_edata->tree_indices, tree_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->depths, other_edata->depths, split_sizes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->values, other_edata->values, value_sizes, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_indices, other_edata->feature_indices, cond_sizes*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_values, other_edata->feature_values, cond_sizes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->edge_weights, other_edata->edge_weights, edge_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->is_numerics, other_edata->is_numerics, cond_sizes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->inequality_directions, other_edata->inequality_directions, edge_size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->categorical_values, other_edata->categorical_values, cond_sizes * sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyHostToDevice); 
    return edata;
}

void ensemble_data_dealloc_cuda(ensembleData *edata){
    cudaFree(edata->bias);
    delete edata; 
}

splitDataGPU* allocate_split_data(ensembleMetaData *metadata, const int n_candidates){
    splitDataGPU *split_data = new splitDataGPU;
    int nodes_per_evaluation = (metadata->grow_policy == GREEDY) ? 1 : (1 << metadata->max_depth);
    size_t data_alloc_size = sizeof(float) * n_candidates + 
                    sizeof(float) * metadata->output_dim  +
                    sizeof(float) * n_candidates * metadata->output_dim * 2 + 
                    sizeof(float) * n_candidates * 2 + 
                    sizeof(int)*3 + sizeof(int) + sizeof(float);
    if (metadata->split_score_func == Cosine)
        data_alloc_size += sizeof(float) * n_candidates * 4;
    if (metadata->grow_policy == OBLIVIOUS)
        data_alloc_size += sizeof(float) * n_candidates * nodes_per_evaluation;

    char *data_alloc;
    cudaError_t err = cudaMalloc((void**)&data_alloc, data_alloc_size);
    if (err != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                << " when trying to allocate " << (data_alloc_size / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return nullptr;
    }
    cudaMemset(data_alloc, 0, data_alloc_size);
    size_t trace = 0;
    split_data->split_scores = (float *)(data_alloc + trace);
    trace += sizeof(float)*n_candidates;
    split_data->node_mean = (float *)(data_alloc + trace);
    trace += sizeof(float)*metadata->output_dim;
    split_data->left_sum = (float *)(data_alloc + trace);
    trace += sizeof(float)*n_candidates*metadata->output_dim;
    split_data->right_sum  = (float *)(data_alloc + trace);
    trace += sizeof(float)*n_candidates*metadata->output_dim;
    split_data->left_count = (float *)(data_alloc + trace);
    trace += sizeof(float)*n_candidates;
    split_data->right_count  = (float *)(data_alloc + trace);
    trace += sizeof(float)*n_candidates;
    split_data->tree_counters = (int *)(data_alloc + trace);
    trace += sizeof(int) * 3;
    split_data->best_score = (float *)(data_alloc + trace);
    trace += sizeof(float);
    split_data->best_idx = (int *)(data_alloc + trace);
    trace += sizeof(int);

    split_data->left_norms = nullptr;
    split_data->right_norms = nullptr;
    split_data->left_dot = nullptr;
    split_data->right_dot = nullptr;

    if (metadata->split_score_func == Cosine){
        split_data->left_norms = (float *)(data_alloc + trace);
        trace += sizeof(float)*n_candidates;
        split_data->right_norms  = (float *)(data_alloc + trace);
        trace += sizeof(float)*n_candidates;
        split_data->left_dot = (float *)(data_alloc + trace);
        trace += sizeof(float)*n_candidates;
        split_data->right_dot = (float *)(data_alloc + trace);
        trace += sizeof(float)*n_candidates;
    }
    split_data->oblivious_split_scores = nullptr;
    if (metadata->grow_policy == OBLIVIOUS){
        split_data->oblivious_split_scores = (float *)(data_alloc + trace);
    }
    split_data->size = data_alloc_size;
    return split_data;
}

void allocate_ensemble_memory_cuda(ensembleMetaData *metadata, ensembleData *edata){
    int leaf_idx = metadata->n_leaves, tree_idx = metadata->n_trees; 
    if  ((leaf_idx >= metadata->max_leaves) || (tree_idx >= metadata->max_trees)){
        int new_size_leaves = metadata->n_leaves + metadata->max_leaves_batch;
        int new_tree_size = metadata->n_trees + metadata->max_trees_batch;
        metadata->max_leaves = new_size_leaves;
        metadata->max_trees = new_tree_size;
        ensembleData *new_data = ensemble_data_alloc_cuda(metadata);
        cudaMemcpy(new_data->bias, edata->bias, metadata->output_dim*sizeof(float), cudaMemcpyDeviceToDevice);
#ifdef DEBUG
        cudaMemcpy(new_data->n_samples, edata->n_samples, leaf_idx*sizeof(int), cudaMemcpyDeviceToDevice);
#endif 
        cudaMemcpy(new_data->values, edata->values, leaf_idx*metadata->output_dim*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->tree_indices, edata->tree_indices, tree_idx*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->inequality_directions, edata->inequality_directions, leaf_idx*metadata->max_depth*sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->edge_weights, edata->edge_weights, leaf_idx*metadata->max_depth*sizeof(float), cudaMemcpyDeviceToDevice);
        if (metadata->grow_policy == GREEDY){
            cudaMemcpy(new_data->depths, edata->depths, leaf_idx*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_indices, edata->feature_indices, leaf_idx*metadata->max_depth*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_values, edata->feature_values, leaf_idx*metadata->max_depth*sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->is_numerics, edata->is_numerics, leaf_idx*metadata->max_depth*sizeof(bool), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->categorical_values, edata->categorical_values, leaf_idx*metadata->max_depth*sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(new_data->depths, edata->depths, tree_idx*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_indices, edata->feature_indices, tree_idx*metadata->max_depth*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_values, edata->feature_values, tree_idx*metadata->max_depth*sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->is_numerics, edata->is_numerics, tree_idx*metadata->max_depth*sizeof(bool), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->categorical_values, edata->categorical_values, tree_idx*metadata->max_depth*sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);
        }
        cudaFree(edata->bias);
        edata->bias = new_data->bias;
#ifdef DEBUG
        edata->n_samples = new_data->n_samples;
#endif
        edata->depths = new_data->depths;
        edata->tree_indices = new_data->tree_indices;
        edata->values = new_data->values;
        edata->inequality_directions = new_data->inequality_directions;
        edata->feature_indices = new_data->feature_indices;
        edata->feature_values = new_data->feature_values;
        edata->edge_weights = new_data->edge_weights;
        edata->is_numerics = new_data->is_numerics;
        edata->categorical_values = new_data->categorical_values;
        delete new_data;
    }
}

