//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#include "types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WARP_SIZE 32
#define MAX_CANDIDATES_PER_GRID 256
#define THREADS_PER_BLOCK 256
#define MAX_GRIDS 10
#define MAX_TREES_PER_THREAD 10
#define MIN_SAMPLES_PER_GRID 250
#define FULL_MASK 0xffffffff
#define MAX_BLOCKS_PER_GRID 65535
#define MAX_THREADS_PER_BLOCK 1024

#define BLOCK_COLS 32
#define BLOCK_ROWS 32

#define INITAL_MAX_GPU_TREES 250000 //250k
#define GPU_TREES_BATCH  100000 // 100 K

struct SGDOptimizerGPU{
    int start_idx;
    int stop_idx;
    float init_lr;
    schedulerFunc scheduler;
};

struct splitDataGPU{
    float *split_scores;
    float *node_mean;
    float *left_sum;
    float *right_sum;
    float *left_count;
    float *right_count;
    int *tree_counters;
    float *best_score;
    int *best_idx;
    // cosine score
    float *left_norms;
    float *right_norms;
    float *left_dot;
    float *right_dot;
    float *oblivious_split_scores;
    size_t size;
};

struct TreeNodeGPU {
    int n_samples; 
    int depth;
    int n_num_features;
    int n_cat_features;
    int output_dim;
    int node_idx;
    float score;
    int *sample_indices;
    int* feature_indices;
    float* feature_values;
    float* edge_weights;
    bool* inequality_directions;
    bool* is_numerics;
    char* categorical_values;
};

struct candidatesData {
    const int n_candidates;
    const int *candidate_indices;
    const float *candidate_values;
    const bool *candidate_numeric;
    const char *candidate_categories;
    
};

#ifdef __cplusplus
extern "C" {
#endif
ensembleData* ensemble_data_alloc_cuda(ensembleMetaData *metadata);
ensembleData* ensemble_copy_data_alloc_cuda(ensembleMetaData *metadata);
splitDataGPU* allocate_split_data(ensembleMetaData *metadata, const int n_candidates);
void ensemble_data_dealloc_cuda(ensembleData *edata);
ensembleData* ensemble_data_copy_gpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata);
ensembleData* ensemble_data_copy_cpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata);
ensembleData* ensemble_data_copy_gpu_cpu(ensembleMetaData *metadata, ensembleData *other_edata);
void allocate_ensemble_memory_cuda(ensembleMetaData *metadata, ensembleData *edata);
#ifdef __cplusplus
}
#endif

#endif 