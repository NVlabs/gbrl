//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
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
ensembleData* ensemble_data_copy_gpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData* edata);
ensembleData* ensemble_data_copy_cpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData* edata);
ensembleData* ensemble_data_copy_gpu_cpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData* edata);
void allocate_ensemble_memory_cuda(ensembleMetaData *metadata, ensembleData *edata);
#ifdef __cplusplus
}
#endif

#endif 