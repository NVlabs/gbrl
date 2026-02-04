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
 * @file cuda_types.h
 * @brief CUDA-specific data types and GPU configuration constants
 * 
 * Defines GPU kernel parameters, optimization constants, and CUDA-specific
 * data structures for gradient boosting on NVIDIA GPUs.
 */

#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#include "types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel execution parameters
#define WARP_SIZE 32                  /**< CUDA warp size for synchronization */
#define MAX_CANDIDATES_PER_GRID 256   /**< Maximum split candidates per grid */
#define THREADS_PER_BLOCK 256         /**< Default threads per block */
#define MAX_GRIDS 10                  /**< Maximum concurrent grids */
#define MAX_TREES_PER_THREAD 10       /**< Maximum trees per thread */
#define MIN_SAMPLES_PER_GRID 250      /**< Minimum samples to justify grid */
#define FULL_MASK 0xffffffff          /**< Full warp mask for warp operations */
#define MAX_BLOCKS_PER_GRID 65535     /**< CUDA maximum blocks per grid */
#define MAX_THREADS_PER_BLOCK 1024    /**< CUDA maximum threads per block */
#define DEFAULT_N_STREAMS 8           /**< Default number of CUDA streams */

// Matrix operation tile sizes
#define BLOCK_COLS 32                 /**< Block column size for matrix ops */
#define BLOCK_ROWS 32                 /**< Block row size for matrix ops */

// GPU memory management parameters
#define INITAL_MAX_GPU_TREES 250000   /**< Initial GPU tree capacity (250k) */
#define GPU_TREES_BATCH 100000        /**< Tree batch size for allocation (100k) */

/**
 * @brief SGD optimizer configuration for GPU
 * 
 * Defines learning rate schedule for a range of trees on GPU.
 */
struct SGDOptimizerGPU {
    int start_idx;            /**< Starting tree index */
    int stop_idx;             /**< Stopping tree index */
    float init_lr;            /**< Initial learning rate */
    float stop_lr;            /**< Final learning rate (for Linear scheduler) */
    int T;                    /**< Total iterations (for Linear scheduler) */
    schedulerFunc scheduler;  /**< Scheduler type */
};

/**
 * @brief GPU split evaluation data
 * 
 * Holds intermediate computation buffers for parallel split scoring on GPU.
 */
struct splitDataGPU {
    float *split_scores;            /**< Split quality scores */
    float *node_mean;               /**< Node mean values */
    float *left_sum;                /**< Left child sum statistics */
    float *right_sum;               /**< Right child sum statistics */
    float *left_count;              /**< Left child sample counts */
    float *right_count;             /**< Right child sample counts */
    int *tree_counters;             /**< Tree processing counters */
    float *best_score;              /**< Best split scores */
    int *best_idx;                  /**< Best split indices */
    float *left_dot;                /**< Left child dot products (cosine) */
    float *right_dot;               /**< Right child dot products (cosine) */
    float *oblivious_split_scores;  /**< Oblivious tree split scores */
    size_t size;                    /**< Total allocated size */
};

/**
 * @brief GPU tree node representation
 * 
 * Stores node data on GPU for parallel tree operations.
 * Members ordered for optimal GPU memory alignment: floats/ints first, then pointers.
 */
struct TreeNodeGPU {
    // Scalar types first (optimal alignment)
    int n_samples;
    int depth;
    int n_num_features;
    int n_cat_features;
    int output_dim;
    int node_idx;
    int n_objs;
    float conflict_rho;
    // Pointers last (proper GPU memory alignment)
    int *sample_indices;
    int* feature_indices;
    float* feature_values;
    float* edge_weights;
    float* scores;
    float *densities;
    float *mean_values;
    bool* inequality_directions;
    bool* is_numerics;
    char* categorical_values;
};

/**
 * @brief Split candidate data for GPU evaluation
 * 
 * Immutable candidate set for parallel split scoring.
 */
struct candidatesData {
    const int n_candidates;          /**< Number of candidates */
    const int *candidate_indices;    /**< Feature indices */
    const float *candidate_values;   /**< Threshold values */
    const bool *candidate_numeric;   /**< Numerical feature flags */
    const char *candidate_categories; /**< Categorical values */
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate ensemble data on GPU
 * 
 * @param metadata Ensemble configuration
 * @return Pointer to GPU ensemble data
 */
ensembleData* ensemble_data_alloc_cuda(ensembleMetaData *metadata);

/**
 * @brief Allocate copy-target ensemble data on GPU
 * 
 * @param metadata Ensemble configuration
 * @return Pointer to GPU ensemble data for copy operations
 */
ensembleData* ensemble_copy_data_alloc_cuda(ensembleMetaData *metadata);

/**
 * @brief Allocate GPU split evaluation buffers
 * 
 * @param metadata Ensemble configuration
 * @param n_candidates Number of split candidates
 * @return Pointer to GPU split data structure
 */
splitDataGPU* allocate_split_data(
    ensembleMetaData *metadata,
    const int n_candidates
);

/**
 * @brief Deallocate ensemble data from GPU
 * 
 * @param edata Ensemble data to free
 */
void ensemble_data_dealloc_cuda(ensembleData *edata);

/**
 * @brief Copy ensemble data between GPU locations
 * 
 * @param metadata Ensemble configuration
 * @param other_edata Source ensemble data (GPU)
 * @param edata Target ensemble data (GPU)
 * @return Pointer to updated target ensemble data
 */
ensembleData* ensemble_data_copy_gpu_gpu(
    ensembleMetaData *metadata,
    ensembleData *other_edata,
    ensembleData* edata
);

/**
 * @brief Allocate GPU memory for compressed ensemble data
 * 
 * Allocates device memory for a compressed ensemble with specified number
 * of trees and leaves. Memory layout optimized for GPU access patterns.
 * 
 * @param metadata Ensemble metadata specifying structure and dimensions
 * @param n_compressed_leaves Number of leaves in compressed ensemble
 * @param n_compressed_trees Number of trees in compressed ensemble
 * @return Pointer to allocated ensembleData structure on GPU
 */
ensembleData* ensemble_compressed_data_alloc_cuda(ensembleMetaData *metadata, const int n_compressed_leaves, const int n_compressed_trees);

/**
 * @brief Copy ensemble data from CPU to GPU
 * 
 * @param metadata Ensemble configuration
 * @param other_edata Source ensemble data (CPU)
 * @param edata Target ensemble data (GPU)
 * @return Pointer to updated GPU ensemble data
 */
ensembleData* ensemble_data_copy_cpu_gpu(
    ensembleMetaData *metadata,
    ensembleData *other_edata,
    ensembleData* edata
);

/**
 * @brief Copy ensemble data from GPU to CPU
 * 
 * @param metadata Ensemble configuration
 * @param other_edata Source ensemble data (GPU)
 * @param edata Target ensemble data (CPU)
 * @return Pointer to updated CPU ensemble data
 */
ensembleData* ensemble_data_copy_gpu_cpu(
    ensembleMetaData *metadata,
    ensembleData *other_edata,
    ensembleData* edata
);


/**
 * @brief Copy compressed ensemble data between GPU memory locations
 * 
 * Performs GPU-to-GPU copy of selected trees and leaves for ensemble compression.
 * Uses CUDA kernels for efficient parallel copying. Copies all associated metadata
 * including feature mappings, categorical values, and tree structures.
 * 
 * @param metadata Ensemble metadata defining structure
 * @param other_edata Source ensemble data on GPU
 * @param edata Target ensemble data structure on GPU (pre-allocated)
 * @param n_compressed_leaves Number of leaves in compressed ensemble
 * @param n_compressed_trees Number of trees in compressed ensemble
 * @param leaf_indices Device pointer to leaf indices to copy
 * @param tree_indices Device pointer to tree indices to copy
 * @param new_tree_indices Device pointer to new tree starting indices
 * @return Pointer to updated target ensemble data
 */
ensembleData* ensemble_compressed_data_copy_gpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData* edata, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices);

/**
 * @brief Allocate GPU memory for ensemble
 * 
 * @param metadata Ensemble configuration
 * @param edata Ensemble data structure to populate
 */
void allocate_ensemble_memory_cuda(
    ensembleMetaData *metadata,
    ensembleData *edata
);

#ifdef __cplusplus
}
#endif

#endif 