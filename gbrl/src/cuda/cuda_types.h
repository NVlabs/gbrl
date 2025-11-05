//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
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
 */
struct TreeNodeGPU {
    int n_samples;                /**< Number of samples in node */
    int depth;                    /**< Node depth in tree */
    int n_num_features;           /**< Number of numerical features */
    int n_cat_features;           /**< Number of categorical features */
    int output_dim;               /**< Output dimensionality */
    int node_idx;                 /**< Node index in tree */
    float score;                  /**< Node split score */
    int *sample_indices;          /**< Indices of samples in node */
    int *feature_indices;         /**< Feature indices along path */
    float *feature_values;        /**< Feature threshold values along path */
    float *edge_weights;          /**< Edge weights along path */
    bool *inequality_directions;  /**< Split directions along path */
    bool *is_numerics;            /**< Feature type flags along path */
    char *categorical_values;     /**< Categorical split values along path */
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