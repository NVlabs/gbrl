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
 * @file cuda_types.cu
 * @brief Implementation of CUDA data structure management and GPU memory operations
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
#include <omp.h>

#include "cuda_types.h"
#include "cuda_utils.h"
#include "types.h"

/**
 * @brief CUDA kernel for selective copy of integer arrays
 * 
 * Copies elements from source to destination based on selection indices.
 * Each thread processes one element from the indices array.
 * 
 * @param n_elements Number of elements to copy
 * @param indices Array of indices to copy from source
 * @param dest Destination array
 * @param src Source array
 * @param stride Number of values per element (for 2D arrays)
 */
__global__ void selective_copyi(const int n_elements, const int* __restrict__ indices, int* __restrict__ dest, const int* __restrict__ src, const int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements){
        int src_idx = indices[idx];
        for (int i = 0; i < stride; ++i){
            dest[idx * stride + i] = src[src_idx * stride + i];
        }
    }
}

/**
 * @brief CUDA kernel for selective copy of float arrays
 * 
 * @param n_elements Number of elements to copy
 * @param indices Array of indices to copy from source
 * @param dest Destination array
 * @param src Source array
 * @param stride Number of values per element
 */
__global__ void selective_copyf(const int n_elements, const int* __restrict__ indices, float* __restrict__ dest, const float* __restrict__ src, const int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements){
        int src_idx = indices[idx];
        for (int i = 0; i < stride; ++i){
            dest[idx * stride + i] = src[src_idx * stride + i];
        }
    }
}

/**
 * @brief CUDA kernel for selective copy of boolean arrays
 * 
 * @param n_elements Number of elements to copy
 * @param indices Array of indices to copy from source
 * @param dest Destination array
 * @param src Source array
 * @param stride Number of values per element
 */
__global__ void selective_copyb(const int n_elements, const int* __restrict__ indices, bool* __restrict__ dest, const bool* __restrict__ src, const int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements){
        int src_idx = indices[idx];
        for (int i = 0; i < stride; ++i){
            dest[idx * stride + i] = src[src_idx * stride + i];
        }
    }
}

/**
 * @brief CUDA kernel for selective copy of character arrays
 * 
 * Copies character strings (MAX_CHAR_SIZE bytes per string) based on indices.
 * 
 * @param n_elements Number of elements to copy
 * @param indices Array of indices to copy from source
 * @param dest Destination array
 * @param src Source array
 * @param stride Number of strings per element
 */
__global__ void selective_copyc(const int n_elements, const int* __restrict__ indices, char* __restrict__ dest, const char* __restrict__ src, const int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements){
        int src_idx = indices[idx];
        for (int i = 0; i < stride; ++i){
            for (int j = 0; j < MAX_CHAR_SIZE; ++j){
                dest[(idx * stride + i) * MAX_CHAR_SIZE + j] = src[(src_idx * stride + i) * MAX_CHAR_SIZE + j];
            }
        }
    }
}

// Implementation notes for ensemble_data_alloc_cuda:
// Allocates unified memory block with feature mapping arrays (4 vectors):
// - feature_mapping: Original to internal index mapping (stored for export)
// - mapping_numerics: Feature type flags (stored for export)
// - reverse_num_feature_mapping: Internal numerical to original mapping (used in computation)
// - reverse_cat_feature_mapping: Internal categorical to original mapping (used in computation)
// Only reverse mappings are actively used; forward mappings maintained for serialization.
ensembleData* ensemble_data_alloc_cuda(ensembleMetaData *metadata){
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        return nullptr;
    }

    // Allocate sub-structs on CPU (they hold pointers to GPU memory)
    edata->ensemble_info = new ensembleInfo;
    edata->leaf_data = new leafData;
    edata->feature_data = new featureData;
    edata->mono_constraints = new monotonicConstraints;
    edata->mono_constraints->n_constraints = 0;  // Initialize to 0
    edata->feature_mappings = new featureMapping;

    char *data;
    size_t bias_size = metadata->output_dim * sizeof(float);
    // Feature mapping: 3x int arrays (feature_mapping, reverse_num, reverse_cat)
    size_t feature_mapping_size = metadata->input_dim * sizeof(int);
    size_t feature_size = metadata->input_dim * sizeof(float);
    // Feature type flags: 1x bool array (mapping_numerics)
    size_t feature_numerics_size = metadata->input_dim * sizeof(bool);
    size_t tree_size = metadata->max_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->max_trees : metadata->max_leaves;
    size_t value_sizes = metadata->output_dim * metadata->max_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->max_depth * metadata->max_leaves;
    size_t mono_size = metadata->n_mono_constraints * sizeof(int);
    size_t data_size = bias_size 
                     + feature_mapping_size * 3  // 3 int arrays: feature_mapping, reverse_num_feature_mapping, reverse_cat_feature_mapping
                     + feature_size 
                     + tree_size
                     + split_sizes * sizeof(int) // depths
                     + value_sizes 
                     + mono_size * 3 // monotonic constraints: feature_idx, output_idx, constraint
                     + edge_size * (sizeof(bool) + sizeof(float)) // inequality directions + edge weights
                     + cond_sizes * (sizeof(int) + sizeof(float) + sizeof(bool) + sizeof(char)*MAX_CHAR_SIZE)
                     + feature_numerics_size;  // 1 bool array: mapping_numerics
#ifdef DEBUG 
    size_t sample_size = metadata->max_leaves * sizeof(int);
    data_size += sample_size;
#endif
    cudaError_t alloc_error = allocateCudaMemory((void**)&data, data_size, "when trying to allocate memory for ensemble_data_alloc");
    if (alloc_error != cudaSuccess) {
        return nullptr;
    }
    cudaMemset(data, 0, data_size);
    size_t trace = 0;
    
    // Keep original allocation ordering: float first, then int, then char/bool
    edata->bias = (float*)(data + trace);
    trace += bias_size;
    // Feature mapping arrays (4 total: 3 int, 1 bool)
    edata->feature_mappings->feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_mappings->reverse_num_feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_mappings->reverse_cat_feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_data->feature_weights = (float*)(data + trace);
    trace += feature_size;
#ifdef DEBUG 
    edata->n_samples = (int *)(data + trace);
    trace += sample_size;
#endif
    edata->ensemble_info->tree_indices = (int *)(data + trace);
    trace += tree_size;
    edata->ensemble_info->depths = (int *)(data + trace);
    trace += split_sizes * sizeof(int);
    edata->leaf_data->values = (float *)(data + trace);
    trace += value_sizes;
    edata->feature_data->feature_indices = (int *)(data + trace);
    trace += cond_sizes * sizeof(int);
    edata->mono_constraints->feature_idx = (int *)(data + trace);
    trace += mono_size;
    edata->mono_constraints->output_idx = (int *)(data + trace);
    trace += mono_size;
    edata->mono_constraints->constraint = (int *)(data + trace);
    trace += mono_size;
    edata->feature_data->feature_values = (float *)(data + trace);
    trace += cond_sizes * sizeof(float);
    edata->leaf_data->edge_weights = (float *)(data + trace);
    trace += edge_size * sizeof(float);
    edata->feature_data->is_numerics = (bool *)(data + trace);
    trace += cond_sizes * sizeof(bool);
    edata->feature_data->inequality_directions = (bool *)(data + trace);
    trace += edge_size * sizeof(bool);
    edata->feature_mappings->mapping_numerics = (bool *)(data + trace);
    trace += feature_numerics_size;
    edata->feature_data->categorical_values = (char *)(data + trace);
    edata->alloc_data_size = data_size;
    return edata;
}

// Implementation notes for ensemble_copy_data_alloc_cuda:
// Same as ensemble_data_alloc_cuda but allocates exact size based on current state
// (n_trees, n_leaves) instead of maximum capacity.
ensembleData* ensemble_copy_data_alloc_cuda(ensembleMetaData *metadata){
    // Same function as normal alloc just allocates exact amount
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        return nullptr;
    }

    // Allocate sub-structs on CPU (they hold pointers to GPU memory)
    edata->ensemble_info = new ensembleInfo;
    edata->leaf_data = new leafData;
    edata->feature_data = new featureData;
    edata->mono_constraints = new monotonicConstraints;
    edata->mono_constraints->n_constraints = 0;  // Initialize to 0
    edata->feature_mappings = new featureMapping;

    char *data;
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t feature_mapping_size = metadata->input_dim * sizeof(int);
    size_t feature_size = metadata->input_dim * sizeof(float);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->n_leaves*metadata->max_depth;
    size_t mono_size = metadata->n_mono_constraints * sizeof(int);

    size_t data_size = bias_size
                     + feature_mapping_size * 3  // 3 int arrays: feature_mapping, reverse_num_feature_mapping, reverse_cat_feature_mapping
                     + feature_size  // feature_data->feature_weights
                     + tree_size
                     + split_sizes * sizeof(int) // depths
                     + value_sizes 
                     + mono_size * 3 // monotonic constraints: feature_idx, output_idx, constraint
                     + edge_size * (sizeof(bool) + sizeof(float)) // inequality directions + edge_weights
                     + sizeof(bool) * metadata->input_dim  // 1 bool array: mapping_numerics
                     + cond_sizes * (sizeof(int) + sizeof(float) + sizeof(bool) + sizeof(char)*MAX_CHAR_SIZE); 
#ifdef DEBUG 
    size_t sample_size = metadata->max_leaves * sizeof(int);
    data_size += sample_size;
#endif
    cudaError_t alloc_error = allocateCudaMemory((void**)&data, data_size, "when trying to allocate memory for ensemble_copy_data");
    if (alloc_error != cudaSuccess) {
        return nullptr;
    }
    cudaMemset(data, 0, data_size);
    size_t trace = 0;
    
    // Keep original allocation ordering: float first, then int, then char/bool
    edata->bias = (float*)(data + trace);
    trace += bias_size;
    // Feature mapping arrays (4 total: 3 int, 1 bool)
    edata->feature_mappings->feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_mappings->reverse_num_feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_mappings->reverse_cat_feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_data->feature_weights = (float*)(data + trace);
    trace += feature_size;
#ifdef DEBUG 
    edata->n_samples = (int *)(data + trace);
    trace += sample_size;
#endif
    edata->ensemble_info->tree_indices = (int *)(data + trace);
    trace += tree_size;
    edata->ensemble_info->depths = (int *)(data + trace);
    trace += split_sizes * sizeof(int);
    edata->leaf_data->values = (float *)(data + trace);
    trace += value_sizes;
    edata->feature_data->feature_indices = (int *)(data + trace);
    trace += cond_sizes * sizeof(int);
    edata->mono_constraints->feature_idx = (int *)(data + trace);
    trace += mono_size;
    edata->mono_constraints->output_idx = (int *)(data + trace);
    trace += mono_size;
    edata->mono_constraints->constraint = (int *)(data + trace);
    trace += mono_size;
    edata->feature_data->feature_values = (float *)(data + trace);
    trace += cond_sizes * sizeof(float);
    edata->leaf_data->edge_weights = (float *)(data + trace);
    trace += edge_size * sizeof(float);
    edata->feature_data->is_numerics = (bool *)(data + trace);
    trace += cond_sizes * sizeof(bool);
    edata->feature_data->inequality_directions = (bool *)(data + trace);
    trace += edge_size * sizeof(bool);
    edata->feature_mappings->mapping_numerics = (bool *)(data + trace);
    trace += metadata->input_dim * sizeof(bool);
    edata->feature_data->categorical_values = (char *)(data + trace);

    metadata->max_trees = metadata->n_trees;
    metadata->max_leaves = metadata->n_leaves;
    edata->alloc_data_size = data_size;
    return edata;
}

ensembleData* ensemble_compressed_data_alloc_cuda(ensembleMetaData *metadata, const int n_compressed_leaves, const int n_compressed_trees){
    // Same function as normal alloc just allocates exact amount
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        return nullptr;
    }

    // Allocate sub-structs on CPU (they hold pointers to GPU memory)
    edata->ensemble_info = new ensembleInfo;
    edata->leaf_data = new leafData;
    edata->feature_data = new featureData;
    edata->mono_constraints = new monotonicConstraints;
    edata->mono_constraints->n_constraints = 0;  // Initialize to 0
    edata->feature_mappings = new featureMapping;

    char *data;
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t feature_mapping_size = metadata->input_dim * sizeof(int);
    size_t feature_size = metadata->input_dim * sizeof(float);
    size_t tree_size = n_compressed_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? n_compressed_trees : n_compressed_leaves;
    size_t value_sizes = metadata->output_dim * n_compressed_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = n_compressed_leaves*metadata->max_depth;
    size_t mono_size = metadata->n_mono_constraints * sizeof(int);

    size_t data_size = bias_size
                     + feature_mapping_size * 3  // 3 int arrays: feature_mapping, reverse_num_feature_mapping, reverse_cat_feature_mapping
                     + feature_size  // feature_data->feature_weights
                     + tree_size
                     + split_sizes * sizeof(int) // depths
                     + value_sizes 
                     + mono_size * 3 // monotonic constraints: feature_idx, output_idx, constraint
                     + edge_size * (sizeof(bool) + sizeof(float)) // inequality directions + edge_weights
                     + sizeof(bool) * metadata->input_dim  // 1 bool array: mapping_numerics
                     + cond_sizes * (sizeof(int) + sizeof(float) + sizeof(bool) + sizeof(char)*MAX_CHAR_SIZE); 
#ifdef DEBUG 
    size_t sample_size = n_compressed_leaves * sizeof(int);
    data_size += sample_size;
#endif
    cudaError_t alloc_error = allocateCudaMemory((void**)&data, data_size, "when trying to allocate memory for ensemble_compressed_data");
    if (alloc_error != cudaSuccess) {
        return nullptr;
    }
    cudaMemset(data, 0, data_size);
    size_t trace = 0;
    
    // Keep original allocation ordering: float first, then int, then char/bool
    edata->bias = (float*)(data + trace);
    trace += bias_size;
    // Feature mapping arrays (4 total: 3 int, 1 bool)
    edata->feature_mappings->feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_mappings->reverse_num_feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_mappings->reverse_cat_feature_mapping = (int*)(data + trace);
    trace += feature_mapping_size;
    edata->feature_data->feature_weights = (float*)(data + trace);
    trace += feature_size;
#ifdef DEBUG 
    edata->n_samples = (int *)(data + trace);
    trace += sample_size;
#endif
    edata->ensemble_info->tree_indices = (int *)(data + trace);
    trace += tree_size;
    edata->ensemble_info->depths = (int *)(data + trace);
    trace += split_sizes * sizeof(int);
    edata->leaf_data->values = (float *)(data + trace);
    trace += value_sizes;
    edata->feature_data->feature_indices = (int *)(data + trace);
    trace += cond_sizes * sizeof(int);
    edata->mono_constraints->feature_idx = (int *)(data + trace);
    trace += mono_size;
    edata->mono_constraints->output_idx = (int *)(data + trace);
    trace += mono_size;
    edata->mono_constraints->constraint = (int *)(data + trace);
    trace += mono_size;
    edata->feature_data->feature_values = (float *)(data + trace);
    trace += cond_sizes * sizeof(float);
    edata->leaf_data->edge_weights = (float *)(data + trace);
    trace += edge_size * sizeof(float);
    edata->feature_data->is_numerics = (bool *)(data + trace);
    trace += cond_sizes * sizeof(bool);
    edata->feature_data->inequality_directions = (bool *)(data + trace);
    trace += edge_size * sizeof(bool);
    edata->feature_mappings->mapping_numerics = (bool *)(data + trace);
    trace += metadata->input_dim * sizeof(bool);
    edata->feature_data->categorical_values = (char *)(data + trace);

    metadata->n_trees = n_compressed_trees; 
    metadata->n_leaves = n_compressed_leaves; 
    edata->alloc_data_size = data_size;
    return edata;
}

ensembleData* ensemble_data_copy_gpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData *edata){
    if (edata == nullptr)
        edata = ensemble_copy_data_alloc_cuda(metadata);
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t feature_size = metadata->input_dim * sizeof(float);
    size_t feature_mapping_size = metadata->input_dim * sizeof(int);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->n_leaves*metadata->max_depth;
    size_t mono_size = metadata->n_mono_constraints * sizeof(int);

    cudaMemcpy(edata->bias, other_edata->bias, bias_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->feature_mapping, other_edata->feature_mappings->feature_mapping, feature_mapping_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->reverse_num_feature_mapping, other_edata->feature_mappings->reverse_num_feature_mapping, feature_mapping_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->reverse_cat_feature_mapping, other_edata->feature_mappings->reverse_cat_feature_mapping, feature_mapping_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_data->feature_weights, other_edata->feature_data->feature_weights, feature_size, cudaMemcpyDeviceToDevice);
#ifdef DEBUG 
    size_t sample_size = metadata->n_leaves * sizeof(int);
    cudaMemcpy(edata->n_samples, other_edata->n_samples, sample_size, cudaMemcpyDeviceToDevice);
#endif
    cudaMemcpy(edata->ensemble_info->tree_indices, other_edata->ensemble_info->tree_indices, tree_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->ensemble_info->depths, other_edata->ensemble_info->depths, split_sizes * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->leaf_data->values, other_edata->leaf_data->values, value_sizes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->mono_constraints->feature_idx, other_edata->mono_constraints->feature_idx, mono_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->mono_constraints->output_idx, other_edata->mono_constraints->output_idx, mono_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->mono_constraints->constraint, other_edata->mono_constraints->constraint, mono_size, cudaMemcpyDeviceToDevice);
    // CUDA split scoring reads this field, not the metadata copy, so a copy
    // that leaves it at 0 silently disables constraint-aware scoring.
    edata->mono_constraints->n_constraints = metadata->n_mono_constraints;
    cudaMemcpy(edata->feature_data->feature_indices, other_edata->feature_data->feature_indices, cond_sizes * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_data->feature_values, other_edata->feature_data->feature_values, cond_sizes * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->leaf_data->edge_weights, other_edata->leaf_data->edge_weights, edge_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_data->is_numerics, other_edata->feature_data->is_numerics, cond_sizes * sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_data->inequality_directions, other_edata->feature_data->inequality_directions, edge_size * sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->mapping_numerics, other_edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_data->categorical_values, other_edata->feature_data->categorical_values, cond_sizes * sizeof(char) * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice); 
    return edata;
}

ensembleData* ensemble_data_copy_gpu_cpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData *edata){
    if (edata == nullptr)
        edata = ensemble_copy_data_alloc(metadata);
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t feature_size = metadata->input_dim * sizeof(float);
    size_t feature_mapping_size = metadata->input_dim * sizeof(int);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t edge_size = metadata->n_leaves*metadata->max_depth;
    size_t mono_size = metadata->n_mono_constraints * sizeof(int);
    
    cudaMemcpy(edata->bias, other_edata->bias, bias_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_data->feature_weights, other_edata->feature_data->feature_weights, feature_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_mappings->feature_mapping, other_edata->feature_mappings->feature_mapping, feature_mapping_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_mappings->reverse_num_feature_mapping, other_edata->feature_mappings->reverse_num_feature_mapping, feature_mapping_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_mappings->reverse_cat_feature_mapping, other_edata->feature_mappings->reverse_cat_feature_mapping, feature_mapping_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_mappings->mapping_numerics, other_edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool), cudaMemcpyDeviceToHost);
#ifdef DEBUG 
    size_t sample_size = metadata->n_leaves * sizeof(int);
    cudaMemcpy(edata->n_samples, other_edata->n_samples, sample_size, cudaMemcpyDeviceToHost);
#endif
    cudaMemcpy(edata->ensemble_info->tree_indices, other_edata->ensemble_info->tree_indices, tree_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->ensemble_info->depths, other_edata->ensemble_info->depths, split_sizes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->leaf_data->values, other_edata->leaf_data->values, value_sizes, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->mono_constraints->feature_idx, other_edata->mono_constraints->feature_idx, mono_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->mono_constraints->output_idx, other_edata->mono_constraints->output_idx, mono_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->mono_constraints->constraint, other_edata->mono_constraints->constraint, mono_size, cudaMemcpyDeviceToHost);
    // CUDA split scoring reads this field, not the metadata copy, so a copy
    // that leaves it at 0 silently disables constraint-aware scoring.
    edata->mono_constraints->n_constraints = metadata->n_mono_constraints;
    cudaMemcpy(edata->feature_data->feature_indices, other_edata->feature_data->feature_indices, cond_sizes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_data->feature_values, other_edata->feature_data->feature_values, cond_sizes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->leaf_data->edge_weights, other_edata->leaf_data->edge_weights, edge_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_data->is_numerics, other_edata->feature_data->is_numerics, cond_sizes * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_data->inequality_directions, other_edata->feature_data->inequality_directions, edge_size * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(edata->feature_data->categorical_values, other_edata->feature_data->categorical_values, cond_sizes * sizeof(char) * MAX_CHAR_SIZE, cudaMemcpyDeviceToHost); 
    return edata;
}

ensembleData* ensemble_data_copy_cpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData *edata){
    if (edata == nullptr)
        edata = ensemble_copy_data_alloc_cuda(metadata);
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t feature_size = metadata->input_dim * sizeof(float);
    size_t feature_mapping_size = metadata->input_dim * sizeof(int);
    size_t tree_size = metadata->n_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
    size_t value_sizes = metadata->output_dim * metadata->n_leaves * sizeof(float);
    size_t cond_sizes = split_sizes*metadata->max_depth;
    size_t mono_size = metadata->n_mono_constraints * sizeof(int);
    size_t edge_size = metadata->n_leaves*metadata->max_depth;
    cudaMemcpy(edata->bias, other_edata->bias, bias_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_mappings->feature_mapping, other_edata->feature_mappings->feature_mapping, feature_mapping_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_mappings->reverse_num_feature_mapping, other_edata->feature_mappings->reverse_num_feature_mapping, feature_mapping_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_mappings->reverse_cat_feature_mapping, other_edata->feature_mappings->reverse_cat_feature_mapping, feature_mapping_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_data->feature_weights, other_edata->feature_data->feature_weights, feature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_mappings->mapping_numerics, other_edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool), cudaMemcpyHostToDevice);
#ifdef DEBUG 
    size_t sample_size = metadata->n_leaves * sizeof(int);
    cudaMemcpy(edata->n_samples, other_edata->n_samples, sample_size, cudaMemcpyHostToDevice);
#endif
    cudaMemcpy(edata->mono_constraints->feature_idx, other_edata->mono_constraints->feature_idx, mono_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->mono_constraints->output_idx, other_edata->mono_constraints->output_idx, mono_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->mono_constraints->constraint, other_edata->mono_constraints->constraint, mono_size, cudaMemcpyHostToDevice);
    // CUDA split scoring reads this field, not the metadata copy, so a copy
    // that leaves it at 0 silently disables constraint-aware scoring.
    edata->mono_constraints->n_constraints = metadata->n_mono_constraints;
    cudaMemcpy(edata->ensemble_info->tree_indices, other_edata->ensemble_info->tree_indices, tree_size, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->ensemble_info->depths, other_edata->ensemble_info->depths, split_sizes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->leaf_data->values, other_edata->leaf_data->values, value_sizes, cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_data->feature_indices, other_edata->feature_data->feature_indices, cond_sizes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_data->feature_values, other_edata->feature_data->feature_values, cond_sizes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->leaf_data->edge_weights, other_edata->leaf_data->edge_weights, edge_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_data->is_numerics, other_edata->feature_data->is_numerics, cond_sizes * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_data->inequality_directions, other_edata->feature_data->inequality_directions, edge_size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(edata->feature_data->categorical_values, other_edata->feature_data->categorical_values, cond_sizes * sizeof(char) * MAX_CHAR_SIZE, cudaMemcpyHostToDevice); 
    return edata;
}

ensembleData* ensemble_compressed_data_copy_gpu_gpu(ensembleMetaData *metadata, ensembleData *other_edata, ensembleData *edata, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices){
    if (edata == nullptr)
        edata = ensemble_compressed_data_alloc_cuda(metadata, n_compressed_leaves, n_compressed_trees);
    size_t bias_size = metadata->output_dim * sizeof(float);
    size_t feature_size = metadata->input_dim * sizeof(float);
    size_t feature_mapping_size = metadata->input_dim * sizeof(int);
    size_t tree_size = n_compressed_trees * sizeof(int);
    size_t split_sizes = (metadata->grow_policy == OBLIVIOUS) ? n_compressed_trees : n_compressed_leaves;
    size_t data_size = sizeof(int)*(n_compressed_leaves + n_compressed_trees); 
    int *data;
    int *leaf_indices_gpu;
    int *tree_indices_gpu;

    cudaError_t alloc_error = allocateCudaMemory((void**)&data, data_size, "when trying to allocate memory for ensemble_copy_data_cpu_gpu");
    if (alloc_error != cudaSuccess) {
        return nullptr;
    }
    size_t trace = 0;
    leaf_indices_gpu = (int *)(data + trace);
    trace += n_compressed_leaves;
    tree_indices_gpu = (int *)(data + trace);
    cudaMemcpy(leaf_indices_gpu, leaf_indices, sizeof(int)*(n_compressed_leaves), cudaMemcpyHostToDevice);
    cudaMemcpy(tree_indices_gpu, tree_indices, sizeof(int)*(n_compressed_trees), cudaMemcpyHostToDevice);
    const int *split_indices = (metadata->grow_policy == OBLIVIOUS) ? tree_indices_gpu : leaf_indices_gpu;
    cudaMemcpy(edata->bias, other_edata->bias, bias_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->feature_mapping, other_edata->feature_mappings->feature_mapping, feature_mapping_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->reverse_num_feature_mapping, other_edata->feature_mappings->reverse_num_feature_mapping, feature_mapping_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->reverse_cat_feature_mapping, other_edata->feature_mappings->reverse_cat_feature_mapping, feature_mapping_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_mappings->mapping_numerics, other_edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool), cudaMemcpyDeviceToDevice);
    cudaMemcpy(edata->feature_data->feature_weights, other_edata->feature_data->feature_weights, feature_size, cudaMemcpyDeviceToDevice);
#ifdef DEBUG 
    size_t sample_size = n_compressed_leaves * sizeof(int);
    cudaMemcpy(edata->n_samples, other_edata->n_samples, sample_size, cudaMemcpyDeviceToDevice);
#endif
    cudaMemcpy(edata->ensemble_info->tree_indices, new_tree_indices, tree_size, cudaMemcpyHostToDevice);
    int n_blocks = split_sizes / THREADS_PER_BLOCK + 1;
    selective_copyi<<<n_blocks, THREADS_PER_BLOCK>>>(split_sizes, split_indices, edata->ensemble_info->depths, other_edata->ensemble_info->depths, 1);

    selective_copyi<<<n_blocks, THREADS_PER_BLOCK>>>(split_sizes, split_indices, edata->feature_data->feature_indices, other_edata->feature_data->feature_indices, metadata->max_depth);
    selective_copyf<<<n_blocks, THREADS_PER_BLOCK>>>(split_sizes, split_indices, edata->feature_data->feature_values, other_edata->feature_data->feature_values, metadata->max_depth);
    selective_copyb<<<n_blocks, THREADS_PER_BLOCK>>>(split_sizes, split_indices, edata->feature_data->is_numerics, other_edata->feature_data->is_numerics, metadata->max_depth);
    selective_copyc<<<n_blocks, THREADS_PER_BLOCK>>>(split_sizes, split_indices, edata->feature_data->categorical_values, other_edata->feature_data->categorical_values, metadata->max_depth);

    n_blocks = n_compressed_leaves / THREADS_PER_BLOCK + 1;
    selective_copyf<<<n_blocks, THREADS_PER_BLOCK>>>(n_compressed_leaves, leaf_indices_gpu, edata->leaf_data->values, other_edata->leaf_data->values, metadata->output_dim);
    selective_copyf<<<n_blocks, THREADS_PER_BLOCK>>>(n_compressed_leaves, leaf_indices_gpu, edata->leaf_data->edge_weights, other_edata->leaf_data->edge_weights, metadata->max_depth);
    selective_copyb<<<n_blocks, THREADS_PER_BLOCK>>>(n_compressed_leaves, leaf_indices_gpu, edata->feature_data->inequality_directions, other_edata->feature_data->inequality_directions, metadata->max_depth);
    cudaDeviceSynchronize();
    cudaFree(data);
    return edata;
}

void ensemble_data_dealloc_cuda(ensembleData *edata){
    cudaFree(edata->bias);
    // Delete sub-structs allocated on CPU
    delete edata->ensemble_info;
    delete edata->leaf_data;
    delete edata->feature_data;
    delete edata->mono_constraints;
    delete edata->feature_mappings;
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
        data_alloc_size += sizeof(float) * n_candidates * 2;
    if (metadata->grow_policy == OBLIVIOUS)
        data_alloc_size += sizeof(float) * n_candidates * nodes_per_evaluation;

    char *data_alloc;
    cudaError_t err = allocateCudaMemory((void**)&data_alloc, data_alloc_size, "when trying to allocate memory for allocate_split_data");
    if (err != cudaSuccess) {
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

    split_data->left_dot = nullptr;
    split_data->right_dot = nullptr;

    if (metadata->split_score_func == Cosine){
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
        cudaMemcpy(new_data->bias, edata->bias, metadata->output_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->feature_mappings->feature_mapping, edata->feature_mappings->feature_mapping, metadata->input_dim * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->feature_mappings->reverse_num_feature_mapping, edata->feature_mappings->reverse_num_feature_mapping, metadata->input_dim * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->feature_mappings->reverse_cat_feature_mapping, edata->feature_mappings->reverse_cat_feature_mapping, metadata->input_dim * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->feature_data->feature_weights, edata->feature_data->feature_weights, metadata->input_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->feature_mappings->mapping_numerics, edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool), cudaMemcpyDeviceToDevice);
#ifdef DEBUG
        cudaMemcpy(new_data->n_samples, edata->n_samples, leaf_idx * sizeof(int), cudaMemcpyDeviceToDevice);
#endif 
        cudaMemcpy(new_data->leaf_data->values, edata->leaf_data->values, leaf_idx * metadata->output_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->ensemble_info->tree_indices, edata->ensemble_info->tree_indices, tree_idx * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->feature_data->inequality_directions, edata->feature_data->inequality_directions, leaf_idx * metadata->max_depth * sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->leaf_data->edge_weights, edata->leaf_data->edge_weights, leaf_idx * metadata->max_depth * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->mono_constraints->feature_idx, edata->mono_constraints->feature_idx, metadata->n_mono_constraints * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->mono_constraints->output_idx, edata->mono_constraints->output_idx, metadata->n_mono_constraints * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_data->mono_constraints->constraint, edata->mono_constraints->constraint, metadata->n_mono_constraints * sizeof(int), cudaMemcpyDeviceToDevice);
        // edata is the container that survives: new_data->mono_constraints is
        // deleted below once its arrays have been handed over.  CUDA split
        // scoring reads this field rather than the metadata copy, so it must be
        // set on edata or constraint-aware scoring silently stops after growth.
        edata->mono_constraints->n_constraints = metadata->n_mono_constraints;
        if (metadata->grow_policy == GREEDY){
            cudaMemcpy(new_data->ensemble_info->depths, edata->ensemble_info->depths, leaf_idx * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->feature_indices, edata->feature_data->feature_indices, leaf_idx * metadata->max_depth * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->feature_values, edata->feature_data->feature_values, leaf_idx * metadata->max_depth * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->is_numerics, edata->feature_data->is_numerics, leaf_idx * metadata->max_depth * sizeof(bool), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->categorical_values, edata->feature_data->categorical_values, leaf_idx * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpy(new_data->ensemble_info->depths, edata->ensemble_info->depths, tree_idx * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->feature_indices, edata->feature_data->feature_indices, tree_idx * metadata->max_depth * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->feature_values, edata->feature_data->feature_values, tree_idx * metadata->max_depth * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->is_numerics, edata->feature_data->is_numerics, tree_idx * metadata->max_depth * sizeof(bool), cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_data->feature_data->categorical_values, edata->feature_data->categorical_values, tree_idx * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);
        }
        cudaFree(edata->bias);
        edata->bias = new_data->bias;
        edata->feature_data->feature_weights = new_data->feature_data->feature_weights;
        edata->feature_mappings->reverse_num_feature_mapping = new_data->feature_mappings->reverse_num_feature_mapping;
        edata->feature_mappings->reverse_cat_feature_mapping = new_data->feature_mappings->reverse_cat_feature_mapping;
        edata->feature_mappings->mapping_numerics = new_data->feature_mappings->mapping_numerics;
        edata->feature_mappings->feature_mapping = new_data->feature_mappings->feature_mapping;
#ifdef DEBUG
        edata->n_samples = new_data->n_samples;
#endif
        edata->ensemble_info->depths = new_data->ensemble_info->depths;
        edata->ensemble_info->tree_indices = new_data->ensemble_info->tree_indices;
        edata->leaf_data->values = new_data->leaf_data->values;
        edata->feature_data->inequality_directions = new_data->feature_data->inequality_directions;
        edata->feature_data->feature_indices = new_data->feature_data->feature_indices;
        edata->feature_data->feature_values = new_data->feature_data->feature_values;
        edata->mono_constraints->feature_idx = new_data->mono_constraints->feature_idx;
        edata->mono_constraints->output_idx = new_data->mono_constraints->output_idx;
        edata->mono_constraints->constraint = new_data->mono_constraints->constraint;
        edata->leaf_data->edge_weights = new_data->leaf_data->edge_weights;
        edata->feature_data->is_numerics = new_data->feature_data->is_numerics;
        edata->feature_data->categorical_values = new_data->feature_data->categorical_values;
        
        // Delete only the sub-struct containers from new_data (not arrays - they're now owned by edata)
        delete new_data->ensemble_info;
        delete new_data->leaf_data;
        delete new_data->feature_data;
        delete new_data->mono_constraints;
        delete new_data->feature_mappings;
        delete new_data;
    }
}

