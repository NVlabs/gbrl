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
 * @file cuda_compression.cu
 * @brief CUDA kernels for GPU-accelerated tree ensemble compression
 * 
 * @warning EXPERIMENTAL - UNDER ACTIVE RESEARCH
 * This module implements tree ensemble compression algorithms that are currently
 * under active research and development. The API, behavior, and results may change
 * without notice. This feature has not been fully validated and is intended for
 * internal research use only.
 * 
 * Implements GPU-accelerated versions of tree compression operations including
 * matrix representation generation and ensemble compression. Provides significant
 * speedup over CPU implementation for large ensembles.
 */

#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>

#include "cuda_compression.h"
#include "cuda_types.h"
#include "cuda_utils.h"

/**
 * @brief Generate matrix representation of tree ensemble on GPU
 * 
 * CUDA implementation that generates binary activation matrix A and scaled
 * value matrix V for the entire tree ensemble. Dispatches appropriate kernel
 * based on tree type (greedy/oblivious) and feature types (numerical/mixed).
 * 
 * @param dataset Input dataset (observations must already be on GPU)
 * @param metadata Ensemble metadata
 * @param edata Ensemble data on GPU
 * @param opts Array of GPU optimizer structures
 * @param n_opts Number of optimizers
 * @param matrix Output matrix representation (allocated on CPU)
 */
void get_matrix_representation_cuda(dataSet *dataset, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, matrixRepresentation *matrix){
    int n_samples = dataset->n_samples;
    int output_dim = metadata->output_dim;
    float *device_batch_obs;
    char *device_batch_cat_obs;
    char *device_data;
    float *device_V;
    bool *device_A;
    // assuming row-major order
    size_t A_size = dataset->n_samples * (metadata->n_leaves+1) * sizeof(bool);
    size_t V_size = (metadata->n_leaves+1) * output_dim * sizeof(float);
    size_t obs_matrix_size = dataset->n_samples * metadata->n_num_features * sizeof(float);
    size_t cat_obs_matrix_size = dataset->n_samples * metadata->n_cat_features * sizeof(char) * MAX_CHAR_SIZE;
    
    // Calculate allocation size based on what data is already on device
    size_t extra_alloc_size = 0;
    bool obs_on_device = (dataset->obs != nullptr && dataset->obs->data != nullptr && dataset->obs->device != cpu);
    bool cat_on_device = (dataset->categorical_obs != nullptr && dataset->categorical_obs->data != nullptr && dataset->categorical_obs->device != cpu);
    
    if (!obs_on_device) extra_alloc_size += obs_matrix_size;
    if (!cat_on_device) extra_alloc_size += cat_obs_matrix_size;
    
    cudaError_t alloc_error = allocateCudaMemory((void**)&device_data, extra_alloc_size + A_size + V_size, "when trying to allocate matrix representation");
    if (alloc_error != cudaSuccess) {
        return;
    }
    cudaMemset(device_data, 0, extra_alloc_size + A_size + V_size);

    size_t trace = 0;
    device_V = (float *)(device_data + trace);
    trace += V_size;
    device_A = (bool *)(device_data + trace);
    trace += A_size;
    
    // Handle obs data - device-aware copy
    if (dataset->obs != nullptr && dataset->obs->data != nullptr) {
        if (obs_on_device) {
            device_batch_obs = const_cast<float*>(dataset->obs->data);
        } else {
            device_batch_obs = (float*)(device_data + trace);
            trace += obs_matrix_size;
            cudaMemcpy(device_batch_obs, dataset->obs->data, obs_matrix_size, cudaMemcpyHostToDevice);
        }
    } else {
        device_batch_obs = nullptr;
    }
    
    // Handle categorical obs data - device-aware copy
    if (dataset->categorical_obs != nullptr && dataset->categorical_obs->data != nullptr) {
        if (cat_on_device) {
            device_batch_cat_obs = const_cast<char*>(dataset->categorical_obs->data);
        } else {
            device_batch_cat_obs = (char*)(device_data + trace);
            cudaMemcpy(device_batch_cat_obs, dataset->categorical_obs->data, cat_obs_matrix_size, cudaMemcpyHostToDevice);
        }
    } else {
        device_batch_cat_obs = nullptr;
    }
    
    int n_blocks, threads_per_block;
    get_grid_dimensions(dataset->n_samples, n_blocks, threads_per_block);
    
    // Validate that required feature buffers exist before proceeding
    if (metadata->n_num_features > 0 && device_batch_obs == nullptr) {
        std::cerr << "ERROR: Numerical features expected but dataset obs buffer is null" << std::endl;
        matrix->A = nullptr;
        matrix->V = nullptr;
        matrix->n_leaves = 0;
        cudaFree(device_data);
        return;
    }
    if (metadata->n_cat_features > 0 && device_batch_cat_obs == nullptr) {
        std::cerr << "ERROR: Categorical features expected but dataset categorical_obs buffer is null" << std::endl;
        matrix->A = nullptr;
        matrix->V = nullptr;
        matrix->n_leaves = 0;
        cudaFree(device_data);
        return;
    }
    
    cudaMemcpy(device_V, edata->bias, sizeof(float)*output_dim, cudaMemcpyDeviceToDevice);
    
    if (n_opts == 0){
        std::cerr << "No optimizers." << std::endl;
        matrix->A = nullptr;
        matrix->V = nullptr;
        matrix->n_leaves = 0;
        cudaFree(device_data);
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Replace 0 with your device ID if you have multiple devices

    threads_per_block = WARP_SIZE*((dataset->n_samples + WARP_SIZE - 1) / WARP_SIZE);
    if (threads_per_block > deviceProp.maxThreadsPerBlock)
        threads_per_block = deviceProp.maxThreadsPerBlock;

    if (metadata->grow_policy == GREEDY){
        if (metadata->n_cat_features == 0)
            get_representation_kernel_numerical_only<<<metadata->n_leaves, threads_per_block>>>(device_batch_obs, dataset->n_samples, metadata->n_num_features, edata->feature_data->feature_indices, edata->ensemble_info->depths, edata->feature_data->feature_values, edata->feature_data->inequality_directions, edata->leaf_data->values, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
        else
            get_representation_kernel_tree_wise<<<metadata->n_leaves, threads_per_block>>>(device_batch_obs, device_batch_cat_obs, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_data->feature_indices, edata->ensemble_info->depths, edata->feature_data->feature_values, edata->feature_data->inequality_directions, edata->leaf_data->values, edata->feature_data->categorical_values, edata->feature_data->is_numerics, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
    } else{
        
        if (metadata->n_cat_features == 0)
            get_representation_oblivious_kernel_numerical_only<<<metadata->n_trees, threads_per_block>>>(device_batch_obs, dataset->n_samples, metadata->n_num_features, edata->feature_data->feature_indices, edata->ensemble_info->depths, edata->feature_data->feature_values, edata->feature_data->inequality_directions, edata->leaf_data->values, edata->ensemble_info->tree_indices, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
        else
            get_representation_oblivious_kernel_tree_wise<<<metadata->n_trees, threads_per_block>>>(device_batch_obs, device_batch_cat_obs, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_data->feature_indices, edata->ensemble_info->depths, edata->feature_data->feature_values, edata->feature_data->inequality_directions, edata->leaf_data->values, edata->ensemble_info->tree_indices, edata->feature_data->categorical_values, edata->feature_data->is_numerics, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
    }
    cudaDeviceSynchronize();
    n_blocks = metadata->n_leaves / THREADS_PER_BLOCK + 1; 
    get_V_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(device_V, edata->leaf_data->values, opts, n_opts, metadata->output_dim, metadata->n_leaves);
    cudaDeviceSynchronize();
    // Allocate by element count, not byte size
    int A_elems = n_samples * (metadata->n_leaves + 1);
    int V_elems = (metadata->n_leaves + 1) * output_dim;
    matrix->A = new bool[A_elems];
    cudaMemcpy(matrix->A, device_A, A_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_samples; i++)
        matrix->A[i*(metadata->n_leaves + 1)] = true;
    matrix->V = new float[V_elems];
    cudaMemcpy(matrix->V, device_V, V_size, cudaMemcpyDeviceToHost);
    // Copy results back to CPU
    matrix->n_leaves = metadata->n_leaves;
    cudaFree(device_data);
    int *tree_indices = new int[metadata->n_trees];
    cudaMemcpy(tree_indices, edata->ensemble_info->tree_indices,  sizeof(int)*metadata->n_trees, cudaMemcpyDeviceToHost);
    matrix->n_leaves_per_tree = new int[metadata->n_trees];
    for (int i = 0; i < metadata->n_trees - 1; ++i )
        matrix->n_leaves_per_tree[i] = tree_indices[i+1] - tree_indices[i];
    matrix->n_leaves_per_tree[metadata->n_trees - 1] = metadata->n_leaves - tree_indices[metadata->n_trees - 1];
    matrix->n_trees = metadata->n_trees;
    delete[] tree_indices;
}

/**
 * @brief Compress tree ensemble on GPU by selecting subset and applying correction
 * 
 * Creates compressed ensemble with selected subset of trees and leaves, then applies
 * correction matrix W to maintain prediction accuracy. Uses GPU kernels for efficient
 * matrix operations. Original ensemble is reused to avoid memory fragmentation.
 * 
 * @param metadata Ensemble metadata (modified to reflect compressed dimensions)
 * @param edata Ensemble data on GPU (will be modified in-place)
 * @param opts Array of GPU optimizer structures
 * @param n_opts Number of optimizers
 * @param n_compressed_leaves Number of leaves in compressed ensemble
 * @param n_compressed_trees Number of trees in compressed ensemble
 * @param leaf_indices CPU array of leaf indices to retain
 * @param tree_indices CPU array of tree indices to retain
 * @param new_tree_indices CPU array of new starting indices for trees
 * @param W CPU array containing correction matrix (shape: n_compressed_leaves+1 x output_dim)
 * @return Pointer to compressed ensemble data on GPU
 */
ensembleData * compress_ensemble_cuda(ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W){
    // First create the compressed ensemble (this updates metadata->n_leaves to n_compressed_leaves)
    ensembleData* compressed_edata = ensemble_compressed_data_copy_gpu_gpu(metadata, edata, nullptr, n_compressed_leaves, n_compressed_trees, leaf_indices, tree_indices, new_tree_indices);
    
    // Check for allocation failure
    if (compressed_edata == nullptr) {
        std::cerr << "ERROR: Failed to create compressed ensemble data" << std::endl;
        return nullptr;
    }
    
    // Now apply W correction matrix to the compressed ensemble values
    // W is sized (n_compressed_leaves + 1, output_dim) which matches metadata->n_leaves after compression
    float *device_W;
    size_t W_size = (n_compressed_leaves + 1) * metadata->output_dim * sizeof(float);
    cudaError_t alloc_error = allocateCudaMemory((void**)&device_W, W_size, "when trying to allocate compress ensemble W");
    if (alloc_error != cudaSuccess) {
        ensemble_data_dealloc_cuda(compressed_edata);
        return nullptr;
    }
    cudaMemcpy(device_W, W, W_size, cudaMemcpyHostToDevice);
    int n_blocks = (n_compressed_leaves + 1) / THREADS_PER_BLOCK + 1;
    add_W_matrix_to_values_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(device_W, compressed_edata->leaf_data->values, compressed_edata->bias, opts, n_opts, n_compressed_leaves, metadata->output_dim);
    cudaDeviceSynchronize();
    cudaFree(device_W);

    // Reset original data and copy compressed data into it
    // This strategy avoids memory fragmentation
    cudaMemset(edata->bias, 0, edata->alloc_data_size);
    edata = ensemble_data_copy_gpu_gpu(metadata, compressed_edata, edata);
    ensemble_data_dealloc_cuda(compressed_edata);
    return edata;
}

/**
 * @brief CUDA kernel for oblivious tree representation with mixed features
 * 
 * Computes binary activation matrix for oblivious trees supporting both
 * numerical and categorical features. Each block processes one tree, threads
 * process samples in parallel. Uses binary indexing for efficient leaf lookup.
 * 
 * @param obs Numerical observations on GPU
 * @param categorical_obs Categorical observations on GPU
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param feature_indices Feature indices for split conditions
 * @param depths Tree depths
 * @param feature_values Split threshold values
 * @param inequality_directions Split directions (> or ==)
 * @param leaf_values Leaf values (unused in representation)
 * @param tree_indices Starting leaf index for each tree
 * @param categorical_values Categorical split values
 * @param is_numerics Boolean array indicating feature types
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param n_leaves Total number of leaves
 * @param A Output binary activation matrix
 */
__global__ void get_representation_oblivious_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features, 
                                                   const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values,
                                                   const int* __restrict__ tree_indices, const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, const int output_dim, const int max_depth,
                                                   const int n_leaves, bool* __restrict__ A){
    bool decision;
    int tree_idx = blockIdx.x;
    int leaf_idx, initial_leaf_idx = __ldg(tree_indices + tree_idx);
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        leaf_idx = 0;
        for (int depth_idx = 0; depth_idx < __ldg(depths + tree_idx); depth_idx++){ 
            if (is_numerics[tree_idx * max_depth + depth_idx])
                decision = (__ldg(&obs[sample_idx*n_num_features + __ldg(feature_indices + tree_idx * max_depth + depth_idx)])) > (__ldg(feature_values + tree_idx * max_depth + depth_idx));
            else{
                decision = true;
                for (int i = 0; i < MAX_CHAR_SIZE; ++i) {
                    if (categorical_values[(tree_idx * max_depth + depth_idx)*MAX_CHAR_SIZE + i] != categorical_obs[(sample_idx*n_cat_features + __ldg(feature_indices + tree_idx * max_depth + depth_idx))*MAX_CHAR_SIZE + i]){
                        decision = false;
                        break;
                    } else if (categorical_values[(tree_idx * max_depth + depth_idx)*MAX_CHAR_SIZE + i] == '\0' || categorical_obs[(sample_idx*n_cat_features + __ldg(feature_indices + tree_idx * max_depth + depth_idx))*MAX_CHAR_SIZE + i] == '\0')
                        break;
                }
            }
            leaf_idx |= (decision <<  (__ldg(depths + tree_idx) - 1 - depth_idx));
        }
        A[sample_idx*(n_leaves + 1) + leaf_idx + 1 + initial_leaf_idx] = true;
    }
}

/**
 * @brief CUDA kernel for oblivious tree representation with numerical features only
 * 
 * Optimized version for oblivious trees with only numerical features. Avoids
 * categorical feature checks for better performance. Each block handles one tree.
 * 
 * @param obs Numerical observations on GPU
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param feature_indices Feature indices for split conditions
 * @param depths Tree depths
 * @param feature_values Split threshold values
 * @param inequality_directions Split directions
 * @param leaf_values Leaf values (unused)
 * @param tree_indices Starting leaf index for each tree
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param n_leaves Total number of leaves
 * @param A Output binary activation matrix
 */
__global__ void get_representation_oblivious_kernel_numerical_only(const float* __restrict__ obs, const int n_samples, const int n_num_features, 
                                                        const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values,
                                                        const int* __restrict__ tree_indices, const int output_dim, const int max_depth,
                                                        const int n_leaves, bool* __restrict__ A){
    
    int leaf_idx, initial_leaf_idx = __ldg(tree_indices + blockIdx.x);
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        leaf_idx = 0;
        for (int depth_idx = 0; depth_idx < __ldg(depths + blockIdx.x); depth_idx++){ 
            bool decision = (__ldg(&obs[sample_idx*n_num_features + __ldg(feature_indices + blockIdx.x * max_depth + depth_idx)])) > (__ldg(feature_values + blockIdx.x * max_depth + depth_idx));
            leaf_idx |= (decision << (__ldg(depths + blockIdx.x) - 1 - depth_idx));
        }
        A[sample_idx*(n_leaves + 1) + leaf_idx + initial_leaf_idx + 1] = true;
    }
}

/**
 * @brief CUDA kernel for greedy tree representation with numerical features only
 * 
 * Processes greedy (non-oblivious) trees with only numerical features. Each
 * block handles one leaf, traversing split conditions from leaf to root to
 * determine if samples reach that leaf.
 * 
 * @param obs Numerical observations on GPU
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param feature_indices Feature indices for split conditions
 * @param depths Depth of each leaf
 * @param feature_values Split threshold values
 * @param inequality_directions Split directions
 * @param leaf_values Leaf values (unused)
 * @param output_dim Output dimensionality
 * @param max_depth Maximum tree depth
 * @param n_leaves Total number of leaves
 * @param A Output binary activation matrix
 */
__global__ void get_representation_kernel_numerical_only(const float* __restrict__ obs, const int n_samples, const int n_num_features, const int* __restrict__ feature_indices,
                                              const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, 
                                              const int output_dim, const int max_depth,
                                              const int n_leaves, bool* __restrict__ A){
    int cond_idx = blockIdx.x * max_depth;
    int depth_idx; // Initialize mask to all bits set
    bool passed;
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        depth_idx = __ldg(depths + blockIdx.x) - 1;
        passed = true;
        while (depth_idx >= 0 && passed) {
            passed = (__ldg(&obs[sample_idx*n_num_features + __ldg(feature_indices + cond_idx + depth_idx)]) > __ldg(feature_values + cond_idx + depth_idx) == inequality_directions[cond_idx + depth_idx]);
            depth_idx--;
        }
        if (passed){
            A[sample_idx*(n_leaves + 1) + blockIdx.x + 1] = true;
        }
    }
}

/**
 * @brief CUDA kernel for greedy tree representation with mixed features
 * 
 * Processes greedy trees with both numerical and categorical features. Each
 * block processes one leaf, checking all split conditions to determine sample
 * membership. Handles string comparison for categorical features.
 * 
 * @param obs Numerical observations on GPU
 * @param categorical_obs Categorical observations on GPU
 * @param n_samples Number of samples
 * @param n_num_features Number of numerical features
 * @param n_cat_features Number of categorical features
 * @param feature_indices Feature indices
 * @param depths Leaf depths
 * @param feature_values Split thresholds
 * @param inequality_directions Split directions
 * @param leaf_values Leaf values (unused)
 * @param categorical_values Categorical split values
 * @param is_numerics Feature type indicators
 * @param output_dim Output dimensionality
 * @param max_depth Maximum depth
 * @param n_leaves Total leaves
 * @param A Output activation matrix
 */
__global__ void get_representation_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features, 
                                         const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, 
                                         const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, const int output_dim, const int max_depth,
                                         const int n_leaves, bool* __restrict__ A){
    
    bool equal, passed;
    int cond_idx = blockIdx.x * max_depth, depth_idx;
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        passed = true;
        depth_idx = __ldg(depths + blockIdx.x) - 1;
        while(depth_idx >= 0 && passed){
            if (is_numerics[cond_idx + depth_idx]){
                passed = __ldg(&obs[sample_idx*n_num_features + __ldg(&feature_indices[cond_idx + depth_idx])]) > __ldg(&feature_values[cond_idx + depth_idx]) == inequality_directions[cond_idx + depth_idx];
            } 
            else {
                equal = true;
                for (int i = 0; i < MAX_CHAR_SIZE; ++i) {
                    if (categorical_values[(cond_idx + depth_idx)*MAX_CHAR_SIZE + i] != categorical_obs[(sample_idx*n_cat_features + __ldg(&feature_indices[cond_idx + depth_idx]))*MAX_CHAR_SIZE + i]){
                        equal = false;
                        break;
                    } else if (categorical_values[(cond_idx + depth_idx)*MAX_CHAR_SIZE + i] == '\0' || categorical_obs[(sample_idx*n_cat_features + __ldg(&feature_indices[cond_idx + depth_idx]))*MAX_CHAR_SIZE + i] == '\0')
                        break;
                }
                passed = equal == inequality_directions[cond_idx + depth_idx];
            }
            depth_idx--;
        }
        if (passed){
            A[sample_idx*(n_leaves + 1) + blockIdx.x + 1] = true;
        }
    }
}

/**
 * @brief CUDA kernel to apply correction matrix W to leaf values
 * 
 * Subtracts scaled W values from leaf values (inverse scaling by learning rate)
 * and adds bias correction. Optimized for 1-2 optimizer common case with
 * unrolled loops.
 * 
 * @param W Correction matrix on GPU
 * @param leaf_values Leaf values to modify on GPU
 * @param bias Bias values on GPU
 * @param opts Array of GPU optimizer structures
 * @param n_opts Number of optimizers
 * @param n_leaves Number of leaves
 * @param output_dim Output dimensionality
 */
__global__ void add_W_matrix_to_values_kernel(const float * __restrict__ W, float* __restrict__ leaf_values, float* __restrict__ bias, SGDOptimizerGPU** opts, const int n_opts, const int n_leaves, const int output_dim){
    int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    if (idx < n_leaves){
        int value_idx = idx*output_dim;
        int offset_value = (idx + 1)*output_dim;
        if (n_opts == 1){
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                leaf_values[value_idx + i] -= __ldg(W + offset_value + i)  / opts[0]->init_lr;
            }
        } 
        else if (n_opts == 2) {
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                leaf_values[value_idx + i] -= __ldg(W + offset_value + i)  / opts[0]->init_lr;
            }
            for (int i = opts[1]->start_idx; i < opts[1]->stop_idx; ++i){
                leaf_values[value_idx + i] -= __ldg(W + offset_value + i)  / opts[1]->init_lr;
            }
        }
        else {
            for (int opt_idx = 0; opt_idx < n_opts; ++opt_idx){
                for (int i = opts[opt_idx]->start_idx; i < opts[opt_idx]->stop_idx; ++i)
                    leaf_values[value_idx + i] -= __ldg(W + offset_value + i)  / opts[opt_idx]->init_lr;
            }
            }
        if (idx == 0){
            for (int i = 0; i < output_dim; ++i){
                bias[value_idx + i] += __ldg(W + value_idx + i);
            }
        }
    }
}

/**
 * @brief CUDA kernel to extract and scale leaf values into matrix V
 * 
 * Copies leaf values and scales by negative learning rate to populate V matrix.
 * Optimized for common case of 1-2 optimizers with loop unrolling.
 * 
 * @param V Output value matrix on GPU
 * @param leaf_values Source leaf values on GPU
 * @param opts Array of GPU optimizer structures
 * @param n_opts Number of optimizers
 * @param output_dim Output dimensionality
 * @param n_leaves Number of leaves
 */
__global__  void get_V_kernel(float* __restrict__ V, const float* __restrict__ leaf_values, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int n_leaves){
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx < n_leaves){
        if (n_opts == 1){
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i)
                V[(leaf_idx + 1)*output_dim + i] = -__ldg(&opts[0]->init_lr) * __ldg(leaf_values + leaf_idx*output_dim + i);
        } 
        else if (n_opts == 2) {
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i)
                 V[(leaf_idx + 1)*output_dim + i] = -__ldg(&opts[0]->init_lr) * __ldg(leaf_values + leaf_idx*output_dim + i);

            for (int i = opts[1]->start_idx; i < opts[1]->stop_idx; ++i)
                 V[(leaf_idx + 1)*output_dim + i] = -__ldg(&opts[1]->init_lr) * __ldg(leaf_values + leaf_idx*output_dim + i);
        } 
        else {
            for (int opt_idx = 0; opt_idx < n_opts; ++opt_idx){
                for (int i = opts[opt_idx]->start_idx; i < opts[opt_idx]->stop_idx; ++i)
                    V[(leaf_idx + 1)*output_dim + i] = -__ldg(&opts[opt_idx]->init_lr) * __ldg(leaf_values + leaf_idx*output_dim + i);
            }
        }
    }
}