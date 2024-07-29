
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>

#include "cuda_compression.h"
#include "cuda_utils.h"

void get_matrix_representation_cuda(dataSet *dataset, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, matrixRepresentation *matrix){
    int n_samples = dataset->n_samples;
    float *device_batch_obs;
    char *device_batch_cat_obs;
    char *device_data;
    float *device_V;
    bool *device_A;
    // assuming row-major order
    size_t A_size = dataset->n_samples * (metadata->n_leaves+1) * sizeof(bool);
    size_t V_size = (metadata->n_leaves+1) * metadata->output_dim * sizeof(float);
    size_t obs_matrix_size = dataset->n_samples * metadata->n_num_features * sizeof(float);
    size_t cat_obs_matrix_size = dataset->n_samples * metadata->n_cat_features * sizeof(char) * MAX_CHAR_SIZE;
    cudaError_t alloc_error = cudaMalloc((void**)&device_data, obs_matrix_size + cat_obs_matrix_size + A_size + V_size);
    if (alloc_error != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA predict_cuda error: " << cudaGetErrorString(alloc_error)
                << " when trying to allocate " << ((obs_matrix_size + cat_obs_matrix_size + A_size + V_size) / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return;
    }

    // Allocate host buffer
    char* host_data = new char[obs_matrix_size + cat_obs_matrix_size + A_size + V_size];
    memset(host_data, 0, obs_matrix_size + cat_obs_matrix_size + A_size + V_size);
    // Copy data into host buffer
    std::memcpy(host_data, dataset->obs, obs_matrix_size);
    std::memcpy(host_data + obs_matrix_size + V_size + A_size,  dataset->categorical_obs, cat_obs_matrix_size);
    
    cudaMemcpy(device_data, host_data, obs_matrix_size + cat_obs_matrix_size + A_size + V_size, cudaMemcpyHostToDevice);
    delete[] host_data;

    size_t trace = 0;
    device_batch_obs = (float*)device_data;
    trace += obs_matrix_size;
    device_V = (float *)(device_data + trace);
    trace += V_size;
    device_A = (bool *)(device_data + trace);
    trace += A_size;
    device_batch_cat_obs = (char *)(device_data + trace);
    
    int n_blocks, threads_per_block;
    get_grid_dimensions(dataset->n_samples, n_blocks, threads_per_block);
    cudaMemcpy(device_V, edata->bias, sizeof(float)*metadata->output_dim, cudaMemcpyDeviceToDevice);
    
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
            get_representation_kernel_numerical_only<<<metadata->n_leaves, threads_per_block>>>(device_batch_obs, dataset->n_samples, metadata->n_num_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
        else
            get_representation_kernel_tree_wise<<<metadata->n_leaves, threads_per_block>>>(device_batch_obs, device_batch_cat_obs, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->categorical_values, edata->is_numerics, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
    } else{
        
        if (metadata->n_cat_features == 0)
            get_representation_oblivious_kernel_numerical_only<<<metadata->n_trees, threads_per_block>>>(device_batch_obs, dataset->n_samples, metadata->n_num_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->tree_indices, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
        else
            get_representation_oblivious_kernel_tree_wise<<<metadata->n_trees, threads_per_block>>>(device_batch_obs, device_batch_cat_obs, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->tree_indices, edata->categorical_values, edata->is_numerics, metadata->output_dim, metadata->max_depth, metadata->n_leaves, device_A);
    }
    cudaDeviceSynchronize();
    n_blocks = metadata->n_leaves / THREADS_PER_BLOCK + 1; 
    get_V_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(device_V, edata->values, opts, n_opts, metadata->output_dim, metadata->n_leaves);
    cudaDeviceSynchronize();
    matrix->A = new bool[A_size];
    cudaMemcpy(matrix->A, device_A, A_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_samples; i++)
        matrix->A[i*(metadata->n_leaves + 1)] = true;
    matrix->V = new float[V_size];
    cudaMemcpy(matrix->V, device_V, V_size, cudaMemcpyDeviceToHost);
    // Copy results back to CPU
    matrix->n_leaves = metadata->n_leaves;
    cudaFree(device_data);
    int *tree_indices = new int[metadata->n_trees];
    cudaMemcpy(tree_indices, edata->tree_indices,  sizeof(int)*metadata->n_trees, cudaMemcpyDeviceToHost);
    matrix->n_leaves_per_tree = new int[metadata->n_trees];
    for (int i = 0; i < metadata->n_trees - 1; ++i )
        matrix->n_leaves_per_tree[i] = tree_indices[i+1] - tree_indices[i];
    matrix->n_leaves_per_tree[metadata->n_trees - 1] = metadata->n_leaves - tree_indices[metadata->n_trees - 1];
    matrix->n_trees = metadata->n_trees;
    delete[] tree_indices;
}


ensembleData* compress_ensemble_cuda(ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W){
    float *device_W;
    size_t W_size = metadata->n_leaves * metadata->output_dim * sizeof(float);
    cudaError_t alloc_error = cudaMalloc((void**)&device_W, W_size);
    if (alloc_error != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA predict_cuda error: " << cudaGetErrorString(alloc_error)
                << " when trying to allocate " << ((W_size) / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return nullptr;
    }
    cudaMemcpy(device_W, W, W_size, cudaMemcpyHostToDevice);
    int n_blocks = metadata->n_leaves / THREADS_PER_BLOCK + 1;
    add_W_matrix_to_values_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(device_W, edata->values, edata->bias, opts, n_opts, metadata->n_leaves, metadata->output_dim);
    cudaDeviceSynchronize();

    ensembleData* compressed_edata = ensemble_compressed_data_copy_gpu_gpu(metadata, edata, n_compressed_leaves, n_compressed_trees, leaf_indices, tree_indices, new_tree_indices);

    cudaFree(device_W);
    ensemble_data_dealloc_cuda(edata);
    return compressed_edata;


}

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

__global__ void add_W_matrix_to_values_kernel(const float * __restrict__ W, float* __restrict__ leaf_values, float* __restrict__ bias, SGDOptimizerGPU** opts, const int n_opts, const int n_leaves, const int output_dim){
    int idx = blockIdx.x*blockDim.x + threadIdx.x; 
    if (idx < n_leaves){
        int value_idx = idx*output_dim;
        int offset_value = (idx + 1)*output_dim;
        if (n_opts == 1){
                for (int i = opts[0]->start_idx; i < opts[0]->end_idx; ++i){
                    leaf_values[value_idx + i] -= __ldg(W + offset_value + i)  / opts[0]->init_lr;
                }
            } 
            else {
                for (int i = opts[0]->start_idx; i < opts[0]->end_idx; ++i){
                    leaf_values[value_idx + i] -= __ldg(W + offset_value + i)  / opts[0]->init_lr;
                }
                for (int i = opts[1]->start_idx; i < opts[1]->end_idx; ++i){
                    leaf_values[value_idx + i] -= __ldg(W + offset_value + i)  / opts[1]->init_lr;
                }
        }
        if (idx == 0){
            if (n_opts == 1){
                for (int i = opts[0]->start_idx; i < opts[0]->end_idx; ++i){
                    bias[value_idx + i] += __ldg(W + value_idx + i);
                }
            } 
            else {
                for (int i = opts[0]->start_idx; i < opts[0]->end_idx; ++i){
                    bias[value_idx + i] += __ldg(W + value_idx + i);
                }
                for (int i = opts[1]->start_idx; i < opts[1]->end_idx; ++i){
                    bias[value_idx + i] += __ldg(W + value_idx + i);
                }
        }
        }
    }
}


__global__  void get_V_kernel(float* __restrict__ V, const float* __restrict__ leaf_values, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int n_leaves){
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx < n_leaves){
        if (n_opts == 1){
            for (int i = opts[0]->start_idx; i < opts[0]->end_idx; ++i)
                V[(leaf_idx + 1)*output_dim + i] = -opts[0]->init_lr * __ldg(leaf_values + leaf_idx*output_dim + i);
        } 
        else {
            for (int i = opts[0]->start_idx; i < opts[0]->end_idx; ++i)
                 V[(leaf_idx + 1)*output_dim + i] = -opts[0]->init_lr * __ldg(leaf_values + leaf_idx*output_dim + i);
            for (int i = opts[1]->start_idx; i < opts[1]->end_idx; ++i)
                 V[(leaf_idx + 1)*output_dim + i] = -opts[1]->init_lr * __ldg(leaf_values + leaf_idx*output_dim + i);
            
        }
    }
}