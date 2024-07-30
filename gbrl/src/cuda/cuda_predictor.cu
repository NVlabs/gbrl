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

#include "cuda_predictor.h"
#include "cuda_fitter.h"
#include "cuda_utils.h"

SGDOptimizerGPU** deepCopySGDOptimizerVectorToGPU(const std::vector<Optimizer*>& host_opts) {
    // Allocate memory for array of SGDOptimizerGPU pointers
    cudaError_t error;
    SGDOptimizerGPU** device_opts;
    int n_opts = host_opts.size();
    error = cudaMalloc((void**)&device_opts, sizeof(SGDOptimizerGPU*) * n_opts);
    if (error != cudaSuccess) {
    // Handle the error (e.g., print an error message and exit)
        std::cout << "Cuda error: " << error << " when trying to allocate device_opts." <<std::endl;
        return nullptr;
    }
    // For each leaf, deep copy it to GPU and store its pointer in the array
    for (int i = 0; i < n_opts; ++i) {
        SGDOptimizerGPU* device_opt;
        error = cudaMalloc((void**)&device_opt, sizeof(SGDOptimizerGPU));
        if (error != cudaSuccess) {
        // Handle the error (e.g., print an error message and exit)
            std::cout << "Cuda error: " << error << " when trying to allocate device_opt " << i << "." <<std::endl;
            for (int j = 0; j < i; ++j)
                cudaFree(device_opts[j]);
            cudaFree(device_opts);
            return nullptr;
        }
        cudaMemset(device_opt, 0, sizeof(SGDOptimizerGPU));
        cudaMemcpy(&(device_opt->start_idx), &(host_opts[i]->start_idx), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&(device_opt->stop_idx), &(host_opts[i]->stop_idx), sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&(device_opt->init_lr), &(host_opts[i]->scheduler->init_lr), sizeof(float), cudaMemcpyHostToDevice);
        
        schedulerFunc scheduler = host_opts[i]->scheduler->getType();
        cudaMemcpy(&(device_opt->scheduler), &scheduler, sizeof(schedulerFunc), cudaMemcpyHostToDevice);
        cudaMemcpy(&(device_opts[i]), &device_opt, sizeof(SGDOptimizerGPU*), cudaMemcpyHostToDevice);
    }
    // Return the pointer to the array of SGDOptimizerGPU pointers
    return device_opts;
}

void freeSGDOptimizer(SGDOptimizerGPU **device_ops, const int n_opts){
    if (device_ops != nullptr){
        // Copy the GPU pointer array to host
        SGDOptimizerGPU** host_ops = new SGDOptimizerGPU*[n_opts];
        cudaMemcpy(host_ops, device_ops, sizeof(SGDOptimizerGPU*) * n_opts, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_opts; ++i){
            // This requires copying each SGDOptimizerGPU structure from GPU to CPU
            if (host_ops[i] != nullptr)
            {
                cudaFree(host_ops[i]);
            }
        }
        cudaFree(device_ops);
        delete[] host_ops;
    }
}

__global__ void add_vec_to_mat_kernel(const float *vec, float *mat, const int n_samples, const int n_cols){
    extern __shared__ float vec_s[];
    int thread_id = threadIdx.x;
    if (thread_id == 0){
        for (int i = 0; i < n_cols; ++i){
            vec_s[i] = vec[i];
        }     
    }

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples){
        int row_idx = idx*n_cols;
        for (int i = 0; i < n_cols; ++i)
            mat[row_idx + i] += vec_s[i];
    }
}

void predict_cuda(dataSet *dataset, float *host_preds, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, int start_tree_idx, int stop_tree_idx){

    float *device_batch_obs, *device_preds;
    char *device_batch_cat_obs;
    char *device_data;
    // assuming row-major order
    size_t preds_matrix_size = dataset->n_samples * metadata->output_dim * sizeof(float);
    size_t obs_matrix_size = dataset->n_samples * metadata->n_num_features * sizeof(float);
    size_t cat_obs_matrix_size = dataset->n_samples * metadata->n_cat_features * sizeof(char) * MAX_CHAR_SIZE;
    cudaError_t alloc_error = cudaMalloc((void**)&device_data, obs_matrix_size + preds_matrix_size + cat_obs_matrix_size);
    if (alloc_error != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA predict_cuda error: " << cudaGetErrorString(alloc_error)
                << " when trying to allocate " << ((obs_matrix_size + preds_matrix_size + cat_obs_matrix_size) / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return;
    }

    // Allocate host buffer
    char* host_data = new char[preds_matrix_size + obs_matrix_size + cat_obs_matrix_size];
    // Copy data into host buffer
    std::memcpy(host_data, dataset->obs, obs_matrix_size);
    std::memcpy(host_data + obs_matrix_size, host_preds, preds_matrix_size);
    std::memcpy(host_data + obs_matrix_size + preds_matrix_size,  dataset->categorical_obs, cat_obs_matrix_size);
    
    cudaMemcpy(device_data, host_data, obs_matrix_size + cat_obs_matrix_size  + preds_matrix_size, cudaMemcpyHostToDevice);
    delete[] host_data;

    size_t trace = 0;
    device_batch_obs = (float*)device_data;
    trace += obs_matrix_size;
    device_preds = (float *)(device_data + trace);
    trace += preds_matrix_size;
    device_batch_cat_obs = (char *)(device_data + trace);
    
    int n_blocks, threads_per_block;
    get_grid_dimensions(dataset->n_samples, n_blocks, threads_per_block);
    size_t shared_n_cols_size = metadata->output_dim * sizeof(float);
    add_vec_to_mat_kernel<<<n_blocks, threads_per_block, shared_n_cols_size>>>(edata->bias, device_preds, dataset->n_samples, metadata->output_dim);
    cudaDeviceSynchronize();
    
    if (stop_tree_idx > metadata->n_trees){
        std::cerr << "Given stop_tree_idx idx: " << stop_tree_idx << " greater than number of trees in model: " << metadata->n_trees << std::endl;
        cudaMemcpy(host_preds, device_preds, preds_matrix_size, cudaMemcpyDeviceToHost);
        cudaFree(device_data);
        return;
    } 
    if (metadata->n_trees == 0){
        cudaMemcpy(host_preds, device_preds, preds_matrix_size, cudaMemcpyDeviceToHost);
        cudaFree(device_data);
        return; 
    }
    
    if (stop_tree_idx == 0)
        stop_tree_idx = metadata->n_trees;

    if (n_opts == 0){
        std::cerr << "No optimizers." << std::endl;
        cudaMemcpy(host_preds, device_preds, preds_matrix_size, cudaMemcpyDeviceToHost);
        cudaFree(device_data);
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Replace 0 with your device ID if you have multiple devices

    threads_per_block = WARP_SIZE*((dataset->n_samples + WARP_SIZE - 1) / WARP_SIZE);
    if (threads_per_block > deviceProp.maxThreadsPerBlock)
        threads_per_block = deviceProp.maxThreadsPerBlock;

    if (metadata->grow_policy == GREEDY){
        int start_leaf_idx = 0, stop_leaf_idx = metadata->n_leaves;
        if (start_tree_idx > 0)
            cudaMemcpy(&start_leaf_idx, edata->tree_indices + start_tree_idx, sizeof(int), cudaMemcpyDeviceToHost);
        if (stop_tree_idx < metadata->n_trees)
            cudaMemcpy(&stop_leaf_idx, edata->tree_indices + stop_tree_idx, sizeof(int), cudaMemcpyDeviceToHost);
        int n_leaves = stop_leaf_idx - start_leaf_idx;
        if (metadata->n_cat_features == 0)
            predict_kernel_numerical_only<<<n_leaves, threads_per_block>>>(device_batch_obs, device_preds, dataset->n_samples, metadata->n_num_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, opts, n_opts, metadata->output_dim, metadata->max_depth, start_leaf_idx);
        else
            predict_kernel_tree_wise<<<n_leaves, threads_per_block>>>(device_batch_obs, device_batch_cat_obs, device_preds, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->categorical_values, edata->is_numerics, opts, n_opts, metadata->output_dim, metadata->max_depth, start_leaf_idx);
    } else{
        int n_trees = stop_tree_idx - start_tree_idx;
        if (metadata->n_cat_features == 0)
            predict_oblivious_kernel_numerical_only<<<n_trees, threads_per_block>>>(device_batch_obs, device_preds, dataset->n_samples, metadata->n_num_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->tree_indices, opts, n_opts, metadata->output_dim, metadata->max_depth, start_tree_idx);
        else
            predict_oblivious_kernel_tree_wise<<<n_trees, threads_per_block>>>(device_batch_obs, device_batch_cat_obs, device_preds, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->tree_indices, edata->categorical_values, edata->is_numerics, opts, n_opts, metadata->output_dim, metadata->max_depth, start_tree_idx);
    }
    cudaDeviceSynchronize();
    // Copy results back to CPU
    cudaMemcpy(host_preds, device_preds, preds_matrix_size, cudaMemcpyDeviceToHost);
    cudaFree(device_data);
}


void predict_cuda_no_host(dataSet *dataset, float *device_preds, ensembleMetaData *metadata, ensembleData *edata, SGDOptimizerGPU** opts, const int n_opts, int start_tree_idx, int stop_tree_idx, const bool add_bias){

    int n_blocks, threads_per_block;
    get_grid_dimensions(dataset->n_samples, n_blocks, threads_per_block);
    if (add_bias){
        size_t shared_n_cols_size = metadata->output_dim * sizeof(float);
        add_vec_to_mat_kernel<<<n_blocks, threads_per_block, shared_n_cols_size>>>(edata->bias, device_preds, dataset->n_samples, metadata->output_dim);
        cudaDeviceSynchronize();
    }
    
    if (stop_tree_idx > metadata->n_trees){
        std::cerr << "Given stop_tree_idx idx: " << stop_tree_idx << " greater than number of trees in model: " << metadata->n_trees << std::endl;
        return;
    } 
    if (metadata->n_trees == 0){
        return; 
    }
    
    if (stop_tree_idx == 0)
        stop_tree_idx = metadata->n_trees;

    if (n_opts == 0){
        std::cerr << "No optimizers." << std::endl;
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Replace 0 with your device ID if you have multiple devices

    threads_per_block = WARP_SIZE*((dataset->n_samples + WARP_SIZE - 1) / WARP_SIZE);
    if (threads_per_block > deviceProp.maxThreadsPerBlock)
        threads_per_block = deviceProp.maxThreadsPerBlock;
    if (metadata->grow_policy == GREEDY){
        int start_leaf_idx = 0, stop_leaf_idx = metadata->n_leaves;
        if (start_tree_idx > 0)
            cudaMemcpy(&start_leaf_idx, edata->tree_indices + start_tree_idx, sizeof(int), cudaMemcpyDeviceToHost);
        if (stop_tree_idx < metadata->n_trees)
            cudaMemcpy(&stop_leaf_idx, edata->tree_indices + stop_tree_idx, sizeof(int), cudaMemcpyDeviceToHost);
        
        int n_leaves = stop_leaf_idx - start_leaf_idx;
        if (dataset->n_samples > n_leaves)
            predict_sample_wise_kernel_tree_wise<<<dataset->n_samples, threads_per_block>>>(dataset->obs, dataset->categorical_obs, device_preds, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->categorical_values, edata->is_numerics, opts, n_opts, metadata->output_dim, metadata->max_depth, start_leaf_idx, n_leaves);
        else if (metadata->n_cat_features == 0){
            predict_kernel_numerical_only<<<n_leaves, threads_per_block>>>(dataset->obs, device_preds, dataset->n_samples, metadata->n_num_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, opts, n_opts, metadata->output_dim, metadata->max_depth, start_leaf_idx);
        }
        else
            predict_kernel_tree_wise<<<n_leaves, threads_per_block>>>(dataset->obs, dataset->categorical_obs, device_preds, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->categorical_values, edata->is_numerics, opts, n_opts, metadata->output_dim, metadata->max_depth, start_leaf_idx);
    } else {
        int n_trees = stop_tree_idx - start_tree_idx;
        if (metadata->n_cat_features == 0)
            predict_oblivious_kernel_numerical_only<<<n_trees, threads_per_block>>>(dataset->obs, device_preds, dataset->n_samples, metadata->n_num_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->tree_indices, opts, n_opts, metadata->output_dim, metadata->max_depth, start_tree_idx);
        else
            predict_oblivious_kernel_tree_wise<<<n_trees, threads_per_block>>>(dataset->obs, dataset->categorical_obs, device_preds, dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, edata->feature_indices, edata->depths, edata->feature_values, edata->inequality_directions, edata->values, edata->tree_indices, edata->categorical_values, edata->is_numerics, opts, n_opts, metadata->output_dim, metadata->max_depth, start_tree_idx);
    }
    cudaDeviceSynchronize();
}


__global__ void predict_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int n_cat_features, 
                                         const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, 
                                         const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int leaf_offset){
    
    bool equal, passed;
    int cond_idx = (blockIdx.x + leaf_offset) * max_depth, depth_idx;
    int value_idx = (blockIdx.x + leaf_offset) * output_dim;
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        passed = true;
        depth_idx = __ldg(depths + blockIdx.x + leaf_offset) - 1;
        while(depth_idx >= 0 && passed){
            if (is_numerics[cond_idx + depth_idx]){
                passed = __ldg(&obs[sample_idx*n_num_features + __ldg(&feature_indices[cond_idx + depth_idx])]) > __ldg(&feature_values[cond_idx + depth_idx]) == inequality_directions[cond_idx + depth_idx];
            } else {
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
            if (n_opts == 1){
                for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                    atomicAdd(&preds[sample_idx*output_dim + i], -opts[0]->init_lr * __ldg(leaf_values + value_idx + i));
                }
            } else if (n_opts == 2) {
                for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                    atomicAdd(&preds[sample_idx*output_dim + i], -opts[0]->init_lr * __ldg(leaf_values + value_idx + i));
                }
                for (int i = opts[1]->start_idx; i < opts[1]->stop_idx; ++i){
                    atomicAdd(&preds[sample_idx*output_dim + i], -opts[1]->init_lr * __ldg(leaf_values + value_idx + i));
                }
            } else {
                for (int opt_idx = 0; opt_idx < n_opts; ++opt_idx){
                    for (int i = opts[opt_idx]->start_idx; i < opts[opt_idx]->stop_idx; ++i)
                        atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[opt_idx]->init_lr) * __ldg(leaf_values + value_idx + i));
                }
            }
        }
    }
}

__global__ void predict_sample_wise_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int n_cat_features, 
                                                     const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, 
                                                     const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int leaf_offset, const int n_leaves){
    bool equal, passed;
    int cond_idx, depth_idx, value_idx;
    
    for (int leaf_idx = threadIdx.x; leaf_idx < n_leaves; leaf_idx += blockDim.x){
        cond_idx = (leaf_idx + leaf_offset) * max_depth;
        passed = true;
        depth_idx = __ldg(depths + leaf_idx + leaf_offset) - 1;
        while(depth_idx >= 0 && passed){
            if (is_numerics[cond_idx + depth_idx]){
                passed = __ldg(&obs[blockIdx.x*n_num_features + __ldg(feature_indices + cond_idx + depth_idx)]) > __ldg(feature_values + cond_idx + depth_idx) == inequality_directions[cond_idx + depth_idx];
            } else {
                equal = true;
                for (int i = 0; i < MAX_CHAR_SIZE; ++i) {
                    if (categorical_values[(cond_idx + depth_idx)*MAX_CHAR_SIZE + i] != categorical_obs[(blockIdx.x*n_cat_features + __ldg(feature_indices + cond_idx + depth_idx))*MAX_CHAR_SIZE + i]){
                        equal = false;
                        break;
                    } else if (categorical_values[(cond_idx + depth_idx)*MAX_CHAR_SIZE + i] == '\0' || categorical_obs[(blockIdx.x*n_cat_features + __ldg(feature_indices + cond_idx + depth_idx))*MAX_CHAR_SIZE + i] == '\0')
                        break;
                }
                passed = equal == inequality_directions[cond_idx + depth_idx];
            }
            depth_idx--;
        }
        value_idx = (leaf_idx + leaf_offset) * output_dim;
        if (passed){
            if (n_opts == 1){
                for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                    atomicAdd(&preds[blockIdx.x*output_dim + i], -opts[0]->init_lr * __ldg(leaf_values + value_idx + i));
                }
            } else if (n_opts == 2) {
                for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                    atomicAdd(&preds[blockIdx.x*output_dim + i], -opts[0]->init_lr * __ldg(leaf_values + value_idx + i));
                }
                for (int i = opts[1]->start_idx; i < opts[1]->stop_idx; ++i){
                    atomicAdd(&preds[blockIdx.x*output_dim + i], -opts[1]->init_lr * __ldg(leaf_values + value_idx + i));
                }
            } else {
                for (int opt_idx = 0; opt_idx < n_opts; ++opt_idx){
                    for (int i = opts[opt_idx]->start_idx; i < opts[opt_idx]->stop_idx; ++i)
                        atomicAdd(&preds[blockIdx.x*output_dim + i], -__ldg(&opts[opt_idx]->init_lr) * __ldg(leaf_values + value_idx + i));
                }
            }
        }
    }
}

__global__ void predict_kernel_numerical_only(const float* __restrict__ obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int* __restrict__ feature_indices,
                                              const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values, 
                                              SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int leaf_offset){
    int cond_idx = (blockIdx.x + leaf_offset) * max_depth;
    int value_idx = (blockIdx.x + leaf_offset) * output_dim;
    int depth_idx; // Initialize mask to all bits set
    bool passed;
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        depth_idx = __ldg(depths + (leaf_offset + blockIdx.x)) - 1;
        passed = true;
        while (depth_idx >= 0 && passed) {
            passed = (__ldg(&obs[sample_idx*n_num_features + __ldg(feature_indices + cond_idx + depth_idx)]) > __ldg(feature_values + cond_idx + depth_idx) == inequality_directions[cond_idx + depth_idx]);
            depth_idx--;
        }
        if (passed){
            if (n_opts == 1){
                for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                    atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[0]->init_lr) * __ldg(leaf_values + value_idx + i));
                }
            } else if (n_opts == 2) {
                for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                    atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[0]->init_lr) * __ldg(leaf_values + value_idx + i));
                }
                for (int i = opts[1]->start_idx; i < opts[1]->stop_idx; ++i){
                    atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[1]->init_lr) * __ldg(leaf_values + value_idx + i));
                }
            } else {
                for (int opt_idx = 0; opt_idx < n_opts; ++opt_idx){
                    for (int i = opts[opt_idx]->start_idx; i < opts[opt_idx]->stop_idx; ++i)
                        atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[opt_idx]->init_lr) * __ldg(leaf_values + value_idx + i));
                }
            }
        }
    }
}

__global__ void predict_oblivious_kernel_numerical_only(const float* __restrict__ obs, float* __restrict__ preds, const int n_samples, const int n_num_features, 
                                                        const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values,
                                                        const int* __restrict__ tree_indices, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int tree_offset){
    int leaf_idx, initial_leaf_idx = __ldg(tree_indices + blockIdx.x + tree_offset);
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        leaf_idx = 0;
        for (int depth_idx = 0; depth_idx < __ldg(depths + blockIdx.x + tree_offset); depth_idx++){ 
            bool decision = (__ldg(&obs[sample_idx*n_num_features + __ldg(feature_indices + (blockIdx.x + tree_offset) * max_depth + depth_idx)])) > (__ldg(feature_values + (blockIdx.x + tree_offset) * max_depth + depth_idx));
            leaf_idx |= (decision <<  (__ldg(depths + blockIdx.x + tree_offset) - 1 - depth_idx));
        }
        if (n_opts == 1){
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[0]->init_lr) * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
        } else if (n_opts == 2){
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[0]->init_lr) * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
            for (int i = opts[1]->start_idx; i < opts[1]->stop_idx; ++i){
                atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[1]->init_lr) * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
        } else {
            for (int opt_idx=0; opt_idx < n_opts; ++opt_idx){
                for (int i = opts[opt_idx]->start_idx; i < opts[opt_idx]->stop_idx; ++i)
                    atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[opt_idx]->init_lr) * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
        }
    }
}


__global__ void predict_oblivious_kernel_tree_wise(const float* __restrict__ obs, const char* __restrict__ categorical_obs, float* __restrict__ preds, const int n_samples, const int n_num_features, const int n_cat_features, 
                                                   const int* __restrict__ feature_indices, const int* __restrict__ depths, const float* __restrict__ feature_values, const bool* __restrict__ inequality_directions, const float* __restrict__ leaf_values,
                                                   const int* __restrict__ tree_indices, const char* __restrict__ categorical_values, const bool* __restrict__ is_numerics, SGDOptimizerGPU** opts, const int n_opts, const int output_dim, const int max_depth, const int tree_offset){
    bool decision;
    int tree_idx = blockIdx.x + tree_offset;
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

        if (n_opts == 1){
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                atomicAdd(&preds[sample_idx*output_dim + i], -opts[0]->init_lr * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
        } else if (n_opts == 2) {
            for (int i = opts[0]->start_idx; i < opts[0]->stop_idx; ++i){
                atomicAdd(&preds[sample_idx*output_dim + i], -opts[0]->init_lr * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
            for (int i = opts[1]->start_idx; i < opts[1]->stop_idx; ++i){
                atomicAdd(&preds[sample_idx*output_dim + i], -opts[1]->init_lr * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
        } else {
            for (int opt_idx = 0; opt_idx < n_opts; ++opt_idx){
                for (int i = opts[opt_idx]->start_idx; i < opts[opt_idx]->stop_idx; ++i)
                    atomicAdd(&preds[sample_idx*output_dim + i], -__ldg(&opts[opt_idx]->init_lr) * __ldg(leaf_values + (initial_leaf_idx + leaf_idx)*output_dim + i));
            }
        }
    }
}



