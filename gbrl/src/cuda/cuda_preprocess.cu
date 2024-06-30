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

#include <iostream>
#include <float.h>

#include "cuda_preprocess.h"
#include "cuda_predictor.h"
#include "split_candidate_generator.h"
#include "cuda_utils.h"



__global__ void iota_kernel(int *arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = idx;
    }
}

void preprocess_matrices(float* __restrict__ grads, float* __restrict__ grads_norm, const int n_rows, const int n_cols, const scoreFunc split_score_func){
    size_t shared_mem;
    if (split_score_func == Cosine){
        int n_threads = ((WARP_SIZE + n_cols - 1) / WARP_SIZE)*WARP_SIZE;
        if (n_threads > THREADS_PER_BLOCK)
            n_threads = THREADS_PER_BLOCK;
        shared_mem = sizeof(float)*n_threads;
        rowwise_squared_norm<<<n_rows, n_threads, shared_mem>>>(grads, n_cols, grads_norm, n_rows);
    } else if (split_score_func == L2){
        shared_mem = sizeof(float)*THREADS_PER_BLOCK*2;
        center_matrix<<<n_cols, THREADS_PER_BLOCK, shared_mem>>>(grads, n_cols, n_rows);
    }   
    cudaDeviceSynchronize();
}

__global__ void center_matrix(float* __restrict__ input, const int n_cols, const int n_rows)
{
    /*
    Center row-major matrix.
    */
    extern __shared__ float sdata[];

    float *mean = &sdata[0];
    float *var = &sdata[blockDim.x];
    /* #################################
    calculating colwise mean
    ################################# */ 
    float x = 0.0f, tmp;
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_rows; i += blockDim.x) {
        x += input[i * n_cols + blockIdx.x];
    }
    // load thread partial sum into shared memory
    mean[threadIdx.x] = x;
    __syncthreads();
    // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            mean[threadIdx.x] += mean[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0) {
        mean[0] /= static_cast<float>(n_rows);
    }
    __syncthreads();
     /* #################################
    calculating colwise var
    ################################# */ 
    x = 0.0f;
    for(int i=threadIdx.x; i < n_rows; i += blockDim.x) {
        tmp = input[i * n_cols + blockIdx.x] - mean[0];
        x += tmp*tmp;
    }
    // load thread partial sum into shared memory
    var[threadIdx.x] = x;
    __syncthreads();
    // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            var[threadIdx.x] += var[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0) {
        if (n_rows > 1)
            var[0] /= static_cast<float>(n_rows - 1);
    }
    __syncthreads();


     for(int i=threadIdx.x; i < n_rows; i += blockDim.x) {
        input[i * n_cols + blockIdx.x] =  (input[i * n_cols + blockIdx.x] - mean[0]) / (sqrtf(var[0]) + 1.0e-8);
    }
}


__global__ void rowwise_squared_norm(const float* __restrict__ input, const int n_cols, float* __restrict__ per_row_results, const int n_rows)
{
    /*
    Calculate the row-wise squared norm of an input matrix, where each block works on a different row.
    */
    extern __shared__ float sdata[];

    float x = 0.0f;
    // Accumulate per thread partial sum of squares
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        float val = input[blockIdx.x * n_cols + j];
        x += val * val;
    }
    // load thread partial sum into shared memory
    sdata[threadIdx.x] = x;
    __syncthreads();
    // tree reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (threadIdx.x == 0) {
        per_row_results[blockIdx.x] = sdata[0];
    }
}

__global__ void get_colwise_max(const float* __restrict__ input, const int n_cols, float* __restrict__ per_col_max, const int n_rows) {
    extern __shared__ float sdata[];

    float max_val = -FLT_MAX;
    // Find per thread maximum
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        max_val =  fmaxf(max_val, input[i * n_cols + blockIdx.x]);
    }
    // Load thread max into shared memory
    sdata[threadIdx.x] = max_val;
    __syncthreads();

    // Tree reduction for max
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) 
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + offset]);
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        per_col_max[blockIdx.x] = sdata[0];
    }
}

__global__ void get_colwise_min(const float* __restrict__ input, const int n_cols, float* __restrict__ per_col_max, const int n_rows) {
    extern __shared__ float sdata[];

    float min_val = FLT_MAX;
    // Find per thread maximum
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        min_val =  fminf(min_val, input[i * n_cols + blockIdx.x]);
    }
    // Load thread max into shared memory
    sdata[threadIdx.x] = min_val;
    __syncthreads();

    // Tree reduction for max
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) 
            sdata[threadIdx.x] = fminf(sdata[threadIdx.x], sdata[threadIdx.x + offset]);
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (threadIdx.x == 0) {
        per_col_max[blockIdx.x] = sdata[0];
    }
}

__global__ void bitonic_sort_kernel(const float* __restrict__ input, int* __restrict__ indices, int n_rows, int n_features) {
    int col = blockIdx.x;  // Column index this block is processing
    int thread_id = threadIdx.x;  // Thread ID within the block

    // Initialize indices with IOTA pattern for each column
    for (int i = thread_id; i < n_rows; i += blockDim.x) {
        indices[col * n_rows + i] = i;
    }
    __syncthreads();

    // Bitonic sort
    for (int k = 2; k / 2 < n_rows; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int idx = thread_id; idx < n_rows; idx += blockDim.x) {
                int partner_index = idx ^ j;  // Calculate partner index for comparison

                // Ensure both current and partner indices are within range
                if (partner_index > idx && partner_index < n_rows) {
                    // Determine if current thread is part of increasing or decreasing sequence
                    bool is_increasing = (idx & k) == 0;

                    // Calculate global indices for the elements to be compared
                    int idx1 = col * n_rows + idx;
                    int idx2 = col * n_rows + partner_index;

                    // Comparison and swap condition
                    bool swap_cond = is_increasing ? 
                                     (__ldg(&input[indices[idx1] * n_features + col]) > __ldg(&input[indices[idx2] * n_features + col])) :
                                     (__ldg(&input[indices[idx1] * n_features + col]) < __ldg(&input[indices[idx2] * n_features + col]));

                    // Swap indices if needed
                    if (swap_cond) {
                        int temp = indices[idx1];
                        indices[idx1] = indices[idx2];
                        indices[idx2] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}



int* sort_indices_cuda(const float* __restrict__ obs, const int n_samples, const int n_features){
    int *indices;
    cudaError_t err = cudaMalloc((void**)&indices, sizeof(int)*n_samples*n_features);
    if (err != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                << " when trying to allocate " << ((sizeof(int)*n_samples*n_features) / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return nullptr;
    }

    bitonic_sort_kernel<<<n_features, THREADS_PER_BLOCK>>>(obs, indices, n_samples, n_features);
    cudaDeviceSynchronize();
    return indices;
}


int quantile_candidates_cuda(const float* __restrict__ obs, const int* __restrict__ sorted_indices, int* candidate_indices, float* candidate_values,
                                       char *candidate_categories, bool *candidate_numerics,
                                       const int n_samples, const int n_features, 
                                       const int n_bins)
{   

    size_t shared_mem = sizeof(int)*n_bins;
    quantile_kernel<<<n_features, n_bins, shared_mem>>>(obs, sorted_indices, n_samples, n_features, n_bins, candidate_indices, candidate_values, candidate_categories, candidate_numerics);
    cudaDeviceSynchronize();
    int n_elements = n_bins*n_features;
    
    int n_candidates = 1;
    int *pos_ptr;
    cudaMalloc((void**)&pos_ptr, sizeof(int));
    cudaMemcpy(pos_ptr, &n_candidates, sizeof(int), cudaMemcpyHostToDevice);
    int n_blocks = n_elements / THREADS_PER_BLOCK + 1;
    place_unique_elements_infront_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(candidate_indices, candidate_values, candidate_categories, candidate_numerics, n_elements, pos_ptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&n_candidates, pos_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(pos_ptr);
    return n_candidates;
    // return n_elements;
}

int uniform_candidates_cuda(const float* __restrict__ obs, int* candidate_indices, float* candidate_values, 
                                      char *candidate_categories, bool *candidate_numerics,
                                      const int n_samples, const int n_features, 
                                      const int n_bins)
{   
    float *col_max, *col_min;
    cudaMalloc((void**)&col_max, sizeof(float)*n_features);
    cudaMalloc((void**)&col_min, sizeof(float)*n_features);

    size_t shared_mem = sizeof(float)*THREADS_PER_BLOCK;
    get_colwise_min<<<n_features, THREADS_PER_BLOCK, shared_mem>>>(obs, n_features, col_min, n_samples);
    get_colwise_max<<<n_features, THREADS_PER_BLOCK, shared_mem>>>(obs, n_features, col_max, n_samples);
    cudaDeviceSynchronize();
    int n_threads = ((WARP_SIZE + n_bins - 1) / WARP_SIZE ) * WARP_SIZE;
    linspace_kernel<<<n_features, n_threads>>>(col_min, col_max, n_features, n_bins, candidate_indices, candidate_values, candidate_categories, candidate_numerics);
    cudaDeviceSynchronize();

    cudaFree(col_max);
    cudaFree(col_min);
    return n_bins*n_features;
}

__global__ void quantile_kernel(const float* __restrict__ obs, const int* __restrict__ sorted_indices, const int n_samples, const int n_features, const int n_bins,  int* __restrict__ candidate_indices, float* __restrict__ candidate_values, char* __restrict__ candidate_cateogories, bool* __restrict__ candidate_numerics){
    extern __shared__ int bin_counts[];
    
        int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
        bin_counts[threadIdx.x] = n_samples / (n_bins + 1);
        int reminder = n_samples % (n_bins + 1);

        // no need to syncthreads here
        for (int i=threadIdx.x; i < reminder; i += blockDim.x)
            bin_counts[threadIdx.x] += 1;
        __syncthreads();

        int cumulative_count = 0;
        for (int i = 0; i <= threadIdx.x; ++i)
            cumulative_count += bin_counts[i];
        const int split_point = __ldg(&sorted_indices[blockIdx.x*n_samples + cumulative_count - 1]);
        candidate_indices[global_idx] = blockIdx.x;
        candidate_values[global_idx] = obs[split_point*n_features + blockIdx.x];
        candidate_numerics[global_idx] = true;
}

__global__ void linspace_kernel(const float* __restrict__ min_vec, const float* __restrict__ max_vec, int n_features, int n_bins, int* __restrict__ candidate_indices, float* __restrict__ candidate_values, char* __restrict__ candidate_cateogories, bool* __restrict__ candidate_numerics) {
    int feature_idx = blockIdx.x;
    int bin_idx = threadIdx.x;

    if (feature_idx < n_features && bin_idx < n_bins) {
        float feature_min = min_vec[feature_idx];
        float feature_max = max_vec[feature_idx];
        float bin_width = (feature_max - feature_min) / n_bins;
        int candidate_idx = feature_idx * n_bins + bin_idx;
        candidate_indices[candidate_idx] = feature_idx;
        candidate_values[candidate_idx] = feature_min + bin_idx * bin_width;
        candidate_numerics[candidate_idx] = true;
    }
}

__global__ void place_unique_elements_infront_kernel(int* __restrict__ candidate_indices, float* __restrict__ candidate_values, char* __restrict__ candidate_cateogories, bool* __restrict__ candidate_numerics, int size, int* pos_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // Start from the second element (1st index)
    for (int i = idx + 1; i < size; i += stride) {
        if (candidate_indices[i] != candidate_indices[i - 1] || candidate_values[i] != candidate_values[i - 1]) {
            int pos = atomicAdd(pos_idx, 1);
            candidate_indices[pos] = candidate_indices[i];
            candidate_values[pos] = candidate_values[i];
            candidate_numerics[pos] = candidate_numerics[i];
        }
    }
}

__global__ void linspaceKernel(const float* min_vec, const float* max_vec, int n_features, int n_bins, splitCandidate* split_candidates) {
    int feature_idx = blockIdx.x;
    int bin_idx = threadIdx.x;

    if (feature_idx < n_features && bin_idx < n_bins) {
        float feature_min = min_vec[feature_idx];
        float feature_max = max_vec[feature_idx];
        float bin_width = (feature_max - feature_min) / n_bins;

        float feature_value = feature_min + bin_idx * bin_width;
        int candidate_idx = feature_idx * n_bins + bin_idx;
        split_candidates[candidate_idx].feature_idx = feature_idx;
        split_candidates[candidate_idx].feature_value = feature_value;
    }
}


  int process_candidates_cuda(const float* gpu_obs, const char* categorical_obs, const float *gpu_grad_norms, int* candidate_indices, float *candidate_values, char* candidate_categories, bool* candidate_numerics, const int n_samples, const int n_num_features, const int n_cat_features, const int n_bins, const generatorType generator_type ){
    int n_candidates = 0;
    if (n_num_features > 0){
        if (generator_type == Uniform){
            n_candidates = uniform_candidates_cuda(gpu_obs, candidate_indices, candidate_values, candidate_categories, candidate_numerics, n_samples, n_num_features, n_bins);
        } else if (generator_type == Quantile){
            int *indices = sort_indices_cuda(gpu_obs, n_samples, n_num_features);
            n_candidates = quantile_candidates_cuda(gpu_obs, indices, candidate_indices, candidate_values, candidate_categories, candidate_numerics, n_samples, n_num_features, n_bins);
            cudaFree(indices);
        }
        else {
            std::cerr << "Error! Unknown generator type: " << generator_type << std::endl;
        }
    }

    if (n_cat_features > 0){
        float* cpu_feature_values = new float[n_bins*n_cat_features];
        int* cpu_feature_inds = new int[n_bins*n_cat_features];
        char* cpu_feature_categories = new char[n_bins*n_cat_features*MAX_CHAR_SIZE];
        bool *cpu_numerics = new bool[n_bins*n_cat_features];
        float *grad_norms = new float[n_samples];
        cudaMemcpy(grad_norms, gpu_grad_norms, sizeof(float)*n_samples, cudaMemcpyDeviceToHost);
        int n_cat_candidates = processCategoricalCandidates_func(categorical_obs, grad_norms, n_samples, n_cat_features, n_bins, cpu_feature_inds, cpu_feature_values, cpu_feature_categories, cpu_numerics);
        
        cudaMemcpy(candidate_indices + n_candidates, cpu_feature_inds, sizeof(int)*n_cat_candidates, cudaMemcpyHostToDevice);
        cudaMemcpy(candidate_values + n_candidates, cpu_feature_values, sizeof(float)*n_cat_candidates, cudaMemcpyHostToDevice);
        cudaMemcpy(candidate_numerics + n_candidates, cpu_numerics, sizeof(bool)*n_cat_candidates, cudaMemcpyHostToDevice);
        cudaMemcpy(candidate_categories + n_candidates*MAX_CHAR_SIZE, cpu_feature_categories, sizeof(char)*n_cat_candidates*MAX_CHAR_SIZE, cudaMemcpyHostToDevice);

        n_candidates += n_cat_candidates;

        delete[] grad_norms;
        delete[] cpu_feature_values;
        delete[] cpu_feature_inds;
        delete[] cpu_feature_categories;
        delete[] cpu_numerics;
    }

    return n_candidates;
  }

  float* transpose_matrix(const float *mat, float *trans_mat, const int width, const int height){
    int n_blocks, threads_per_block;
    get_grid_dimensions(width*height, n_blocks, threads_per_block);
    transpose1D<<<n_blocks, threads_per_block>>>(trans_mat, mat, width, height);
    cudaDeviceSynchronize();
    return trans_mat;
  }

  __global__ void transpose1D(float *out, const float *in, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    if (tid < width * height) {
        int row = tid / width;
        int col = tid % width;
        int transposedIndex = col * height + row;
        out[transposedIndex] = in[tid];
    }
}

