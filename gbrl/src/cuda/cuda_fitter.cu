//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda_fitter.cu
 * @brief Implementation of CUDA kernels for tree fitting on GPU
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <limits>

#include "utils.h"
#include "cuda_fitter.h"
#include "cuda_preprocess.h"
#include "cuda_predictor.h"
#include "cuda_utils.h"
#include "cuda_types.h"


void calc_parallelism(
    const int n_candidates,
    const int output_dim,
    int &threads_per_block,
    const scoreFunc split_score_func) {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (n_candidates > MAX_BLOCKS_PER_GRID){
        std::cerr << "n_candidates: " << n_candidates << " > " << MAX_BLOCKS_PER_GRID << " max blocks per grid." << std::endl;
    }

    threads_per_block = THREADS_PER_BLOCK;

    int shared_mem;
    if (split_score_func == Cosine)
        shared_mem = 2 * (output_dim + 3) * sizeof(float);
    else if (split_score_func == L2)
        shared_mem = 2 * (output_dim + 1) * sizeof(float);
    while (threads_per_block*shared_mem > deviceProp.sharedMemPerBlock){
        if (threads_per_block == 1){
            std::cerr << "output_dim " << output_dim << "too large! cannot work with so many columns! use cpu version" << std::endl;
        }
        threads_per_block >>= 1;
    }
}


void calc_oblivious_parallelism(
    const int n_candidates,
    const int output_dim,
    int &threads_per_block,
    const scoreFunc split_score_func,
    size_t &shared_mem,
    const int depth,
    const int n_objs) {

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (n_candidates > MAX_BLOCKS_PER_GRID){
        std::cerr << "n_candidates: " << n_candidates << " > " << MAX_BLOCKS_PER_GRID << " max blocks per grid." << std::endl;
    }

    threads_per_block = THREADS_PER_BLOCK;
    size_t floats_per_thread = 0;

    if (split_score_func == Cosine) {
        // [Mean L/R (Vector)] + [Count L/R, Dot L/R (Scalar)] + [Label Stats (6)]
        // 2 * n_objs * dim + 4 * n_objs + 6
        floats_per_thread = 2 * n_objs * (output_dim + 2) + 6;
    } 
    else if (split_score_func == L2) {
        // [Sum L/R (Vector)] + [Count L/R (Scalar)] + [Label Stats (6)]
        // 2 * n_objs * dim + 2 * n_objs + 6
        floats_per_thread = 2 * n_objs * (output_dim + 1) + 6;
    }

    shared_mem = floats_per_thread * sizeof(float);
    while (threads_per_block*shared_mem*(1 << depth) > deviceProp.sharedMemPerBlock){
        if (threads_per_block == 1){
            std::cerr << "output_dim " << output_dim << "too large! cannot work with so many columns! use cpu version" << std::endl;
        }
        threads_per_block >>= 1;
    }
}


__global__ void update_best_candidate_cuda(
    float* __restrict__ split_scores,
    int n_candidates,
    int* __restrict__ best_idx,
    float* __restrict__ best_score
) {

    // Allocate shared memory for intermediate best scores and indices
    __shared__ float s_best_scores[THREADS_PER_BLOCK];
    __shared__ int s_best_indices[THREADS_PER_BLOCK];

    if (threadIdx.x == 0){
        *best_score = -CUDART_INF_F;
        *best_idx = -1;
    }
    // Initialize shared memory
    s_best_scores[threadIdx.x] = -CUDART_INF_F;
    s_best_indices[threadIdx.x] = -1;
    __syncthreads();
    // Each thread processes multiple elements
    for (int i = threadIdx.x; i < n_candidates; i += blockDim.x) {
        if (split_scores[i] > s_best_scores[threadIdx.x]) {
            s_best_scores[threadIdx.x] = split_scores[i];
            s_best_indices[threadIdx.x] = i;
        }
    }
    // Synchronize threads within the block
    __syncthreads();
    // Sequential reduction in the first warp
    if (threadIdx.x < WARP_SIZE) { // Assuming warp size is 32
        for (int i = threadIdx.x + WARP_SIZE; i < blockDim.x; i += WARP_SIZE) {
            if (s_best_scores[i] > s_best_scores[threadIdx.x]) {
                s_best_scores[threadIdx.x] = s_best_scores[i];
                s_best_indices[threadIdx.x] = s_best_indices[i];
            }
        }
    }
    __syncthreads();
    // Sequential reduction in the first thread of the block
    if (threadIdx.x == 0) {
        for (int i = 0; i < WARP_SIZE; ++i) {
            if (s_best_scores[i] > *best_score) {
                *best_score = s_best_scores[i];
                *best_idx = s_best_indices[i];
            }
        }
    }
}

__global__ void reduce_split_scores_kernel(
    float* __restrict__ split_scores,      // In/Out: [Obj0][Obj1]... -> [Total][Garbage]...
    const TreeNodeGPU* __restrict__ node,   // Size: n_objs
    const int n_candidates,
    const int n_objs)
{
    // Grid-Stride Loop over Candidates
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int cand_idx = idx; cand_idx < n_candidates; cand_idx += stride) {
        
        float total_gain = 0.0f;
        bool is_valid = true;

        // Sum across all objective layers
        for (int k = 0; k < n_objs; ++k) {
            
            // Jump to the K-th layer
            int offset = cand_idx + (k * n_candidates);
            float score = split_scores[offset];

            // 1. Hard Constraint Check
            // If ANY objective marked this split as invalid (-inf), 
            // the whole split is invalid.
            if (score == -CUDART_INF_F) { // Check for -CUDART_INF_F
                is_valid = false;
                break;
            }

            // 2. Weighted Accumulation
            total_gain += score * node->densities[k];
        }

        // 3. Write Back
        // We overwrite the slot for Objective 0 with the final Total Gain.
        // This is safe because 'cand_idx' is processed by a single thread.
        split_scores[cand_idx] = is_valid ? total_gain : -CUDART_INF_F;
    }
}

void evaluate_greedy_splits(
    dataSet *dataset,
    ensembleData *edata,
    TreeNodeGPU *node,
    candidatesData *candidata,
    ensembleMetaData *metadata,
    splitDataGPU* split_data,
    const int threads_per_block,
    const int parent_n_samples){

    cudaMemset(split_data->split_scores, 0, split_data->size);
    int n_blocks, tpb; 
    get_grid_dimensions(parent_n_samples * candidata->n_candidates * metadata->n_objs, n_blocks, tpb);
    if (metadata->split_score_func == Cosine){
        split_conditional_sum_kernel<<<n_blocks, tpb>>>(
            dataset->obs->data,
            dataset->categorical_obs->data,
            dataset->build_grads->data,
            node,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            candidata->n_candidates,
            dataset->n_samples,
            metadata->n_objs,
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count
        );
        cudaDeviceSynchronize();
        split_conditional_dot_kernel<<<n_blocks, tpb>>>(
            dataset->obs->data,
            dataset->categorical_obs->data,
            dataset->build_grads->data,
            node,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            candidata->n_candidates,
            dataset->n_samples,
            metadata->n_objs,
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count,
            split_data->left_dot,
            split_data->right_dot
        );
        cudaDeviceSynchronize();
        get_grid_dimensions(candidata->n_candidates * metadata->n_objs, n_blocks, tpb);
        split_cosine_score_kernel<<<n_blocks, tpb>>>(
            node,
            edata->feature_weights,
            split_data->split_scores,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            edata->reverse_num_feature_mapping,
            edata->reverse_cat_feature_mapping,
            candidata->n_candidates,
            metadata->n_objs,
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count,
            split_data->left_dot,
            split_data->right_dot,
            metadata->min_data_in_leaf,
            metadata->n_num_features);
     } else if (metadata->split_score_func == L2){
        split_conditional_sum_kernel<<<n_blocks, tpb>>>(
            dataset->obs->data,
            dataset->categorical_obs->data,
            dataset->build_grads->data,
            node,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            candidata->n_candidates,
            dataset->n_samples,
            metadata->n_objs,
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count);
        cudaDeviceSynchronize();
        get_grid_dimensions(candidata->n_candidates * metadata->n_objs, n_blocks, tpb);
        split_l2_score_kernel<<<n_blocks, tpb>>>(
            node,
            edata->feature_weights,
            split_data->split_scores,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            edata->reverse_num_feature_mapping,
            edata->reverse_cat_feature_mapping,
            candidata->n_candidates,
            metadata->n_objs,
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count,
            metadata->min_data_in_leaf,
            metadata->n_num_features);

    }

    cudaDeviceSynchronize();
    reduce_split_scores_kernel<<<(candidata->n_candidates + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(
        split_data->split_scores,
        node,
        candidata->n_candidates,
        metadata->n_objs
    );

    if (dataset->obj_labels->data != nullptr){
        cudaDeviceSynchronize();
        int threads = metadata->output_dim;
        // Need shared memory for 2 float arrays of size 'threads'
        size_t smem = 2 * threads * sizeof(float);

        calc_node_conflict_kernel<<<1, threads, smem>>>(
            split_data->node_mean,
            node,
            metadata->n_objs,
            metadata->output_dim
        );
        cudaDeviceSynchronize();
        get_tpb_dimensions(candidata->n_candidates * parent_n_samples, candidata->n_candidates, tpb);
        size_t shared_mem = sizeof(float) * 6 * tpb;
        split_impurity_penalty_kernel<<<candidata->n_candidates, tpb, shared_mem>>>(
            dataset->obj_labels->data,
            dataset->obs->data,
            dataset->categorical_obs->data,
            node,
            split_data->split_scores,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            candidata->n_candidates,
            dataset->n_samples,
            metadata->lambda_penalty
        );
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
#ifdef DEBUG
    if (metadata->verbose > 1){
        print_candidate_scores<<<1, THREADS_PER_BLOCK>>>(candidata->candidate_indices, candidata->candidate_values,  candidata->candidate_categories, candidata->candidate_numeric, split_data->split_scores, candidata->n_candidates);
        cudaDeviceSynchronize();
    }
#endif
    update_best_candidate_cuda<<<1, THREADS_PER_BLOCK>>>(split_data->split_scores, candidata->n_candidates, split_data->best_idx, split_data->best_score); 
    cudaDeviceSynchronize();
}

void evaluate_oblivious_splits_cuda(
    dataSet *dataset,
    ensembleData *edata,
    TreeNodeGPU **nodes,
    const int depth,
    candidatesData *candidata,
    ensembleMetaData *metadata,
    splitDataGPU *split_data){

    int tpb;
    int n_nodes = (1 << depth);
    size_t per_thread_shared_mem, shared_mem;

    int n_streams = 8;
    if (n_streams > n_nodes)
        n_streams = n_nodes;

    cudaStream_t streams[n_streams];
    for (int k = 0; k < n_streams; ++k) cudaStreamCreate(&streams[k]);
   
    calc_oblivious_parallelism(candidata->n_candidates, metadata->output_dim, tpb, metadata->split_score_func, per_thread_shared_mem, depth, metadata->n_objs);
    shared_mem = per_thread_shared_mem * tpb;
    for (int i = 0; i < n_nodes; ++i){

        cudaStream_t current_stream = streams[i % n_streams];

        // A. Conflict Calculation (Async)
        if (dataset->obj_labels->data != nullptr) {

            const dim3 n_threads_per_blockdim3(BLOCK_COLS, BLOCK_ROWS);
            // Pointer offset for this node's means
            size_t mean_offset = (size_t)i * metadata->n_objs * metadata->output_dim;
            node_column_mean_reduce<<<(metadata->output_dim * metadata->n_objs + BLOCK_COLS - 1) / BLOCK_COLS, n_threads_per_blockdim3, 0, current_stream>>>(
                dataset->build_grads->data,
                split_data->node_mean + mean_offset,
                metadata->output_dim,
                dataset->n_samples,
                nodes[i],
                metadata->n_objs);

            int threads_rho = metadata->output_dim;
            size_t smem_rho = 2 * threads_rho * sizeof(float);

            calc_node_conflict_kernel<<<1, threads_rho, smem_rho, current_stream>>>(
                split_data->node_mean + mean_offset,
                nodes[i], 
                metadata->n_objs,
                metadata->output_dim
            );
        }

        if (metadata->split_score_func == Cosine){
            split_score_cosine_cuda<<<candidata->n_candidates, tpb, shared_mem, current_stream>>>(
                dataset->obs->data,
                dataset->categorical_obs->data,
                dataset->build_grads->data,
                edata->feature_weights,
                dataset->obj_labels->data,
                nodes[i],
                candidata->candidate_indices,
                candidata->candidate_values,
                candidata->candidate_categories,
                candidata->candidate_numeric,
                edata->reverse_num_feature_mapping,
                edata->reverse_cat_feature_mapping,
                metadata->min_data_in_leaf,
                split_data->oblivious_split_scores + candidata->n_candidates*i,
                dataset->n_samples,
                metadata->n_num_features,
                metadata->n_objs,
                metadata->lambda_penalty);
        } else if (metadata->split_score_func == L2){
            split_score_l2_cuda<<<candidata->n_candidates, tpb, shared_mem, current_stream>>>(
                dataset->obs->data, dataset->categorical_obs->data,
                dataset->build_grads->data,
                edata->feature_weights,
                dataset->obj_labels->data,
                nodes[i],
                candidata->candidate_indices,
                candidata->candidate_values,
                candidata->candidate_categories,
                candidata->candidate_numeric,
                edata->reverse_num_feature_mapping,
                edata->reverse_cat_feature_mapping,
                metadata->min_data_in_leaf,
                split_data->oblivious_split_scores + candidata->n_candidates*i,
                dataset->n_samples,
                metadata->n_num_features,
                metadata->n_objs,
                metadata->lambda_penalty);
        }
       
    }

    cudaDeviceSynchronize();

    // Cleanup Streams
    for (int k = 0; k < n_streams; ++k) cudaStreamDestroy(streams[k]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    const dim3 n_threads_per_blockdim3(BLOCK_COLS, BLOCK_ROWS);
    column_sums_reduce<<<(candidata->n_candidates + BLOCK_COLS - 1) / BLOCK_COLS, n_threads_per_blockdim3>>>(split_data->oblivious_split_scores, split_data->split_scores, candidata->n_candidates, n_nodes);
    cudaDeviceSynchronize();
#ifdef DEBUG
    if (metadata->verbose > 1){
        print_candidate_scores<<<1, THREADS_PER_BLOCK>>>(candidata->candidate_indices, candidata->candidate_values,  candidata->candidate_categories, candidata->candidate_numeric, split_data->split_scores, candidata->n_candidates);
        cudaDeviceSynchronize();
    }
#endif
    update_best_candidate_cuda<<<1, THREADS_PER_BLOCK>>>(split_data->split_scores, candidata->n_candidates, split_data->best_idx, split_data->best_score); 
    cudaDeviceSynchronize();
}


__global__ void split_score_cosine_cuda(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const float* __restrict__ feature_weights,
    const float* __restrict__ obj_labels,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int min_data_in_leaf,
    float* __restrict__ split_scores,
    const int global_n_samples,
    const int n_num_features,
    const int n_objs,
    const float lambda_penalty){
    extern __shared__ float sdata[];

    int n_samples = __ldg(&node->n_samples), n_cols = __ldg(&node->output_dim);
    int cand_idx = blockIdx.x;

    if (split_scores[cand_idx] == -CUDART_INF_F)
        return;

    if (__ldg(&node->depth) > 0 && min_data_in_leaf == 0){
        if (candidate_numeric[cand_idx]){
            for (int i = 0; i < __ldg(&node->depth); ++i){
                if (node->is_numerics[i] && __ldg(&node->feature_values[i]) == __ldg(&candidate_values[cand_idx]) && __ldg(&node->feature_indices[i]) == __ldg(&candidate_indices[cand_idx])){
                    split_scores[cand_idx] = -CUDART_INF_F;
                    return;
                }
            }
        } else {
            for (int i = 0; i < __ldg(&node->depth); ++i){
                if (!node->is_numerics[i] && strcmpCuda(node->categorical_values + i * MAX_CHAR_SIZE, candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0 && node->feature_indices[i] == candidate_indices[cand_idx]){
                    split_scores[cand_idx] = -CUDART_INF_F;
                    return;
                }
            }
        }
    }

    int vec_stride = n_objs * n_cols;

    int thread_offset = 0;
    float *left_mean = &sdata[0];
    thread_offset += blockDim.x * vec_stride;
    float *l_count = &sdata[thread_offset];
    thread_offset += blockDim.x * n_objs;
    float *l_dot_sum = &sdata[thread_offset];
    thread_offset += blockDim.x * n_objs;
    float* right_mean = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += blockDim.x * vec_stride;
    float* r_count = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += blockDim.x * n_objs;
    float* r_dot_sum = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += blockDim.x * n_objs;
    float *s_labels  = &sdata[thread_offset]; // For impurity penalty

    for (int k = 0; k < n_objs; ++k){

        r_dot_sum[threadIdx.x * n_objs + k] = 0.0f;
        l_dot_sum[threadIdx.x * n_objs + k] = 0.0f;
        l_count[threadIdx.x * n_objs + k] = 0.0f;
        r_count[threadIdx.x * n_objs + k] = 0.0f;

        for (int d = 0; d < n_cols; ++d){
            right_mean[threadIdx.x*vec_stride + k * n_cols + d ] = 0.0f;
            left_mean[threadIdx.x*vec_stride + k * n_cols + d ] = 0.0f;
        }
    }

    // Layout: [L_c, L_s, L_sq, R_c, R_s, R_sq]
    for(int i=0; i<6; ++i) s_labels[threadIdx.x*6 + i] = 0.0f;

    
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = __ldg(&node->sample_indices[i]); // Access the specific sample
        bool passed = candidate_numeric[cand_idx] && __ldg(&obs[sample_idx +  global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx]);
        passed = passed || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + __ldg(&candidate_indices[cand_idx]))* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        
        float lbl = (obj_labels) ? __ldg(&obj_labels[sample_idx]) : 0.0f;
        if (passed){
            // Label Stats
            s_labels[threadIdx.x*6 + 3] += 1.0f; s_labels[threadIdx.x*6 + 4] += lbl; s_labels[threadIdx.x*6 + 5] += lbl*lbl;
            for (int k = 0; k < n_objs; ++k){
                size_t g_base = (size_t)k * global_n_samples * n_cols + (size_t)sample_idx * n_cols;
                int s_base = threadIdx.x * vec_stride + k * n_cols;

                r_count[threadIdx.x * n_objs + k] += 1;
                for (int d = 0; d < n_cols; ++d){
                    right_mean[s_base + d] += __ldg(&grads[g_base + d]);
                }
            }
        } 
        else {

            s_labels[threadIdx.x*6 + 0] += 1.0f; s_labels[threadIdx.x*6 + 1] += lbl; s_labels[threadIdx.x*6 + 2] += lbl*lbl;
            for (int k = 0; k < n_objs; ++k){
                size_t g_base = (size_t)k * global_n_samples * n_cols + (size_t)sample_idx * n_cols;
                int s_base = threadIdx.x * vec_stride + k * n_cols;

                l_count[threadIdx.x * n_objs + k] += 1;
                for (int d = 0; d < n_cols; ++d){
                    left_mean[s_base + d] += __ldg(&grads[g_base + d]);
                }
            }
        }
    }
    __syncthreads();
     // // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {

            // Reduce Labels
            for(int j=0; j<6; ++j) s_labels[threadIdx.x*6 + j] += s_labels[(threadIdx.x+offset)*6 + j];
            
            for (int k = 0; k < n_objs; ++k){

                int s_base_curr = threadIdx.x * vec_stride + k * n_cols;
                int s_base_next = (threadIdx.x + offset) * vec_stride + k * n_cols;


                for (int d = 0; d < n_cols; ++d){
                    left_mean[s_base_curr + d] += left_mean[s_base_next + d];
                    right_mean[s_base_curr + d] += right_mean[s_base_next + d];
                }
                l_count[threadIdx.x * n_objs + k] += l_count[(threadIdx.x + offset) * n_objs + k];
                r_count[threadIdx.x * n_objs + k] += r_count[(threadIdx.x + offset) * n_objs + k];
                
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        if (l_count[threadIdx.x] < static_cast<float>(min_data_in_leaf) || r_count[threadIdx.x] < static_cast<float>(min_data_in_leaf)){
            split_scores[cand_idx] = -CUDART_INF_F;
        } 
    }

    __syncthreads();
    if (split_scores[cand_idx] == -CUDART_INF_F) return;

    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = __ldg(&node->sample_indices[i]); // Access the spec
        bool passed = candidate_numeric[cand_idx] && __ldg(&obs[sample_idx + global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx]);
        passed = passed || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + __ldg(&candidate_indices[cand_idx]))* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        if (passed){
            for (int k = 0; k < n_objs; ++k){
                size_t g_base = (size_t)k * global_n_samples * n_cols + (size_t)sample_idx * n_cols;
                int s_base = 0 * vec_stride + k * n_cols;
                
                for (int d = 0; d < n_cols; ++d){
                    r_dot_sum[threadIdx.x * n_objs + k] += __ldg(&grads[g_base + d]) * right_mean[s_base + d];
                }
            }

        } else {
            for (int k = 0; k < n_objs; ++k){
                size_t g_base = (size_t)k * global_n_samples * n_cols + (size_t)sample_idx * n_cols;
                int s_base = 0 * vec_stride + k * n_cols;

                for (int d = 0; d < n_cols; ++d){
                    l_dot_sum[threadIdx.x * n_objs + k] += __ldg(&grads[g_base + d]) * left_mean[s_base + d];
                }
            }
        }
    }
    __syncthreads();
    
     // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            for (int k = 0; k < n_objs; ++k) {
                r_dot_sum[threadIdx.x * n_objs + k] += r_dot_sum[(threadIdx.x + offset) * n_objs + k];  
                l_dot_sum[threadIdx.x * n_objs + k] += l_dot_sum[(threadIdx.x + offset) * n_objs + k];  
            }
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (threadIdx.x == 0){
        float total_gain = 0.0f;

        for (int k = 0; k < n_objs; k++){
            int base = k * n_cols;
            float cosine = 0.0f, l_mean_norm = 0.0f, r_mean_norm = 0.0f;
            for (int d = 0; d < n_cols; ++d){
                l_mean_norm += left_mean[base + d] * left_mean[base + d];
                r_mean_norm += right_mean[base + d] * right_mean[base + d];
            }
            l_mean_norm = (l_count[k] > 0.0f) ? l_mean_norm / (l_count[k]*l_count[k]) : 0.0f;
            r_mean_norm = (r_count[k] > 0.0f) ? r_mean_norm / (r_count[k]*r_count[k]) : 0.0f;
            l_dot_sum[k] = (l_count[k] > 0.0f) ? l_dot_sum[k] / l_count[k] : 0.0f;
            r_dot_sum[k] = (r_count[k] > 0.0f) ? r_dot_sum[k] / r_count[k] : 0.0f;

            float denominator = l_count[0]* l_mean_norm + r_count[0] * r_mean_norm;
            if (denominator > 0.0f) {
                cosine = (l_dot_sum[k] + r_dot_sum[k]) / sqrtf(denominator);
            }
            total_gain += cosine * node->densities[k];
        }

        float penalty = 0.0f;
        if (obj_labels != nullptr && node->conflict_rho > 1e-6f) {
            float lc = s_labels[0], ls = s_labels[1], lsq = s_labels[2];
            float rc = s_labels[3], rs = s_labels[4], rsq = s_labels[5];
            
            float l_sse = (lc > 0) ? (lsq - (ls*ls)/lc) : 0.0f;
            float r_sse = (rc > 0) ? (rsq - (rs*rs)/rc) : 0.0f;
            float pc = lc+rc, ps = ls+rs, psq = lsq+rsq;
            float p_sse = (pc > 0) ? (psq - (ps*ps)/pc) : 0.0f;

            float H = (p_sse > 1e-10f) ? (l_sse + r_sse) / p_sse : 0.0f;
            if (H > 1.0f) H = 1.0f;

            penalty = lambda_penalty * node->conflict_rho * H;
            if (penalty > 1.0f) penalty = 1.0f;

            total_gain *= (1.0f - penalty);
        }

        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];
        
        split_scores[cand_idx] = total_gain * __ldg(feature_weights + feat_idx);
    }  
}


__global__ void split_score_l2_cuda(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const float* __restrict__ feature_weights,
    const float* __restrict__ obj_labels,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int min_data_in_leaf,
    float* __restrict__ split_scores,
    const int global_n_samples,
    const int n_num_features,
    const int n_objs,
    const float lambda_penalty
){

    extern __shared__ float sdata[];

    int n_samples = node->n_samples, n_cols = node->output_dim;
    int cand_idx = blockIdx.x;

    if (split_scores[cand_idx] == -CUDART_INF_F)
        return;

    if (node->depth > 0 && min_data_in_leaf == 0){
        if (candidate_numeric[cand_idx]){
            for (int i = 0; i < node->depth; ++i){
                if (node->is_numerics[i] && __ldg(&node->feature_values[i]) == __ldg(&candidate_values[cand_idx]) && __ldg(&node->feature_indices[i]) == __ldg(&candidate_indices[cand_idx])){
                    split_scores[cand_idx] = -CUDART_INF_F;
                    return;
                }
            }
        } else{
            for (int i = 0; i < node->depth; ++i){
                if (!node->is_numerics[i] && strcmpCuda(node->categorical_values + i * MAX_CHAR_SIZE, candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0 && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                    split_scores[cand_idx] = -CUDART_INF_F;
                    return;
                }
            }
        }
    }
    int threads_per_block = blockDim.x;

    int vec_stride = n_objs * n_cols;

    int thread_offset = 0;
    float *left_sum = &sdata[thread_offset];
    thread_offset += threads_per_block * vec_stride;
    float *l_count = &sdata[thread_offset];
    thread_offset += threads_per_block * n_objs;
    float* right_sum = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += threads_per_block * vec_stride;
    float* r_count = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += threads_per_block * n_objs;

    // Label Stats (L_c, L_s, L_sq, R_c, R_s, R_sq)
    float *s_labels = &sdata[thread_offset];

for (int k = 0; k < n_objs; ++k){
        l_count[threadIdx.x * n_objs + k] = 0.0f;
        r_count[threadIdx.x * n_objs + k] = 0.0f;
        for (int d = 0; d < n_cols; ++d){
            right_sum[threadIdx.x*vec_stride + k * n_cols + d] = 0.0f;
            left_sum[threadIdx.x*vec_stride + k * n_cols + d] = 0.0f;
        }
    }
    // Init Labels
    for(int i=0; i<6; ++i) s_labels[threadIdx.x*6 + i] = 0.0f;

    __syncthreads();
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = __ldg(&node->sample_indices[i]); // Access the spec

        float lbl = (obj_labels) ? __ldg(&obj_labels[sample_idx]) : 0.0f;

        if ((candidate_numeric[cand_idx] && __ldg(&obs[__ldg(&candidate_indices[cand_idx])*global_n_samples + sample_idx]) > __ldg(&candidate_values[cand_idx])) || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + __ldg(&candidate_indices[cand_idx]))* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0)){
            
            s_labels[threadIdx.x*6 + 3] += 1.0f; 
            s_labels[threadIdx.x*6 + 4] += lbl; 
            s_labels[threadIdx.x*6 + 5] += lbl*lbl;

            for (int k = 0; k < n_objs; ++k){
                size_t g_base = (size_t)k * global_n_samples * n_cols + (size_t)sample_idx * n_cols;
                int s_base = threadIdx.x * vec_stride + k * n_cols;

                r_count[threadIdx.x * n_objs + k] += 1.0f;
                for (int d = 0; d < n_cols; ++d){
                    right_sum[s_base + d] += __ldg(&grads[g_base + d]);
                }
            }
        } else {
            // Label Stats (Left)
            s_labels[threadIdx.x*6 + 0] += 1.0f; 
            s_labels[threadIdx.x*6 + 1] += lbl; 
            s_labels[threadIdx.x*6 + 2] += lbl*lbl;

            for (int k = 0; k < n_objs; ++k){
                size_t g_base = (size_t)k * global_n_samples * n_cols + (size_t)sample_idx * n_cols;
                int s_base = threadIdx.x * vec_stride + k * n_cols;

                l_count[threadIdx.x * n_objs + k] += 1.0f;
                for (int d = 0; d < n_cols; ++d){
                    left_sum[s_base + d] += __ldg(&grads[g_base + d]);
                }
            }
        }
    }
    __syncthreads();

     // // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            // Reduce Labels
            for(int j=0; j<6; ++j) s_labels[threadIdx.x*6 + j] += s_labels[(threadIdx.x + offset)*6 + j];

            // Reduce Sums and Counts
            for (int k = 0; k < n_objs; ++k){

                int s_base_curr = threadIdx.x * vec_stride + k * n_cols;
                int s_base_next = (threadIdx.x + offset) * vec_stride + k * n_cols;

                for (int d = 0; d < n_cols; ++d){
                    left_sum[s_base_curr + d]  += left_sum[s_base_next + d];
                    right_sum[s_base_curr + d] += right_sum[s_base_next + d];
                }
                l_count[threadIdx.x * n_objs + k]   += l_count[(threadIdx.x + offset) * n_objs + k];
                r_count[threadIdx.x * n_objs + k]   += r_count[(threadIdx.x + offset) * n_objs + k];
            }
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (threadIdx.x == 0) {
        if (l_count[0] < static_cast<float>(min_data_in_leaf) || r_count[0] < static_cast<float>(min_data_in_leaf)){
            split_scores[cand_idx] = -CUDART_INF_F;
            return;
        }  

        float total_gain = 0.0f;

        for (int k = 0; k < n_objs; ++k){
            int base = k * n_cols;
            float l_sq_sum = 0.0f, r_sq_sum = 0.0f;
            
            for (int d = 0; d < n_cols; ++d){
                l_sq_sum += left_sum[base + d] * left_sum[base + d];
                r_sq_sum += right_sum[base + d] * right_sum[base + d];
            }

            // Gain = ||Sum||^2 / N
            float l_gain = (l_count[k] > 0.0f) ? l_sq_sum / l_count[k] : 0.0f;
            float r_gain = (r_count[k] > 0.0f) ? r_sq_sum / r_count[k] : 0.0f;

            // Weighted Sum
            total_gain += (l_gain + r_gain) * node->densities[k];
        }
        // --- SPLIT-RL Penalty (Impurity H) ---
        float penalty = 0.0f;
        if (obj_labels != nullptr && node->conflict_rho > 1e-6f) {
            float lc = s_labels[0], ls = s_labels[1], lsq = s_labels[2];
            float rc = s_labels[3], rs = s_labels[4], rsq = s_labels[5];
            
            float l_sse = (lc > 0) ? (lsq - (ls*ls)/lc) : 0.0f;
            float r_sse = (rc > 0) ? (rsq - (rs*rs)/rc) : 0.0f;
            float pc = lc+rc, ps = ls+rs, psq = lsq+rsq;
            float p_sse = (pc > 0) ? (psq - (ps*ps)/pc) : 0.0f;

            float H = (p_sse > 1e-10f) ? (l_sse + r_sse) / p_sse : 0.0f;
            if (H > 1.0f) H = 1.0f;

            penalty = lambda_penalty * node->conflict_rho * H;
            if (penalty > 1.0f) penalty = 1.0f;
            
            total_gain *= (1.0f - penalty);
        }

        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];
        split_scores[cand_idx] = total_gain * __ldg(feature_weights + feat_idx);
    }  
}

__global__ void split_impurity_penalty_kernel(
    const float* __restrict__ obj_labels,        // Renamed from 'labels'
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const TreeNodeGPU* __restrict__ node,
    float* __restrict__ split_scores,            // In/Out: Gain -> Penalized Gain
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int n_candidates,
    const int global_n_samples,
    const float lambda_penalty)
{
    // One block per candidate
    int cand_idx = blockIdx.x;
    if (cand_idx >= n_candidates) return;

    // 1. Strict Equality Check for Invalid Splits
    if (split_scores[cand_idx] == -CUDART_INF_F) return;
    
    // Optimization: If node has no conflict, no need to calculate impurity
    float rho = node->conflict_rho;
    if (rho < 1e-6f) return; 

    // --- SHARED MEMORY SETUP ---
    extern __shared__ float sdata[];
    
    // Layout: 6 arrays of size [blockDim.x]
    // 0:L_Count, 1:L_Sum, 2:L_SqSum, 3:R_Count, 4:R_Sum, 5:R_SqSum
    int bdim = blockDim.x;
    
    float* l_count  = &sdata[0];
    float* l_sum    = &sdata[bdim];
    float* l_sq     = &sdata[bdim * 2];
    float* r_count  = &sdata[bdim * 3];
    float* r_sum    = &sdata[bdim * 4];
    float* r_sq     = &sdata[bdim * 5];

    // Initialize Local Registers
    l_count[threadIdx.x] = 0.0f; l_sum[threadIdx.x] = 0.0f; l_sq[threadIdx.x] = 0.0f;
    r_count[threadIdx.x] = 0.0f; r_sum[threadIdx.x] = 0.0f; r_sq[threadIdx.x] = 0.0f;

    __syncthreads();

    // 2. ACCUMULATE (Grid-Stride Loop)
    int n_samples = node->n_samples;
    for(int i = threadIdx.x; i < n_samples; i += bdim) {
        int sample_idx = __ldg(&node->sample_indices[i]);
        float val = __ldg(&obj_labels[sample_idx]); // Renamed access
        
        // Check Split
        bool is_greater = false;
        if (candidate_numeric[cand_idx]) {
             float f_val = __ldg(&obs[sample_idx + global_n_samples * __ldg(&candidate_indices[cand_idx])]);
             is_greater = f_val > __ldg(&candidate_values[cand_idx]);
        } else {
             const char* s_cat = &categorical_obs[(sample_idx*node->n_cat_features + __ldg(&candidate_indices[cand_idx]))* MAX_CHAR_SIZE];
             const char* c_cat = candidate_categories + cand_idx * MAX_CHAR_SIZE;
             is_greater = (strcmpCuda(s_cat, c_cat) == 0);
        }

        if (is_greater) {
            r_count[threadIdx.x] += 1.0f;
            r_sum[threadIdx.x]   += val;
            r_sq[threadIdx.x]    += val * val;
        } else {
            l_count[threadIdx.x] += 1.0f;
            l_sum[threadIdx.x]   += val;
            l_sq[threadIdx.x]    += val * val;
        }
    }
    __syncthreads();

    // 3. TREE REDUCTION
    for(int offset = bdim / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            l_count[threadIdx.x] += l_count[threadIdx.x + offset];
            l_sum[threadIdx.x]   += l_sum[threadIdx.x + offset];
            l_sq[threadIdx.x]    += l_sq[threadIdx.x + offset];
            
            r_count[threadIdx.x] += r_count[threadIdx.x + offset];
            r_sum[threadIdx.x]   += r_sum[threadIdx.x + offset];
            r_sq[threadIdx.x]    += r_sq[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // 4. FINAL CALCULATION (Thread 0)
    if (threadIdx.x == 0) {
        float lc = l_count[0];
        float rc = r_count[0];
        float ls = l_sum[0];
        float rs = r_sum[0];
        float lsq = l_sq[0];
        float rsq = r_sq[0];

        // SSE = Sum(x^2) - (Sum x)^2 / N
        float l_sse = (lc > 1e-6f) ? (lsq - (ls * ls) / lc) : 0.0f;
        float r_sse = (rc > 1e-6f) ? (rsq - (rs * rs) / rc) : 0.0f;
        
        // Parent SSE
        float pc = lc + rc;
        float ps = ls + rs;
        float psq = lsq + rsq;
        float p_sse = (pc > 1e-6f) ? (psq - (ps * ps) / pc) : 0.0f;

        // H: Relative Impurity (Remaining Variance / Original Variance)
        // Range: [0, 1]
        float H_impurity = 1.0f; 
        if (p_sse > 1e-10f) {
            H_impurity = (l_sse + r_sse) / p_sse;
        } else {
            H_impurity = 0.0f;
        }
        
        // Clamp H
        if (H_impurity > 1.0f) H_impurity = 1.0f;
        if (H_impurity < 0.0f) H_impurity = 0.0f;

        // --- APPLY PENALTY ---
        // Score *= (1 - lambda * rho * H)
        float penalty_factor = lambda_penalty * rho * H_impurity;
        
        if (penalty_factor > 1.0f) penalty_factor = 1.0f;
        if (penalty_factor < 0.0f) penalty_factor = 0.0f;

        split_scores[cand_idx] *= (1.0f - penalty_factor);
    }
}

__global__ void calc_node_conflict_kernel(
    const float* __restrict__ node_means, // Stacked: [Obj0][Obj1]...
    TreeNodeGPU* __restrict__ node,       // Output: node->conflict_rho
    const int n_objs,
    const int n_cols)
{
    // 1. One Block per Node (Launched with threads = n_cols)
    int d = threadIdx.x;
    if (d >= n_cols) return;

    // Shared memory for 2 reductions (Numerator and Denominator)
    extern __shared__ float sdata[];
    float* s_num = &sdata[0];
    float* s_den = &sdata[blockDim.x];

    // 2. Per-Dimension Accumulation
    // We iterate K (objectives) locally in registers.
    float sum_of_components = 0.0f; // (\sum \mu)^2
    float sum_of_squares = 0.0f;    // \sum (\mu^2)

    for (int k = 0; k < n_objs; ++k) {
        float val = node_means[k * n_cols + d];
        sum_of_components += val;
        sum_of_squares += val * val;
    }

    // Store partial results for reduction
    s_num[d] = sum_of_components * sum_of_components; // Contribution to ||Sum Mu||^2
    s_den[d] = sum_of_squares;                        // Contribution to Sum ||Mu||^2
    
    __syncthreads();

    // 3. Parallel Reduction (Summing across dimensions D)
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (d < offset) {
            s_num[d] += s_num[d + offset];
            s_den[d] += s_den[d + offset];
        }
        __syncthreads();
    }

    // 4. Final Calculation (Thread 0)
    if (d == 0) {
        float numerator = s_num[0];   // || Sum \mu ||^2
        float denominator = s_den[0]; // Sum || \mu ||^2

        float rho = 0.0f;
        
        // Avoid division by zero
        if (denominator > 1e-12f) {
            float ratio = numerator / denominator;
            
            // Numerical stability clamp (Ratio should be <= 1.0 mathematically)
            if (ratio > 1.0f) ratio = 1.0f; 
            
            rho = 1.0f - ratio;
        }

        // Store result in the node struct
        // Ensure you added 'float conflict_rho;' to TreeNodeGPU definition
        node->conflict_rho = rho;
    }
}

__global__ void split_conditional_sum_kernel(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int n_candidates,
    const int global_n_samples,
    const int n_objs,
    float* __restrict__ left_sum,
    float* __restrict__ right_sum,
    float* __restrict__ left_count,
    float* __restrict__ right_count){
    // Accumulate per thread partial sum
    size_t global_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int output_dim = __ldg(&node->output_dim);

    int n_node_samples = __ldg(&node->n_samples);
    // The size of one full "pass" over the node (Legacy size)
    size_t node_pass_stride = (size_t)n_node_samples * n_candidates; 
    
    // Total items to process across all objectives
    size_t total_items = (size_t)n_objs * node_pass_stride;
    if (global_idx < total_items){

        // If n_objs == 1: obj_idx is 0, rem is global_idx. 
        // This is ZERO REGRESSION for the single-objective case.
        int obj_idx = global_idx / node_pass_stride;
        size_t rem = global_idx % node_pass_stride;
        
        int sample_row = rem / n_candidates;
        int cand_idx = rem % n_candidates;

        size_t sum_obj_offset = (size_t)obj_idx * n_candidates * output_dim;
        size_t count_obj_offset = (size_t)obj_idx * n_candidates;

        int sample_idx = __ldg(&node->sample_indices[sample_row]); // Access the spec
        bool is_greater = (candidate_numeric[cand_idx] && __ldg(&obs[sample_idx +  global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx])) || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + candidate_indices[cand_idx])* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        
        int row_idx = sample_idx*output_dim;
        if (is_greater){
            for (int d = 0; d < output_dim; ++d){
                float eff_grad = __ldg(&grads[(row_idx + d) + obj_idx * global_n_samples * output_dim]);
                atomicAdd(right_sum + sum_obj_offset + cand_idx * output_dim + d, eff_grad);
            }
            atomicAdd(right_count + count_obj_offset + cand_idx, 1);
        } else {
            for (int d = 0; d < output_dim; ++d){
                float eff_grad = __ldg(&grads[(row_idx + d) + obj_idx * global_n_samples * output_dim]);
                atomicAdd(left_sum + sum_obj_offset + cand_idx * output_dim + d, eff_grad);
            }
            atomicAdd(left_count + count_obj_offset + cand_idx, 1);
        }
    }
}

__global__ void split_conditional_dot_kernel(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const TreeNodeGPU* __restrict__ node,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int n_candidates,
    const int global_n_samples,
    const int n_objs,
    const float* __restrict__ left_sum,
    const float* __restrict__ right_sum,
    const float* __restrict__ left_count,
    const float* __restrict__ right_count,
    float* __restrict__ ldot,
    float* __restrict__ rdot){

    int n_node_samples = __ldg(&node->n_samples);
    int n_cols = __ldg(&node->output_dim);

    // Stride for one full pass over the node (Legacy size)
    size_t node_pass_stride = (size_t)n_node_samples * n_candidates; 
    
    // Total items to process across all objectives
    size_t total_items = (size_t)n_objs * node_pass_stride;

    size_t global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (global_idx < total_items){
        int obj_idx = global_idx / node_pass_stride;
        size_t rem = global_idx % node_pass_stride;
        int sample_row = rem / n_candidates;
        int cand_idx = rem % n_candidates;

        float cdot = 0.0f;
        int cand_row = cand_idx*n_cols;

        int sample_idx = __ldg(&node->sample_indices[sample_row]); // Access the spec
        int row_idx = sample_idx*n_cols;

        bool is_greater = (candidate_numeric[cand_idx] && __ldg(&obs[sample_idx +  global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx])) || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + candidate_indices[cand_idx])* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        if (is_greater){
            for (int d = 0; d < n_cols; ++d){
                float eff_grads = __ldg(&grads[(row_idx + d) + obj_idx * global_n_samples * n_cols]);
                cdot += eff_grads * __ldg(&right_sum[obj_idx * n_candidates * n_cols + cand_row + d]);
            }
            cdot /= __ldg(&right_count[obj_idx * n_candidates + cand_idx]);
            atomicAdd(rdot + obj_idx * n_candidates + cand_idx, cdot);
        } else {
            for (int d = 0; d < n_cols; ++d){
                float eff_grads = __ldg(&grads[(row_idx + d) + obj_idx * global_n_samples * n_cols]);
                cdot += eff_grads * __ldg(&left_sum[obj_idx * n_candidates * n_cols + cand_row + d]);
            }
            cdot /= __ldg(&left_count[obj_idx * n_candidates + cand_idx]);
            atomicAdd(ldot + obj_idx * n_candidates + cand_idx, cdot);
        }
    }
}

__global__ void split_cosine_score_kernel(
    const TreeNodeGPU* __restrict__ node,
    const float* __restrict__ feature_weights,
    float* __restrict__ split_scores,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int n_candidates,
    const int n_objs,
    const float* __restrict__ lsum,
    const float* __restrict__ rsum,
    const float* __restrict__ lcount,
    const float* __restrict__ rcount,
    const float* __restrict__ ldot,
    const float* __restrict__ rdot,
    const int min_data_in_leaf, 
    const int n_num_features){

    int global_idx = blockIdx.x*blockDim.x + threadIdx.x;

    int cand_idx = global_idx % n_candidates;
    int obj_idx = global_idx / n_candidates;

    int cand_offset = obj_idx * n_candidates + cand_idx;

    int n_cols = __ldg(&node->output_dim);

    int cand_row = cand_idx*n_cols;
    int sum_offset = obj_idx * n_candidates * n_cols + cand_row;
    float lvalue, rvalue;

    if (split_scores[cand_offset] == -INFINITY)
        return;

    if (global_idx < n_candidates * n_objs){
        if (node->depth > 0 && min_data_in_leaf == 0){
            if (candidate_numeric[cand_idx]){
                for (int i = 0; i < node->depth; ++i){
                    if (node->is_numerics[i] && __ldg(&node->feature_values[i]) == __ldg(&candidate_values[cand_idx]) && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_offset] = -CUDART_INF_F;
                        return;
                    }
                }   
            } else {
                for (int i = 0; i < node->depth; ++i){
                    if (!node->is_numerics[i] && strcmpCuda(node->categorical_values + i * MAX_CHAR_SIZE, candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0 && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_offset] = -CUDART_INF_F;
                        return;
                    }
                }
            }
        }

        if (lcount[cand_offset] < static_cast<float>(min_data_in_leaf) || rcount[cand_offset] < static_cast<float>(min_data_in_leaf)){
            split_scores[cand_offset] = -CUDART_INF_F;
            return;
        } 

        float l_mean_norm = 0.0f, r_mean_norm = 0.0f;
        for (int d = 0; d < n_cols; ++d){
            lvalue = __ldg(lsum + sum_offset + d);
            rvalue = __ldg(rsum + sum_offset + d);
            l_mean_norm += lvalue * lvalue;
            r_mean_norm += rvalue * rvalue;
        }
        lvalue = __ldg(&lcount[cand_offset]);
        rvalue = __ldg(&rcount[cand_offset]);

        l_mean_norm = (lvalue > 0.0f) ? l_mean_norm / (lvalue * lvalue) : 0.0f;
        r_mean_norm = (rvalue > 0.0f) ? r_mean_norm / (rvalue * rvalue) : 0.0f;

        float denominator =  lvalue * l_mean_norm + rvalue * r_mean_norm;
        float numerator = ldot[cand_offset] + rdot[cand_offset];
        if (denominator == 0.0f){
            split_scores[cand_offset] = -CUDART_INF_F;
            return;
        }
        float cos = numerator / sqrtf(denominator) - node->scores[obj_idx];
        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];
        split_scores[cand_offset] = cos * __ldg(feature_weights + feat_idx);
    }
}

__global__ void split_l2_score_kernel(
    const TreeNodeGPU* __restrict__ node,
    const float* __restrict__ feature_weights,
    float* __restrict__ split_scores,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ r_num_mapping,
    const int* __restrict__ r_cat_mapping,
    const int n_candidates,
    const int n_objs,
    const float* __restrict__ lsum,
    const float* __restrict__ rsum,
    const float* __restrict__ lcount,
    const float* __restrict__ rcount,
    const int min_data_in_leaf,
    const int n_num_features){

    int global_idx = blockIdx.x*blockDim.x + threadIdx.x;

    int cand_idx = global_idx % n_candidates;
    int obj_idx = global_idx / n_candidates;

    int n_cols = __ldg(&node->output_dim);

    float lvalue, rvalue;

    int cand_row = cand_idx*n_cols;

    int cand_offset = obj_idx * n_candidates + cand_idx;
    int sum_offset = obj_idx * n_candidates * n_cols + cand_row;

    if (split_scores[cand_offset] == -INFINITY)
        return;
        
    if (global_idx < n_candidates * n_objs){
        if (node->depth > 0 && min_data_in_leaf == 0){
            if (candidate_numeric[cand_idx]){
                for (int i = 0; i < node->depth; ++i){
                    if (node->is_numerics[i] && __ldg(&node->feature_values[i]) == __ldg(&candidate_values[cand_idx]) && __ldg(&node->feature_indices[i]) == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_offset] = -CUDART_INF_F;
                        return;
                    }
                }   
            } else {
                for (int i = 0; i < node->depth; ++i){
                    if (!node->is_numerics[i] && strcmpCuda(node->categorical_values + i * MAX_CHAR_SIZE, candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0 && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_offset] = -CUDART_INF_F;
                        return;
                    }
                }
            }
        }

        if (lcount[cand_offset] < static_cast<float>(min_data_in_leaf) || rcount[cand_offset] < static_cast<float>(min_data_in_leaf)){
            split_scores[cand_offset] = -CUDART_INF_F;
            return;
        } 

        float l_mean_norm = 0.0f, r_mean_norm = 0.0f;
        for (int d = 0; d < n_cols; ++d){
            lvalue = __ldg(lsum + sum_offset + d);
            rvalue = __ldg(rsum + sum_offset + d);
            l_mean_norm += lvalue * lvalue;
            r_mean_norm += rvalue * rvalue;
        }
        lvalue = __ldg(&lcount[cand_offset]);
        rvalue = __ldg(&rcount[cand_offset]);
        l_mean_norm = (lvalue > 0.0f) ? l_mean_norm / lvalue : 0.0f; // n_count * l2 norm 
        r_mean_norm = (rvalue > 0.0f) ? r_mean_norm / rvalue : 0.0f; // n_count * l2 norm 

        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];
        split_scores[cand_offset] = ((l_mean_norm + r_mean_norm) - node->scores[obj_idx]) * __ldg(feature_weights + feat_idx);    
    }
}


__global__ void print_candidate_scores(
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    float* __restrict__ split_scores,
    const int n_candidates){

    if (threadIdx.x == 0)
    {
    for (int i = 0;  i < n_candidates; i += 1){

        if (candidate_numeric[i]){
            printf("Candidate %d: %f -> is_numeric --score-- %f\n",candidate_indices[i], candidate_values[i], split_scores[i]);
        }
        else{
            printf("Candidate %d: ", candidate_indices[i]);
            for (int k = 0; k < MAX_CHAR_SIZE; k++){
                if (candidate_categories[i * MAX_CHAR_SIZE + k] == '\0')
                    break;
                printf("%c", candidate_categories[i * MAX_CHAR_SIZE + k]);
            }
            printf(" -> is_categorical --score-- %f\n", split_scores[i]);
        }
        
  
    }
    }
}



__global__ void column_sums_reduce(
    const float * __restrict__ in,
    float * __restrict__ out,
    size_t n_cols,
    size_t n_rows){

  __shared__ float sdata[BLOCK_ROWS][BLOCK_COLS + 1]; // +1 to avoid bank conflicts
  size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
  size_t width_stride = gridDim.x*blockDim.x;
  // bitwise round-up
//   size_t full_width = (n_cols & (~((unsigned long long)(BLOCK_COLS -1)))) + ((n_cols & (BLOCK_COLS-1)) ? BLOCK_COLS : 0); // round up to next block
  size_t full_width = (n_cols + BLOCK_COLS - 1) / BLOCK_COLS * BLOCK_COLS;

  for (size_t col = idx; col < full_width; col += width_stride){          // grid-stride loop across matrix width
    float partial_sum = 0.0f;
    if (col < n_cols) { //
        size_t in_ptr = col + threadIdx.y * n_cols;
        for (size_t row = threadIdx.y; row < n_rows; row += BLOCK_ROWS) { // Block-stride loop
            partial_sum += in[in_ptr];
            in_ptr += BLOCK_ROWS * n_cols;
        }
    }
    
    sdata[threadIdx.y][threadIdx.x] = partial_sum;
    __syncthreads();

    float val = sdata[threadIdx.x][threadIdx.y];
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    __syncthreads();
    if (threadIdx.x == 0) 
        sdata[0][threadIdx.y] = val;

    __syncthreads();
    if ((threadIdx.y == 0) && ((col) < n_cols)) 
        out[col] = sdata[0][threadIdx.x];

  }
}

__global__ void reduce_leaf_sum(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,       // Stacked: [Obj0][Obj1]...
    float* __restrict__ values,
    const TreeNodeGPU* __restrict__ node,
    const int n_samples,                   // Global sample count (loop limit)
    const int global_idx,                  // Offset into 'values' array
    const int n_objs
) {
    // Dynamic Shared Memory Layout:
    // 1. Sums for each objective: [n_objs * blockDim.x]
    // 2. Count: [blockDim.x]
    extern __shared__ float sdata[];
    
    float* s_sums  = sdata;
    float* s_count = &sdata[n_objs * blockDim.x];

    int d = blockIdx.x; // Current Output Dimension (Action Dim)
    int output_dim = node->output_dim;

    // 1. Initialize Shared Memory
    for (int k = 0; k < n_objs; ++k) {
        s_sums[k * blockDim.x + threadIdx.x] = 0.0f;
    }
    s_count[threadIdx.x] = 0.0f;
    
    // Clear Global Output
    if (threadIdx.x == 0) values[global_idx + d] = 0.0f;
    
    __syncthreads();

    // 2. Iterate over ALL Samples
    // (Note: This re-checks the tree path for every sample. 
    //  If you have sample_indices available, iterating those is much faster.)
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x) {
        
        bool passed = true;
        
        // --- PATH TRAVERSAL CHECK ---
        // Verify if this sample actually falls into this leaf
        int cat_row_idx = sample_idx * node->n_cat_features;
        
        for (int condIdx = node->depth - 1; condIdx >= 0; --condIdx) {
            bool condition_met = false;
            int feat_idx = node->feature_indices[condIdx];
            
            if (node->is_numerics[condIdx]) {
                // Column-major access for obs
                float val = obs[sample_idx + n_samples * feat_idx];
                condition_met = val > node->feature_values[condIdx];
            } else {
                // Categorical check
                condition_met = (strcmpCuda(&categorical_obs[(cat_row_idx + feat_idx) * MAX_CHAR_SIZE], 
                                            node->categorical_values + condIdx * MAX_CHAR_SIZE) == 0);
            }
            
            // Check against the direction recorded in the node
            // inequality_directions: 1 for Right (> or ==), 0 for Left
            if (condition_met != node->inequality_directions[condIdx]) {
                passed = false;
                break;
            }
        }

        // --- ACCUMULATE ---
        if (passed) {
            s_count[threadIdx.x] += 1.0f;

            // Base index for this sample and dimension
            size_t base_idx = (size_t)sample_idx * output_dim + d;
            // Accumulate gradient for EACH objective
            for (int k = 0; k < n_objs; ++k) {
                // Jump to the correct layer
                size_t obj_stride = (size_t)k * n_samples * output_dim;
                float g = __ldg(&grads[base_idx + obj_stride]);
                
                // Store in shared memory slot for Obj K
                s_sums[k * blockDim.x + threadIdx.x] += g;
            }
        }
    }
    __syncthreads();

    // 3. REDUCTION
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            // Reduce Count
            s_count[threadIdx.x] += s_count[threadIdx.x + offset];

            // Reduce Sums for all objectives
            for (int k = 0; k < n_objs; ++k) {
                int base = k * blockDim.x;
                s_sums[base + threadIdx.x] += s_sums[base + threadIdx.x + offset];
            }
        }
        __syncthreads();
    }

    // 4. FINALIZE (Thread 0)
    if (threadIdx.x == 0) {
        float total_count = s_count[0];
        
        if (total_count > 0.0f) {
            float weighted_value = 0.0f;

            // Weighted Mixture of Means
            // V = Sum( c_k * (Sum_G_k / N) )
            for (int k = 0; k < n_objs; ++k) {
                float total_sum_k = s_sums[k * blockDim.x];
                float mean_k = total_sum_k / total_count;
                
                weighted_value += mean_k * node->densities[k];
// #ifdef DEBUG
//             printf("N_objectives: %d, k: %d, Leaf Dim %d: Count = %f, Weighted Value = %f, total_sum_k: %f mean_k: %f, node->densities[%d]: %f\n", n_objs, k, d, total_count, weighted_value, total_sum_k, mean_k, k, node->densities[k]);
// #endif 
            }

            values[global_idx + d] = weighted_value;
        }
    }
}


__global__ void node_column_mean_reduce(
    const float * __restrict__ in,
    float * __restrict__ out,
    size_t n_cols,
    size_t global_n_rows,
    const TreeNodeGPU* __restrict__ node,
    const int n_objs){

  __shared__ float sdata[BLOCK_ROWS][BLOCK_COLS + 1];
  size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
  size_t virtual_col = n_cols * n_objs;
  size_t width_stride = gridDim.x*blockDim.x;
  size_t n_rows = node->n_samples;

  // bitwise round-up
  size_t full_width = (virtual_col & (~((unsigned long long)(BLOCK_COLS -1)))) + ((virtual_col & (BLOCK_COLS-1)) ? BLOCK_COLS : 0); // round up to next block

  for (size_t global_col = idx; global_col < full_width; global_col+=width_stride){ // grid-stride loop across matrix width
    
    size_t col = global_col % n_cols;
    size_t obj_idx = global_col / n_cols;

    sdata[threadIdx.y][threadIdx.x] = 0;
    for (size_t row = threadIdx.y; row < n_rows; row+=BLOCK_ROWS){ // block-stride loop across matrix height
        float eff_in = 0.0f;
        eff_in += in[(col + node->sample_indices[row]*n_cols) + obj_idx * global_n_rows * n_cols];
        sdata[threadIdx.y][threadIdx.x] += (global_col < virtual_col) ? eff_in : 0;
    }
    __syncthreads();
    float tmp = sdata[threadIdx.x][threadIdx.y];
    for (int i = WARP_SIZE >>1; i > 0; i >>= 1)                       // warp-wise parallel sum reduction
      tmp += __shfl_xor_sync(0xFFFFFFFFU, tmp, i);
    __syncthreads();
    if (threadIdx.x == 0) 
        sdata[0][threadIdx.y]  = tmp;
    __syncthreads();
    if ((threadIdx.y == 0) && (global_col < virtual_col)) 
        out[global_col] = sdata[0][threadIdx.x] / static_cast<float>(n_rows);
  }
}

__global__ void node_l2_kernel(
    TreeNodeGPU* __restrict__ node,
    const float* __restrict__ mean,
    const int n_objs
    ){

    int obj_idx = blockIdx.x;
    
    int idx = threadIdx.x;
    if (idx == 0){
        float mean_squared_norm = 0.0f;
        for (int i = 0; i < node->output_dim; ++i)
            mean_squared_norm += (mean[obj_idx * node->output_dim + i]*mean[obj_idx * node->output_dim + i]);
            
        node->scores[obj_idx] = (node->node_idx > 0) ? mean_squared_norm * static_cast<float>(node->n_samples) : 0.0f; 
    }
}


__global__ void node_cosine_kernel(
    TreeNodeGPU* __restrict__ node,
    const float* __restrict__ grads,
    float* __restrict__ mean,
    const int n_objs, // number of policy objectives
    const int global_n_samples
    ){

    extern __shared__ float sdata[];
    int n_samples = node->n_samples, n_cols = node->output_dim;    
    int obj_idx = blockIdx.x;
    int thread_offset = 0;
    float *dot_sum = &sdata[thread_offset];
    dot_sum[threadIdx.x] = 0.0f;
    __syncthreads();
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = node->sample_indices[i]; // Access the spec
        int row_idx = sample_idx*n_cols;
        
        for (int d = 0; d < n_cols ; ++d){
            float mixture_grads = grads[(row_idx + d) + obj_idx * global_n_samples * n_cols];
            dot_sum[threadIdx.x] += mixture_grads * mean[obj_idx * n_cols + d];
        }
    }
    __syncthreads();

     // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            dot_sum[threadIdx.x] += dot_sum[threadIdx.x + offset]; 
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (threadIdx.x == 0){
        float cosine = 0.0f;
        float mean_norm = 0.0f;
        for (int d = 0; d < n_cols; ++d)
            mean_norm += mean[obj_idx * n_cols + d]*mean[obj_idx * n_cols + d];
        float denominator = static_cast<float>(n_samples) * mean_norm;
        if (denominator > 0) {
            cosine = dot_sum[0] / sqrtf(denominator);
        }
        node->scores[obj_idx] = (node->node_idx > 0) ? cosine : 0.0f;
    }  
}


TreeNodeGPU* allocate_root_tree_node(
    dataSet *dataset,
    ensembleMetaData *metadata){

    cudaError_t error;
    TreeNodeGPU* node;
    error = allocateCudaMemory((void**)&node, sizeof(TreeNodeGPU), "when trying to allocate TreeNodeGPU");
    if (error != cudaSuccess) {
        return nullptr;
    }
    // Allocate temporary node on host to set the value
    TreeNodeGPU tempNode;
    tempNode.depth = 0;
    tempNode.n_samples = dataset->n_samples;
    tempNode.n_num_features = metadata->n_num_features;
    tempNode.n_cat_features = metadata->n_cat_features;
    tempNode.output_dim = metadata->output_dim;
    tempNode.n_objs = metadata->n_objs;
    tempNode.node_idx = 0;
    tempNode.conflict_rho = 0.0f;

    tempNode.sample_indices = nullptr;
    tempNode.feature_indices = nullptr;
    tempNode.feature_values = nullptr;
    tempNode.edge_weights = nullptr;
    tempNode.inequality_directions = nullptr;
    tempNode.is_numerics = nullptr;
    tempNode.categorical_values = nullptr;
    tempNode.scores = nullptr;
    tempNode.densities = nullptr;

    size_t data_size = sizeof(int) * dataset->n_samples + // sample_indices
                        sizeof(float) * metadata->n_objs + // scores
                        sizeof(float) * metadata->n_objs;  // densities

    char *data;
    error = allocateCudaMemory((void**)&data, data_size, "when trying to allocate root data");
    if (error != cudaSuccess) {
        cudaFree(node);
        return nullptr;
    }
    tempNode.sample_indices = (int*)data;
    size_t trace = sizeof(int) * dataset->n_samples;
    tempNode.scores = (float*)(data + trace);
    trace += sizeof(float) * metadata->n_objs;
    tempNode.densities = (float*)(data + trace);
    cudaMemset(data, 0, data_size);

    int n_blocks = dataset->n_samples / THREADS_PER_BLOCK + 1;
    iota_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(tempNode.sample_indices, dataset->n_samples);
    n_blocks = metadata->n_objs / THREADS_PER_BLOCK + 1;
    ones_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(tempNode.densities, metadata->n_objs);

    cudaDeviceSynchronize();

    cudaMemcpy(node, &tempNode, sizeof(TreeNodeGPU), cudaMemcpyHostToDevice);
    // cudaMemcpy(&(node->sample_indices), &tempNode.sample_indices, sizeof(int*), cudaMemcpyHostToDevice);
    // cudaMemcpy(&(node->densities), &tempNode.densities, sizeof(float*), cudaMemcpyHostToDevice);
    // cudaMemcpy(&(node->scores), &tempNode.scores, sizeof(float*), cudaMemcpyHostToDevice);

    if (dataset->obj_labels->data != nullptr){
        cudaDeviceSynchronize();
        int threads_per_block;
        get_tpb_dimensions(dataset->n_samples, metadata->n_objs, threads_per_block);
        size_t shared_mem_size = threads_per_block * sizeof(float);
        calc_node_densities_kernel<<<metadata->n_objs, threads_per_block, shared_mem_size>>>(node, nullptr, dataset->obj_labels->data);
        cudaDeviceSynchronize();
    }
    return node;
}

void allocate_child_tree_node(
    TreeNodeGPU* host_parent,
    TreeNodeGPU** device_child,
    const int n_objs){

    TreeNodeGPU host_child;
    int n_samples = host_parent->n_samples;
    int depth = host_parent->depth + 1;

    host_child.depth = depth;
    host_child.n_samples = n_samples;
    host_child.output_dim = host_parent->output_dim;
    host_child.node_idx = -1;
    host_child.conflict_rho = 0.0f;
    host_child.n_num_features = host_parent->n_num_features;
    host_child.n_cat_features = host_parent->n_cat_features;
    host_child.n_objs = host_parent->n_objs;
    host_child.sample_indices = nullptr;
    host_child.feature_indices = nullptr;
    host_child.feature_values = nullptr;
    host_child.inequality_directions = nullptr;
    host_child.edge_weights = nullptr;
    host_child.is_numerics = nullptr;
    host_child.categorical_values = nullptr;
    host_child.scores = nullptr;
    host_child.densities = nullptr;

    char* device_memory_block;
    size_t conditions_size = sizeof(int) * n_samples // sample_indices
                + sizeof(int) * depth    // feature_indices
                + sizeof(float) * depth  // feature_values
                + sizeof(float) * depth   // edge_weights
                + sizeof(bool) * depth   // inequality_directions
                + sizeof(bool) * depth   // is_numerics
                + sizeof(float) * n_objs // scores
                + sizeof(float) * n_objs // densities
                + sizeof(char) * depth * MAX_CHAR_SIZE; // categorical_values

    cudaError_t error = allocateCudaMemory((void**)&device_memory_block, conditions_size, "CUDA allocate child tree node error:");
    if (error != cudaSuccess) {
        return;
    }
    cudaMemset(device_memory_block, 0, conditions_size);
    size_t trace = 0;
    host_child.sample_indices = (int*)device_memory_block;
    trace += sizeof(int) * n_samples;
    host_child.feature_indices = (int*)(device_memory_block + trace);
    trace += sizeof(int) * depth;
    host_child.feature_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * depth;
    host_child.edge_weights = (float*)(device_memory_block + trace);
    trace += sizeof(float) * depth;
    host_child.scores = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_objs;
    host_child.densities = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_objs;
    host_child.inequality_directions = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * depth;
    host_child.is_numerics = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * depth;
    host_child.categorical_values = (char*)(device_memory_block + trace);

    int n_blocks = n_objs / THREADS_PER_BLOCK + 1;
    ones_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(host_child.densities, n_objs);

    cudaDeviceSynchronize();

    error = allocateCudaMemory((void**)&(*device_child), sizeof(TreeNodeGPU), "CUDA allocate child tree node error when trying to allocate child:");
    if (error != cudaSuccess){
        cudaFree(device_memory_block);
        *device_child = nullptr;
        return;
    }
    cudaMemcpy(*device_child, &host_child, sizeof(TreeNodeGPU), cudaMemcpyHostToDevice);
}

void allocate_child_tree_nodes(
    dataSet *dataset,
    TreeNodeGPU* parent_node,
    TreeNodeGPU* host_parent,
    TreeNodeGPU** left_child,
    TreeNodeGPU** right_child,
    candidatesData *candidata,
    splitDataGPU *split_data,
    ensembleMetaData *metadata){

    int n_samples = host_parent->n_samples;
    int depth = host_parent->depth + 1;
    allocate_child_tree_node(host_parent, left_child, metadata->n_objs);
    allocate_child_tree_node(host_parent, right_child, metadata->n_objs);

    int n_blocks, threads_per_block;
    get_grid_dimensions(n_samples, n_blocks, threads_per_block);
    partition_samples_kernel<<<n_blocks, threads_per_block>>>(dataset->obs->data, dataset->categorical_obs->data, parent_node, *left_child, *right_child, candidata->candidate_indices, candidata->candidate_values, candidata->candidate_categories, candidata->candidate_numeric, split_data->best_idx, split_data->tree_counters, dataset->n_samples) ;
    cudaDeviceSynchronize();
    int n_threads = WARP_SIZE*((MAX_CHAR_SIZE + WARP_SIZE - 1) / WARP_SIZE);
    update_child_nodes_kernel<<<depth, n_threads>>>(parent_node, *left_child, *right_child, split_data->tree_counters, candidata->candidate_indices, candidata->candidate_values, candidata->candidate_numeric, candidata->candidate_categories, split_data->best_idx, split_data->best_score);
    cudaDeviceSynchronize();
    if (dataset->obj_labels->data != nullptr){
        get_tpb_dimensions(n_samples, metadata->n_objs, threads_per_block);
        size_t shared_mem_size = threads_per_block * sizeof(float);
        calc_node_densities_kernel<<<metadata->n_objs, threads_per_block, shared_mem_size>>>(*left_child, parent_node, dataset->obj_labels->data);
        calc_node_densities_kernel<<<metadata->n_objs, threads_per_block, shared_mem_size>>>(*right_child, parent_node, dataset->obj_labels->data);
        cudaDeviceSynchronize();
    }
    
}

void add_leaf_node(
    const TreeNodeGPU *node,
    const int depth,
    ensembleMetaData *metadata,
    ensembleData *edata,
    dataSet *dataset){
    int leaf_idx = metadata->n_leaves, tree_idx = metadata->n_trees; 
    if (depth > 0){
        int n_threads = WARP_SIZE*((MAX_CHAR_SIZE + WARP_SIZE - 1) / WARP_SIZE);
        int global_idx = (metadata->grow_policy == GREEDY) ? leaf_idx : tree_idx;
        copy_node_to_data<<<depth, n_threads>>>(node, edata->depths, edata->feature_indices, edata->feature_values, edata->edge_weights, edata->inequality_directions, edata->is_numerics, edata->categorical_values,
            edata->densities, 
#ifdef DEBUG
            edata->n_samples,
#endif
            global_idx, leaf_idx, metadata->max_depth, metadata->n_objs);
        cudaDeviceSynchronize();
    }

    int threads_per_block = WARP_SIZE*((dataset->n_samples  + WARP_SIZE - 1 )/ WARP_SIZE);
    if (threads_per_block > THREADS_PER_BLOCK) {
        threads_per_block = THREADS_PER_BLOCK;
    }
    size_t shared_mem = sizeof(float)*2*threads_per_block * (1 + metadata->n_objs);
    reduce_leaf_sum<<<metadata->output_dim, threads_per_block, shared_mem>>>(dataset->obs->data, dataset->categorical_obs->data, dataset->grads->data, edata->values, node, dataset->n_samples, leaf_idx*metadata->output_dim, metadata->n_objs);
    cudaDeviceSynchronize();
       
    metadata->n_leaves += 1;
}

__global__ void copy_node_to_data(
    const TreeNodeGPU* __restrict__ node,
    int* __restrict__ depths,
    int* __restrict__ feature_indices,
    float* __restrict__ feature_values,
    float* __restrict__ edge_weights,
    bool* __restrict__ inequality_directions,
    bool* __restrict__ is_numerics,
    char * __restrict__  categorical_values,
    float * __restrict__  densities,
#ifdef DEBUG
    int* __restrict__ n_samples,
#endif
    const int global_idx,
    const int leaf_idx,
    const int max_depth,
    const int n_objs
)
    {
    if (blockIdx.x == 0 && threadIdx.x == 0){
        depths[global_idx] = node->depth;
        for (int i = 0; i < n_objs; ++i)
            densities[leaf_idx * n_objs + i] = node->densities[i];
#ifdef DEBUG
            n_samples[leaf_idx] = node->n_samples;
#endif
    }

    if (blockIdx.x < node->depth){
        if (threadIdx.x == 0){
            feature_indices[global_idx*max_depth + blockIdx.x] = node->feature_indices[blockIdx.x];
            feature_values[global_idx*max_depth + blockIdx.x] = node->feature_values[blockIdx.x];
            inequality_directions[leaf_idx*max_depth + blockIdx.x] = node->inequality_directions[blockIdx.x];
            edge_weights[leaf_idx*max_depth + blockIdx.x] = node->edge_weights[blockIdx.x];
            is_numerics[global_idx*max_depth + blockIdx.x] = node->is_numerics[blockIdx.x];
        }  
        if (threadIdx.x < MAX_CHAR_SIZE){
            categorical_values[(global_idx*max_depth + blockIdx.x) * MAX_CHAR_SIZE + threadIdx.x] = node->categorical_values[blockIdx.x * MAX_CHAR_SIZE + threadIdx.x];
        }   
    }
}

__global__ void print_tree_indices_kernel(const int* __restrict__ tree_indices, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 )
    {
        for (int i = 0 ; i < size ; ++i)
            printf("tree_indices[%d] = %d\n", i, tree_indices[i]);
    }
        
}

__global__ void partition_samples_kernel(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    TreeNodeGPU* __restrict__ parent_node,
    TreeNodeGPU* __restrict__ left_child,
    TreeNodeGPU* __restrict__ right_child,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const char* __restrict__ candidate_categories,
    const bool* __restrict__ candidate_numeric,
    const int* __restrict__ best_idx,
    int* __restrict__ tree_counters,
    const int global_n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0){
        tree_counters[0] = 0;
        tree_counters[1] = 0;
    }
    __syncthreads();

    if (idx < parent_node->n_samples) {
        int sample_idx = parent_node->sample_indices[idx];
        int best_idx_ = *best_idx;
        bool is_numeric = candidate_numeric[best_idx_];

        bool is_greater;
        if (is_numeric){
            is_greater = __ldg(&obs[sample_idx +  global_n_samples *  __ldg(&candidate_indices[best_idx_])]) > __ldg(&candidate_values[best_idx_]);
        } else {
            is_greater = strcmpCuda(categorical_obs + (sample_idx * parent_node->n_cat_features + candidate_indices[best_idx_])*MAX_CHAR_SIZE, candidate_categories + best_idx_ * MAX_CHAR_SIZE) == 0;
        }
        // Determine if the sample goes to the left or right partition
        int pos;
        if (is_greater) {
            pos = atomicAdd(tree_counters + 1, 1);
            right_child->sample_indices[pos] = sample_idx;
        } else {
            pos = atomicAdd(tree_counters + 0, 1);
            left_child->sample_indices[pos] = sample_idx;
        }
    }
}

void free_tree_node(TreeNodeGPU* node){
    if (node != nullptr){
        cudaError_t err;
        TreeNodeGPU *temp_node = new TreeNodeGPU;
        cudaMemcpy(temp_node, node, sizeof(TreeNodeGPU), cudaMemcpyDeviceToHost);
        // sample indices points to the start of the large memory block allocated -> need to release the entire block
        if (temp_node->sample_indices != nullptr){
           err = cudaFree(temp_node->sample_indices); 
            if (err != cudaSuccess) {
                std::cerr << "CUDA Error in freeing sample_indices: " << cudaGetErrorString(err) << std::endl;
            }
        }
        err = cudaFree(node);
        if (err != cudaSuccess) {
                std::cerr << "CUDA Error in freeing node: " << cudaGetErrorString(err) << std::endl;
        }
        delete temp_node;
    }
}

__global__ void print_tree_node(const TreeNodeGPU* __restrict__ node){
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx == 0){
        printf("##### TreenodeGPU %d #####\n", node->node_idx);
        printf("%d samples %d num_features %d cat_features %d output dim %d depth \n", node->n_samples, node->n_num_features, node->n_cat_features, node->output_dim, node->depth);
        printf("conflict_rho: %f\n", node->conflict_rho);
        if (node->n_objs > 1){
            printf("scores: [");
            for (int i = 0; i < node->n_objs; ++i){
                printf("%f", node->scores[i]);
                if (i < node->n_objs - 1)
                    printf(", ");
            }
            printf("]\n");
        } else{
            printf("score: %f\n", node->scores[0]);
        }
        printf("sample indices [");
        for (int i = 0; i < node->n_samples; ++i){
            printf("%d", node->sample_indices[i]);
            if (i < node->n_samples - 1)
                printf(", ");
        }
        printf("]\n");
        printf("split_conditions : [");
        for (int i = 0; i < node->depth; ++i){
            if (node->is_numerics[i]){
                 printf("%d: (%d = %d > %f)", i, node->inequality_directions[i], node->feature_indices[i], node->feature_values[i]); 
            } else {
                printf("%d: %d = (%d == ", i, node->inequality_directions[i], node->feature_indices[i] + node->n_num_features);
                for (int j = 0; j < MAX_CHAR_SIZE; j++){
                    if (node->categorical_values[i * MAX_CHAR_SIZE + j] == '\0')
                        break;
                    printf("%c", node->categorical_values[i * MAX_CHAR_SIZE + j]);

                }
                printf(")");
                
            }
           
            if (i < node->depth - 1)
                printf(", ");
        }
        printf("]\n");
        printf("##### END ######\n");
     }
}

__global__ void print_vector_kernel(const float* __restrict__ vec, const int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0){
        printf("vec: [");
        for (int i = 0; i < size; ++i){
            printf("%f", vec[i]);
            if (i < size - 1)
                printf(", ");
        }
        printf("]\n");
    }
}

__global__ void update_child_nodes_kernel(
    const TreeNodeGPU* __restrict__ parent_node,
    TreeNodeGPU* __restrict__ left_child,
    TreeNodeGPU* __restrict__ right_child, 
    int* __restrict__ tree_counters,
    const int* __restrict__ candidate_indices,
    const float* __restrict__ candidate_values,
    const bool* __restrict__ candidate_numeric,
    const char* __restrict__ candidate_categories,
    const int* __restrict__ best_idx,
    const float* __restrict__ best_score
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x < parent_node->depth){
        if (threadIdx.x == 0){
            left_child->feature_indices[blockIdx.x] = parent_node->feature_indices[blockIdx.x];
            left_child->feature_values[blockIdx.x] = parent_node->feature_values[blockIdx.x];
            left_child->inequality_directions[blockIdx.x] = parent_node->inequality_directions[blockIdx.x];
            left_child->edge_weights[blockIdx.x] = parent_node->edge_weights[blockIdx.x];
            left_child->is_numerics[blockIdx.x] = parent_node->is_numerics[blockIdx.x];
            right_child->feature_indices[blockIdx.x] = parent_node->feature_indices[blockIdx.x];
            right_child->feature_values[blockIdx.x] = parent_node->feature_values[blockIdx.x];
            right_child->inequality_directions[blockIdx.x] = parent_node->inequality_directions[blockIdx.x];
            right_child->edge_weights[blockIdx.x] = parent_node->edge_weights[blockIdx.x];
            right_child->is_numerics[blockIdx.x] = parent_node->is_numerics[blockIdx.x];
        }
        if (threadIdx.x < MAX_CHAR_SIZE){
            left_child->categorical_values[blockIdx.x * MAX_CHAR_SIZE + threadIdx.x] = parent_node->categorical_values[blockIdx.x * MAX_CHAR_SIZE + threadIdx.x];
            right_child->categorical_values[blockIdx.x * MAX_CHAR_SIZE + threadIdx.x] = parent_node->categorical_values[blockIdx.x * MAX_CHAR_SIZE + threadIdx.x];
        }

    } else if (blockIdx.x == parent_node->depth){
        if (threadIdx.x == 0){
            left_child->feature_indices[blockIdx.x] = candidate_indices[*best_idx];
            left_child->feature_values[blockIdx.x] = candidate_values[*best_idx];
            left_child->inequality_directions[blockIdx.x] = false;
            left_child->edge_weights[blockIdx.x] = (tree_counters[0] + tree_counters[1] > 0) ? static_cast<float>(tree_counters[0]) / (static_cast<float>(tree_counters[0]) + static_cast<float>(tree_counters[1])) : 0.0f;
            left_child->is_numerics[blockIdx.x] = candidate_numeric[*best_idx];
            right_child->feature_indices[blockIdx.x] = candidate_indices[*best_idx];
            right_child->feature_values[blockIdx.x] = candidate_values[*best_idx];
            right_child->inequality_directions[blockIdx.x] = true;
            right_child->edge_weights[blockIdx.x] = (tree_counters[0] + tree_counters[1] > 0) ? static_cast<float>(tree_counters[1]) / (static_cast<float>(tree_counters[0]) + static_cast<float>(tree_counters[1])) : 0.0f;
            right_child->is_numerics[blockIdx.x] = candidate_numeric[*best_idx];
        }
        if (threadIdx.x < MAX_CHAR_SIZE){
            left_child->categorical_values[blockIdx.x * MAX_CHAR_SIZE + threadIdx.x] = candidate_categories[(*best_idx)*MAX_CHAR_SIZE + threadIdx.x] ;
            right_child->categorical_values[blockIdx.x * MAX_CHAR_SIZE + threadIdx.x] = candidate_categories[(*best_idx)*MAX_CHAR_SIZE + threadIdx.x];
        }
    }

    if (idx == 0){
        left_child->node_idx = tree_counters[2] + 1;
        right_child->node_idx = tree_counters[2] + 2;
        left_child->n_samples = tree_counters[0];
        right_child->n_samples = tree_counters[1];
        tree_counters[2] += 2;
    }
}


__global__ void calc_node_densities_kernel(
    TreeNodeGPU* __restrict__ node,
    const TreeNodeGPU* __restrict__ parent_node,
    const float* __restrict__ obj_labels){

    int obj_idx = blockIdx.x;

    extern __shared__ float s_label_count[];
    if (node->n_samples == 0)
        return;

    s_label_count[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int idx = threadIdx.x; idx < node->n_samples; idx += blockDim.x) {
        int sample_idx = node->sample_indices[idx];
        if (static_cast<int>(obj_labels[sample_idx]) == obj_idx)
            s_label_count[threadIdx.x] += 1;
    }
    __syncthreads();
    // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            s_label_count[threadIdx.x] += s_label_count[threadIdx.x + offset]; 
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        node->densities[obj_idx] = s_label_count[threadIdx.x] / static_cast<float>(node->n_samples);

// #ifdef DEBUG
//     printf("Node %d Obj %d: Density: %.4f (Count: %.0f / %d)\n", 
//                node->node_idx, 
//                obj_idx, 
//                node->densities[obj_idx], 
//                s_label_count[threadIdx.x], 
//                node->n_samples);
// #endif
    }
}

void fit_tree_oblivious_cuda(
    dataSet *dataset,
    ensembleData *edata,
    ensembleMetaData *metadata,
    candidatesData *candidata,
    splitDataGPU *split_data){

    allocate_ensemble_memory_cuda(metadata, edata);
    cudaMemcpy(edata->tree_indices + metadata->n_trees, &metadata->n_leaves, sizeof(int), cudaMemcpyHostToDevice);

    TreeNodeGPU **tree_nodes = (TreeNodeGPU **)malloc((1 << metadata->max_depth) * sizeof(TreeNodeGPU *));
    // for oblivious trees
    TreeNodeGPU **child_tree_nodes = (TreeNodeGPU **)malloc((1 << metadata->max_depth) * sizeof(TreeNodeGPU *));

    int crnt_node_ptr_idx = 0, host_status;
    TreeNodeGPU *crnt_node;
    TreeNodeGPU *root_node = allocate_root_tree_node(dataset, metadata);
    tree_nodes[crnt_node_ptr_idx] = root_node;
    crnt_node_ptr_idx++;
    TreeNodeGPU host_node;
    int depth = 0;

    int threads_per_block;
    calc_parallelism(candidata->n_candidates, metadata->output_dim, threads_per_block, metadata->split_score_func);

    while(depth < metadata->max_depth){
        cudaMemset(split_data->split_scores, 0, split_data->size);
        
        evaluate_oblivious_splits_cuda(dataset, edata, tree_nodes, depth, candidata, metadata, split_data);
        cudaMemcpy(&host_status, split_data->best_idx, sizeof(int), cudaMemcpyDeviceToHost);
        if (host_status < 0)
            break;
        for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
            TreeNodeGPU *left_child = nullptr, *right_child = nullptr;
            crnt_node = tree_nodes[node_idx];
            cudaMemcpy(&host_node, crnt_node, sizeof(TreeNodeGPU), cudaMemcpyDeviceToHost);
            allocate_child_tree_nodes(dataset, crnt_node, &host_node, &left_child, &right_child, candidata, split_data, metadata);
            child_tree_nodes[node_idx*2] = left_child;
            child_tree_nodes[node_idx*2+ 1] = right_child;
            free_tree_node(crnt_node);
        }
        depth += 1;
        for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
            tree_nodes[node_idx] = child_tree_nodes[node_idx];
            child_tree_nodes[node_idx] = nullptr;
        }
    }
    for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
        add_leaf_node(tree_nodes[node_idx], depth, metadata, edata, dataset);
        free_tree_node(tree_nodes[node_idx]);
    }

    root_node = nullptr;
    metadata->n_trees++;
    free(tree_nodes);
    free(child_tree_nodes);
}

void fit_tree_greedy_cuda(
    dataSet *dataset,
    ensembleData *edata,
    ensembleMetaData *metadata,
    candidatesData *candidata,
    splitDataGPU *split_data){

    allocate_ensemble_memory_cuda(metadata, edata);
    cudaMemcpy(edata->tree_indices + metadata->n_trees, &metadata->n_leaves, sizeof(int), cudaMemcpyHostToDevice);
      
    TreeNodeGPU **tree_nodes = (TreeNodeGPU **)malloc((1 << metadata->max_depth) * sizeof(TreeNodeGPU *));
    
    int crnt_node_ptr_idx = 0, host_status;
    TreeNodeGPU *crnt_node;
    TreeNodeGPU *root_node = allocate_root_tree_node(dataset, metadata);
    tree_nodes[crnt_node_ptr_idx] = root_node;
    crnt_node_ptr_idx++;

    float host_score;
    TreeNodeGPU host_node;

    int threads_per_block;
    calc_parallelism(candidata->n_candidates, metadata->output_dim, threads_per_block, metadata->split_score_func);
    
    while (crnt_node_ptr_idx > 0)  {
        crnt_node_ptr_idx--; 
        crnt_node = tree_nodes[crnt_node_ptr_idx]; 
        if (crnt_node == nullptr){
            std::cerr << "Error crnt_node is nullptr" << std::endl;
            break;
        }

        cudaMemcpy(&host_node, crnt_node, sizeof(TreeNodeGPU), cudaMemcpyDeviceToHost);
        host_status = 0;

        if (candidata->n_candidates == 0 || host_node.n_samples == 0 || host_node.depth == metadata->max_depth){
            host_status = -1;
        }
        if (host_status == 0){
            size_t shmsize;
            const dim3 n_threads_per_blockdim3(BLOCK_COLS, BLOCK_ROWS);
            node_column_mean_reduce<<<(metadata->output_dim * metadata->n_objs + BLOCK_COLS - 1) / BLOCK_COLS, n_threads_per_blockdim3>>>(
                dataset->build_grads->data,
                split_data->node_mean,
                metadata->output_dim,
                dataset->n_samples,
                crnt_node,
                metadata->n_objs);
            cudaDeviceSynchronize();
            if (metadata->split_score_func == Cosine){
                shmsize = sizeof(float) * THREADS_PER_BLOCK;
                node_cosine_kernel<<<metadata->n_objs, THREADS_PER_BLOCK, shmsize>>>(
                    crnt_node,
                    dataset->build_grads->data,
                    split_data->node_mean,
                    metadata->n_objs,
                    dataset->n_samples);
            } else if (metadata->split_score_func == L2){
                node_l2_kernel<<<metadata->n_objs, WARP_SIZE>>>(
                    crnt_node,
                    split_data->node_mean,
                    metadata->n_objs);
            } else{
                std::cerr << "error invalid split score func." << std::endl;
                continue;
            }   

            cudaDeviceSynchronize();
            evaluate_greedy_splits(dataset, edata, crnt_node, candidata, metadata, split_data, threads_per_block, host_node.n_samples);
        }
        cudaMemcpy(&host_score, split_data->best_score, sizeof(float), cudaMemcpyDeviceToHost);

        if (host_score >= 0 && host_status == 0){   
            TreeNodeGPU *left_child = nullptr, *right_child = nullptr;
            allocate_child_tree_nodes(dataset, crnt_node, &host_node, &left_child, &right_child, candidata, split_data, metadata);
            tree_nodes[crnt_node_ptr_idx] = right_child;
            crnt_node_ptr_idx++;
            tree_nodes[crnt_node_ptr_idx] = left_child;
            crnt_node_ptr_idx++;
        } else {
            add_leaf_node(crnt_node, host_node.depth, metadata, edata, dataset);
        }
        free_tree_node(crnt_node);
        crnt_node = nullptr;
    }

    root_node = nullptr;
    metadata->n_trees++;
    free(tree_nodes);
}

__device__ int strcmpCuda(const char* __restrict__ str_a,
                          const char* __restrict__ str_b){
    int match = 0;
    unsigned i = 0;
    unsigned done = 0;
    while ((i < MAX_CHAR_SIZE) && (match == 0) && !done){
        if ((str_a[i] == 0) || (str_b[i] == 0)) 
            done = 1;
        else if (str_a[i] != str_b[i]){
            match = i+1;
            if (((int)str_a[i] - (int)str_b[i]) < 0) 
                match = 0 - (i + 1);
        }
        i++;
    }
    return match;
  }