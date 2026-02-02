//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
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
    const int depth) {

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
    while (threads_per_block*shared_mem*(1 << depth) > deviceProp.sharedMemPerBlock){
        if (threads_per_block == 1){
            std::cerr << "output_dim " << output_dim << "too large! cannot work with so many columns! use cpu version" << std::endl;
        }
        threads_per_block >>= 1;
    }
}

__device__ __forceinline__ void check_and_project_split(
    float* __restrict__ left_val, 
    float* __restrict__ right_val, 
    int constraint) 
{
    // Constraint: 0 (None), 1 (Increasing), -1 (Decreasing)
    if (constraint == 0) return;

    float l = *left_val;
    float r = *right_val;

    // Violation Check:
    // Inc (1): Right must be >= Left. Violation if R < L.
    // Dec (-1): Right must be <= Left. Violation if R > L.
    bool violation = (constraint == 1 && r < l) || (constraint == -1 && r > l);

    if (violation) {
        // Project to unweighted mean (fastest for scoring approximation)
        float pooled = 0.5f * (l + r);
        *left_val = pooled;
        *right_val = pooled;
    }
}

template<int N>
__device__ __forceinline__ void run_pava_registers(float* val) {
    float stack_val[N];
    int   stack_cnt[N]; 
    int top = -1;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        float curr_val = val[i];
        int   curr_cnt = 1;

        // Merge Backwards
        while (top >= 0) {
            // Check Monotonicity: Prev <= Curr
            if (stack_val[top] <= curr_val) break; 

            // Violation: Merge
            float prev_sum = stack_val[top] * stack_cnt[top];
            float curr_sum = curr_val * curr_cnt;
            int new_cnt = stack_cnt[top] + curr_cnt;
            
            curr_val = (prev_sum + curr_sum) / new_cnt;
            curr_cnt = new_cnt;
            top--; 
        }

        // Push
        top++;
        stack_val[top] = curr_val;
        stack_cnt[top] = curr_cnt;
    }

    // Flatten back
    int ptr = 0;
    for (int i = 0; i <= top; ++i) {
        float avg = stack_val[i];
        int count = stack_cnt[i];
        #pragma unroll
        for (int k = 0; k < count; ++k) {
            val[ptr++] = avg;
        }
    }
}

__global__ void enforce_oblivious_pava_kernel(
    float* __restrict__ values,       // [n_leaves * output_dim]
    const int* __restrict__ tree_constraints, // [depth] constraints for this tree
    const int leaf_start_idx,         
    const int output_dim,
    const int depth) 
{
    // One thread per output dimension (e.g., 2 for Actor-Critic, 1 for Value)
    int dim_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (dim_idx >= output_dim) return;

    // Hardcoded max size for registers (Depth 6 = 64 leaves). 
    // For Depth 4, this loop will just run 16 times.
    const int MAX_N = 64; 
    int n_leaves = 1 << depth;
    
    // 1. Load Global -> Registers
    float val[MAX_N];
    int   perm[MAX_N];
    unsigned int keys[MAX_N];

    int offset = leaf_start_idx * output_dim + dim_idx;

    // Stride loading
    for (int i = 0; i < n_leaves; ++i) {
        val[i] = values[offset + i * output_dim];
        perm[i] = i; 
        
        // 2. Generate Sort Key (Gray Code / Path Logic)
        unsigned int key = 0;
        for (int d = 0; d < depth; ++d) {
            int c = tree_constraints[d]; 
            // In oblivious trees, bit 'd' of index 'i' is the decision at depth 'd'
            // (Assumes standard MSB->LSB or LSB->MSB mapping. Verify your build order!)
            // Here assuming: bit (depth - 1 - d) corresponds to layer d.
            int bit = (i >> (depth - 1 - d)) & 1; 

            if (c == 1)       key = (key << 1) | bit;       // Inc: 0 < 1
            else if (c == -1) key = (key << 1) | (1 - bit); // Dec: 1 < 0
            else              key = (key << 1) | 0;         // None
        }
        keys[i] = key;
    }

    // 3. Bubble Sort in Registers (Fastest for N <= 32)
    for (int i = 0; i < n_leaves - 1; ++i) {
        for (int j = 0; j < n_leaves - i - 1; ++j) {
            if (keys[j] > keys[j+1]) {
                // Swap Everything
                float tv = val[j]; val[j] = val[j+1]; val[j+1] = tv;
                unsigned int tk = keys[j]; keys[j] = keys[j+1]; keys[j+1] = tk;
                int tp = perm[j]; perm[j] = perm[j+1]; perm[j+1] = tp;
            }
        }
    }

    // 4. Run PAVA
    // We instantiate for 64. If n_leaves < 64, we need to be careful.
    // Better strategy: Call template based on depth.
    if (depth <= 4)      run_pava_registers<16>(val); 
    else if (depth == 5) run_pava_registers<32>(val);
    else                 run_pava_registers<64>(val);

    // 5. Scatter Write Back
    for (int i = 0; i < n_leaves; ++i) {
        int target = perm[i]; // Original index
        values[offset + target * output_dim] = val[i];
    }
}

__global__ void update_best_candidate_cuda(
    float* __restrict__ split_scores,
    int n_candidates,
    int* __restrict__ best_idx,
    float* __restrict__ best_score,
    const TreeNodeGPU* __restrict__ node) {

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
// #ifdef DEBUG
//         printf("split_score[%d]: %f - %f\n", i, split_scores[i], node->score);
// #endif
        split_scores[i] -= node->score;
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

void evaluate_greedy_splits(
    dataSet *dataset,
    ensembleData *edata,
    const TreeNodeGPU *node,
    candidatesData *candidata,
    ensembleMetaData *metadata,
    splitDataGPU* split_data,
    const int threads_per_block,
    const int parent_n_samples){

    cudaMemset(split_data->split_scores, 0, split_data->size);
    int n_blocks, tpb; 
    get_grid_dimensions(parent_n_samples*candidata->n_candidates, n_blocks, tpb);
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
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count,
            split_data->left_dot,
            split_data->right_dot);
        cudaDeviceSynchronize();
        get_grid_dimensions(candidata->n_candidates, n_blocks, tpb);
        split_cosine_score_kernel<<<n_blocks, tpb>>>(
            node,
            edata->feature_data->feature_weights,
            split_data->split_scores,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            edata->feature_mappings->reverse_num_feature_mapping,
            edata->feature_mappings->reverse_cat_feature_mapping,
            candidata->n_candidates,
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
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count);
        cudaDeviceSynchronize();
        get_grid_dimensions(candidata->n_candidates, n_blocks, tpb);
        split_l2_score_kernel<<<n_blocks, tpb>>>(
            node,
            edata->feature_data->feature_weights,
            split_data->split_scores,
            candidata->candidate_indices,
            candidata->candidate_values,
            candidata->candidate_categories,
            candidata->candidate_numeric,
            edata->feature_mappings->reverse_num_feature_mapping,
            edata->feature_mappings->reverse_cat_feature_mapping,
            candidata->n_candidates,
            split_data->left_sum,
            split_data->right_sum,
            split_data->left_count,
            split_data->right_count,
            metadata->min_data_in_leaf,
            metadata->n_num_features);

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
    update_best_candidate_cuda<<<1, THREADS_PER_BLOCK>>>(split_data->split_scores, candidata->n_candidates, split_data->best_idx, split_data->best_score, node); 
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
    size_t shared_mem;
   
    calc_oblivious_parallelism(candidata->n_candidates, metadata->output_dim, tpb, metadata->split_score_func, depth);
    
    // Get monotonic constraint pointers (may be nullptr if no constraints)
    const int* mono_feat_idx = edata->mono_constraints ? edata->mono_constraints->feature_idx : nullptr;
    const int* mono_out_idx = edata->mono_constraints ? edata->mono_constraints->output_idx : nullptr;
    const int* mono_constr = edata->mono_constraints ? edata->mono_constraints->constraint : nullptr;
    const int n_mono = edata->mono_constraints ? edata->mono_constraints->n_constraints : 0;
    
    for (int i = 0; i < n_nodes; ++i){
        if (metadata->split_score_func == Cosine){
            shared_mem = sizeof(float)*2*(metadata->output_dim + 2)*tpb;
            split_score_cosine_cuda<<<candidata->n_candidates, tpb, shared_mem>>>(
                dataset->obs->data,
                dataset->categorical_obs->data,
                dataset->build_grads->data,
                edata->feature_data->feature_weights,
                nodes[i],
                candidata->candidate_indices,
                candidata->candidate_values,
                candidata->candidate_categories,
                candidata->candidate_numeric,
                edata->feature_mappings->reverse_num_feature_mapping,
                edata->feature_mappings->reverse_cat_feature_mapping,
                metadata->min_data_in_leaf,
                split_data->oblivious_split_scores + candidata->n_candidates*i,
                dataset->n_samples,
                metadata->n_num_features,
                mono_feat_idx,
                mono_out_idx,
                mono_constr,
                n_mono);
        } else if (metadata->split_score_func == L2){
            shared_mem = sizeof(float)*2*(metadata->output_dim + 1)*tpb;
            split_score_l2_cuda<<<candidata->n_candidates, tpb, shared_mem>>>(
                dataset->obs->data, dataset->categorical_obs->data,
                dataset->build_grads->data,
                edata->feature_data->feature_weights,
                nodes[i],
                candidata->candidate_indices,
                candidata->candidate_values,
                candidata->candidate_categories,
                candidata->candidate_numeric,
                edata->feature_mappings->reverse_num_feature_mapping,
                edata->feature_mappings->reverse_cat_feature_mapping,
                metadata->min_data_in_leaf,
                split_data->oblivious_split_scores + candidata->n_candidates*i,
                dataset->n_samples,
                metadata->n_num_features,
                mono_feat_idx,
                mono_out_idx,
                mono_constr,
                n_mono);
        }
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    const dim3 n_threads_per_blockdim3(BLOCK_COLS, BLOCK_ROWS);
    column_sums_reduce<<<(candidata->n_candidates + BLOCK_COLS - 1) / BLOCK_COLS, n_threads_per_blockdim3>>>(split_data->oblivious_split_scores, split_data->split_scores, candidata->n_candidates, n_nodes);
    cudaDeviceSynchronize();
    update_best_candidate_cuda<<<1, THREADS_PER_BLOCK>>>(split_data->split_scores, candidata->n_candidates, split_data->best_idx, split_data->best_score, nodes[0]); 
    cudaDeviceSynchronize();
}

__global__ void split_score_cosine_cuda(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const float* __restrict__ feature_weights,
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
    const int* __restrict__ mono_feature_idx,
    const int* __restrict__ mono_output_idx,
    const int* __restrict__ mono_constraint,
    const int n_mono_constraints){
    extern __shared__ float sdata[];
    int n_samples = __ldg(&node->n_samples), n_cols = __ldg(&node->output_dim);
    int cand_idx = blockIdx.x;

    if (split_scores[cand_idx] == -INFINITY)
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
    int thread_offset = 0;
    float *left_mean = &sdata[0];
    thread_offset += blockDim.x * n_cols;
    float *l_count = &sdata[thread_offset];
    thread_offset += blockDim.x;
    float *l_dot_sum = &sdata[thread_offset];
    thread_offset += blockDim.x;
    float* right_mean = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += blockDim.x * n_cols;
    float* r_count = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += blockDim.x;
    float* r_dot_sum = &sdata[thread_offset]; // Assuming each part is n_cols floats long

    r_count[threadIdx.x] = 0.0f;
    l_count[threadIdx.x] = 0.0f;
    r_dot_sum[threadIdx.x] = 0.0f;
    l_dot_sum[threadIdx.x] = 0.0f;
    for (int d = 0; d < n_cols; ++d){
        right_mean[threadIdx.x*n_cols + d] = 0.0f;
        left_mean[threadIdx.x*n_cols + d] = 0.0f;
    }
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = __ldg(&node->sample_indices[i]); // Access the specific sample
        bool passed = candidate_numeric[cand_idx] && __ldg(&obs[sample_idx +  global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx]);
        passed = passed || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + __ldg(&candidate_indices[cand_idx]))* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        
        if (passed){
            for (int d = 0; d < n_cols; ++d){
                right_mean[threadIdx.x*n_cols + d] += __ldg(&grads[sample_idx*n_cols + d]);
            }
            r_count[threadIdx.x] += 1;
        } else {
            for (int d = 0; d < n_cols; ++d){
                left_mean[threadIdx.x*n_cols + d] += __ldg(&grads[sample_idx*n_cols + d]);
            }
            l_count[threadIdx.x] += 1;
        }
    }
    __syncthreads();
     // // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            for (int d = 0; d < n_cols; ++d){
                left_mean[threadIdx.x*n_cols + d] += left_mean[(threadIdx.x + offset)*n_cols + d];
                right_mean[threadIdx.x*n_cols + d] += right_mean[(threadIdx.x + offset)*n_cols + d];
            }
            l_count[threadIdx.x] += l_count[threadIdx.x + offset];
            r_count[threadIdx.x] += r_count[threadIdx.x + offset];
            
        }
        __syncthreads();
    }

    if (l_count[0] < static_cast<float>(min_data_in_leaf) || r_count[0] < static_cast<float>(min_data_in_leaf)){
        split_scores[cand_idx] = -CUDART_INF_F;
        return;
    } 



    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = __ldg(&node->sample_indices[i]); // Access the spec
        bool passed = candidate_numeric[cand_idx] && __ldg(&obs[sample_idx + global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx]);
        passed = passed || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + __ldg(&candidate_indices[cand_idx]))* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        if (passed){
            for (int d = 0; d < n_cols; ++d){
                r_dot_sum[threadIdx.x] += __ldg(&grads[sample_idx*n_cols + d]) * right_mean[d];
            }
        } else {
            for (int d = 0; d < n_cols; ++d){
                l_dot_sum[threadIdx.x] += __ldg(&grads[sample_idx*n_cols + d]) * left_mean[d];
            }
        }
    }
    __syncthreads();
    
     // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            r_dot_sum[threadIdx.x] += r_dot_sum[threadIdx.x + offset];  
            l_dot_sum[threadIdx.x] += l_dot_sum[threadIdx.x + offset];  
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (threadIdx.x == 0){
        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];
        
        // Convert sums to means first (left_mean and right_mean hold sums at this point)
        for (int d = 0; d < n_cols; ++d){
            left_mean[d] = (l_count[0] > 0.0f) ? left_mean[d] / l_count[0] : 0.0f;
            right_mean[d] = (r_count[0] > 0.0f) ? right_mean[d] / r_count[0] : 0.0f;
        }
        
        // Check monotonic constraints and apply PAVA-like pooling if violated
        for (int c = 0; c < n_mono_constraints; ++c) {
            if (mono_feature_idx[c] != feat_idx) continue;
            
            int out_idx = mono_output_idx[c];
            int direction = mono_constraint[c];
            
            float l_val = left_mean[out_idx];
            float r_val = right_mean[out_idx];
            
            // Check violation: Inc(+1) requires r >= l, Dec(-1) requires r <= l
            bool violation = (direction == 1 && r_val < l_val) ||
                            (direction == -1 && r_val > l_val);
            
            if (violation) {
                // Pool the means (weighted average) for this output dimension
                float total_cnt = l_count[0] + r_count[0];
                float pooled = (l_count[0] * l_val + r_count[0] * r_val) / total_cnt;
                left_mean[out_idx] = pooled;
                right_mean[out_idx] = pooled;
            }
        }
        
        // Compute final score with potentially adjusted means
        float cosine = 0.0f, l_mean_norm = 0.0f, r_mean_norm = 0.0f;
        for (int d = 0; d < n_cols; ++d){
            l_mean_norm += left_mean[d] * left_mean[d];
            r_mean_norm += right_mean[d] * right_mean[d];
        }
        l_dot_sum[0] = (l_count[0] > 0.0f) ? l_dot_sum[0] / l_count[0] : 0.0f;
        r_dot_sum[0] = (r_count[0] > 0.0f) ? r_dot_sum[0] / r_count[0] : 0.0f;
        float denominator = l_count[0]* l_mean_norm + r_count[0] * r_mean_norm;
        if (denominator > 0.0f) {
            cosine = (l_dot_sum[0] + r_dot_sum[0]) / sqrtf(denominator);
        }
        
        split_scores[cand_idx] = cosine * __ldg(feature_weights + feat_idx);
    }  
}


__global__ void split_score_l2_cuda(
    const float* __restrict__ obs,
    const char* __restrict__ categorical_obs,
    const float* __restrict__ grads,
    const float* __restrict__ feature_weights,
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
    const int* __restrict__ mono_feature_idx,
    const int* __restrict__ mono_output_idx,
    const int* __restrict__ mono_constraint,
    const int n_mono_constraints){
    extern __shared__ float sdata[];

    int n_samples = node->n_samples, n_cols = node->output_dim;
    int cand_idx = blockIdx.x;

    if (split_scores[cand_idx] == -INFINITY)
        return;

    if (node->depth > 0 && min_data_in_leaf == 0){
        if (candidate_numeric[cand_idx]){
            for (int i = 0; i < node->depth; ++i){
                if (node->is_numerics[i] && __ldg(&node->feature_values[i]) == __ldg(&candidate_values[cand_idx]) && __ldg(&node->feature_indices[i]) == __ldg(&candidate_indices[cand_idx])){
                    split_scores[cand_idx] = -CUDART_INF_F;
                    return;
                }
            }
        } else {
            for (int i = 0; i < node->depth; ++i){
                if (!node->is_numerics[i] && strcmpCuda(node->categorical_values + i * MAX_CHAR_SIZE, candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0 && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                    split_scores[cand_idx] = -CUDART_INF_F;
                    return;
                }
            }
        }
    }
    int threads_per_block = blockDim.x;

    int thread_offset = 0;
    float *left_mean = &sdata[thread_offset];
    thread_offset += threads_per_block * n_cols;
    float *l_count = &sdata[thread_offset];
    thread_offset += threads_per_block;
    float* right_mean = &sdata[thread_offset]; // Assuming each part is n_cols floats long
    thread_offset += threads_per_block * n_cols;
    float* r_count = &sdata[thread_offset]; // Assuming each part is n_cols floats long

    r_count[threadIdx.x] = 0.0f;
    l_count[threadIdx.x] = 0.0f;
    for (int d = 0; d < n_cols; ++d){
        right_mean[threadIdx.x * n_cols + d] = 0.0f;
        left_mean[threadIdx.x * n_cols + d] = 0.0f;
    }
    __syncthreads();
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = __ldg(&node->sample_indices[i]); // Access the spec
        int row_idx = sample_idx*n_cols;
        if ((candidate_numeric[cand_idx] && __ldg(&obs[__ldg(&candidate_indices[cand_idx])*global_n_samples + sample_idx]) > __ldg(&candidate_values[cand_idx])) || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + __ldg(&candidate_indices[cand_idx]))* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0)){
            for (int d = 0; d < n_cols; ++d){
                right_mean[threadIdx.x*n_cols + d] += __ldg(&grads[row_idx + d]);
            }
            r_count[threadIdx.x] += 1;
        } else {
            for (int d = 0; d < n_cols; ++d){
                left_mean[threadIdx.x*n_cols + d] += __ldg(&grads[row_idx + d]);
            }
            l_count[threadIdx.x] += 1;
        }
    }
    __syncthreads();

     // // tree reduction
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            for (int d = 0; d < n_cols; ++d){
                left_mean[threadIdx.x*n_cols + d]  += left_mean[(threadIdx.x + offset)*n_cols + d];
                right_mean[threadIdx.x*n_cols + d] += right_mean[(threadIdx.x + offset)*n_cols + d];
            }
            l_count[threadIdx.x]   += l_count[threadIdx.x + offset];
            r_count[threadIdx.x]   += r_count[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if (threadIdx.x == 0) {
        float l_mean_norm = 0.0f, r_mean_norm = 0.0f;
        if (l_count[0] < static_cast<float>(min_data_in_leaf) || r_count[0] < static_cast<float>(min_data_in_leaf)){
            split_scores[cand_idx] = -CUDART_INF_F;
            return;
        }  

        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];

        // Convert sums to means
        for (int d = 0; d < n_cols; ++d){
            left_mean[d] = (l_count[0] > 0) ? left_mean[d] / l_count[0] : 0.0f;
            right_mean[d] = (r_count[0] > 0) ? right_mean[d] / r_count[0] : 0.0f;
        }

        // Check monotonic constraints and apply pooling if violated
        for (int c = 0; c < n_mono_constraints; ++c) {
            if (mono_feature_idx[c] != feat_idx) continue;
            
            int out_idx = mono_output_idx[c];
            int direction = mono_constraint[c];
            
            float l_val = left_mean[out_idx];
            float r_val = right_mean[out_idx];
            
            // Check violation: Inc(+1) requires l <= r, Dec(-1) requires l >= r
            bool violation = (direction == 1 && l_val > r_val) ||
                            (direction == -1 && l_val < r_val);
            
            if (violation) {
                // Pool the means for this output dimension
                float pooled = (l_val + r_val) * 0.5f;
                left_mean[out_idx] = pooled;
                right_mean[out_idx] = pooled;
            }
        }

        // Compute final score with potentially adjusted means
        for (int d = 0; d < n_cols; ++d){
            l_mean_norm += left_mean[d] * left_mean[d];
            r_mean_norm += right_mean[d] * right_mean[d];
        }
        
        split_scores[cand_idx] = (l_count[0]*l_mean_norm + r_count[0]*r_mean_norm) * __ldg(feature_weights + feat_idx);
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
    float* __restrict__ left_sum,
    float* __restrict__ right_sum,
    float* __restrict__ left_count,
    float* __restrict__ right_count){
    // Accumulate per thread partial sum
    int global_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int output_dim = __ldg(&node->output_dim);
    if (global_idx < __ldg(&node->n_samples) * n_candidates){
        int sample_row = global_idx / n_candidates;
        int cand_idx = global_idx % n_candidates; 
        int sample_idx = __ldg(&node->sample_indices[sample_row]); // Access the spec
        bool is_greater = (candidate_numeric[cand_idx] && __ldg(&obs[sample_idx +  global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx])) || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + candidate_indices[cand_idx])* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        
        if (is_greater){
            for (int d = 0; d < output_dim; ++d){
                atomicAdd(right_sum + cand_idx*output_dim + d, __ldg(&grads[sample_idx*output_dim + d]));
            }
            atomicAdd(right_count + cand_idx, 1);
        } else {
            for (int d = 0; d < output_dim; ++d){
                atomicAdd(left_sum + cand_idx*output_dim + d, __ldg(&grads[sample_idx*output_dim + d]));
            }
            atomicAdd(left_count + cand_idx, 1);
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
    float* __restrict__ left_sum,
    float* __restrict__ right_sum,
    float* __restrict__ left_count,
    float* __restrict__ right_count,
    float* __restrict__ ldot,
    float* __restrict__ rdot){

    int n_cols = __ldg(&node->output_dim), n_samples = __ldg(&node->n_samples);
    int global_idx = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (global_idx < __ldg(&node->n_samples) * n_candidates){
        int sample_row = global_idx / n_candidates;
        int cand_idx = global_idx % n_candidates; 
        float cdot = 0.0f;
        int cand_row = cand_idx*n_cols;

        int sample_idx = __ldg(&node->sample_indices[sample_row]); // Access the spec
        int row_idx = sample_idx*n_cols;

        bool is_greater = (candidate_numeric[cand_idx] && __ldg(&obs[sample_idx +  global_n_samples * __ldg(&candidate_indices[cand_idx])]) > __ldg(&candidate_values[cand_idx])) || (!candidate_numeric[cand_idx] && strcmpCuda(&categorical_obs[(sample_idx*node->n_cat_features + candidate_indices[cand_idx])* MAX_CHAR_SIZE], candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0);
        if (is_greater){
            for (int d = 0; d < n_cols; ++d){
                cdot += __ldg(&grads[row_idx + d]) * __ldg(&right_sum[cand_row + d]);
            }
            cdot /= __ldg(&right_count[cand_idx]);
            atomicAdd(rdot + cand_idx, cdot);
        } else {
            for (int d = 0; d < n_cols; ++d){
                cdot += __ldg(&grads[row_idx + d]) * __ldg(&left_sum[cand_row + d]);
            }
            cdot /= __ldg(&left_count[cand_idx]);
            atomicAdd(ldot + cand_idx, cdot);
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
    float* __restrict__ lsum,
    float* __restrict__ rsum,
    float* __restrict__ lcount,
    float* __restrict__ rcount,
    float* __restrict__ ldot,
    float* __restrict__ rdot,
    const int min_data_in_leaf, 
    const int n_num_features){

    int cand_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int n_cols = __ldg(&node->output_dim);
    int cand_row = cand_idx*n_cols;
    float lvalue, rvalue;

    if (split_scores[cand_idx] == -INFINITY)
        return;

    if (cand_idx < n_candidates){
        if (node->depth > 0 && min_data_in_leaf == 0){
            if (candidate_numeric[cand_idx]){
                for (int i = 0; i < node->depth; ++i){
                    if (node->is_numerics[i] && __ldg(&node->feature_values[i]) == __ldg(&candidate_values[cand_idx]) && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_idx] = -CUDART_INF_F;
                        return;
                    }
                }   
            } else {
                for (int i = 0; i < node->depth; ++i){
                    if (!node->is_numerics[i] && strcmpCuda(node->categorical_values + i * MAX_CHAR_SIZE, candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0 && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_idx] = -CUDART_INF_F;
                        return;
                    }
                }
            }
        }

        if (lcount[cand_idx] < static_cast<float>(min_data_in_leaf) || rcount[cand_idx] < static_cast<float>(min_data_in_leaf)){
            split_scores[cand_idx] = -CUDART_INF_F;
            return;
        } 

        float l_mean_norm = 0.0f, r_mean_norm = 0.0f;
        for (int d = 0; d < n_cols; ++d){
            lvalue = __ldg(lsum + cand_row + d);
            rvalue = __ldg(rsum + cand_row + d);
            l_mean_norm += lvalue * lvalue;
            r_mean_norm += rvalue * rvalue;
        }
        lvalue = __ldg(&lcount[cand_idx]);
        rvalue = __ldg(&rcount[cand_idx]);

        l_mean_norm = (lvalue > 0.0f) ? l_mean_norm / (lvalue * lvalue) : 0.0f;
        r_mean_norm = (rvalue > 0.0f) ? r_mean_norm / (rvalue * rvalue) : 0.0f;

        float denominator =  lvalue * l_mean_norm + rvalue * r_mean_norm;
        float numerator = ldot[cand_idx] + rdot[cand_idx];
        if (denominator == 0.0f){
            split_scores[cand_idx] = -CUDART_INF_F;
            return;
        }
        float cos = numerator / sqrtf(denominator);
        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];
        split_scores[cand_idx] = cos * __ldg(feature_weights + feat_idx);
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
    float* __restrict__ lsum,
    float* __restrict__ rsum,
    float* __restrict__ lcount,
    float* __restrict__ rcount,
    const int min_data_in_leaf,
    const int n_num_features){

    int cand_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int n_cols = __ldg(&node->output_dim);
    int cand_row = cand_idx*n_cols;
    float lvalue, rvalue;

    if (split_scores[cand_idx] == -INFINITY)
        return;
        
    if (cand_idx < n_candidates){
        if (node->depth > 0 && min_data_in_leaf == 0){
            if (candidate_numeric[cand_idx]){
                for (int i = 0; i < node->depth; ++i){
                    if (node->is_numerics[i] && __ldg(&node->feature_values[i]) == __ldg(&candidate_values[cand_idx]) && __ldg(&node->feature_indices[i]) == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_idx] = -CUDART_INF_F;
                        return;
                    }
                }   
            } else {
                for (int i = 0; i < node->depth; ++i){
                    if (!node->is_numerics[i] && strcmpCuda(node->categorical_values + i * MAX_CHAR_SIZE, candidate_categories + cand_idx * MAX_CHAR_SIZE) == 0 && node->feature_indices[i] == __ldg(&candidate_indices[cand_idx])){
                        split_scores[cand_idx] = -CUDART_INF_F;
                        return;
                    }
                }
            }
        }

        if (lcount[cand_idx] < static_cast<float>(min_data_in_leaf) || rcount[cand_idx] < static_cast<float>(min_data_in_leaf)){
            split_scores[cand_idx] = -CUDART_INF_F;
            return;
        } 

        float l_mean_norm = 0.0f, r_mean_norm = 0.0f;
        for (int d = 0; d < n_cols; ++d){
            lvalue = __ldg(lsum + cand_row + d);
            rvalue = __ldg(rsum + cand_row + d);
            l_mean_norm += lvalue * lvalue;
            r_mean_norm += rvalue * rvalue;
        }
        lvalue = __ldg(&lcount[cand_idx]);
        rvalue = __ldg(&rcount[cand_idx]);
        l_mean_norm = (lvalue > 0.0f) ? l_mean_norm / lvalue : 0.0f; // n_count * l2 norm 
        r_mean_norm = (rvalue > 0.0f) ? r_mean_norm / rvalue : 0.0f; // n_count * l2 norm 

        int tmp_idx = __ldg(&candidate_indices[cand_idx]);
        int feat_idx = (candidate_numeric[cand_idx]) ? r_num_mapping[tmp_idx] : r_cat_mapping[tmp_idx];
        split_scores[cand_idx] = (l_mean_norm + r_mean_norm) * __ldg(feature_weights + feat_idx);    
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
    const float* __restrict__ grads,
    float* __restrict__ values,
    const TreeNodeGPU* __restrict__ node,
    const int n_samples,
    const int global_idx,
    const int policy_dim){

    extern __shared__ float sdata[];

    int thread_offset = 0;
    float *sums = &sdata[thread_offset];
    thread_offset += blockDim.x;
    float *sum_count = &sdata[thread_offset];
    sums[threadIdx.x] = 0.0f; // Initialize shared memory
    sum_count[threadIdx.x] = 0.0f; // Initialize shared memory
    values[global_idx + blockIdx.x] = 0.0f;

    __syncthreads();

    bool passed;
    int cat_row_idx;
    for (int sample_idx = threadIdx.x; sample_idx < n_samples; sample_idx += blockDim.x){
        cat_row_idx = sample_idx*node->n_cat_features;
        passed = false;
        for (int condIdx = node->depth - 1; condIdx >= 0; --condIdx){
            if (node->is_numerics[condIdx]){
                passed = obs[sample_idx  + n_samples*node->feature_indices[condIdx]] > node->feature_values[condIdx] == node->inequality_directions[condIdx];
            } else {
                passed = (strcmpCuda(&categorical_obs[(cat_row_idx + node->feature_indices[condIdx])*MAX_CHAR_SIZE],  node->categorical_values + condIdx*MAX_CHAR_SIZE) == 0) == node->inequality_directions[condIdx];
            }
            if (!passed)
                break;
        }
        if (passed){
            sums[threadIdx.x] += grads[sample_idx * node->output_dim + blockIdx.x];
            sum_count[threadIdx.x] += 1;       
        }
    }
    __syncthreads();

    // Perform reduction in shared memory
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset) {
            sums[threadIdx.x]  += sums[threadIdx.x + offset];
            sum_count[threadIdx.x] += sum_count[threadIdx.x + offset];  

        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        if (sum_count[threadIdx.x] > 0){
            values[global_idx + blockIdx.x] = (sums[threadIdx.x] / sum_count[threadIdx.x]);
        }
    }
}


__global__ void node_column_mean_reduce(
    const float * __restrict__ in,
    float * __restrict__ out,
    size_t n_cols,
    const TreeNodeGPU* __restrict__ node){

  __shared__ float sdata[BLOCK_ROWS][BLOCK_COLS + 1];
  size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
  size_t width_stride = gridDim.x*blockDim.x;
  size_t n_rows = node->n_samples;
//   if (threadIdx.y >= n_rows)
//     return;
  // bitwise round-up
  size_t full_width = (n_cols & (~((unsigned long long)(BLOCK_COLS -1)))) + ((n_cols & (BLOCK_COLS-1)) ? BLOCK_COLS : 0); // round up to next block

  for (size_t col = idx; col < full_width; col+=width_stride){          // grid-stride loop across matrix width
    sdata[threadIdx.y][threadIdx.x] = 0;
    for (size_t row = threadIdx.y; row < n_rows; row+=BLOCK_ROWS){ // block-stride loop across matrix height
        sdata[threadIdx.y][threadIdx.x] += (col < n_cols) ? in[col + node->sample_indices[row]*n_cols]  : 0;
    }
    __syncthreads();
    float tmp = sdata[threadIdx.x][threadIdx.y];
    for (int i = WARP_SIZE >>1; i > 0; i >>= 1)                       // warp-wise parallel sum reduction
      tmp += __shfl_xor_sync(0xFFFFFFFFU, tmp, i);
    __syncthreads();
    if (threadIdx.x == 0) 
        sdata[0][threadIdx.y]  = tmp;
    __syncthreads();
    if ((threadIdx.y == 0) && (col < n_cols)) 
        out[col] = sdata[0][threadIdx.x] / static_cast<float>(n_rows);
  }
}

__global__ void node_l2_kernel(
    TreeNodeGPU* __restrict__ node,
    const float* __restrict__ mean){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0){
        float mean_squared_norm = 0.0f;
        for (int i = 0; i < node->output_dim; ++i)
            mean_squared_norm += (mean[i]*mean[i]);
        node->score = (node->node_idx > 0) ? mean_squared_norm*static_cast<float>(node->n_samples) : 0.0f; 
    }
}

__global__ void node_cosine_kernel(
    TreeNodeGPU* __restrict__ node,
    const float* __restrict__ grads,
    float* __restrict__ mean){

    extern __shared__ float sdata[];
    int n_samples = node->n_samples, n_cols = node->output_dim;    
    int thread_offset = 0;
    float *dot_sum = &sdata[thread_offset];
    dot_sum[threadIdx.x] = 0.0f;
    __syncthreads();
    // Accumulate per thread partial sum
    for(int i=threadIdx.x; i < n_samples; i += blockDim.x) {
        int sample_idx = node->sample_indices[i]; // Access the spec
        int row_idx = sample_idx*n_cols;
        
        for (int d = 0; d < n_cols ; ++d){
            dot_sum[threadIdx.x] += grads[row_idx + d]*mean[d];
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
            mean_norm += mean[d]*mean[d];
        float denominator = static_cast<float>(n_samples) * mean_norm;
        if (denominator > 0) {
            cosine = dot_sum[0] / sqrtf(denominator);
        }
        node->score = (node->node_idx > 0) ? cosine : 0.0f;
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
    tempNode.node_idx = 0;
    tempNode.score = 0.0f;

    tempNode.sample_indices = nullptr;
    tempNode.feature_indices = nullptr;
    tempNode.feature_values = nullptr;
    tempNode.edge_weights = nullptr;
    tempNode.inequality_directions = nullptr;
    tempNode.is_numerics = nullptr;
    tempNode.categorical_values = nullptr;

    int *sample_indices;
    error = allocateCudaMemory((void**)&sample_indices, sizeof(int)*dataset->n_samples, "when trying to allocate root sample_indices");
    if (error != cudaSuccess) {
        cudaFree(node);
        return nullptr;
    }
    int n_blocks = dataset->n_samples / THREADS_PER_BLOCK + 1;
    iota_kernel<<<n_blocks, THREADS_PER_BLOCK>>>(sample_indices, dataset->n_samples);
    cudaDeviceSynchronize();

    cudaMemcpy(node, &tempNode, sizeof(TreeNodeGPU), cudaMemcpyHostToDevice);
    if (sample_indices != nullptr){
        cudaMemcpy(&(node->sample_indices), &sample_indices, sizeof(int*), cudaMemcpyHostToDevice);
    }
    return node;
}

void allocate_child_tree_node(
    TreeNodeGPU* host_parent,
    TreeNodeGPU** device_child){

    TreeNodeGPU host_child;
    int n_samples = host_parent->n_samples;
    int depth = host_parent->depth + 1;

    host_child.depth = depth;
    host_child.n_samples = n_samples;
    host_child.output_dim = host_parent->output_dim;
    host_child.node_idx = -1;
    host_child.score = 0.0f;
    host_child.n_num_features = host_parent->n_num_features;
    host_child.n_cat_features = host_parent->n_cat_features;
    host_child.sample_indices = nullptr;
    host_child.feature_indices = nullptr;
    host_child.feature_values = nullptr;
    host_child.inequality_directions = nullptr;
    host_child.edge_weights = nullptr;
    host_child.is_numerics = nullptr;
    host_child.categorical_values = nullptr;

    char* device_memory_block;
    size_t conditions_size = sizeof(int) * n_samples // sample_indices
                + sizeof(int) * depth    // feature_indices
                + sizeof(float) * depth  // feature_values
                + sizeof(float) * depth   // edge_weights
                + sizeof(bool) * depth   // inequality_directions
                + sizeof(bool) * depth   // is_numerics
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
    host_child.inequality_directions = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * depth;
    host_child.is_numerics = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * depth;
    host_child.categorical_values = (char*)(device_memory_block + trace);

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
    allocate_child_tree_node(host_parent, left_child);
    allocate_child_tree_node(host_parent, right_child);

    int n_blocks, threads_per_block;
    get_grid_dimensions(n_samples, n_blocks, threads_per_block);
    partition_samples_kernel<<<n_blocks, threads_per_block>>>(dataset->obs->data, dataset->categorical_obs->data, parent_node, *left_child, *right_child, candidata->candidate_indices, candidata->candidate_values, candidata->candidate_categories, candidata->candidate_numeric, split_data->best_idx, split_data->tree_counters, dataset->n_samples) ;
    cudaDeviceSynchronize();
    int n_threads = WARP_SIZE*((MAX_CHAR_SIZE + WARP_SIZE - 1) / WARP_SIZE);
    update_child_nodes_kernel<<<depth, n_threads>>>(parent_node, *left_child, *right_child, split_data->tree_counters, candidata->candidate_indices, candidata->candidate_values, candidata->candidate_numeric, candidata->candidate_categories, split_data->best_idx, split_data->best_score);
    cudaDeviceSynchronize();
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
        copy_node_to_data<<<depth, n_threads>>>(node, edata->ensemble_info->depths, edata->feature_data->feature_indices, edata->feature_data->feature_values, edata->leaf_data->edge_weights, edata->feature_data->inequality_directions, edata->feature_data->is_numerics, edata->feature_data->categorical_values, global_idx, leaf_idx, metadata->max_depth);
        cudaDeviceSynchronize();
    }

    int threads_per_block = WARP_SIZE*((dataset->n_samples  + WARP_SIZE - 1 )/ WARP_SIZE);
    if (threads_per_block > THREADS_PER_BLOCK) {
        threads_per_block = THREADS_PER_BLOCK;
    }
    size_t shared_mem = sizeof(float)*2*threads_per_block;
    reduce_leaf_sum<<<metadata->output_dim, threads_per_block, shared_mem>>>(dataset->obs->data, dataset->categorical_obs->data, dataset->grads->data, edata->leaf_data->values, node, dataset->n_samples, leaf_idx*metadata->output_dim, metadata->policy_dim);
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
    const int global_idx,
    const int leaf_idx,
    const int max_depth){
    if (blockIdx.x == 0 && threadIdx.x == 0){
        depths[global_idx] = node->depth;
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
        // printf("best_idx %d, is_numeric %d\n", best_idx_, is_numeric);
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
        printf("%d samples %d num_features %d cat_features %d output dim %d depth score %f\n", node->n_samples, node->n_num_features, node->n_cat_features, node->output_dim, node->depth, node->score);
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
        left_child->score = best_score[0];
        right_child->score = best_score[0];
        left_child->n_samples = tree_counters[0];
        right_child->n_samples = tree_counters[1];
        tree_counters[2] += 2;
    }
}

void fit_tree_oblivious_cuda(
    dataSet *dataset,
    ensembleData *edata,
    ensembleMetaData *metadata,
    candidatesData *candidata,
    splitDataGPU *split_data){

    allocate_ensemble_memory_cuda(metadata, edata);
    cudaMemcpy(edata->ensemble_info->tree_indices + metadata->n_trees, &metadata->n_leaves, sizeof(int), cudaMemcpyHostToDevice);

    // Save starting leaf index for monotonic constraints
    int start_leaf_idx = metadata->n_leaves;
    int tree_idx = metadata->n_trees;

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

    // Apply monotonic constraints using PAVA after all leaves are computed
    if (metadata->n_mono_constraints > 0 && depth > 0) {
        apply_monotonic_constraints_cuda(edata, metadata, tree_idx, depth, start_leaf_idx);
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
    cudaMemcpy(edata->ensemble_info->tree_indices + metadata->n_trees, &metadata->n_leaves, sizeof(int), cudaMemcpyHostToDevice);
      
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
            node_column_mean_reduce<<<(metadata->output_dim + BLOCK_COLS - 1) / BLOCK_COLS, n_threads_per_blockdim3 >>>(
                dataset->build_grads->data,
                split_data->node_mean,
                metadata->output_dim,
                crnt_node);
            cudaDeviceSynchronize();
            if (metadata->split_score_func == Cosine){
                shmsize = sizeof(float) * THREADS_PER_BLOCK;
                node_cosine_kernel<<<1, THREADS_PER_BLOCK, shmsize>>>(
                    crnt_node,
                    dataset->build_grads->data,
                    split_data->node_mean);
            } else if (metadata->split_score_func == L2){
                node_l2_kernel<<<1, WARP_SIZE>>>(
                    crnt_node,
                    split_data->node_mean);
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

// ============================================================================
// Monotonic Constraints Implementation (PAVA for Oblivious Trees)
// ============================================================================

/**
 * For oblivious trees, monotonic constraints create a partial order on leaves.
 * 
 * Consider a tree of depth D. Each leaf has an index from 0 to 2^D - 1.
 * The binary representation of the leaf index tells us the path:
 *   - bit k = 0 means we went LEFT at depth k (feature < threshold)
 *   - bit k = 1 means we went RIGHT at depth k (feature >= threshold)
 * 
 * For a monotonic constraint on feature F used at depth d:
 *   - If INCREASING (+1): leaves with bit d=1 should have >= value than leaves with bit d=0
 *   - If DECREASING (-1): leaves with bit d=1 should have <= value than leaves with bit d=0
 * 
 * When multiple monotonic features exist, they define a partial order.
 * We linearize this by treating the monotonic bits as a number and sorting.
 * 
 * Algorithm (following CatBoost's approach):
 * 1. Find which depths use monotonic features for a given output
 * 2. Build a linear order on leaves consistent with the partial order
 * 3. Apply PAVA (Pool Adjacent Violators Algorithm) along that order
 * 
 * PAVA for isotonic regression:
 * - Process leaves in order
 * - Maintain a stack of "level sets" (contiguous groups with same adjusted value)
 * - When adding a new element, if it violates monotonicity with the previous level set,
 *   merge them and take the weighted average
 * - Continue until no violations remain
 */

/**
 * @brief Build the linear order of leaves for monotonic constraints
 * 
 * For leaves in an oblivious tree, this creates an ordering such that
 * if leaf A must have value <= leaf B according to monotonic constraints,
 * then A appears before B in the order.
 * 
 * @param tree_mono_constraints Array of size tree_depth: 0 (no constraint), +1 (increasing), -1 (decreasing)
 * @param tree_depth Depth of the tree
 * @param n_leaves Number of leaves (2^tree_depth)
 * @param leaf_order Output: ordered indices of leaves
 */
__device__ void build_leaf_order(
    const int* tree_mono_constraints,
    int tree_depth,
    int n_leaves,
    int* leaf_order
) {
    // Count monotonic and non-monotonic splits
    int mono_count = 0;
    int mono_depths[32];  // Assuming max depth <= 32
    int mono_dirs[32];    // Direction for each monotonic depth
    
    for (int d = 0; d < tree_depth; ++d) {
        if (tree_mono_constraints[d] != 0) {
            mono_dirs[mono_count] = tree_mono_constraints[d];
            mono_depths[mono_count] = d;
            mono_count++;
        }
    }
    
    // For each leaf, compute its position in the linear order
    // The order is determined by the monotonic bits interpreted as a number
    // with appropriate direction flipping for decreasing constraints
    for (int leaf_idx = 0; leaf_idx < n_leaves; ++leaf_idx) {
        int rank = 0;
        for (int m = 0; m < mono_count; ++m) {
            int d = mono_depths[m];
            int bit = (leaf_idx >> d) & 1;
            
            // For decreasing constraint, flip the bit contribution
            if (mono_dirs[m] == -1) {
                bit = 1 - bit;
            }
            
            // Build rank with highest-significance monotonic feature first
            rank = (rank << 1) | bit;
        }
        
        // Also need to handle non-monotonic features to place leaves correctly
        // For non-monotonic features, leaves with different non-mono bits are in different "subtrees"
        // We process each subtree independently in PAVA
        
        // For now, compute a combined index: (non_mono_bits, rank)
        // This groups leaves by their non-monotonic path, then orders by monotonic rank
        int non_mono_bits = 0;
        int non_mono_count = 0;
        for (int d = 0; d < tree_depth; ++d) {
            if (tree_mono_constraints[d] == 0) {
                int bit = (leaf_idx >> d) & 1;
                non_mono_bits |= (bit << non_mono_count);
                non_mono_count++;
            }
        }
        
        // Combined key: non_mono_bits * (1 << mono_count) + rank
        int combined = (non_mono_bits << mono_count) | rank;
        leaf_order[combined] = leaf_idx;
    }
}

/**
 * @brief PAVA kernel for applying isotonic regression to leaf values
 * 
 * Each block handles one output dimension.
 * Within each block, thread 0 performs sequential PAVA (inherently sequential algorithm).
 * 
 * For trees with non-monotonic features, leaves are grouped into subtrees.
 * PAVA is applied independently to each subtree.
 */
__global__ void pava_kernel(
    float* __restrict__ values,
    const int constraint_depth,    // Which depth has the constraint we're enforcing
    const int constraint_dir,      // Direction: +1 (increasing) or -1 (decreasing)
    const int tree_depth,
    const int start_leaf_idx,
    const int n_leaves_in_tree,
    const int output_dim,
    const int target_output        // Which output dimension to process
) {
    // Each block handles one "plane" of leaves where all other depths are fixed
    // and only the constraint_depth varies
    int plane_idx = blockIdx.x;
    int n_planes = n_leaves_in_tree / 2;  // 2^(tree_depth-1) planes
    
    if (plane_idx >= n_planes) return;
    
    // Only thread 0 does the work (PAVA is sequential)
    if (threadIdx.x != 0) return;
    
    // CRITICAL FIX: Use proper bit ordering where depth 0 (root) is MSB
    // Bit position for this depth: (tree_depth - 1 - constraint_depth)
    int bit_pos = tree_depth - 1 - constraint_depth;
    int bit_mask = 1 << bit_pos;
    
    // Build the two leaf indices for this plane
    // Use plane_idx to enumerate all combinations of other bits
    int base_leaf = 0;
    int plane_bit = 0;
    for (int d = 0; d < tree_depth; ++d) {
        int d_bit_pos = tree_depth - 1 - d;
        if (d == constraint_depth) continue;  // Skip the constraint depth
        
        // Extract bit from plane_idx
        int bit = (plane_idx >> plane_bit) & 1;
        base_leaf |= (bit << d_bit_pos);
        plane_bit++;
    }
    
    // The two leaves in this plane
    int leaf0 = base_leaf;  // constraint bit = 0
    int leaf1 = base_leaf | bit_mask;  // constraint bit = 1
    
    int global_leaf0 = start_leaf_idx + leaf0;
    int global_leaf1 = start_leaf_idx + leaf1;
    
    // Get current values
    float val0 = values[global_leaf0 * output_dim + target_output];
    float val1 = values[global_leaf1 * output_dim + target_output];
    
    // For increasing constraint (+1): leaf0 (bit=0) should have value <= leaf1 (bit=1)
    // For decreasing constraint (-1): leaf0 (bit=0) should have value >= leaf1 (bit=1)
    bool violation = (constraint_dir == 1 && val0 > val1) ||
                     (constraint_dir == -1 && val0 < val1);
    
    if (violation) {
        // Pool the values (simple average for 2 points)
        float pooled = (val0 + val1) / 2.0f;
        values[global_leaf0 * output_dim + target_output] = pooled;
        values[global_leaf1 * output_dim + target_output] = pooled;
    }
}

void apply_monotonic_constraints_cuda(
    ensembleData *edata,
    ensembleMetaData *metadata,
    int tree_idx,
    int tree_depth,
    int start_leaf_idx
) {
    if (metadata->n_mono_constraints <= 0 || tree_depth <= 0) return;
    
    int n_leaves_in_tree = 1 << tree_depth;
    int n_planes = n_leaves_in_tree / 2;  // Number of pairs of leaves
    
    // Copy feature indices for this tree to host
    int* h_feature_indices = new int[tree_depth];
    cudaMemcpy(h_feature_indices, 
               edata->feature_data->feature_indices + tree_idx * metadata->max_depth,
               tree_depth * sizeof(int), 
               cudaMemcpyDeviceToHost);
    
    // FIX: Copy inequality directions for this tree
    int* h_inequality_directions = new int[tree_depth];
    int ineq_base = start_leaf_idx * metadata->max_depth;
    cudaMemcpy(h_inequality_directions,
               edata->feature_data->inequality_directions + ineq_base,
               tree_depth * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // FIX: Copy reverse feature mapping to convert internal->global indices
    int* h_reverse_mapping = new int[metadata->n_num_features];
    cudaMemcpy(h_reverse_mapping,
               edata->feature_mappings->reverse_num_feature_mapping,
               metadata->n_num_features * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // Copy monotonic constraints to host
    int* h_mono_feature_idx = new int[metadata->n_mono_constraints];
    int* h_mono_output_idx = new int[metadata->n_mono_constraints];
    int* h_mono_constraint = new int[metadata->n_mono_constraints];
    
    cudaMemcpy(h_mono_feature_idx, edata->mono_constraints->feature_idx,
               metadata->n_mono_constraints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mono_output_idx, edata->mono_constraints->output_idx,
               metadata->n_mono_constraints * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mono_constraint, edata->mono_constraints->constraint,
               metadata->n_mono_constraints * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Build map: depth -> (effective_constraint, output_idx) for this tree
    int** effective_constraints = new int*[tree_depth];
    for (int d = 0; d < tree_depth; ++d) {
        effective_constraints[d] = new int[metadata->output_dim]();
    }
    
    for (int c = 0; c < metadata->n_mono_constraints; ++c) {
        int global_feature_idx = h_mono_feature_idx[c];
        int constraint_dir = h_mono_constraint[c];
        int constraint_output = h_mono_output_idx[c];
        
        for (int d = 0; d < tree_depth; ++d) {
            // Convert internal feature index to global using reverse mapping
            int internal_idx = h_feature_indices[d];
            int global_idx = h_reverse_mapping[internal_idx];
            
            if (global_idx == global_feature_idx) {
                // If inequality_direction is inverted (0), flip the constraint
                int effective_dir = (h_inequality_directions[d] == 1) ? constraint_dir : -constraint_dir;
                effective_constraints[d][constraint_output] = effective_dir;
            }
        }
    }
    
    // Apply constraints using single-pass PAVA (like CatBoost)
    for (int out_idx = 0; out_idx < metadata->output_dim; ++out_idx) {
        for (int d = 0; d < tree_depth; ++d) {
            int constraint_dir = effective_constraints[d][out_idx];
            if (constraint_dir == 0) continue;
            
            // Apply PAVA for this depth and output
            pava_kernel<<<n_planes, 1>>>(
                edata->leaf_data->values,
                d,                    // constraint_depth
                constraint_dir,       // constraint_dir (+1 or -1)
                tree_depth,
                start_leaf_idx,
                n_leaves_in_tree,
                metadata->output_dim,
                out_idx              // target_output
            );
            cudaDeviceSynchronize();
        }
    }
    
    delete[] h_feature_indices;
    delete[] h_inequality_directions;
    delete[] h_reverse_mapping;
    delete[] h_mono_feature_idx;
    delete[] h_mono_output_idx;
    delete[] h_mono_constraint;
    for (int d = 0; d < tree_depth; ++d) {
        delete[] effective_constraints[d];
    }
    delete[] effective_constraints;
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