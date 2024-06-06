//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <vector>
#include <omp.h>

#include "loss.h"

float MultiRMSE::get_loss_and_gradients(const float *raw_preds, const float *raw_targets, float *raw_grads, const int n_samples, const int output_dim){
    float count_recip = 1.0f /  static_cast<float>(n_samples);
    const int n_threads = static_cast<int>(omp_get_max_threads());
    int n_elements = n_samples*output_dim;
    int elements_per_thread = n_elements / n_threads;
    int row, col;
    float grad_value, loss = 0.0f;
    std::vector<float> losses(n_threads, 0.0f);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int start_idx = thread_id * elements_per_thread;
        int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
#if !defined(_MSC_VER) && !defined(__APPLE__)
    #pragma omp simd
#endif
        for (int i = start_idx; i < end_idx; ++i){
            row = i / output_dim;
            col = i % output_dim;
            grad_value = raw_preds[row*output_dim + col] - raw_targets[row*output_dim + col];
            raw_grads[row*output_dim + col] = grad_value;
            losses[thread_id] += (grad_value * grad_value);
        }   
    }
    for (int thread_id = 0; thread_id < n_threads; ++thread_id){
        loss += losses[thread_id];
    }
    return sqrtf(0.5f * loss * count_recip);
}

float MultiRMSE::get_loss(const float *raw_preds, const float *raw_targets, const int n_samples, const int output_dim){
    float count_recip = 1.0f /  static_cast<float>(n_samples);
    const int n_threads = static_cast<int>(omp_get_max_threads());
    int samples_per_thread = n_samples / n_threads;
    int row;
    float grad_value, loss = 0.0f;
    std::vector<float> losses(n_threads, 0.0f);
    #pragma omp parallel
    {
        #pragma omp for
        for (int thread_id = 0; thread_id < n_threads; ++thread_id) {
            int start_idx = thread_id * samples_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_samples : start_idx + samples_per_thread;
            for (int sample_idx = start_idx; sample_idx < end_idx; ++sample_idx){
                row = sample_idx * output_dim;
#if !defined(_MSC_VER) && !defined(__APPLE__)
    #pragma omp simd
#endif 
                for (int d = 0; d < output_dim; ++d){
                    grad_value = raw_preds[row + d] - raw_targets[row + d];
                    losses[thread_id] += (grad_value * grad_value);
                }
            }
        }
    }
    for (int thread_id = 0; thread_id < n_threads; ++thread_id){
        loss += losses[thread_id];
    }
    return sqrtf(0.5f * loss * count_recip);
}

