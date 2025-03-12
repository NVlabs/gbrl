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
#include "utils.h"

float MultiRMSE::get_loss_and_gradients(const float *raw_preds, const float *raw_targets, float *raw_grads, const int n_samples, const int output_dim, const int par_th){
    float count_recip = 1.0f /  static_cast<float>(n_samples);
    int n_elements = n_samples * output_dim;

    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads < 1)
        n_threads = 1;
   
    int elements_per_thread = n_elements / n_threads;
    float grad_value, loss = 0.0f;
    std::vector<float> losses(n_threads, 0.0f);
    #pragma omp parallel num_threads(n_threads)
    {
        int thread_id = omp_get_thread_num();
        int start_idx = thread_id * elements_per_thread;
        int end_idx = (start_idx + elements_per_thread > n_elements) ? n_elements : start_idx + elements_per_thread;

        #pragma omp simd
        for (int i = start_idx; i < end_idx; ++i){
            grad_value = raw_preds[i] - raw_targets[i];
            raw_grads[i] = grad_value;
            losses[thread_id] += (grad_value * grad_value);
        }   
    }
    for (int thread_id = 0; thread_id < n_threads; ++thread_id){
        loss += losses[thread_id];
    }
    return sqrtf(0.5f * loss * count_recip);
}

float MultiRMSE::get_loss(const float *raw_preds, const float *raw_targets, const int n_samples, const int output_dim, const int par_th){
    float count_recip = 1.0f /  static_cast<float>(n_samples);
    int n_elements = n_samples * output_dim;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads < 1)
        n_threads = 1;
   
    int elements_per_thread = n_elements / n_threads;
    float grad_value, loss = 0.0f;
    std::vector<float> losses(n_threads, 0.0f);
    #pragma omp parallel num_threads(n_threads)
    {
        int thread_id = omp_get_thread_num();
        int start_idx = thread_id * elements_per_thread;
        int end_idx = (start_idx + elements_per_thread > n_elements) ? n_elements : start_idx + elements_per_thread;

        #pragma omp simd
        for (int i = start_idx; i < end_idx; ++i){
            grad_value = raw_preds[i] - raw_targets[i];
            losses[thread_id] += (grad_value * grad_value); 
        }
    }
    for (int thread_id = 0; thread_id < n_threads; ++thread_id){
        loss += losses[thread_id];
    }
    return sqrtf(0.5f * loss * count_recip);
}

