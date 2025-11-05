//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
/**
 * @file cuda_loss.h
 * @brief GPU loss function implementations for gradient boosting
 * 
 * Provides CUDA kernels for computing gradients and loss values
 * on NVIDIA GPUs.
 */

#ifndef CUDA_LOSS_H
#define CUDA_LOSS_H

#include "types.h"
#include "cuda_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute MultiRMSE gradients on GPU
 * 
 * Calculates prediction errors (preds - targets) for multiple outputs.
 * 
 * @param preds Model predictions (n_samples x output_dim)
 * @param targets Ground truth targets (n_samples x output_dim)
 * @param grads Output gradients (n_samples x output_dim)
 * @param output_dim Output dimensionality
 * @param n_samples Number of samples
 * @param n_blocks Number of CUDA blocks
 * @param threads_per_block Threads per CUDA block
 */
void MultiRMSEGrad(
    const float *preds,
    const float *targets,
    float *grads,
    const int output_dim,
    const int n_samples,
    const int n_blocks,
    const int threads_per_block
);

/**
 * @brief Compute MultiRMSE gradients and loss on GPU
 * 
 * Calculates both prediction errors and mean squared error loss.
 * 
 * @param preds Model predictions (n_samples x output_dim)
 * @param targets Ground truth targets (n_samples x output_dim)
 * @param grads Output gradients (n_samples x output_dim)
 * @param loss_tmp Temporary buffer for loss reduction
 * @param output_dim Output dimensionality
 * @param n_samples Number of samples
 * @param n_blocks Number of CUDA blocks
 * @param threads_per_block Threads per CUDA block
 * @return Mean squared error loss value
 */
float MultiRMSEGradandLoss(
    const float *preds,
    const float *targets,
    float *grads,
    float *loss_tmp,
    const int output_dim,
    const int n_samples,
    const int n_blocks,
    const int threads_per_block
);

#ifdef __CUDACC__  // NVCC only
/**
 * @brief CUDA kernel for MultiRMSE gradient computation
 * 
 * @param preds Model predictions
 * @param targets Ground truth targets
 * @param grads Output gradient array
 * @param n_samples Number of samples
 * @param output_dim Output dimensionality
 */
__global__ void multirmse_grad_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ targets,
    float* __restrict__ grads,
    const int n_samples,
    const int output_dim
);

/**
 * @brief CUDA kernel for sum of squares reduction
 * 
 * @param grads Gradient values to square and sum
 * @param size Array size
 * @param result Output sum of squares
 */
__global__ void sum_squares_kernel(
    const float* __restrict__ grads,
    int size,
    float* __restrict__ result
);
#endif

#ifdef __cplusplus
}
#endif

#endif // CUDA_LOSS_H