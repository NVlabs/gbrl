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
#include <omp.h>

#include "cuda_loss.h"
#include "cuda_predictor.h"


void MultiRMSEGrad(const float *preds, const float *targets, float *grads, const int output_dim, const int n_samples, const int n_blocks, const int threads_per_block){
    multirmse_grad_kernel<<<n_blocks, threads_per_block>>>(preds, targets, grads, n_samples, output_dim);
    cudaDeviceSynchronize();
}

float MultiRMSEGradandLoss(const float *preds, const float *targets, float *grads, float *loss_tmp, const int output_dim, const int n_samples, const int n_blocks, const int threads_per_block){
    multirmse_grad_kernel<<<n_blocks, threads_per_block>>>(preds, targets, grads, n_samples, output_dim);
    cudaDeviceSynchronize();
    size_t shared_m = sizeof(float) * threads_per_block;

    sum_squares_kernel<<<n_blocks, threads_per_block, shared_m>>>(grads, n_samples*output_dim, loss_tmp);
    float loss = 0.0f;
    float *result = new float[n_blocks];
    cudaMemcpy(result, loss_tmp, sizeof(float)*n_blocks, cudaMemcpyDeviceToHost);
    #pragma omp simd
    for (int i = 0; i < n_blocks; ++i){
        loss += result[i];
    }
    delete[] result;
    return sqrtf(0.5f * loss / static_cast<float>(n_samples));
}

__global__ void multirmse_grad_kernel(const float* __restrict__ preds, const float* __restrict__ targets, float* __restrict__ grads, const int n_samples, const int output_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples * output_dim){
        int row = idx / output_dim, col = idx % output_dim; 
        grads[row*output_dim + col] = preds[row*output_dim + col] - targets[row*output_dim + col];
    }
}

__global__ void sum_squares_kernel(const float* __restrict__ grads, int size, float* __restrict__ result) {
    extern __shared__ float sdata[];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (idx < size) ? grads[idx] * grads[idx] : 0;
    __syncthreads();

    // Perform binary reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadIdx.x == 0){
        result[blockIdx.x] = sdata[threadIdx.x];
    }
}
