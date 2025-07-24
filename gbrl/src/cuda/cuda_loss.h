//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
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
#ifndef CUDA_LOSS_H
#define CUDA_LOSS_H

#include "types.h"
#include "cuda_types.h"

#ifdef __cplusplus
extern "C" {
#endif
void MultiRMSEGrad(const float *preds, const float *targets, float *grads, const int output_dim, const int n_samples, const int n_blocks, const int threads_per_block);
float MultiRMSEGradandLoss(const float *preds, const float *targets, float *grads, float *loss_tmp, const int output_dim, const int n_samples, const int n_blocks, const int threads_per_block);
#ifdef __CUDACC__  // This macro is defined by NVCC
__global__ void multirmse_grad_kernel(const float* __restrict__ preds, const float* __restrict__ targets, float* __restrict__ grads, const int n_samples, const int output_dim);
__global__ void sum_squares_kernel(const float* __restrict__ grads, int size, float* __restrict__ result);
#endif 

#ifdef __cplusplus
} // extern C
#endif

#endif // end CUDA_LOSS_H