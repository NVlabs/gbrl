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
}
#endif

#endif // end CUDA_MATH_H