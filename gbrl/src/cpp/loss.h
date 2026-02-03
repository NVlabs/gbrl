//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
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

/**
 * @file loss.h
 * @brief Loss functions for gradient boosting
 * 
 * Provides loss function implementations for computing gradients
 * and loss values during training.
 */

#ifndef LOSS_H
#define LOSS_H

#include <utility>

/**
 * @brief Multi-output Root Mean Squared Error loss function
 * 
 * Implements RMSE loss for multi-dimensional outputs, computing both
 * loss values and gradients efficiently for gradient boosting.
 */
class MultiRMSE {
    public:
        /**
         * @brief Compute loss value and gradients
         * 
         * @param raw_preds Raw prediction values (n_samples x output_dim)
         * @param raw_targets Target values (n_samples x output_dim)
         * @param raw_grads Output gradient array (n_samples x output_dim)
         * @param n_samples Number of samples
         * @param output_dim Dimensionality of output space
         * @param par_th Parallelization threshold
         * @return Total loss value
         */
        static float get_loss_and_gradients(
            const float *raw_preds,
            const float *raw_targets,
            float *raw_grads,
            const int n_samples,
            const int output_dim,
            const int par_th
        );
        
        /**
         * @brief Compute loss value only (no gradients)
         * 
         * @param raw_preds Raw prediction values (n_samples x output_dim)
         * @param raw_targets Target values (n_samples x output_dim)
         * @param n_samples Number of samples
         * @param output_dim Dimensionality of output space
         * @param par_th Parallelization threshold
         * @return Total loss value
         */
        static float get_loss(
            const float *raw_preds,
            const float *raw_targets,
            const int n_samples,
            const int output_dim,
            const int par_th
        );
};

#endif // LOSS_H