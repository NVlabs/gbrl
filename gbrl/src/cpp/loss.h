//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
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