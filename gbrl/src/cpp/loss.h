//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef LOSS_H
#define LOSS_H

#include <utility>

class MultiRMSE{
    public:
        static float get_loss_and_gradients(const float *raw_preds, const float *raw_targets, float *raw_grads, const int n_samples, const int output_dim);
        static float get_loss(const float *raw_preds, const float *raw_targets, const int n_samples, const int output_dim);
};


#endif //