//////////////////////////////////////////////////////////////////////////////
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
//  property and proprietary rights in and to this material, related
//  documentation and any modifications thereto. Any use, reproduction,
//  disclosure or distribution of this material and related documentation
//  without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.
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