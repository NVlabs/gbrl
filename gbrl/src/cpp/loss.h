#ifndef LOSS_H
#define LOSS_H

#include <utility>

class MultiRMSE{
    public:
        static float get_loss_and_gradients(const float *raw_preds, const float *raw_targets, float *raw_grads, const int n_samples, const int output_dim);
        static float get_loss(const float *raw_preds, const float *raw_targets, const int n_samples, const int output_dim);
};


#endif //