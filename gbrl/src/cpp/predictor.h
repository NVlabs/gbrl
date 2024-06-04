//////////////////////////////////////////////////////////////////////////////
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
//  property and proprietary rights in and to this material, related
//  documentation and any modifications thereto. Any use, reproduction,
//  disclosure or distribution of this material and related documentation
//  without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.
//
//////////////////////////////////////////////////////////////////////////////
#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "types.h"
#include "optimizer.h"

class Predictor {
    public:
        static void momentum_over_leaves(const float *obs, const char *categorical_obs, float *momentum, const ensembleData *edata, const ensembleMetaData *metadata, int start_tree_idx, const int stop_tree_idx, const int sample_idx);
        static void momentum_over_trees(const float *obs, const char *categorical_obs, float *momentum, const ensembleData *edata, const ensembleMetaData *metadata, int start_tree_idx, const int stop_tree_idx, const int sample_idx);
        static void predict_cpu(dataSet *dataset, float *preds, const ensembleData *edata, const ensembleMetaData *metadata, int start_tree_idx, int stop_tree_idx, const bool parallel_predict, std::vector<Optimizer*> opts);
        static void predict_over_leaves(const float *obs, const char *categorical_obs, float *theta, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, std::vector<Optimizer*> opts);
        static void predict_over_trees(const float *obs, const char *categorical_obs, float *theta, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, std::vector<Optimizer*> opts);
};

#endif 