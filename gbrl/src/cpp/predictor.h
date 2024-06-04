//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
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