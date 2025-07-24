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