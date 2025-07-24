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
#ifndef SPLIT_CANDIDATE_GENERATOR_H
#define SPLIT_CANDIDATE_GENERATOR_H

#include <utility> 
#include <vector>

#include "types.h"


class SplitCandidateGenerator {
    public:
        SplitCandidateGenerator(const int n_samples, const int n_num_features, const int n_cat_features, const int n_bins, const int par_th, const generatorType &generator_type);
        ~SplitCandidateGenerator();
        void generateNumericalSplitCandidates(const float *obs, int* const* sorted_indices);
        void quantileSplitCandidates(const float *obs, int* const* sorted_indices);
        void uniformSplitCandidates(const float *obs);
        void processCategoricalCandidates(const char *categorical_obs, const float *grad_norms);
        int computeQuantiles(const float *obs, FloatVector &quantiles, const int *sorted_feature_indices, const int feature_idx, splitCandidate *_split_candidates, int _n_candidates);
        
        int n_samples;
        int n_num_features;
        int n_cat_features;
        int n_bins;
        int par_th;
        generatorType generator_type;

        splitCandidate *split_candidates;
        int n_candidates = 0;
};


std::ostream& operator<<(std::ostream& os, const splitCandidate& obj);

int processCategoricalCandidates_func(const char *categorical_obs, const float *grad_norms, const int n_samples, const int n_cat_features, const int n_bins, int* feature_inds, float *feature_values, char* category_values, bool* numerics);
float scoreCosine(const int *indices, const int n_samples, const float *grads, const int n_cols);
float scoreL2(const int *indices, const int n_samples, const float *grads, const int n_col);

#endif 