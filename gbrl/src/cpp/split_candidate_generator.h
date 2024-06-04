//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
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
        int computeQuantiles(const float *obs, FloatVector &quantiles, const int *sorted_feature_indices, const int feature_idx, splitCandidate *split_candidates, int n_candidates);
        
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
float scoreCosine(const int *indices, const int n_samples, const float *grads, const float *grads_norm_raw, const int n_cols);
float scoreL2(const int *indices, const int n_samples, const float *grads, const int n_col);

#endif 