//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file split_candidate_generator.h
 * @brief Generation of split candidates for tree building
 * 
 * Provides functionality for generating candidate split points for
 * decision tree nodes, supporting both uniform and quantile-based
 * generation strategies for numerical and categorical features.
 */

#ifndef SPLIT_CANDIDATE_GENERATOR_H
#define SPLIT_CANDIDATE_GENERATOR_H

#include <utility>
#include <vector>

#include "types.h"

/**
 * @brief Generates candidate split points for tree building
 * 
 * Creates potential split candidates using various strategies (uniform,
 * quantile-based) for both numerical and categorical features. Candidates
 * are evaluated to find the best split at each node during tree growth.
 */
class SplitCandidateGenerator {
    public:
        /**
         * @brief Construct a split candidate generator
         * 
         * @param n_samples Number of training samples
         * @param n_num_features Number of numerical features
         * @param n_cat_features Number of categorical features
         * @param n_bins Number of candidate bins per feature
         * @param par_th Parallelization threshold
         * @param generator_type Generation strategy (Uniform/Quantile)
         */
        SplitCandidateGenerator(
            const int n_samples,
            const int n_num_features,
            const int n_cat_features,
            const int n_bins,
            const int par_th,
            const generatorType &generator_type
        );
        
        /**
         * @brief Destroy generator and free resources
         */
        ~SplitCandidateGenerator();
        
        /**
         * @brief Generate candidates for numerical features
         * 
         * @param obs Observation matrix (numerical features)
         * @param sorted_indices Pre-sorted indices for each feature
         */
        void generateNumericalSplitCandidates(
            const float *obs,
            int* const* sorted_indices
        );
        
        /**
         * @brief Generate candidates using quantile-based strategy
         * 
         * @param obs Observation matrix
         * @param sorted_indices Pre-sorted indices for each feature
         */
        void quantileSplitCandidates(
            const float *obs,
            int* const* sorted_indices
        );
        
        /**
         * @brief Generate candidates using uniform spacing strategy
         * 
         * @param obs Observation matrix
         */
        void uniformSplitCandidates(const float *obs);
        
        /**
         * @brief Process categorical features to generate candidates
         * 
         * @param categorical_obs Categorical observation matrix
         * @param grad_norms Gradient norms for prioritizing categories
         */
        void processCategoricalCandidates(
            const char *categorical_obs,
            const float *grad_norms
        );
        
        /**
         * @brief Compute quantile split points for a feature
         * 
         * @param obs Observation matrix
         * @param quantiles Output vector of quantile values
         * @param sorted_feature_indices Sorted indices for this feature
         * @param feature_idx Index of the feature
         * @param _split_candidates Output array for candidates
         * @param _n_candidates Current number of candidates
         * @return Updated number of candidates
         */
        int computeQuantiles(
            const float *obs,
            FloatVector &quantiles,
            const int *sorted_feature_indices,
            const int feature_idx,
            splitCandidate *_split_candidates,
            int _n_candidates
        );
        
        int n_samples;                  /**< Number of samples */
        int n_num_features;             /**< Number of numerical features */
        int n_cat_features;             /**< Number of categorical features */
        int n_bins;                     /**< Number of bins per feature */
        int par_th;                     /**< Parallelization threshold */
        generatorType generator_type;   /**< Generation strategy */

        splitCandidate *split_candidates; /**< Array of generated candidates */
        int n_candidates = 0;           /**< Total number of candidates */
};

/**
 * @brief Output stream operator for split candidates
 * 
 * @param os Output stream
 * @param obj Split candidate to output
 * @return Reference to output stream
 */
std::ostream& operator<<(std::ostream& os, const splitCandidate& obj);

/**
 * @brief Process categorical candidates (standalone function)
 * 
 * Generates split candidates for categorical features based on
 * gradient norms.
 * 
 * @param categorical_obs Categorical observations
 * @param grad_norms Gradient norms
 * @param n_samples Number of samples
 * @param n_cat_features Number of categorical features
 * @param n_bins Number of bins
 * @param feature_inds Output feature indices
 * @param feature_values Output feature values
 * @param category_values Output category values
 * @param numerics Output flags for numerical features
 * @return Number of candidates generated
 */
int processCategoricalCandidates_func(
    const char *categorical_obs,
    const float *grad_norms,
    const int n_samples,
    const int n_cat_features,
    const int n_bins,
    int* feature_inds,
    float *feature_values,
    char* category_values,
    bool* numerics
);

/**
 * @brief Compute cosine-based score for sample indices
 * 
 * @param indices Sample indices
 * @param n_samples Number of samples
 * @param grads Gradient matrix
 * @param n_cols Number of columns
 * @return Cosine score
 */
float scoreCosine(
    const int *indices,
    const int n_samples,
    const float *grads,
    const int n_cols
);

/**
 * @brief Compute L2-based score for sample indices
 * 
 * @param indices Sample indices
 * @param n_samples Number of samples
 * @param grads Gradient matrix
 * @param n_col Number of columns
 * @return L2 score
 */
float scoreL2(
    const int *indices,
    const int n_samples,
    const float *grads,
    const int n_col
);

#endif // SPLIT_CANDIDATE_GENERATOR_H 