//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
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

/**
 * @file predictor.h
 * @brief Prediction functionality for gradient boosted ensembles
 * 
 * Provides methods for generating predictions from trained gradient boosted
 * tree ensembles, supporting both leaf-wise and tree-wise traversal strategies.
 */

#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "types.h"
#include "optimizer.h"

/**
 * @brief Prediction engine for gradient boosted ensembles
 * 
 * Implements prediction algorithms for traversing tree ensembles and
 * accumulating leaf values to produce final predictions. Supports
 * different traversal orders and parallelization strategies.
 */
class Predictor {
    public:
        /**
         * @brief Compute momentum (accumulate predictions) leaf-wise
         * 
         * Traverses ensemble in leaf-first order, accumulating predictions.
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param momentum Output array for accumulated predictions
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         * @param start_tree_idx Starting tree index (inclusive)
         * @param stop_tree_idx Stopping tree index (exclusive)
         * @param sample_idx Sample index to predict for
         */
        static void momentum_over_leaves(
            const float *obs,
            const char *categorical_obs,
            float *momentum,
            const ensembleData *edata,
            const ensembleMetaData *metadata,
            int start_tree_idx,
            const int stop_tree_idx,
            const int sample_idx
        );
        
        /**
         * @brief Compute momentum (accumulate predictions) tree-wise
         * 
         * Traverses ensemble tree-by-tree, accumulating predictions.
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param momentum Output array for accumulated predictions
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         * @param start_tree_idx Starting tree index (inclusive)
         * @param stop_tree_idx Stopping tree index (exclusive)
         * @param sample_idx Sample index to predict for
         */
        static void momentum_over_trees(
            const float *obs,
            const char *categorical_obs,
            float *momentum,
            const ensembleData *edata,
            const ensembleMetaData *metadata,
            int start_tree_idx,
            const int stop_tree_idx,
            const int sample_idx
        );
        
        /**
         * @brief Generate predictions on CPU
         * 
         * Main prediction function that selects appropriate strategy
         * and produces predictions for all samples in dataset.
         * 
         * @param dataset Input dataset
         * @param preds Output predictions array (n_samples x output_dim)
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         * @param start_tree_idx Starting tree index (inclusive)
         * @param stop_tree_idx Stopping tree index (exclusive)
         * @param parallel_predict Whether to use parallel prediction
         * @param opts Vector of optimizers for leaf value transformations
         */
        static void predict_cpu(
            dataSet *dataset,
            float *preds,
            const ensembleData *edata,
            const ensembleMetaData *metadata,
            int start_tree_idx,
            int stop_tree_idx,
            const bool parallel_predict,
            std::vector<Optimizer*> opts
        );
        
        /**
         * @brief Predict for single sample traversing leaves
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param theta Output prediction array
         * @param sample_idx Sample index
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         * @param start_tree_idx Starting tree index
         * @param stop_tree_idx Stopping tree index
         * @param opts Optimizers for transformations
         */
        static void predict_over_leaves(
            const float *obs,
            const char *categorical_obs,
            float *theta,
            const int sample_idx,
            const ensembleData *edata,
            const ensembleMetaData *metadata,
            const int start_tree_idx,
            const int stop_tree_idx,
            std::vector<Optimizer*> opts
        );
        
        /**
         * @brief Predict for single sample traversing trees
         * 
         * @param obs Numerical observations
         * @param categorical_obs Categorical observations
         * @param theta Output prediction array
         * @param sample_idx Sample index
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         * @param start_tree_idx Starting tree index
         * @param stop_tree_idx Stopping tree index
         * @param opts Optimizers for transformations
         */
        static void predict_over_trees(
            const float *obs,
            const char *categorical_obs,
            float *theta,
            const int sample_idx,
            const ensembleData *edata,
            const ensembleMetaData *metadata,
            const int start_tree_idx,
            const int stop_tree_idx,
            std::vector<Optimizer*> opts
        );
};

#endif // PREDICTOR_H 