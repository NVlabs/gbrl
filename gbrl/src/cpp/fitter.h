//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file fitter.h
 * @brief Tree fitting and training for gradient boosting
 * 
 * Provides functionality for fitting gradient boosted trees, including
 * tree growing, leaf value optimization, and ensemble updates. Supports
 * both greedy and oblivious tree growing strategies.
 */

#ifndef FITTER_H
#define FITTER_H

#include "types.h"
#include "split_candidate_generator.h"
#include "node.h"
#include "optimizer.h"

/**
 * @brief Tree fitting engine for gradient boosting
 * 
 * Implements algorithms for growing decision trees and optimizing
 * their parameters during gradient boosting training. Handles both
 * CPU-based fitting with various growth strategies.
 */
class Fitter {
    public:
        /**
         * @brief Perform single boosting step on CPU
         * 
         * Adds one new tree to the ensemble by fitting to current gradients.
         * 
         * @param dataset Training dataset
         * @param edata Ensemble data (modified in-place)
         * @param metadata Ensemble metadata
         */
        static void step_cpu(
            dataSet *dataset,
            ensembleData *edata,
            ensembleMetaData *metadata
        );
        
        /**
         * @brief Fit ensemble for multiple iterations on CPU
         * 
         * Main training loop that grows the ensemble over multiple iterations,
         * computing gradients and adding trees to minimize the loss.
         * 
         * @param dataset Training dataset
         * @param targets Target values (n_samples x output_dim)
         * @param edata Ensemble data (modified in-place)
         * @param metadata Ensemble metadata (modified in-place)
         * @param iterations Number of boosting iterations
         * @param loss_type Loss function to use
         * @param opts Vector of optimizers for leaf value updates
         * @return Final loss value after training
         */
        static float fit_cpu(
            dataSet *dataset,
            const float *targets,
            ensembleData *edata,
            ensembleMetaData *metadata,
            const int iterations,
            lossType loss_type,
            std::vector<Optimizer*> opts
        );
        
        /**
         * @brief Fit single tree using greedy growth strategy
         * 
         * Grows tree by recursively selecting best split at each node
         * independently, following a depth-first or breadth-first order.
         * 
         * @param dataset Training dataset
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         * @param generator Split candidate generator
         * @return Number of leaves created
         */
        static int fit_greedy_tree(
            dataSet *dataset,
            ensembleData *edata,
            ensembleMetaData *metadata,
            const SplitCandidateGenerator &generator
        );
        
        /**
         * @brief Fit single tree using oblivious growth strategy
         * 
         * Grows symmetric tree where all nodes at same depth use the
         * same split feature, resulting in a balanced tree structure.
         * 
         * @param dataset Training dataset
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         * @param generator Split candidate generator
         * @return Number of leaves created
         */
        static int fit_oblivious_tree(
            dataSet *dataset,
            ensembleData *edata,
            ensembleMetaData *metadata,
            const SplitCandidateGenerator &generator
        );
        
        /**
         * @brief Optimize leaf values for recently added leaves
         * 
         * Computes optimal prediction values for leaves by averaging
         * gradients of samples falling in each leaf.
         * 
         * @param dataset Training dataset
         * @param edata Ensemble data (modified in-place)
         * @param metadata Ensemble metadata
         * @param added_leaves Number of leaves that were just added
         */
        static void fit_leaves(
            dataSet *dataset,
            ensembleData *edata,
            ensembleMetaData *metadata,
            const int added_leaves
        );
        
        /**
         * @brief Update ensemble data structure for a single leaf
         * 
         * Adds node's split conditions and structure information to
         * the ensemble data arrays (per-leaf update).
         * 
         * @param edata Ensemble data (modified in-place)
         * @param metadata Ensemble metadata
         * @param node Tree node to add
         */
        static void update_ensemble_per_leaf(
            ensembleData *edata,
            ensembleMetaData *metadata,
            const TreeNode* node
        );
        
        /**
         * @brief Update ensemble data structure for entire tree
         * 
         * Adds all nodes' split conditions to ensemble data arrays
         * (per-tree batch update for oblivious trees).
         * 
         * @param edata Ensemble data (modified in-place)
         * @param metadata Ensemble metadata
         * @param nodes Vector of tree nodes to add
         * @param n_nodes Number of nodes in vector
         */
        static void update_ensemble_per_tree(
            ensembleData *edata,
            ensembleMetaData *metadata,
            std::vector<TreeNode*> nodes,
            const int n_nodes
        );
        
        /**
         * @brief Calculate optimal value for a specific leaf
         * 
         * Computes leaf prediction by averaging gradients of samples
         * assigned to that leaf.
         * 
         * @param data Training dataset
         * @param edata Ensemble data (modified in-place)
         * @param metadata Ensemble metadata
         * @param leaf_idx Index of leaf within its tree
         * @param tree_idx Index of tree in ensemble
         */
        static void calc_leaf_value(
            dataSet *data,
            ensembleData *edata,
            ensembleMetaData *metadata,
            const int leaf_idx,
            const int tree_idx
        );
        
        /**
         * @brief Apply control variates for variance reduction
         * 
         * Applies control variates technique to reduce variance in
         * gradient estimates, improving training stability.
         * 
         * @param dataset Training dataset
         * @param edata Ensemble data
         * @param metadata Ensemble metadata
         */
        static void control_variates(
            dataSet *dataset,
            ensembleData *edata,
            ensembleMetaData *metadata
        );
};

#endif // FITTER_H