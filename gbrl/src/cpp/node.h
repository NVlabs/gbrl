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
 * @file node.h
 * @brief Tree node structures and operations for gradient boosted trees
 * 
 * Defines the TreeNode class which represents nodes in decision trees,
 * including methods for splitting, scoring, and tree traversal.
 */

#ifndef NODE_HEADER
#define NODE_HEADER

#include <iostream>
#include <string>
#include <cstdint>

#include "split_candidate_generator.h"
#include "types.h"

/**
 * @brief Node in a gradient boosted decision tree
 * 
 * Represents a single node (internal or leaf) in a decision tree. Internal
 * nodes contain split conditions and pointers to child nodes, while leaf
 * nodes contain predictions. The class provides methods for evaluating
 * potential splits and growing the tree.
 */
class TreeNode {
    public:
        /**
         * @brief Construct a new tree node
         * 
         * @param sample_indices Array of indices of samples at this node
         * @param n_samples Number of samples at this node
         * @param n_num_features Number of numerical features
         * @param n_cat_features Number of categorical features
         * @param output_dim Dimensionality of output space
         * @param depth Depth of this node in the tree (root = 0)
         * @param node_idx Index of this node in tree traversal order
         */
        TreeNode(
            int *sample_indices,
            const int n_samples,
            const int n_num_features,
            const int n_cat_features,
            const int output_dim,
            const int depth,
            const int node_idx
        );
        
        /**
         * @brief Destroy the tree node and free resources
         */
        ~TreeNode();
        
        /**
         * @brief Split this node based on a candidate split
         * 
         * Creates left and right child nodes by partitioning samples
         * according to the split condition.
         * 
         * @param obs Numerical observation matrix
         * @param categorical_obs Categorical observation matrix
         * @param _node_idx Index for the new child nodes
         * @param split_candidate Split condition to apply
         * @return Number of nodes created (typically 2 for binary split)
         */
        int splitNode(
            const float *obs,
            const char *categorical_obs,
            const int _node_idx,
            const splitCandidate &split_candidate
        );
        
        /**
         * @brief Compute split quality score for a candidate
         * 
         * @param dataset Training dataset
         * @param split_score_func Scoring function to use (L2 or Cosine)
         * @param split_candidate Candidate split to evaluate
         * @param min_data_in_leaf Minimum samples required per leaf
         * @return Score indicating split quality (higher is better)
         */
        float getSplitScore(
            dataSet *dataset,
            scoreFunc split_score_func,
            const splitCandidate &split_candidate,
            const int min_data_in_leaf
        );
        
        /**
         * @brief Score numerical split using cosine similarity
         * 
         * @param obs Numerical observations
         * @param grads Gradient values
         * @param split_candidate Split to evaluate
         * @param min_data_in_leaf Minimum samples per leaf
         * @return Cosine-based split score
         */
        float splitScoreCosine(
            const float *obs,
            const float *grads,
            const splitCandidate &split_candidate,
            const int min_data_in_leaf
        );
        
        /**
         * @brief Score categorical split using cosine similarity
         * 
         * @param obs Categorical observations
         * @param grads Gradient values
         * @param split_candidate Split to evaluate
         * @param min_data_in_leaf Minimum samples per leaf
         * @return Cosine-based split score
         */
        float splitScoreCosineCategorical(
            const char *obs,
            const float *grads,
            const splitCandidate &split_candidate,
            const int min_data_in_leaf
        );
        
        /**
         * @brief Score numerical split using L2 norm
         * 
         * @param obs Numerical observations
         * @param grads Gradient values
         * @param split_candidate Split to evaluate
         * @param min_data_in_leaf Minimum samples per leaf
         * @return L2-based split score
         */
        float splitScoreL2(
            const float *obs,
            const float *grads,
            const splitCandidate &split_candidate,
            const int min_data_in_leaf
        );
        
        /**
         * @brief Score categorical split using L2 norm
         * 
         * @param obs Categorical observations
         * @param grads Gradient values
         * @param split_candidate Split to evaluate
         * @param min_data_in_leaf Minimum samples per leaf
         * @return L2-based split score
         */
        float splitScoreL2Categorical(
            const char *obs,
            const float *grads,
            const splitCandidate &split_candidate,
            const int min_data_in_leaf
        );
        
        /**
         * @brief Check if this node is a leaf
         * 
         * @return true if node has no children, false otherwise
         */
        bool isLeaf() const;
        
        /**
         * @brief Print tree structure starting from this node
         * 
         * @param node Root node to print from
         */
        static void printTree(TreeNode *node);
        
        /**
         * @brief Stream output operator for node information
         */
        friend std::ostream& operator<<(std::ostream& os, const TreeNode& obj);

        // Node data members
        int *sample_indices = nullptr;      /**< Indices of samples at this node */
        int n_samples;                      /**< Number of samples at this node */
        int n_num_features;                 /**< Number of numerical features */
        int n_cat_features;                 /**< Number of categorical features */
        int output_dim;                     /**< Output dimensionality */
        int depth;                          /**< Depth in tree (root = 0) */
        int node_idx;                       /**< Node index in traversal order */

        float feature_value;                /**< Split threshold value */
        int feature_idx;                    /**< Index of split feature */

        splitCondition *split_conditions = nullptr;  /**< Split condition data */
        
        TreeNode *left_child = nullptr;     /**< Pointer to left child node */
        TreeNode *right_child = nullptr;    /**< Pointer to right child node */
};

/**
 * @brief Output stream operator for TreeNode
 */
std::ostream& operator<<(std::ostream& os, const TreeNode& obj);

/**
 * @brief Print information about a specific leaf in the ensemble
 * 
 * @param global_leaf_idx Global index of leaf across all trees
 * @param leaf_idx Index of leaf within its tree
 * @param tree_idx Index of the tree containing this leaf
 * @param edata Ensemble data
 * @param metadata Ensemble metadata
 */
void print_leaf(
    const int global_leaf_idx,
    const int leaf_idx,
    const int tree_idx,
    const ensembleData *edata,
    const ensembleMetaData *metadata
);

#endif // NODE_HEADER 