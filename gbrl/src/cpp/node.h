//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef NODE_HEADER
#define NODE_HEADER

#include <iostream>
#include <string>
#include <cstdint>

#include "split_candidate_generator.h"
#include "types.h"

class TreeNode {
    public:
        TreeNode(int *sample_indices, const int n_samples, const int n_num_features, const int n_cat_features, const int output_dim, const int depth, const int node_idx);
        ~TreeNode();
        int splitNode(const float *obs, const char *categorical_obs, const int node_idx, const splitCandidate &split_candidate);
        float getSplitScore(dataSet *dataset, scoreFunc split_score_func, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreCosine(const float *obs, const float *grads, const float *grads_norm, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreCosineCategorical(const char *obs, const float *grads, const float *grads_norm, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreL2(const float *obs, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreL2Categorical(const char *obs, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf);
        bool isLeaf() const;
        static void printTree(TreeNode *node);
        friend std::ostream& operator<<(std::ostream& os, const TreeNode& obj);

        int *sample_indices = nullptr;
        int n_samples;
        int n_num_features;
        int n_cat_features;
        int output_dim;
        int depth;
        int node_idx;

        float feature_value;
        int feature_idx;
        splitCondition *split_conditions = nullptr;
        
        TreeNode *left_child = nullptr;
        TreeNode *right_child = nullptr;
};

std::ostream& operator<<(std::ostream& os, const TreeNode& obj);

void print_leaf(const int global_leaf_idx, const int leaf_idx, const int tree_idx, const ensembleData *edata, const ensembleMetaData *metadata);

#endif 