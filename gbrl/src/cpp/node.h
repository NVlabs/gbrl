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
        int splitNode(const float *obs, const char *categorical_obs, const int _node_idx, const splitCandidate &split_candidate);
        float getSplitScore(dataSet *dataset, const float *feature_weights, scoreFunc split_score_func, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreCosine(const float *obs, const float *feature_weights, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreCosineCategorical(const char *obs, const float *feature_weights, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreL2(const float *obs, const float *feature_weights, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf);
        float splitScoreL2Categorical(const char *obs, const float *feature_weights, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf);
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