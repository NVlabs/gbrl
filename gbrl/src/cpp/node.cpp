//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "node.h"
#include "split_candidate_generator.h"
#include "utils.h"
#include "math_ops.h"
#include "types.h"


TreeNode::TreeNode(int *sample_indices, const int n_samples, const int n_num_features, const int n_cat_features, const int output_dim, const int depth, const int node_idx): 
            sample_indices(sample_indices), n_samples(n_samples), n_num_features(n_num_features), n_cat_features(n_cat_features),
            output_dim(output_dim), depth(depth), node_idx(node_idx), feature_value(0.0),
            feature_idx(0){
    if (depth > 0){
        this->split_conditions = new splitCondition[depth];
        for (int d = 0; d < depth; d++){
            this->split_conditions[d].categorical_value = nullptr;
        }
    }
}

TreeNode::~TreeNode(){
    if (this->sample_indices != nullptr){
        delete[] this->sample_indices;
        this->sample_indices = nullptr;
    }
    
    if (this->split_conditions != nullptr){
        for (int d = 0; d < this->depth; ++d){
            if (this->split_conditions[d].categorical_value != nullptr)
                delete[] this->split_conditions[d].categorical_value;
        }
        delete[] this->split_conditions;
        this->split_conditions = nullptr;
    }
    
    delete this->left_child;        // Delete left child node
    delete this->right_child;       // Delete right child node
    // Setting to nullptr is optional in the destructor
    this->left_child = nullptr;
    this->right_child = nullptr;

}

int TreeNode::splitNode(const float *obs, const char *categorical_obs, const int node_idx, const splitCandidate &split_candidate){
    std::vector<int> pre_left_indices(this->n_samples), pre_right_indices(this->n_samples);
    int left_count = 0, right_count = 0;
    bool is_categorical = split_candidate.categorical_value != nullptr;
    int n_features = (is_categorical) ? this->n_cat_features : this->n_num_features;
    // std::cout << "n_num_features: " << this->n_num_features << " n_cat_features: " << this->n_cat_features << " splitting on: " << split_candidate  << std::endl;
    const int *sample_indices = this->sample_indices;
    int sample_idx, row_idx;

    if (is_categorical){
        for (int n = 0; n < this->n_samples; ++n){
            sample_idx = sample_indices[n];
            row_idx = sample_idx*n_features;
            if (strcmp(&categorical_obs[(row_idx + split_candidate.feature_idx) * MAX_CHAR_SIZE],  split_candidate.categorical_value) == 0){
                pre_right_indices[right_count] = sample_idx;
                ++right_count;
            } else {
                pre_left_indices[left_count] = sample_idx;
                ++left_count;
            }
        }
    } else {
        for (int n = 0; n < this->n_samples; ++n){
            sample_idx = sample_indices[n];
            row_idx = sample_idx*n_features;
            if (obs[row_idx + split_candidate.feature_idx] > split_candidate.feature_value){
                pre_right_indices[right_count] = sample_idx;
                ++right_count;
            } else {
                pre_left_indices[left_count] = sample_idx;
                ++left_count;
            }
        }
    }

    int *left_indices = new int[left_count];
    std::copy(pre_left_indices.begin(), pre_left_indices.begin() + left_count, left_indices);

    this->left_child = new TreeNode(left_indices, left_count, this->n_num_features, this->n_cat_features, this->output_dim, this->depth + 1, node_idx + 1);
    if (this->left_child == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        delete[] left_indices;
        return -1;
    }
    int *right_indices = new int[right_count];
    std::copy(pre_right_indices.begin(), pre_right_indices.begin() + right_count, right_indices);
    this->right_child = new TreeNode(right_indices, right_count, this->n_num_features, this->n_cat_features, this->output_dim, this->depth + 1, node_idx + 2);
    if (this->right_child == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        delete[] right_indices;
        delete this->left_child;
        return -1;
    }
    if (this->depth > 0){
        std::copy(this->split_conditions, this->split_conditions + this->depth, this->left_child->split_conditions);
        std::copy(this->split_conditions, this->split_conditions + this->depth, this->right_child->split_conditions);
        for (int d = 0; d < this->depth; ++d){
            if (this->split_conditions[d].categorical_value != nullptr){
                left_child->split_conditions[d].categorical_value = new char[MAX_CHAR_SIZE];
                right_child->split_conditions[d].categorical_value = new char[MAX_CHAR_SIZE];
                std::copy(this->split_conditions[d].categorical_value, this->split_conditions[d].categorical_value + MAX_CHAR_SIZE, left_child->split_conditions[d].categorical_value);
                std::copy(this->split_conditions[d].categorical_value, this->split_conditions[d].categorical_value + MAX_CHAR_SIZE, right_child->split_conditions[d].categorical_value);
            }
        }
    }
    this->left_child->split_conditions[this->depth].feature_idx = split_candidate.feature_idx;
    this->left_child->split_conditions[this->depth].feature_value = split_candidate.feature_value;
    this->left_child->split_conditions[this->depth].edge_weight = (this->n_samples > 0) ? static_cast<float>(left_count) / (static_cast<float>(this->n_samples)) : 0.0f;
    if (split_candidate.categorical_value != nullptr){
        this->left_child->split_conditions[this->depth].categorical_value = new char[MAX_CHAR_SIZE];
        std::copy(split_candidate.categorical_value, split_candidate.categorical_value + MAX_CHAR_SIZE, this->left_child->split_conditions[this->depth].categorical_value);
    }
    this->left_child->split_conditions[this->depth].inequality_direction = false;

    // For right child
    this->right_child->split_conditions[this->depth].feature_idx = split_candidate.feature_idx;
    this->right_child->split_conditions[this->depth].feature_value = split_candidate.feature_value;
    this->right_child->split_conditions[this->depth].edge_weight = (this->n_samples > 0) ? static_cast<float>(right_count) / (static_cast<float>(this->n_samples)) : 0.0f;
    if (split_candidate.categorical_value != nullptr){
        this->right_child->split_conditions[this->depth].categorical_value = new char[MAX_CHAR_SIZE];
        std::copy(split_candidate.categorical_value, split_candidate.categorical_value + MAX_CHAR_SIZE, this->right_child->split_conditions[this->depth].categorical_value);
    }
    this->right_child->split_conditions[this->depth].inequality_direction = true;
    return 0;
}

float TreeNode::getSplitScore(dataSet *dataset, scoreFunc split_score_func, const splitCandidate &split_candidate, const int min_data_in_leaf){
    // make sure that we do not re-use the same split candidate along a path
    bool is_numeric = split_candidate.categorical_value == nullptr;
    if (this->depth > 0){
        if (is_numeric){
            for (int i = 0; i < this->depth; ++i){
                if (this->split_conditions[i].categorical_value == nullptr && this->split_conditions[i].feature_value == split_candidate.feature_value && this->split_conditions[i].feature_idx == split_candidate.feature_idx)
                    return -INFINITY;
            }
        } else {
            for (int i = 0; i < this->depth; ++i){
                if (this->split_conditions[i].categorical_value != nullptr && strcmp(this->split_conditions[i].categorical_value, split_candidate.categorical_value) == 0 && this->split_conditions[i].feature_idx == split_candidate.feature_idx)
                    return -INFINITY;
            }
        }
    }
    switch (split_score_func) {
        case L2: {
            if (is_numeric)
                return this->splitScoreL2(dataset->obs, dataset->build_grads, split_candidate, min_data_in_leaf);
            else
                return this->splitScoreL2Categorical(dataset->categorical_obs, dataset->build_grads, split_candidate, min_data_in_leaf);
        }
        case Cosine: {
            if (is_numeric)
                return this->splitScoreCosine(dataset->obs, dataset->build_grads, dataset->norm_grads, split_candidate, min_data_in_leaf);
            else
                return this->splitScoreCosineCategorical(dataset->categorical_obs, dataset->build_grads, dataset->norm_grads, split_candidate, min_data_in_leaf);
        }
        default: {
            std::cerr << "Unknown scoreFunc." << std::endl;
            return -INFINITY;
        }
    }
}

float TreeNode::splitScoreCosine(const float *obs, const float *grads, const float *grads_norm, const splitCandidate &split_candidate, const int min_data_in_leaf){
    int left_count = 0, right_count = 0;
    int n_features = this->n_num_features, n_cols = this->output_dim;
    int *left_indices = new int[this->n_samples];
    int *right_indices = new int[this->n_samples];

    const int *sample_indices = this->sample_indices;
    float *left_mean = new float[n_cols]; 
    float *right_mean = new float[n_cols]; 
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] = 0;
        right_mean[d] = 0;
    }
    
    float left_norms = 0, right_norms = 0;
    int sample_idx, grad_row;

    for (int n = 0; n < this->n_samples; ++n){
        sample_idx = sample_indices[n];
        grad_row = sample_idx*n_cols;
        if (obs[sample_idx*n_features + split_candidate.feature_idx] > split_candidate.feature_value){
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                right_mean[d] += grads[grad_row + d];
            right_norms += grads_norm[sample_idx];
            right_indices[right_count] = sample_idx;
            ++right_count;
        } else {
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                left_mean[d] += grads[grad_row + d];
            left_norms += grads_norm[sample_idx];
            left_indices[left_count] = sample_idx;
            ++left_count;
        }
    }
    
    if (left_count < min_data_in_leaf || right_count < min_data_in_leaf){
        delete[] left_mean;
        delete[] right_mean;
        delete[] left_indices;
        delete[] right_indices;

        return -INFINITY;
    } 

    float left_count_f = static_cast<float>(left_count), right_count_f = static_cast<float>(right_count);
    float left_count_recip = (left_count > 0 ) ? 1.0f / left_count_f : 0.0f;
    float right_count_recip = (right_count > 0 ) ? 1.0f / right_count_f : 0.0f;
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] *= left_count_recip;
        right_mean[d] *= right_count_recip;
    }

    float left_cosine = cosine_dist(left_indices, grads, left_mean, left_count, n_cols, left_norms);
    float right_cosine = cosine_dist(right_indices, grads, right_mean, right_count, n_cols, right_norms);

    delete[] left_mean;
    delete[] right_mean;
    delete[] left_indices;
    delete[] right_indices;
 
    return left_cosine + right_cosine;
}

float TreeNode::splitScoreCosineCategorical(const char *obs, const float *grads, const float *grads_norm, const splitCandidate &split_candidate, const int min_data_in_leaf){
    int left_count = 0, right_count = 0;
    int n_features = this->n_cat_features, n_cols = this->output_dim;
    int *left_indices = new int[this->n_samples];
    int *right_indices = new int[this->n_samples];

    const int *sample_indices = this->sample_indices;
    float *left_mean = new float[n_cols]; 
    float *right_mean = new float[n_cols]; 
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] = 0;
        right_mean[d] = 0;
    }
    
    float left_norms = 0, right_norms = 0;
    int sample_idx, grad_row;

    for (int n = 0; n < this->n_samples; ++n){
        sample_idx = sample_indices[n];
        grad_row = sample_idx*n_cols;
        if (strcmp(&obs[(sample_idx*n_features + split_candidate.feature_idx) * MAX_CHAR_SIZE],  split_candidate.categorical_value) == 0){
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                right_mean[d] += grads[grad_row + d];
            right_norms += grads_norm[sample_idx];
            right_indices[right_count] = sample_idx;
            ++right_count;
        } else {
         #ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                left_mean[d] += grads[grad_row + d];
            left_norms += grads_norm[sample_idx];
            left_indices[left_count] = sample_idx;
            ++left_count;
        }
    }
    
    if (left_count < min_data_in_leaf || right_count < min_data_in_leaf){
        delete[] left_mean;
        delete[] right_mean;
        delete[] left_indices;
        delete[] right_indices;

        return -INFINITY;
    } 

    float left_count_f = static_cast<float>(left_count), right_count_f = static_cast<float>(right_count);
    float left_count_recip = (left_count > 0 ) ? 1.0f / left_count_f : 0.0f;
    float right_count_recip = (right_count > 0 ) ? 1.0f / right_count_f : 0.0f;
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] *= left_count_recip;
        right_mean[d] *= right_count_recip;
    }

    float left_cosine = cosine_dist(left_indices, grads, left_mean, left_count, n_cols, left_norms);
    float right_cosine = cosine_dist(right_indices, grads, right_mean, right_count, n_cols, right_norms);

    delete[] left_mean;
    delete[] right_mean;
    delete[] left_indices;
    delete[] right_indices;
 
    return left_cosine + right_cosine;
}


float TreeNode::splitScoreL2(const float *obs, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf){
    int left_count = 0, right_count = 0;
    int n_cols = this->output_dim, n_features = this->n_num_features;
    const int *sample_indices = this->sample_indices;

    float *left_mean = new float[n_cols]; 
    float *right_mean = new float[n_cols]; 
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] = 0;
        right_mean[d] = 0;
    }
    int sample_idx, grad_row;

    for (int n = 0; n < this->n_samples; ++n){
        sample_idx = sample_indices[n];
        grad_row = sample_idx*n_cols;
        if (obs[sample_idx*n_features + split_candidate.feature_idx] > split_candidate.feature_value){
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                    right_mean[d] += grads[grad_row + d];
            ++right_count;
        } else {
         #ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                left_mean[d] += grads[grad_row + d];
            ++left_count;
        }
    }

    if (left_count < min_data_in_leaf || right_count < min_data_in_leaf){
        delete[] left_mean;
        delete[] right_mean;
        return -INFINITY;
    } 

    float left_count_f = static_cast<float>(left_count), right_count_f = static_cast<float>(right_count);
    float left_count_recip = (left_count > 0 ) ? 1.0f / left_count : 0.0f;
    float right_count_recip = (right_count > 0) ? 1.0f / right_count_f : 0.0f;
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] *= left_count_recip;
        right_mean[d] *= right_count_recip;
    }
    float left_mean_norm = squared_norm(left_mean, n_cols);
    float right_mean_norm = squared_norm(right_mean, n_cols);
    delete[] left_mean;
    delete[] right_mean;
    return left_count_f*left_mean_norm + right_count_f*right_mean_norm;
}

float TreeNode::splitScoreL2Categorical(const char *obs, const float *grads, const splitCandidate &split_candidate, const int min_data_in_leaf){
    int left_count = 0, right_count = 0;
    int n_cols = this->output_dim, n_features = this->n_cat_features;
    const int *sample_indices = this->sample_indices;

    float *left_mean = new float[n_cols]; 
    float *right_mean = new float[n_cols]; 
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] = 0;
        right_mean[d] = 0;
    }
    int sample_idx, grad_row;

    for (int n = 0; n < this->n_samples; ++n){
        sample_idx = sample_indices[n];
        grad_row = sample_idx*n_cols;
        if (strcmp(&obs[(sample_idx*n_features + split_candidate.feature_idx) * MAX_CHAR_SIZE], split_candidate.categorical_value) == 0){
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                    right_mean[d] += grads[grad_row + d];
            ++right_count;
        } else {
         #ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < n_cols; ++d)
                left_mean[d] += grads[grad_row + d];
            ++left_count;
        }
    }

    if (left_count < min_data_in_leaf || right_count < min_data_in_leaf){
        delete[] left_mean;
        delete[] right_mean;
        return -INFINITY;
    } 


    float left_count_f = static_cast<float>(left_count), right_count_f = static_cast<float>(right_count);
    float left_count_recip = (left_count > 0 ) ? 1.0f / left_count : 0.0f;
    float right_count_recip = (right_count > 0 ) ? 1.0f / right_count_f : 0.0f;
 #ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        left_mean[d] *= left_count_recip;
        right_mean[d] *= right_count_recip;
    }
    float left_mean_norm = squared_norm(left_mean, n_cols);
    float right_mean_norm = squared_norm(right_mean, n_cols);
    delete[] left_mean;
    delete[] right_mean;
    return left_count_f*left_mean_norm + right_count_f*right_mean_norm;
}

bool TreeNode::isLeaf() const {
    return (!this->left_child && !this->right_child);
}

void TreeNode::printTree(TreeNode *node){
    if (node != nullptr && !node->isLeaf()){
        std::cout << *node << std::endl;
        if (node->left_child != nullptr)
            TreeNode::printTree(node->left_child);
        if (node->right_child != nullptr)
            TreeNode::printTree(node->right_child);   
    }
}

std::ostream& operator<<(std::ostream& os, const TreeNode& obj){
    os << "-----NodeWrapper-----" << std::endl;
    os << "node_idx: " <<  obj.node_idx << " n_samples: " << obj.n_samples << ", n_num_features: " << obj.n_num_features << ", n_cat_features: " << obj.n_cat_features << ", output_dim: " << obj.output_dim;
    os << ", depth: " << obj.depth << " feature_idx: " << obj.feature_idx << " feature_value: " << obj.feature_value << std::endl;
    os << "sample indices: [";
    for (int i = 0 ; i < obj.n_samples; ++i){
        os << obj.sample_indices[i]; 
        if (i < obj.n_samples - 1)
            os << ", ";
    }
    os << "]" << std::endl;
    if (obj.depth > 0){
        os << "feature_idxs size: " <<  obj.depth <<" [";
        for (int i = 0 ; i < obj.depth; ++i){
            os << obj.split_conditions[i].feature_idx; 
            if (i < obj.depth - 1)
                os << ", ";
        }
        os << "]" << std::endl;
        os << "feature_values : [";
        for (int i = 0 ; i < obj.depth; ++i){
            os << obj.split_conditions[i].feature_value; 
            if (i < obj.depth - 1)
                os << ", ";
        }
        os << "]" << std::endl;
        os << "inequality_directions: [";
        for (int i = 0 ; i < obj.depth; ++i){
            os << obj.split_conditions[i].inequality_direction; 
            if (i < obj.depth - 1)
                os << ", ";
        }
        os << "]" << std::endl;

        bool left_child = obj.left_child != nullptr;
        bool right_child = obj.right_child != nullptr;
        os << "left_child: " << left_child << ", right_child: " << right_child << std::endl;
    }
    return os;
}


void print_leaf(const int global_leaf_idx, const int leaf_idx, const int tree_idx, const ensembleData *edata, const ensembleMetaData *metadata){
    int idx = (metadata->grow_policy == OBLIVIOUS) ? tree_idx : global_leaf_idx;
    std::cout << "Leaf idx: " << leaf_idx << " tree_idx: " << tree_idx;
    std::cout << " output_dim: " << metadata->output_dim << " depth: " << edata->depths[idx];
#ifdef DEBUG
    std::cout << " n_samples: " << edata->n_samples[global_leaf_idx]  << " value: [";
#else 
    std::cout << " value: [";
 #endif    
    if (edata->values != nullptr){
        for (int i = 0 ; i < metadata->output_dim ; i++){
            std::cout << edata->values[global_leaf_idx * metadata->output_dim + i];
            if (i < metadata->output_dim - 1)
                std::cout << ", ";
        }
    }
    
    int cond_idx = idx * metadata->max_depth;
    std::cout << "] ";
    std::cout << " feature_idxs: [";
    for (int i = 0 ; i < edata->depths[idx] ; i++){
        if (edata->is_numerics[cond_idx + i])
            std::cout << edata->feature_indices[cond_idx + i];
        else
            std::cout << std::to_string(edata->feature_indices[cond_idx + i] + metadata->n_num_features);

        if (i < edata->depths[idx] - 1)
            std::cout << ", ";
    }
    std::cout << "] ";
    std::cout << " inequality_directions: [";
    for (int i = 0 ; i < edata->depths[idx] ; i++){
        std::cout << edata->inequality_directions[global_leaf_idx * metadata->max_depth + i];
        if (i < edata->depths[idx] - 1)
            std::cout << ", ";
    }
    
    std::cout << "] ";
    std::cout << " feature_values: [";
    for (int i = 0 ; i < edata->depths[idx] ; i++){
        if (edata->is_numerics[cond_idx + i]){
            std::cout << edata->feature_values[cond_idx + i];
        }    
        else {
            for (int j = 0; j < MAX_CHAR_SIZE; ++j)
                std::cout << edata->categorical_values[(cond_idx + i)*MAX_CHAR_SIZE + j];
        }
        if (i < edata->depths[idx] - 1)
            std::cout << ", ";
    }
    
    std::cout << "]" << std::endl;
    std::cout << " edge_weights: [";
    for (int i = 0 ; i < edata->depths[idx] ; i++){
        std::cout << edata->edge_weights[global_leaf_idx * metadata->max_depth + i];
        if (i < edata->depths[idx] - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    return;
}





