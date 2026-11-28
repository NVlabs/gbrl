//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
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
 * @file fitter.cpp
 * @brief Implementation of tree fitting and ensemble training functions
 */

#include <deque>
#include <omp.h>
#include <vector>
#include <memory>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "fitter.h"
#include "split_candidate_generator.h"
#include "node.h"
#include "types.h"
#include "utils.h"
#include "predictor.h"
#include "loss.h"
#include "math_ops.h"



void Fitter::step_cpu(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata){
    const int output_dim = metadata->output_dim, par_th = metadata->par_th;
    const int n_trees = metadata->n_trees;
    const int n_objs = metadata->n_objs;
    if (metadata->use_cv && n_trees > 0){
        Fitter::control_variates(dataset, edata, metadata);
    }

    // Copy all objectives' worth of gradients
    float *build_grads = copy_mat(dataset->grads->data, dataset->n_samples*output_dim*n_objs, par_th);
    if (metadata->split_score_func == L2){
        // Normalize each objective's gradient block independently
        for (int k = 0; k < n_objs; ++k) {
            float *obj_block = build_grads + (size_t)k * dataset->n_samples * output_dim;
            float *mean_grads = calculate_mean(obj_block, dataset->n_samples, output_dim, par_th);
            float *std = calculate_std_and_center(obj_block, mean_grads, dataset->n_samples, output_dim, par_th);
            divide_mat_by_vec_inplace(obj_block, std, dataset->n_samples, output_dim, par_th);
            delete[] mean_grads;
            delete[] std;
        }
    } 

    SplitCandidateGenerator generator = SplitCandidateGenerator(dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->n_bins, metadata->par_th, metadata->generator_type);
    int **indices = nullptr;
    const float *obs = dataset->obs->data;
    const int n_num_features = metadata->n_num_features;

    if (metadata->n_num_features > 0){
        indices = new int*[n_num_features];
        for (int i = 0; i < n_num_features; ++i){
            int *col_inds = new int[dataset->n_samples];
            std::iota(col_inds, col_inds + dataset->n_samples, 0);
            // sort indices per column according to the obs column values
            std::sort(col_inds, col_inds + dataset->n_samples,
            [&obs, n_num_features, i](int a, int b) {
                return obs[a*n_num_features + i] < obs[b*n_num_features + i];
            });
            indices[i] = col_inds;
        }
        generator.generateNumericalSplitCandidates(dataset->obs->data, indices);
    }
    if (metadata->n_cat_features > 0)
        generator.processCategoricalCandidates(dataset->categorical_obs->data);

    dataHolder<float> build_grads_holder{build_grads, cpu};
    dataset->build_grads = &build_grads_holder; 
    
    // The fit_*_tree call below already advances the ensemble counts, and the
    // projection after it can throw, so snapshot them for the rollback.
    const int trees_before_step = metadata->n_trees;
    const int leaves_before_step = metadata->n_leaves;

    int added_leaves = 0;
    if (metadata->grow_policy == GREEDY)
        added_leaves = Fitter::fit_greedy_tree(dataset, edata, metadata, generator);
    else 
        added_leaves = Fitter::fit_oblivious_tree(dataset, edata, metadata, generator);
    Fitter::fit_leaves(dataset, edata, metadata, added_leaves);
    
    // Apply monotonic constraints AFTER leaf values are computed
    if (metadata->n_mono_constraints > 0 && metadata->grow_policy == OBLIVIOUS) {
        int tree_idx = metadata->n_trees - 1;
        int tree_depth = edata->ensemble_info->depths[tree_idx];
        int start_leaf_idx = edata->ensemble_info->tree_indices[tree_idx];
        if (tree_depth > 0) {
            try {
                Fitter::apply_monotonic_constraints_cpu(edata, metadata, tree_idx, tree_depth, start_leaf_idx);
            } catch (...) {
                // Un-publish the tree, then release this call's temporaries since
                // the cleanup below is skipped by the unwind.
                metadata->n_trees = trees_before_step;
                metadata->n_leaves = leaves_before_step;
                if (indices != nullptr) {
                    for (int i = 0; i < metadata->n_num_features; ++i)
                        delete[] indices[i];
                    delete[] indices;
                }
                delete[] build_grads;
                throw;
            }
        }
    }

    if (indices != nullptr){
        for (int i = 0; i < metadata->n_num_features; ++i)
            delete[] indices[i];
        delete[] indices;
    }
    
    delete[] build_grads;
    metadata->iteration++;
}

float Fitter::fit_cpu(dataSet *dataset, const float* targets, ensembleData *edata, ensembleMetaData *metadata, const int iterations, lossType loss_type, std::vector<Optimizer*> opts){
    // Each tree is built on one mini-batch of at most batch_size rows, so a
    // dataset too large to build a tree on is still trainable. The loop below
    // advances to the next batch per tree, wrapping back to row 0.
    int batch_start_idx = 0, output_dim = metadata->output_dim;
    int batch_size = metadata->batch_size, par_th = metadata->par_th;
    int batch_n_samples = batch_start_idx + batch_size < dataset->n_samples ? batch_size : dataset->n_samples - batch_start_idx;
    bool is_last_batch;
    int batch_preds_size = metadata->batch_size*metadata->output_dim, last_batch_preds_size = (dataset->n_samples % metadata->batch_size)*metadata->output_dim;
    float batch_loss = INFINITY; 
    float *build_grads, *preds, *grads;
    float *batch_preds = init_zero_mat(batch_preds_size); // Assuming batch_size is the max batch size
    float *batch_build_grads = init_zero_mat(batch_preds_size); // Assuming batch_size is the max batch size
    float *last_batch_preds = init_zero_mat(last_batch_preds_size); // Assuming batch_size is the max batch size
    float *batch_grads = init_zero_mat(batch_preds_size); // Assuming batch_size is the max batch size
    float *last_batch_grads = init_zero_mat(last_batch_preds_size); // Assuming batch_size is the max batch size
    float *last_batch_build_grads = init_zero_mat(last_batch_preds_size); // Assuming batch_size is the max batch size

    SplitCandidateGenerator generator = SplitCandidateGenerator(dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->n_bins, metadata->par_th, metadata->generator_type);
    int **indices = nullptr;
    const float *obs = dataset->obs->data;
    const int n_num_features = metadata->n_num_features;
    if (metadata->n_num_features > 0){
        indices = new int*[metadata->n_num_features];
        for (int i = 0; i < metadata->n_num_features; ++i){
            int *col_inds = new int[dataset->n_samples];
            std::iota(col_inds, col_inds + dataset->n_samples, 0);
            // sort indices per column according to the obs column values
            std::sort(col_inds, col_inds + dataset->n_samples,
            [&obs, n_num_features, i](int a, int b) {
                return obs[a*n_num_features + i] < obs[b*n_num_features + i];
            });
            indices[i] = col_inds;
        }
        generator.generateNumericalSplitCandidates(dataset->obs->data, indices);
    }
    float *full_preds = nullptr;
    if (metadata->n_cat_features > 0)
        generator.processCategoricalCandidates(dataset->categorical_obs->data);
    dataSet batch_dataset;

    dataHolder<const float> obs_holder{nullptr, cpu};
    batch_dataset.obs = &obs_holder;
    dataHolder<float> grads_holder{nullptr, cpu};
    batch_dataset.grads = &grads_holder;
    dataHolder<float> build_grads_holder{nullptr, cpu};
    batch_dataset.build_grads = &build_grads_holder;
    dataHolder<const char> categorical_obs_holder{nullptr, cpu};
    batch_dataset.categorical_obs = &categorical_obs_holder;

    dataHolder<const float> obj_label_holder{nullptr, cpu};
    batch_dataset.obj_labels = &obj_label_holder;

    for (int i = 0; i < iterations; ++i){
        batch_dataset.obs->data = dataset->obs->data + batch_start_idx*metadata->n_num_features; 
        batch_dataset.categorical_obs->data = dataset->categorical_obs->data + batch_start_idx*metadata->n_cat_features*MAX_CHAR_SIZE;
        batch_dataset.n_samples = batch_n_samples; 

        const float *shifted_targets = targets + batch_start_idx*metadata->output_dim; 
        is_last_batch = batch_start_idx + metadata->batch_size > dataset->n_samples;
        if (is_last_batch){
            preds = last_batch_preds;
            memset(preds, 0, last_batch_preds_size * sizeof(float));
        } else {
            preds = batch_preds;
            memset(preds, 0, batch_preds_size * sizeof(float));
        }

        // Boost against every tree in the model; `i` only counts the ones this call added.
        Predictor::predict_cpu(&batch_dataset, preds, edata, metadata, 0, 0, false, opts);
        grads = is_last_batch ? last_batch_grads : batch_grads;
        if (loss_type == MultiRMSE){
            batch_loss = MultiRMSE::get_loss_and_gradients(preds, shifted_targets, grads, batch_dataset.n_samples, metadata->output_dim, par_th);
        }
        batch_dataset.grads->data = grads;
        // Keyed on the model, not the loop counter, like step_cpu: `i` counts
        // only the trees added by this call.
        if (metadata->use_cv && metadata->n_trees > 0){
            Fitter::control_variates(&batch_dataset, edata, metadata);
        }

        int size_preds = is_last_batch ? last_batch_preds_size : batch_preds_size;
        build_grads =  is_last_batch ? last_batch_build_grads : batch_build_grads;
        memcpy(build_grads, grads, sizeof(float) * size_preds);
        if (metadata->split_score_func == L2){
            float *mean_grads = calculate_mean(build_grads, batch_n_samples, output_dim, par_th);
            float *std = calculate_var_and_center(build_grads, mean_grads, batch_n_samples, output_dim, par_th);

            #pragma omp simd
            for (int d = 0; d < output_dim; ++d)
                std[d] = sqrtf(std[d]);
            divide_mat_by_vec_inplace(build_grads, std, batch_dataset.n_samples, metadata->output_dim, metadata->par_th);
            delete[] mean_grads;
            delete[] std;
        } 
        batch_dataset.build_grads->data = build_grads;

        int added_leaves = 0;
        if (metadata->grow_policy == GREEDY)
            added_leaves = Fitter::fit_greedy_tree(&batch_dataset, edata, metadata, generator);
        else 
            added_leaves = Fitter::fit_oblivious_tree(&batch_dataset, edata, metadata, generator);
        Fitter::fit_leaves(&batch_dataset, edata, metadata, added_leaves);
        
        // Apply monotonic constraints AFTER leaf values are computed
        if (metadata->n_mono_constraints > 0 && metadata->grow_policy == OBLIVIOUS) {
            int tree_idx = metadata->n_trees - 1;
            int tree_depth = edata->ensemble_info->depths[tree_idx];
            int start_leaf_idx = edata->ensemble_info->tree_indices[tree_idx];
            if (tree_depth > 0) {
                // Roll the tree back if the projection rejects it (see step_cpu).
                const int trees_before = metadata->n_trees - 1;
                const int leaves_before = metadata->n_leaves - added_leaves;
                try {
                    Fitter::apply_monotonic_constraints_cpu(edata, metadata, tree_idx, tree_depth, start_leaf_idx);
                } catch (...) {
                    metadata->n_trees = trees_before;
                    metadata->n_leaves = leaves_before;
                    // Release everything this call allocated: the cleanup after
                    // the loop is skipped by the unwind.
                    if (indices != nullptr){
                        for (int f = 0; f < metadata->n_num_features; ++f)
                            delete[] indices[f];
                        delete[] indices;
                    }
                    delete[] batch_preds;
                    delete[] last_batch_preds;
                    delete[] batch_grads;
                    delete[] batch_build_grads;
                    delete[] last_batch_grads;
                    delete[] last_batch_build_grads;
                    throw;
                }
            }
        }
        
        // beginning index of new tree
        batch_start_idx += batch_n_samples;
        if (batch_start_idx >= dataset->n_samples)
            batch_start_idx = 0;
        batch_n_samples = batch_start_idx + metadata->batch_size < dataset->n_samples ? metadata->batch_size : dataset->n_samples - batch_start_idx;
        metadata->iteration++;
        if (metadata->verbose > 0){
            std::cout << "Boosting iteration: " << metadata->iteration << " - MultiRMSE Loss: " << batch_loss << std::endl;
        }
        
    }
    if (indices != nullptr){
        for (int i = 0; i < metadata->n_num_features; ++i){
            delete[] indices[i];
        }
        delete[] indices;
    }
    
    full_preds = init_zero_mat(dataset->n_samples*metadata->output_dim);

    // Loss of the model as it now stands, not of its first `iterations` trees.
    Predictor::predict_cpu(dataset, full_preds, edata, metadata, 0, 0, false, opts);
    float full_loss = INFINITY;
    if (loss_type == MultiRMSE){
        full_loss = MultiRMSE::get_loss(full_preds, targets, dataset->n_samples, output_dim, par_th); 
    }
    delete[] batch_preds;
    delete[] full_preds;
    delete[] last_batch_preds;
    delete[] batch_grads;
    delete[] batch_build_grads;
    delete[] last_batch_grads;
    delete[] last_batch_build_grads;
    return full_loss;
}

int Fitter::fit_greedy_tree(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const SplitCandidateGenerator &generator){
    allocate_ensemble_memory(metadata, edata);
    edata->ensemble_info->tree_indices[metadata->n_trees] = metadata->n_leaves;
    int depth = 0, node_idx_cntr = 0, chosen_idx = 0;
    float best_score, parent_score = -INFINITY;
    int n_samples = dataset->n_samples;
    const int n_objs = metadata->n_objs;
    const bool multi_obj = (n_objs > 1);

    int added_leaves = 0;

    int *root_sample_indices = new int[n_samples];
    std::iota(root_sample_indices, root_sample_indices + n_samples, 0);

    std::vector<TreeNode*> tree_nodes; 
    TreeNode *rootNode = new TreeNode(root_sample_indices, n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->output_dim, depth, 0);
    
    // Initialize multi-objective fields on root
    if (multi_obj) {
        Fitter::init_node_multi_obj(rootNode,
            (dataset->obj_labels != nullptr) ? dataset->obj_labels->data : nullptr,
            dataset->build_grads->data, n_objs, n_samples, metadata->output_dim,
            dataset->default_obj_idx, metadata->policy_dim);
    }
    
    tree_nodes.push_back(rootNode);

    int n_candidates = generator.n_candidates;
    splitCandidate *split_candidates = generator.split_candidates;

#ifdef DEBUG
    int n_threads = 1;
#else
    int n_threads = omp_get_max_threads();
#endif
    std::vector<float> best_scores(n_threads, -INFINITY);
    std::vector<int> best_indices(n_threads, -1);
    FloatVector scores(n_candidates);

    while (!tree_nodes.empty())  {
        TreeNode *crnt_node = tree_nodes.back(); 
        tree_nodes.pop_back(); 
        if (crnt_node == nullptr){
            std::cerr << "Error crnt_node is nullptr" << std::endl;
            break;
        }
        bool to_split = true;
        if (crnt_node->depth == metadata->max_depth || crnt_node->n_samples == 0 || n_candidates == 0){
            to_split = false;
        }

        best_score = -INFINITY;
        if (to_split){
            if (!multi_obj) {
                if (metadata->split_score_func == Cosine){
                    parent_score = scoreCosine(crnt_node->sample_indices, crnt_node->n_samples, dataset->build_grads->data, metadata->output_dim);
                } else if (metadata->split_score_func == L2){
                    parent_score = scoreL2(crnt_node->sample_indices, crnt_node->n_samples, dataset->build_grads->data, metadata->output_dim);
                } else{
                    std::cerr << "error invalid split score func!" << std::endl;
                    continue;
                }
            } else {
                // Multi-obj: parent score is sum_k density_k * lambda_k * score_k, each
                // objective block scored like the single-objective parent. This
                // matches the CUDA greedy path (node->scores[k] reduced with
                // density * lambda) and reduces to the single-objective case at n_objs=1.
                parent_score = 0.0f;
                const float *lambda_objs = edata->multi_objective_data->lambda_objs;
                const size_t block = (size_t)dataset->n_samples * metadata->output_dim;
                for (int k = 0; k < n_objs; ++k) {
                    const float *obj_grads = dataset->build_grads->data + (size_t)k * block;
                    float score_k = (metadata->split_score_func == Cosine)
                        ? scoreCosine(crnt_node->sample_indices, crnt_node->n_samples, obj_grads, metadata->output_dim)
                        : scoreL2(crnt_node->sample_indices, crnt_node->n_samples, obj_grads, metadata->output_dim);
                    parent_score += score_k * crnt_node->densities[k] * lambda_objs[k];
                }
            }

            if (crnt_node->depth == 0)
                parent_score = 0.0f;
            // Threads outside the granted team never write their slot, so clear
            // stale results from the previous node.
            std::fill(best_scores.begin(), best_scores.end(), -INFINITY);
            std::fill(best_indices.begin(), best_indices.end(), -1);
#ifndef DEBUG
            #pragma omp parallel num_threads(n_threads)
            {
                int thread_num = omp_get_thread_num();
#else
                int thread_num = 0;
#endif
                float local_best_score = -INFINITY;
                int local_chosen_idx = -1;
                // Partition by the team actually granted, not omp_get_max_threads():
                // OpenMP may run the region with fewer threads (e.g. thread limits on
                // macOS/libomp), and a max-threads split would leave candidates unscored.
#ifndef DEBUG
                const int team_size = omp_get_num_threads();
#else
                const int team_size = 1;
#endif
                const int team_batch = n_candidates / team_size;
                int start_idx = thread_num * team_batch;
                int end_idx = (thread_num == team_size - 1) ? n_candidates : start_idx + team_batch;
                // Process the batch of candidates
                for (int j = start_idx; j < end_idx; ++j) {
                    float score;
                    float penalty_factor = 1.0f;
                    if (multi_obj) {
                        score = crnt_node->getSplitScoreMultiObj(
                            dataset, metadata->split_score_func, split_candidates[j], metadata->min_data_in_leaf,
                            metadata, edata, -1, nullptr, 0, &penalty_factor);
                    } else {
                        score = crnt_node->getSplitScore(dataset, metadata->split_score_func, split_candidates[j], metadata->min_data_in_leaf);
                    }
                    // feature_weights is indexed by ORIGINAL input column, so the
                    // split's type-local index has to go through the reverse
                    // mapping, as the oblivious path does.
                    int global_feature_idx = (split_candidates[j].categorical_value == nullptr)
                        ? edata->feature_mappings->reverse_num_feature_mapping[split_candidates[j].feature_idx]
                        : edata->feature_mappings->reverse_cat_feature_mapping[split_candidates[j].feature_idx];
                    // Unused reverse slots hold -1; reject rather than read out of bounds.
                    if (global_feature_idx < 0 || global_feature_idx >= metadata->input_dim)
                        continue;
                    // As on CUDA: net = w * gain - parent, then the conflict penalty
                    // scales a positive net only (a negative net times 0 would be
                    // -0.0 and pass the >= 0 test). factor is 1 for single-objective.
                    score = score * edata->feature_data->feature_weights[global_feature_idx]
                            - parent_score;
                    if (score > 0.0f)
                        score *= penalty_factor;
                    
#ifdef DEBUG
                    if (metadata->verbose > 1)
                        std::cout << " cand: " <<  j << " score: " <<  score << " parent score: " <<  parent_score << " info: " << split_candidates[j] << std::endl;
#endif

                    if (score > local_best_score) {
                        local_best_score = score;
                        local_chosen_idx = j;
                    }
                }
                best_scores[thread_num] = local_best_score;
                best_indices[thread_num] = local_chosen_idx;
#ifndef DEBUG
            }
#endif        
            
            for (int i = 0; i < n_threads; ++i){
                if (best_scores[i] > best_score){
                    best_score = best_scores[i];
                    chosen_idx = best_indices[i];
                }
            }  
        }

        if (best_score >= 0 && to_split){           
            int status = crnt_node->splitNode(dataset->obs->data, dataset->categorical_obs->data, node_idx_cntr, split_candidates[chosen_idx]);
            if (status == -1){
                std::cerr << "ERROR couldn't split best score" << std::endl;
                break;
            }
            // Initialize multi-obj fields on children
            if (multi_obj) {
                Fitter::init_node_multi_obj(crnt_node->left_child,
                    (dataset->obj_labels != nullptr) ? dataset->obj_labels->data : nullptr,
                    dataset->build_grads->data, n_objs, n_samples, metadata->output_dim,
            dataset->default_obj_idx, metadata->policy_dim);
                Fitter::init_node_multi_obj(crnt_node->right_child,
                    (dataset->obj_labels != nullptr) ? dataset->obj_labels->data : nullptr,
                    dataset->build_grads->data, n_objs, n_samples, metadata->output_dim,
            dataset->default_obj_idx, metadata->policy_dim);
            }
            // assign node values
            tree_nodes.push_back(crnt_node->right_child);
            tree_nodes.push_back(crnt_node->left_child);
            node_idx_cntr += 2;
        } else {
            Fitter::update_ensemble_per_leaf(edata, metadata, crnt_node);
            added_leaves += 1;
        }
    }
    delete rootNode;
    metadata->n_trees += 1;
    return added_leaves;
}

int Fitter::fit_oblivious_tree(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const SplitCandidateGenerator &generator){
    allocate_ensemble_memory(metadata, edata);
    edata->ensemble_info->tree_indices[metadata->n_trees] = metadata->n_leaves;
    
    int depth = 0, node_idx_cntr = 0, chosen_idx = 0;
    float best_score;
    int n_samples = dataset->n_samples;
    const int n_objs = metadata->n_objs;
    const bool multi_obj = (n_objs > 1);

    int added_leaves = 0;

    int *root_sample_indices = new int[n_samples];
    std::iota(root_sample_indices, root_sample_indices + n_samples, 0);

    int max_n_leaves = 1 << metadata->max_depth;

    std::vector<TreeNode*> tree_nodes(max_n_leaves); 
    std::vector<TreeNode*> child_tree_nodes(max_n_leaves); 
    TreeNode *rootNode = new TreeNode(root_sample_indices, n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->output_dim, depth, 0);
    
    // Initialize multi-objective fields on root
    if (multi_obj) {
        Fitter::init_node_multi_obj(rootNode,
            (dataset->obj_labels != nullptr) ? dataset->obj_labels->data : nullptr,
            dataset->build_grads->data, n_objs, n_samples, metadata->output_dim,
            dataset->default_obj_idx, metadata->policy_dim);
    }
    
    tree_nodes[0] = rootNode;

    int n_candidates = generator.n_candidates;
    splitCandidate *split_candidates = generator.split_candidates;
#ifndef DEBUG
    int n_threads = omp_get_max_threads();
#else 
    int n_threads = 1;
#endif 
    std::vector<float> best_scores(n_threads, -INFINITY);
    std::vector<int> best_indices(n_threads, -1);
    FloatVector scores(n_candidates);

    while (depth < metadata->max_depth)  {
        best_score = -INFINITY;
        // Threads outside the granted team never write their slot.
        std::fill(best_scores.begin(), best_scores.end(), -INFINITY);
        std::fill(best_indices.begin(), best_indices.end(), -1);
#ifndef DEBUG
        #pragma omp parallel num_threads(n_threads)
        {
            int thread_num = omp_get_thread_num();
#else 
            int thread_num = 0;
#endif
            float local_best_score = -INFINITY;
            int local_chosen_idx = -1;
            // Partition by the team actually granted, not omp_get_max_threads():
            // OpenMP may run the region with fewer threads (e.g. thread limits on
            // macOS/libomp), and a max-threads split would leave candidates unscored.
#ifndef DEBUG
            const int team_size = omp_get_num_threads();
#else
            const int team_size = 1;
#endif
            const int team_batch = n_candidates / team_size;
            int start_idx = thread_num * team_batch;
            int end_idx = (thread_num == team_size - 1) ? n_candidates : start_idx + team_batch;
            // Process the batch of candidates
            for (int j = start_idx; j < end_idx; ++j) {
                float score = 0.0f;
                // Get the global feature index for constraint lookup
                int global_feature_idx = (split_candidates[j].categorical_value == nullptr)
                    ? edata->feature_mappings->reverse_num_feature_mapping[split_candidates[j].feature_idx]
                    : edata->feature_mappings->reverse_cat_feature_mapping[split_candidates[j].feature_idx];
                // set_feature_mapping fills unused reverse slots with -1, so a
                // mapping that does not cover this split would index
                // feature_weights out of bounds.  Reject the candidate instead.
                if (global_feature_idx < 0 || global_feature_idx >= metadata->input_dim)
                    continue;

                for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
                    TreeNode *crnt_node = tree_nodes[node_idx];
                    if (multi_obj) {
                        score += crnt_node->getSplitScoreMultiObj(
                            dataset, metadata->split_score_func, split_candidates[j], metadata->min_data_in_leaf,
                            metadata, edata, global_feature_idx, edata->mono_constraints, metadata->n_mono_constraints);
                    } else if (metadata->n_mono_constraints > 0) {
                        score += crnt_node->getSplitScoreWithConstraints(
                            dataset, metadata->split_score_func, split_candidates[j], metadata->min_data_in_leaf,
                            global_feature_idx, edata->mono_constraints, metadata->n_mono_constraints
                        );
                    } else {
                        score += crnt_node->getSplitScore(dataset, metadata->split_score_func, split_candidates[j], metadata->min_data_in_leaf);
                    }
                }
                score = score * edata->feature_data->feature_weights[global_feature_idx];
                
#ifdef DEBUG    
                if (metadata->verbose > 1)
                    std::cout << " cand: " <<  j << " score: " <<  score << " info: " << split_candidates[j] << std::endl;
#endif

                if (score > local_best_score) {
                    local_best_score = score;
                    local_chosen_idx = j;
                }
            }
            best_scores[thread_num] = local_best_score;
            best_indices[thread_num] = local_chosen_idx;
#ifndef DEBUG
        }   
#endif

        for (int i = 0; i < n_threads; ++i){
            if (best_scores[i] > best_score){
                best_score = best_scores[i];
                chosen_idx = best_indices[i];
            }
        }  
        if (best_score == -INFINITY)
            break; 

        for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
            TreeNode *crnt_node = tree_nodes[node_idx];
            int status = crnt_node->splitNode(dataset->obs->data, dataset->categorical_obs->data, node_idx_cntr, split_candidates[chosen_idx]);
            if (status == -1){
                std::cerr << "ERROR couldn't split best score" << std::endl;
                break;
            }

            child_tree_nodes[node_idx*2] = crnt_node->left_child;
            child_tree_nodes[node_idx*2+ 1] = crnt_node->right_child;
        }
        depth += 1;
        for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
            tree_nodes[node_idx] = child_tree_nodes[node_idx];
            child_tree_nodes[node_idx] = nullptr;
            // Initialize multi-obj fields on new level nodes
            if (multi_obj && tree_nodes[node_idx] != nullptr) {
                Fitter::init_node_multi_obj(tree_nodes[node_idx],
                    (dataset->obj_labels != nullptr) ? dataset->obj_labels->data : nullptr,
                    dataset->build_grads->data, n_objs, n_samples, metadata->output_dim,
            dataset->default_obj_idx, metadata->policy_dim);
            }
        }
    }

    Fitter::update_ensemble_per_tree(edata, metadata, tree_nodes, 1 << depth);
    
    added_leaves += 1 << depth;
    metadata->n_trees += 1;
    delete rootNode;
    return added_leaves;
}


void Fitter::fit_leaves(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const int added_leaves){
    for (int leaf_idx = 0; leaf_idx < added_leaves; ++leaf_idx){
        Fitter::calc_leaf_value(dataset, edata, metadata, edata->ensemble_info->tree_indices[metadata->n_trees - 1] + leaf_idx, metadata->n_trees - 1);
    }
}

void Fitter::update_ensemble_per_leaf(ensembleData *edata, ensembleMetaData *metadata, const TreeNode* node){
    int idx = metadata->n_leaves; 
#ifdef DEBUG
    edata->n_samples[idx] = node->n_samples;
#endif
    edata->ensemble_info->depths[idx] = node->depth;
    int row_idx = idx*metadata->max_depth;
    if (node->depth > 0){
        for (int i = 0; i < node->depth; ++i){
            if (node->split_conditions[i].categorical_value != nullptr){
                memcpy(edata->feature_data->categorical_values + (row_idx + i)*MAX_CHAR_SIZE, node->split_conditions[i].categorical_value, sizeof(char)*MAX_CHAR_SIZE);
                edata->feature_data->is_numerics[row_idx+ i] = false;
            } else {
                edata->feature_data->is_numerics[row_idx+ i] = true;
            }
            edata->feature_data->feature_indices[row_idx+ i] = node->split_conditions[i].feature_idx;
            edata->feature_data->feature_values[row_idx+ i] = node->split_conditions[i].feature_value;
            edata->feature_data->inequality_directions[row_idx + i] = node->split_conditions[i].inequality_direction;
            edata->leaf_data->edge_weights[row_idx + i] = node->split_conditions[i].edge_weight;
        }
    }
    // Store per-leaf densities (computed during calc_leaf_value for multi-obj)
    // Initialize to uniform for now; actual densities set in calc_leaf_value
    for (int k = 0; k < metadata->n_objs; ++k)
        edata->multi_objective_data->densities[idx * metadata->n_objs + k] = (k == 0) ? 1.0f : 0.0f;
    metadata->n_leaves += 1;
}

void Fitter::update_ensemble_per_tree(ensembleData *edata, ensembleMetaData *metadata, std::vector<TreeNode*> nodes, const int n_nodes){
    int idx = metadata->n_trees; 
    for (int node_idx = 0; node_idx <  n_nodes; node_idx++){
        TreeNode *node = nodes[node_idx];
#ifdef DEBUG
        edata->n_samples[metadata->n_leaves] = node->n_samples;
#endif
        edata->ensemble_info->depths[idx] = node->depth;
        int row_idx = idx*metadata->max_depth;
        if (node->depth > 0){
            for (int i = 0; i < node->depth; ++i){
                if (node->split_conditions[i].categorical_value != nullptr){
                    memcpy(edata->feature_data->categorical_values + (row_idx + i)*MAX_CHAR_SIZE, node->split_conditions[i].categorical_value, sizeof(char)*MAX_CHAR_SIZE);
                    edata->feature_data->is_numerics[row_idx+ i] = false;
                } else {
                    edata->feature_data->is_numerics[row_idx+ i] = true;
                }
                edata->feature_data->feature_indices[row_idx+ i] = node->split_conditions[i].feature_idx;
                edata->feature_data->feature_values[row_idx+ i] = node->split_conditions[i].feature_value;
                edata->feature_data->inequality_directions[metadata->n_leaves*metadata->max_depth + i] = node->split_conditions[i].inequality_direction;
                edata->leaf_data->edge_weights[metadata->n_leaves*metadata->max_depth + i] = node->split_conditions[i].edge_weight;
            }
        }
        // Initialize densities (actual values set in calc_leaf_value for multi-obj)
        for (int k = 0; k < metadata->n_objs; ++k)
            edata->multi_objective_data->densities[metadata->n_leaves * metadata->n_objs + k] = (k == 0) ? 1.0f : 0.0f;
        metadata->n_leaves += 1;
    }
}


void Fitter::calc_leaf_value(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const int leaf_idx, const int tree_idx){
    int output_dim = metadata->output_dim;
    int policy_dim = metadata->policy_dim;
    int n_objs = metadata->n_objs;
    int n_samples = dataset->n_samples;
    const float *obs = dataset->obs->data, *grads = dataset->grads->data;
    const char *categorical_obs = dataset->categorical_obs->data;
    const float *obj_labels = (dataset->obj_labels != nullptr) ? dataset->obj_labels->data : nullptr;
    const float *lambda_objs = edata->multi_objective_data->lambda_objs;
    int depth = (metadata->grow_policy == OBLIVIOUS) ? edata->ensemble_info->depths[tree_idx] : edata->ensemble_info->depths[leaf_idx];
    int cond_idx = (metadata->grow_policy == OBLIVIOUS) ? tree_idx*metadata->max_depth : leaf_idx*metadata->max_depth;
    int ineq_cond = leaf_idx*metadata->max_depth;
    float count = 0;
    bool passed;
    int row_idx, cat_row_idx;

    bool multi_obj = (n_objs > 1);

    // Per-objective gradient sums for density-weighted leaf values
    float *obj_sums = nullptr;
    float *leaf_densities = nullptr;
    float *label_counts = nullptr;
    if (multi_obj) {
        obj_sums = new float[n_objs * output_dim];
        leaf_densities = new float[n_objs];
        label_counts = new float[n_objs];
        for (int k = 0; k < n_objs; ++k) {
            label_counts[k] = 0.0f;
            leaf_densities[k] = 0.0f;
            for (int d = 0; d < output_dim; ++d)
                obj_sums[k * output_dim + d] = 0.0f;
        }
    }

    // Start from zero: this slot may still hold values from a tree that was
    // rolled back after a failed monotonic projection, and the loop below
    // accumulates rather than assigns.
    float *leaf_values = edata->leaf_data->values + leaf_idx * output_dim;
    std::fill(leaf_values, leaf_values + output_dim, 0.0f);

    for (int i = 0; i < n_samples; ++i){
        row_idx = i*metadata->n_num_features;
        cat_row_idx = i*metadata->n_cat_features;
        passed = (depth == 0);
        for (int depth_idx = depth - 1; depth_idx >= 0; --depth_idx){
            passed = (edata->feature_data->is_numerics[cond_idx + depth_idx]) ? (obs[row_idx + edata->feature_data->feature_indices[cond_idx + depth_idx]] > edata->feature_data->feature_values[cond_idx + depth_idx]) == edata->feature_data->inequality_directions[ineq_cond + depth_idx] : (strcmp(&categorical_obs[(cat_row_idx + edata->feature_data->feature_indices[cond_idx + depth_idx]) * MAX_CHAR_SIZE], edata->feature_data->categorical_values + (cond_idx + depth_idx)*MAX_CHAR_SIZE) == 0) == edata->feature_data->inequality_directions[ineq_cond + depth_idx];
            if (!passed)
                break;
        }
        if (passed){
            count += 1;
            if (multi_obj) {
                // Count labels for density computation
                int lbl = (obj_labels != nullptr) ? static_cast<int>(obj_labels[i]) : dataset->default_obj_idx;
                if (lbl >= 0 && lbl < n_objs) label_counts[lbl] += 1.0f;
                // Accumulate per-objective gradient sums
                for (int k = 0; k < n_objs; ++k) {
                    size_t g_base = (size_t)k * n_samples * output_dim + (size_t)i * output_dim;
                    for (int d = 0; d < output_dim; ++d)
                        obj_sums[k * output_dim + d] += grads[g_base + d];
                }
            } else {
                int idx = i*output_dim;
                #pragma omp simd
                for (int d = 0; d < output_dim; ++d)
                    edata->leaf_data->values[leaf_idx*output_dim + d] += grads[idx + d];
            }
        }
    }

    if (count > 0){
        if (multi_obj) {
            // Compute leaf densities from label counts
            for (int k = 0; k < n_objs; ++k)
                leaf_densities[k] = label_counts[k] / count;

            // Store densities in ensemble data
            for (int k = 0; k < n_objs; ++k)
                edata->multi_objective_data->densities[leaf_idx * n_objs + k] = leaf_densities[k];

            // Compute density-weighted leaf values
            for (int d = 0; d < output_dim; ++d) {
                if (d < policy_dim) {
                    // Policy dimensions: weighted mixture across objectives
                    // V = Σ(density_k × λ_k × mean_grad_k)
                    float leaf_value = 0.0f;
                    for (int k = 0; k < n_objs; ++k) {
                        float mean_k = obj_sums[k * output_dim + d] / count;
                        leaf_value += mean_k * leaf_densities[k] * lambda_objs[k];
                    }
                    edata->leaf_data->values[leaf_idx * output_dim + d] = leaf_value;
                } else {
                    // Critic dimensions (d >= policy_dim): plain mean from objective 0 only
                    edata->leaf_data->values[leaf_idx * output_dim + d] = obj_sums[0 * output_dim + d] / count;
                }
            }
        } else {
            for (int d = 0; d < output_dim; ++d){
                edata->leaf_data->values[leaf_idx*output_dim + d] /= count;
            }
        }
    }

    if (multi_obj) {
        delete[] obj_sums;
        delete[] leaf_densities;
        delete[] label_counts;
    }
#ifdef DEBUG
    edata->n_samples[leaf_idx] = static_cast<int>(count);
#endif
}


void Fitter::init_node_multi_obj(
    TreeNode *node,
    const float *obj_labels,
    const float *grads,
    int n_objs,
    int global_n_samples,
    int output_dim,
    int default_obj_idx,
    int policy_dim
) {
    node->n_objs = n_objs;
    node->global_n_samples = global_n_samples;

    // Allocate and compute densities from labels
    node->densities = new float[n_objs];
    if (obj_labels == nullptr || n_objs <= 1) {
        // No labels: all density on the default objective (caller-supplied label index).
        for (int k = 0; k < n_objs; ++k)
            node->densities[k] = (k == default_obj_idx) ? 1.0f : 0.0f;
    } else {
        for (int k = 0; k < n_objs; ++k)
            node->densities[k] = 0.0f;
        for (int i = 0; i < node->n_samples; ++i) {
            int sample_idx = node->sample_indices[i];
            int lbl = static_cast<int>(obj_labels[sample_idx]);
            if (lbl >= 0 && lbl < n_objs)
                node->densities[lbl] += 1.0f;
        }
        if (node->n_samples > 0) {
            float n_inv = 1.0f / static_cast<float>(node->n_samples);
            for (int k = 0; k < n_objs; ++k)
                node->densities[k] *= n_inv;
        }
    }

    // Compute conflict rho from per-objective mean gradients
    node->conflict_rho = 0.0f;
    if (n_objs > 1 && node->n_samples > 0) {
        // Compute per-obj mean gradients, then ρ = 1 - ||Σ_k μ_k||² / Σ_k ||μ_k||²
        float numerator = 0.0f;   // ||Σ_k μ_k||²
        float denominator = 0.0f; // Σ_k ||μ_k||²
        float n_inv = 1.0f / static_cast<float>(node->n_samples);

        // Only policy columns: critic outputs have a single objective (their
        // other blocks are zero) and must not dilute the conflict measure.
        const int conflict_dims = (policy_dim < 0 || policy_dim > output_dim) ? output_dim : policy_dim;
        for (int d = 0; d < conflict_dims; ++d) {
            float sum_of_means = 0.0f;
            float sum_of_sq = 0.0f;
            for (int k = 0; k < n_objs; ++k) {
                // Compute mean of obj k, dim d over samples in this node
                float mean_kd = 0.0f;
                for (int i = 0; i < node->n_samples; ++i) {
                    int sample_idx = node->sample_indices[i];
                    size_t g_idx = (size_t)k * global_n_samples * output_dim + (size_t)sample_idx * output_dim + d;
                    mean_kd += grads[g_idx];
                }
                mean_kd *= n_inv;
                sum_of_means += mean_kd;
                sum_of_sq += mean_kd * mean_kd;
            }
            numerator += sum_of_means * sum_of_means;
            denominator += sum_of_sq;
        }

        if (denominator > 1e-12f) {
            float ratio = numerator / denominator;
            if (ratio > 1.0f) ratio = 1.0f;
            node->conflict_rho = 1.0f - ratio;
        }
    }
}


void Fitter::control_variates(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata){
    int output_dim = metadata->output_dim, par_th = metadata->par_th, n_samples = dataset->n_samples;
    int n_sample_threads = calculate_num_threads(n_samples, par_th);
    void (*momemntFunc)(const float*, const char*, float*, const ensembleData*, const ensembleMetaData*, const int, const int, const int) = nullptr;
    momemntFunc = (metadata->grow_policy == OBLIVIOUS) ? &Predictor::momentum_over_trees : &Predictor::momentum_over_leaves;
    float *momentum = init_zero_mat(n_samples*output_dim);
    if (n_sample_threads > 1) {
        omp_set_num_threads(n_sample_threads);
        int elements_per_thread = n_samples / n_sample_threads; // Determine the size of each batch
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_sample_threads - 1) ? n_samples : start_idx + elements_per_thread;
            for (int sample_idx = start_idx; sample_idx < end_idx; ++sample_idx) {
                momemntFunc(dataset->obs->data, dataset->categorical_obs->data, momentum, edata, metadata, 0, metadata->n_trees, sample_idx);
            }
        }
    // no parallelization
    } else{ 
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx){
            momemntFunc(dataset->obs->data, dataset->categorical_obs->data, momentum, edata, metadata, 0, metadata->n_trees, sample_idx);
        }
    }

    float error_correction = 1.0f / sqrtf(1.0f - powf(metadata->cv_beta, static_cast<float>(metadata->n_trees)));
    _multiply_mat_by_scalar(momentum, error_correction, n_samples, output_dim, par_th);
    float *grads_copy = copy_mat(dataset->grads->data, n_samples*output_dim, par_th);
    float *grads_mean = calculate_mean(grads_copy, n_samples, output_dim, par_th);
    float *momentum_mean = calculate_mean(momentum, n_samples, output_dim, par_th);
    float *variance = calculate_var_and_center(momentum, momentum_mean, n_samples, output_dim, par_th);
    subtract_vec_from_mat(grads_copy, grads_mean, n_samples, output_dim, par_th);
    float *covariance = calculate_row_covariance(grads_copy, momentum, n_samples, output_dim, par_th);
    float* alpha = element_wise_division(covariance, variance, output_dim, par_th);
    for (int i = 0; i < output_dim; ++i){
        if (alpha[i] > 1 )
            alpha[i] = 1;
        if (alpha[i] < -1)
            alpha[i] = -1;
    }
    multiply_mat_by_vec_subtract_result(dataset->grads->data, momentum, alpha, n_samples, output_dim, par_th);
    delete[] grads_mean;
    delete[] momentum_mean;
    delete[] grads_copy;
    delete[] variance;
    delete[] covariance;
    delete[] alpha;
    delete[] momentum;
}

void Fitter::apply_monotonic_constraints_cpu(
    ensembleData *edata,
    ensembleMetaData *metadata,
    int tree_idx,
    int tree_depth,
    int start_leaf_idx
) {
    if (metadata->n_mono_constraints <= 0 || tree_depth <= 0) return;
    
    const int n_leaves = 1 << tree_depth;
    const int policy_dim = metadata->policy_dim;  // Only apply constraints to policy outputs
    const int output_dim = metadata->output_dim;  // For indexing into leaf values array
    
    // Get feature indices for this tree (feature_indices[0] is root split)
    int* feature_indices = new int[tree_depth];
    bool* split_is_numeric = new bool[tree_depth];
    for (int d = 0; d < tree_depth; ++d) {
        feature_indices[d] = edata->feature_data->feature_indices[tree_idx * metadata->max_depth + d];
        split_is_numeric[d] = edata->feature_data->is_numerics[tree_idx * metadata->max_depth + d];
    }
    
    // Get inequality directions for this tree (from first leaf)
    int* inequality_directions = new int[tree_depth];
    int ineq_base = start_leaf_idx * metadata->max_depth;
    for (int d = 0; d < tree_depth; ++d) {
        inequality_directions[d] = edata->feature_data->inequality_directions[ineq_base + d];
    }
    
    // Build map: depth -> (effective_constraint, output_idx) for this tree
    // Only allocate for policy_dim since monotonic constraints only apply to policy outputs
    int** effective_constraints = new int*[tree_depth];
    for (int d = 0; d < tree_depth; ++d) {
        effective_constraints[d] = new int[policy_dim]();
    }
    
    for (int c = 0; c < metadata->n_mono_constraints; ++c) {
        int global_feature_idx = edata->mono_constraints->feature_idx[c];
        int constraint_dir = edata->mono_constraints->constraint[c];
        int constraint_output = edata->mono_constraints->output_idx[c];
        
        for (int d = 0; d < tree_depth; ++d) {
            // CRITICAL: Convert internal feature index to global using reverse mapping
            // feature_indices[d] is the INTERNAL index used by the tree builder
            // We need to map it back to GLOBAL index to compare with constraints
            // Monotonic constraints apply to numerical splits only.  A categorical
            // split carries a categorical internal index, so looking it up in
            // reverse_num_feature_mapping would either alias onto an unrelated
            // numerical feature or read past the end of that mapping.
            if (!split_is_numeric[d]) continue;
            int internal_idx = feature_indices[d];
            if (internal_idx < 0 || internal_idx >= metadata->n_num_features) continue;
            int global_idx = edata->feature_mappings->reverse_num_feature_mapping[internal_idx];
            
            if (global_idx == global_feature_idx) {
                // inequality_directions[d] is false on the left branch
                // (x <= threshold) and true on the right, see node.cpp. Flip the
                // requested direction on the left branch so both sides are
                // compared in the same orientation.
                effective_constraints[d][constraint_output] = (inequality_directions[d] == 1) ? constraint_dir : -constraint_dir;
            }
        }
    }
    
    // Isotonic projection of the leaf values onto the monotone cone, via Dykstra's
    // algorithm.
    //
    // Within one constrained depth the leaf pairs are disjoint, so averaging a
    // violating pair is the exact L2 projection (2-point isotonic).  Across two
    // or more constrained depths the pair sets overlap.
    //
    // Dykstra adds a per-depth correction term z_d: project (v + z_d) instead of v,
    // then fold the residual back into z_d.  That converges to the true projection
    // onto the intersection (Boyle-Dykstra).  Convergence is asymptotic, hence the
    // tolerance test rather than an exact no-change test.
    // Set when the final feasibility scan still finds a violation.
    bool infeasible = false;
    int infeasible_out = -1;
    float worst_gap = 0.0f;
    float *v      = new float[n_leaves];
    float *v_prev = new float[n_leaves];
    float *y      = new float[n_leaves];
    float *z      = new float[tree_depth * n_leaves];

    for (int out_idx = 0; out_idx < policy_dim; ++out_idx) {
        // Skip outputs with no constraint at any depth.
        bool constrained = false;
        for (int d = 0; d < tree_depth && !constrained; ++d)
            if (effective_constraints[d][out_idx] != 0) constrained = true;
        if (!constrained) continue;

        for (int i = 0; i < n_leaves; ++i)
            v[i] = edata->leaf_data->values[(start_leaf_idx + i) * output_dim + out_idx];
        std::fill(z, z + tree_depth * n_leaves, 0.0f);

        bool converged = false;
        for (int pass = 0; pass < MONOTONIC_MAX_PASSES && !converged; ++pass) {
            std::copy(v, v + n_leaves, v_prev);

            for (int d = 0; d < tree_depth; ++d) {
                int constraint_dir = effective_constraints[d][out_idx];
                if (constraint_dir == 0) continue;
                float *z_d = z + d * n_leaves;

                // y = v + z_d
                for (int i = 0; i < n_leaves; ++i) y[i] = v[i] + z_d[i];

                // v = P_d(y): pool violating pairs.  CRITICAL: depth 0 (root) is
                // the MSB, depth (tree_depth-1) the LSB.
                std::copy(y, y + n_leaves, v);
                int bit_mask = 1 << (tree_depth - 1 - d);
                for (int i = 0; i < n_leaves; ++i) {
                    if ((i & bit_mask) != 0) continue;   // handle each pair once
                    int j = i | bit_mask;
                    // constraint_dir == 1 means v[i] <= v[j]
                    bool violation = (constraint_dir == 1) ? (v[i] > v[j]) : (v[i] < v[j]);
                    if (violation) {
                        float pooled = (v[i] + v[j]) * 0.5f;
                        v[i] = pooled;
                        v[j] = pooled;
                    }
                }

                // z_d = y - P_d(y)
                for (int i = 0; i < n_leaves; ++i) z_d[i] = y[i] - v[i];
            }

            float max_delta = 0.0f;
            for (int i = 0; i < n_leaves; ++i)
                max_delta = std::max(max_delta, std::fabs(v[i] - v_prev[i]));
            if (max_delta < MONOTONIC_TOLERANCE) converged = true;
        }

        // Feasibility cleanup.  Dykstra approaches the optimum from outside the
        // feasible set, so it can stop with violations of order MONOTONIC_TOLERANCE.
        // Monotonicity is a hard contract, so run plain projections (z = 0) until
        // nothing moves.  Measured shift away from the optimum is ~1e-6.
        for (int pass = 0; pass < MONOTONIC_MAX_PASSES; ++pass) {
            std::copy(v, v + n_leaves, v_prev);
            for (int d = 0; d < tree_depth; ++d) {
                int constraint_dir = effective_constraints[d][out_idx];
                if (constraint_dir == 0) continue;
                int bit_mask = 1 << (tree_depth - 1 - d);
                for (int i = 0; i < n_leaves; ++i) {
                    if ((i & bit_mask) != 0) continue;
                    int j = i | bit_mask;
                    bool violation = (constraint_dir == 1) ? (v[i] > v[j]) : (v[i] < v[j]);
                    if (violation) {
                        float pooled = (v[i] + v[j]) * 0.5f;
                        v[i] = pooled;
                        v[j] = pooled;
                    }
                }
            }
            if (std::equal(v, v + n_leaves, v_prev)) break;
        }

        // Final feasibility scan.  `converged` only records whether the Dykstra
        // phase met its movement tolerance, not whether the cleanup phase
        // removed every violation.
        for (int d = 0; d < tree_depth; ++d) {
            int constraint_dir = effective_constraints[d][out_idx];
            if (constraint_dir == 0) continue;
            int bit_mask = 1 << (tree_depth - 1 - d);
            for (int i = 0; i < n_leaves; ++i) {
                if ((i & bit_mask) != 0) continue;
                int j = i | bit_mask;
                float gap = (constraint_dir == 1) ? (v[i] - v[j]) : (v[j] - v[i]);
                if (gap > MONOTONIC_TOLERANCE) {
                    infeasible = true;
                    // Keep the reported index paired with the reported gap.
                    if (gap > worst_gap) {
                        worst_gap = gap;
                        infeasible_out = out_idx;
                    }
                }
            }
        }

        for (int i = 0; i < n_leaves; ++i)
            edata->leaf_data->values[(start_leaf_idx + i) * output_dim + out_idx] = v[i];

        // Dykstra converges to the NEAREST monotone point only if it converged.
        // If it hit the pass limit the cleanup phase still guarantees a feasible
        // result, but it is no longer provably the closest one.  Counted so the
        // Python layer can raise a real RuntimeWarning after fit()/step().
        if (!converged) {
            note_monotonic_nonconverged();
            std::cerr << "WARNING: monotonic projection for output " << out_idx
                      << " reached the " << MONOTONIC_MAX_PASSES
                      << "-pass limit; leaf values are monotone but may not be the"
                         " closest monotone values" << std::endl;
        }
    }

    delete[] v;
    delete[] v_prev;
    delete[] y;
    delete[] z;
    
    delete[] feature_indices;
    delete[] split_is_numeric;
    delete[] inequality_directions;
    for (int d = 0; d < tree_depth; ++d) {
        delete[] effective_constraints[d];
    }
    delete[] effective_constraints;

    // Monotonicity is a hard contract, so surface a failure to Python rather
    // than leaving a silently non-monotone model behind.
    if (infeasible) {
        throw std::runtime_error(
            "Monotonic constraints could not be satisfied for output " +
            std::to_string(infeasible_out) + " (worst violation " +
            std::to_string(worst_gap) + ")");
    }
}
