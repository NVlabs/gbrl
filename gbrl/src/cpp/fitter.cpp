//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
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
    if (metadata->use_cv && n_trees > 0){
        Fitter::control_variates(dataset, edata, metadata);
    }

    float *build_grads = copy_mat(dataset->grads, dataset->n_samples*output_dim, par_th);
    if (metadata->split_score_func == L2){
        float *mean_grads = calculate_mean(build_grads, dataset->n_samples, output_dim, par_th);
        float *std = calculate_std_and_center(build_grads, mean_grads, dataset->n_samples, output_dim, par_th);
        divide_mat_by_vec_inplace(build_grads, std, dataset->n_samples, output_dim, par_th);
        delete[] mean_grads;
        delete[] std;
    } 

    float *norm_grads = nullptr;
    if (metadata->split_score_func == Cosine || metadata->n_cat_features > 0){
        norm_grads = init_zero_mat(dataset->n_samples*metadata->output_dim);
        calculate_squared_norm(norm_grads, dataset->grads, dataset->n_samples, metadata->output_dim, par_th);
    }

    SplitCandidateGenerator generator = SplitCandidateGenerator(dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->n_bins, metadata->par_th, metadata->generator_type);
    int **indices = nullptr;
    const float *obs = dataset->obs;
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
        generator.generateNumericalSplitCandidates(dataset->obs, indices);
    }
    if (metadata->n_cat_features > 0)
        generator.processCategoricalCandidates(dataset->categorical_obs, norm_grads);

    dataset->build_grads = build_grads; 
    dataset->norm_grads = norm_grads; 
    int added_leaves = 0;
    if (metadata->grow_policy == GREEDY)
        added_leaves = Fitter::fit_greedy_tree(dataset, edata, metadata, generator);
    else 
        added_leaves = Fitter::fit_oblivious_tree(dataset, edata, metadata, generator);
    Fitter::fit_leaves(dataset, edata, metadata, added_leaves);

    if (indices != nullptr){
        for (int i = 0; i < metadata->n_num_features; ++i)
            delete[] indices[i];
        delete[] indices;
    }
    
    delete[] build_grads;
    if (norm_grads != nullptr)
        delete[] norm_grads;
    metadata->iteration++;
}

float Fitter::fit_cpu(dataSet *dataset, const float* targets, ensembleData *edata, ensembleMetaData *metadata, const int iterations, lossType loss_type, std::vector<Optimizer*> opts){
    int batch_start_idx = 0, output_dim = metadata->output_dim;
    int batch_size = metadata->batch_size, par_th = metadata->par_th;
    int batch_n_samples = batch_start_idx + batch_size < dataset->n_samples ? batch_size : dataset->n_samples - batch_start_idx;
    bool is_last_batch;
    int batch_preds_size = metadata->batch_size*metadata->output_dim, last_batch_preds_size = (dataset->n_samples % metadata->batch_size)*metadata->output_dim;
    float batch_loss = INFINITY; 
    float *build_grads, *preds, *grads, *norm_grads;
    float *batch_preds = init_zero_mat(batch_preds_size); // Assuming batch_size is the max batch size
    float *batch_build_grads = init_zero_mat(batch_preds_size); // Assuming batch_size is the max batch size
    float *last_batch_preds = init_zero_mat(last_batch_preds_size); // Assuming batch_size is the max batch size
    float *batch_grads = init_zero_mat(batch_preds_size); // Assuming batch_size is the max batch size
    float *last_batch_grads = init_zero_mat(last_batch_preds_size); // Assuming batch_size is the max batch size
    float *last_batch_build_grads = init_zero_mat(last_batch_preds_size); // Assuming batch_size is the max batch size
    float *batch_grad_norms = init_zero_mat(metadata->batch_size); // Assuming batch_size is the max batch size
    float *last_batch_grad_norms = init_zero_mat(dataset->n_samples % metadata->batch_size); // Assuming batch_size is the max batch size

    SplitCandidateGenerator generator = SplitCandidateGenerator(dataset->n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->n_bins, metadata->par_th, metadata->generator_type);
    int **indices = nullptr;
    const float *obs = dataset->obs;
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
        generator.generateNumericalSplitCandidates(dataset->obs, indices);
    }
    float *full_preds = nullptr;
    if (metadata->n_cat_features > 0){
        full_preds = init_zero_mat(dataset->n_samples*metadata->output_dim); 
        Predictor::predict_cpu(dataset, full_preds, edata, metadata, 0, iterations, false, opts);
        float *full_grads = init_zero_mat(dataset->n_samples*metadata->output_dim); 
        float *full_grad_norms = init_zero_mat(dataset->n_samples); 
        batch_loss = MultiRMSE::get_loss_and_gradients(full_preds, targets, full_grads, dataset->n_samples, metadata->output_dim);
        calculate_squared_norm(full_grad_norms, full_grads, dataset->n_samples, metadata->output_dim, metadata->par_th);
        generator.processCategoricalCandidates(dataset->categorical_obs, full_grad_norms);
        delete[] full_preds;
        delete[] full_grads;
        delete[] full_grad_norms;
    }
    dataSet batch_dataset;
    batch_dataset.feature_weights = dataset->feature_weights;

    for (int i = 0; i < iterations; ++i){
        batch_dataset.obs = dataset->obs + batch_start_idx*metadata->n_num_features; 
        batch_dataset.categorical_obs = dataset->categorical_obs + batch_start_idx*metadata->n_cat_features*MAX_CHAR_SIZE; ; 
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

        Predictor::predict_cpu(&batch_dataset, preds, edata, metadata, 0, i, false, opts);
        grads = is_last_batch ? last_batch_grads : batch_grads;
        if (loss_type == MultiRMSE){
            batch_loss = MultiRMSE::get_loss_and_gradients(preds, shifted_targets, grads, batch_dataset.n_samples, metadata->output_dim);
        }
        batch_dataset.grads = grads;
        if (metadata->use_cv && i > 0){
            Fitter::control_variates(&batch_dataset, edata, metadata);
        }

        norm_grads = is_last_batch ? last_batch_grad_norms : batch_grad_norms;
        int size_preds = is_last_batch ? last_batch_preds_size: batch_preds_size;
        build_grads =  is_last_batch ? last_batch_build_grads : batch_build_grads;
        memcpy(build_grads, grads, sizeof(float) * size_preds);
        if (metadata->split_score_func == L2){
            float *mean_grads = calculate_mean(build_grads, batch_n_samples, output_dim, par_th);
            float *std = calculate_var_and_center(build_grads, mean_grads, batch_n_samples, output_dim, par_th);
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int i = 0; i < output_dim; ++i)
                std[i] = sqrtf(std[i]);
            divide_mat_by_vec_inplace(build_grads, std, batch_dataset.n_samples, metadata->output_dim, metadata->par_th);
            delete[] mean_grads;
            delete[] std;
        } else {
            calculate_squared_norm(norm_grads, grads, batch_dataset.n_samples, metadata->output_dim, metadata->par_th);
        }
        batch_dataset.build_grads = build_grads; 
        batch_dataset.norm_grads = norm_grads; 

        int added_leaves = 0;
        if (metadata->grow_policy == GREEDY)
            added_leaves = Fitter::fit_greedy_tree(&batch_dataset, edata, metadata, generator);
        else 
            added_leaves = Fitter::fit_oblivious_tree(&batch_dataset, edata, metadata, generator);
        Fitter::fit_leaves(&batch_dataset, edata, metadata, added_leaves);
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

    Predictor::predict_cpu(dataset, full_preds, edata, metadata, 0, iterations, false, opts);
    float full_loss = INFINITY;
    if (loss_type == MultiRMSE){
        full_loss = MultiRMSE::get_loss(full_preds, targets, dataset->n_samples, output_dim); 
    }
    delete[] batch_preds;
    delete[] full_preds;
    delete[] last_batch_preds;
    delete[] batch_grads;
    delete[] batch_build_grads;
    delete[] last_batch_grads;
    delete[] last_batch_build_grads;
    delete[] batch_grad_norms;
    delete[] last_batch_grad_norms;
    return full_loss;
}

int Fitter::fit_greedy_tree(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const SplitCandidateGenerator &generator){
    allocate_ensemble_memory(metadata, edata);
    edata->tree_indices[metadata->n_trees] = metadata->n_leaves;
    int depth = 0, node_idx_cntr = 0, chosen_idx = 0;
    float best_score, parent_score = -INFINITY;
    int n_samples = dataset->n_samples;

    int added_leaves = 0;

    int *root_sample_indices = new int[n_samples];
    std::iota(root_sample_indices, root_sample_indices + n_samples, 0);

    std::vector<TreeNode*> tree_nodes; 
    TreeNode *rootNode = new TreeNode(root_sample_indices, n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->output_dim, depth, 0);
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
    int batch_size = n_candidates / n_threads;

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
            if (metadata->split_score_func == Cosine){
                parent_score = scoreCosine(crnt_node->sample_indices, crnt_node->n_samples, dataset->build_grads, dataset->norm_grads, metadata->output_dim);
            } else if (metadata->split_score_func == L2){
                parent_score = scoreL2(crnt_node->sample_indices, crnt_node->n_samples, dataset->build_grads, metadata->output_dim);
            } else{
                std::cerr << "error invalid split score func!" << std::endl;
                continue;
            }
            if (crnt_node->depth == 0)
                parent_score = 0.0f;
#ifndef DEBUG
            #pragma omp parallel
            {
                int thread_num = omp_get_thread_num();
#else
                int thread_num = 0;
#endif
                float local_best_score = -INFINITY;
                int local_chosen_idx = -1;
                int start_idx = thread_num * batch_size;
                int end_idx = (thread_num == n_threads - 1) ? n_candidates : start_idx + batch_size;
                // Process the batch of candidates
                for (int j = start_idx; j < end_idx; ++j) {
                    float score = crnt_node->getSplitScore(dataset, metadata->split_score_func, split_candidates[j], metadata->min_data_in_leaf);
                    int feat_idx = (split_candidates[j].categorical_value == nullptr) ? split_candidates[j].feature_idx : split_candidates[j].feature_idx + metadata->n_num_features; 
                    score = score * dataset->feature_weights[feat_idx] - parent_score;
#ifdef DEBUG
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
            int status = crnt_node->splitNode(dataset->obs, dataset->categorical_obs, node_idx_cntr, split_candidates[chosen_idx]);
            if (status == -1){
                std::cerr << "ERROR couldn't split best score" << std::endl;
                break;
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
    edata->tree_indices[metadata->n_trees] = metadata->n_leaves;
    
    int depth = 0, node_idx_cntr = 0, chosen_idx = 0;
    float best_score;
    int n_samples = dataset->n_samples;

    int added_leaves = 0;

    int *root_sample_indices = new int[n_samples];
    std::iota(root_sample_indices, root_sample_indices + n_samples, 0);

    int max_n_leaves = 1 << metadata->max_depth;

    std::vector<TreeNode*> tree_nodes(max_n_leaves); 
    std::vector<TreeNode*> child_tree_nodes(max_n_leaves); 
    TreeNode *rootNode = new TreeNode(root_sample_indices, n_samples, metadata->n_num_features, metadata->n_cat_features, metadata->output_dim, depth, 0);
    tree_nodes[0] = rootNode;
    float parent_score = 0.0f;

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
    int batch_size = n_candidates / n_threads;
    
    while (depth < metadata->max_depth)  {
        best_score = -INFINITY;
#ifndef DEBUG
        #pragma omp parallel
        {
            int thread_num = omp_get_thread_num();
#else 
            int thread_num = 0;
#endif
            float local_best_score = -INFINITY;
            int local_chosen_idx = -1;
            int start_idx = thread_num * batch_size;
            int end_idx = (thread_num == n_threads - 1) ? n_candidates : start_idx + batch_size;
            // Process the batch of candidates
            for (int j = start_idx; j < end_idx; ++j) {
                float score = 0.0f;
                for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
                    TreeNode *crnt_node = tree_nodes[node_idx];
                    score += crnt_node->getSplitScore(dataset, metadata->split_score_func, split_candidates[j], metadata->min_data_in_leaf);
                }
                int feat_idx = (split_candidates[j].categorical_value == nullptr) ? split_candidates[j].feature_idx : split_candidates[j].feature_idx + metadata->n_num_features; 
                score = score*dataset->feature_weights[feat_idx] - parent_score;
#ifdef DEBUG
                std::cout << " cand: " <<  j << " score: " <<  score << " parent_score: " << parent_score <<  " info: " << split_candidates[j] << std::endl;
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
        parent_score = best_score;
        for (int node_idx = 0; node_idx < (1 << depth); ++node_idx){
            TreeNode *crnt_node = tree_nodes[node_idx];
            int status = crnt_node->splitNode(dataset->obs, dataset->categorical_obs, node_idx_cntr, split_candidates[chosen_idx]);
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
        Fitter::calc_leaf_value(dataset, edata, metadata, edata->tree_indices[metadata->n_trees - 1] + leaf_idx, metadata->n_trees - 1);
    }
}

void Fitter::update_ensemble_per_leaf(ensembleData *edata, ensembleMetaData *metadata, const TreeNode* node){
    int idx = metadata->n_leaves; 
#ifdef DEBUG
    edata->n_samples[idx] = node->n_samples;
#endif
    edata->depths[idx] = node->depth;
    int row_idx = idx*metadata->max_depth;
    if (node->depth > 0){
        for (int i = 0; i < node->depth; ++i){
            if (node->split_conditions[i].categorical_value != nullptr){
                memcpy(edata->categorical_values + (row_idx + i)*MAX_CHAR_SIZE, node->split_conditions[i].categorical_value, sizeof(char)*MAX_CHAR_SIZE);
                edata->is_numerics[row_idx+ i] = false;
            } else {
                edata->is_numerics[row_idx+ i] = true;
            }
            edata->feature_indices[row_idx+ i] = node->split_conditions[i].feature_idx;
            edata->feature_values[row_idx+ i] = node->split_conditions[i].feature_value;
            edata->inequality_directions[row_idx + i] = node->split_conditions[i].inequality_direction;
            edata->edge_weights[row_idx + i] = node->split_conditions[i].edge_weight;
        }
    }
    metadata->n_leaves += 1;
}

void Fitter::update_ensemble_per_tree(ensembleData *edata, ensembleMetaData *metadata, std::vector<TreeNode*> nodes, const int n_nodes){
    int idx = metadata->n_trees; 
    for (int node_idx = 0; node_idx <  n_nodes; node_idx++){
        TreeNode *node = nodes[node_idx];
#ifdef DEBUG
        edata->n_samples[metadata->n_leaves] = node->n_samples;
#endif
        edata->depths[idx] = node->depth;
        int row_idx = idx*metadata->max_depth;
        if (node->depth > 0){
            for (int i = 0; i < node->depth; ++i){
                if (node->split_conditions[i].categorical_value != nullptr){
                    memcpy(edata->categorical_values + (row_idx + i)*MAX_CHAR_SIZE, node->split_conditions[i].categorical_value, sizeof(char)*MAX_CHAR_SIZE);
                    edata->is_numerics[row_idx+ i] = false;
                } else {
                    edata->is_numerics[row_idx+ i] = true;
                }
                edata->feature_indices[row_idx+ i] = node->split_conditions[i].feature_idx;
                edata->feature_values[row_idx+ i] = node->split_conditions[i].feature_value;
                edata->inequality_directions[metadata->n_leaves*metadata->max_depth + i] = node->split_conditions[i].inequality_direction;
                edata->edge_weights[metadata->n_leaves*metadata->max_depth + i] = node->split_conditions[i].edge_weight;
            }
        }
        metadata->n_leaves += 1;
    }
}


void Fitter::calc_leaf_value(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const int leaf_idx, const int tree_idx){
    int output_dim = metadata->output_dim;
    const float *obs = dataset->obs, *grads = dataset->grads;
    const char *categorical_obs = dataset->categorical_obs;
    int depth = (metadata->grow_policy == OBLIVIOUS) ? edata->depths[tree_idx] : edata->depths[leaf_idx];
    int cond_idx = (metadata->grow_policy == OBLIVIOUS) ? tree_idx*metadata->max_depth : leaf_idx*metadata->max_depth;
    int ineq_cond = leaf_idx*metadata->max_depth;
    float count = 0;
    bool passed;
    int idx, row_idx, cat_row_idx;

    for (int i = 0; i < dataset->n_samples; ++i){
        row_idx = i*metadata->n_num_features;
        cat_row_idx = i*metadata->n_cat_features;
        passed = false;
        for (int depth_idx = depth - 1; depth_idx >= 0; --depth_idx){
            passed = (edata->is_numerics[cond_idx + depth_idx]) ? (obs[row_idx + edata->feature_indices[cond_idx + depth_idx]] > edata->feature_values[cond_idx + depth_idx]) == edata->inequality_directions[ineq_cond + depth_idx] : (strcmp(&categorical_obs[(cat_row_idx + edata->feature_indices[cond_idx + depth_idx]) * MAX_CHAR_SIZE], edata->categorical_values + (cond_idx + depth_idx)*MAX_CHAR_SIZE) == 0) == edata->inequality_directions[ineq_cond + depth_idx];
            if (!passed)
                break;
        }
        if (passed){
            idx = i*output_dim;
#ifndef _MSC_VER
    #pragma omp simd
#endif
            for (int d = 0; d < output_dim; ++d)
                edata->values[leaf_idx*output_dim + d] += grads[idx + d];
            count += 1;
        }
    }
    if (count > 0){
        for (int d = 0; d < output_dim; ++d)
            edata->values[leaf_idx*output_dim + d] /= count;
    }
#ifdef DEBUG
    edata->n_samples[leaf_idx] = static_cast<int>(count);
#endif
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
                momemntFunc(dataset->obs, dataset->categorical_obs, momentum, edata, metadata, 0, metadata->n_trees, sample_idx);
            }
        }
    // no parallelization
    } else{ 
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx){
            momemntFunc(dataset->obs, dataset->categorical_obs, momentum, edata, metadata, 0, metadata->n_trees, sample_idx);
        }
    }

    float error_correction = 1.0f / sqrtf(1.0f - powf(metadata->cv_beta, static_cast<float>(metadata->n_trees)));
    _multiply_mat_by_scalar(momentum, error_correction, n_samples, output_dim, par_th);
    float *grads_copy = copy_mat(dataset->grads, n_samples*output_dim, par_th);
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
    multiply_mat_by_vec_subtract_result(dataset->grads, momentum, alpha, n_samples, output_dim, par_th);
    delete[] grads_mean;
    delete[] momentum_mean;
    delete[] grads_copy;
    delete[] variance;
    delete[] covariance;
    delete[] alpha;
    delete[] momentum;
}