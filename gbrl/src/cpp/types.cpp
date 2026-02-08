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
 * @file types.cpp
 * @brief Implementation of type conversion and data structure management functions
 * 
 * Provides string-to-enum conversions, data structure allocation/deallocation,
 * and serialization utilities for gradient boosting types.
 */

#include <stdexcept>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "types.h"
#include "utils.h"
#include "optimizer.h"
#ifdef USE_CUDA
#include "cuda_types.h"
#endif

scoreFunc stringToScoreFunc(std::string str) {
    if (str == "L2" || str == "l2") return scoreFunc::L2;
    if (str == "Cosine" || str == "cosine") return scoreFunc::Cosine;
    throw std::runtime_error("Invalid score function! Options are: Cosine/L2");
    return scoreFunc::L2;
}

generatorType stringTogeneratorType(std::string str) {
    if (str == "uniform" || str == "Uniform") return generatorType::Uniform;
    if (str == "quantile" || str == "Quantile") return generatorType::Quantile;
    throw std::runtime_error("Invalid generator function! Options are: Uniform/Quantile");
    return generatorType::Uniform;
}

growPolicy stringTogrowPolicy(std::string str) {
    if (str == "oblivious" || str == "Oblivious") return growPolicy::OBLIVIOUS;
    if (str == "greedy" || str == "Greedy") return growPolicy::GREEDY;
    throw std::runtime_error("Invalid generator function! Options are: Greedy/Oblivious");
    return growPolicy::GREEDY;
}

lossType stringTolossType(std::string str) {
    if (str == "MultiRMSE") return lossType::MultiRMSE;
    throw std::runtime_error("Invalid loss function! Options are: MultiRMSE");
    return lossType::MultiRMSE;
}

deviceType stringTodeviceType(std::string str) {
    if (str == "cpu") return deviceType::cpu;
    if (str == "cuda" || str == "gpu") return deviceType::gpu;
    throw std::runtime_error("Invalid device! Options are: cpu/cuda");
}

exportFormat stringToexportFormat(std::string str) {
    if (str == "fxp8") return exportFormat::EXP_FXP8;
    if (str == "fxp16") return exportFormat::EXP_FXP16;
    if (str == "float") return exportFormat::EXP_FLOAT;
    throw std::runtime_error("Invalid exportFormat! Options are: float/fxp8/fxp16");
}
exportType stringToexportType(std::string str) {
    if (str == "compact") return exportType::COMPACT;
    if (str == "full") return exportType::FULL;
    throw std::runtime_error("Invalid exportType Options are: full/compact");
}

optimizerAlgo stringToAlgoType(std::string str) {
    if (str == "Adam" || str == "adam") return optimizerAlgo::Adam;
    if (str == "SGD" || str == "sgd") return optimizerAlgo::SGD;
    throw std::runtime_error("Invalid Optimizer Algorithm! Options are: SGD/Adam");
    return optimizerAlgo::SGD;
}

schedulerFunc stringToSchedulerType(std::string str) {
    if (str == "Const" || str == "const") return schedulerFunc::Const;
    if (str == "Linear" || str == "linear") return schedulerFunc::Linear;
    throw std::runtime_error("Invalid Scheduler type! Options are: Linear/Const");
    return schedulerFunc::Const;
}

std::string scoreFuncToString(scoreFunc func) {
    switch (func) {
        case scoreFunc::L2:
            return "L2";
        case scoreFunc::Cosine:
            return "Cosine";
        default:
            throw std::runtime_error("Invalid score function.");
    }
}

std::string generatorTypeToString(generatorType type) {
    switch (type) {
        case generatorType::Uniform:
            return "Uniform";
        case generatorType::Quantile:
            return "Quantile";
        default:
            throw std::runtime_error("Invalid generator type.");
    }
}

std::string growPolicyToString(growPolicy type) {
    switch (type) {
        case growPolicy::OBLIVIOUS:
            return "Oblivous";
        case growPolicy::GREEDY:
            return "Greedy";
        default:
            throw std::runtime_error("Invalid generator type.");
    }
}


std::string lossTypeToString(lossType type) {
    switch (type) {
        case lossType::MultiRMSE:
            return "MultiRMSE";
        default:
            throw std::runtime_error("Invalid loss type.");
    }
}

std::string deviceTypeToString(deviceType type) {
    switch (type) {
        case deviceType::cpu:
            return "cpu";
        case deviceType::gpu:
            return "cuda"; // Assuming 'cuda' is the preferred string for GPU.
        default:
            throw std::runtime_error("Invalid device type.");
    }
}

std::string algoTypeToString(optimizerAlgo algo) {
    switch (algo) {
        case optimizerAlgo::Adam:
            return "Adam";
        case optimizerAlgo::SGD:
            return "SGD";
        default:
            throw std::runtime_error("Invalid optimizer algorithm.");
    }
}

std::string schedulerTypeToString(schedulerFunc func) {
    switch (func) {
        case schedulerFunc::Const:
            return "Const";
        case schedulerFunc::Linear:
            return "Linear";
        default:
            throw std::runtime_error("Invalid scheduler type.");
    }
}

ensembleMetaData* ensemble_metadata_alloc(
    int max_trees,
    int max_leaves,
    int max_trees_batch,
    int max_leaves_batch,
    int input_dim,
    int output_dim,
    int policy_dim,
    int max_depth,
    int min_data_in_leaf,
    int n_bins,
    int par_th,
    float cv_beta, float lambda_penalty,
    int verbose, int n_objs,
    int batch_size,
    bool use_cv,
    scoreFunc split_score_func,
    generatorType generator_type,
    growPolicy grow_policy,
    int n_mono_constraints){

    ensembleMetaData *metadata = new ensembleMetaData;
    metadata->input_dim = input_dim; 
    metadata->output_dim = output_dim; 
    metadata->policy_dim = policy_dim;
    metadata->max_depth = max_depth; 
    metadata->min_data_in_leaf = min_data_in_leaf; 
    metadata->par_th = par_th; 
    metadata->n_bins = n_bins; 
    metadata->n_leaves = 0;
    metadata->n_trees = 0; 
    metadata->verbose = verbose; 
    metadata->batch_size = batch_size; 
    metadata->grow_policy = grow_policy;
    metadata->generator_type = generator_type;
    metadata->lambda_penalty = lambda_penalty;
    metadata->cv_beta = cv_beta;
    metadata->split_score_func = split_score_func;
    metadata->use_cv = use_cv;
    metadata->max_trees = max_trees;
    metadata->max_leaves = max_leaves; 
    metadata->max_trees_batch = max_trees_batch; 
    metadata->max_leaves_batch = max_leaves_batch;
    metadata->n_num_features = 0;
    metadata->n_cat_features = 0;
    metadata->iteration = 0;
    metadata->n_objs = n_objs;
    metadata->n_mono_constraints = n_mono_constraints;
    return metadata;
}

ensembleData* ensemble_data_alloc(ensembleMetaData *metadata){
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        throw std::runtime_error("Error invalid pointer");
    }
    ensembleData *edata = new ensembleData;
    size_t data_size = 0;
    
    // Allocate sub-structs on CPU
    edata->ensemble_info = new ensembleInfo;
    edata->leaf_data = new leafData;
    edata->feature_data = new featureData;
    edata->mono_constraints = new monotonicConstraints;
    edata->mono_constraints->n_constraints = 0;  // Initialize to 0
    edata->feature_mappings = new featureMapping;
    edata->multi_objective_data = new multiObjectiveData;
    
    edata->bias = new float[metadata->output_dim];
    data_size += sizeof(float) * metadata->output_dim;
    memset(edata->bias, 0, metadata->output_dim * sizeof(float));
    int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->max_trees : metadata->max_leaves;
#ifdef DEBUG
    edata->n_samples = new int[metadata->max_leaves]; // debugging
    memset(edata->n_samples, 0, metadata->max_leaves * sizeof(int));
    data_size += sizeof(int) * metadata->max_leaves;
#endif
    // Ensemble info
    edata->ensemble_info->tree_indices = new int[metadata->max_trees];
    data_size += sizeof(int) * metadata->max_trees;
    memset(edata->ensemble_info->tree_indices, 0, metadata->max_trees * sizeof(int));
    edata->ensemble_info->depths = new int[split_sizes];
    data_size += sizeof(int) * split_sizes;
    memset(edata->ensemble_info->depths, 0, split_sizes * sizeof(int));
    
    // Leaf data
    edata->leaf_data->values = new float[metadata->max_leaves * metadata->output_dim];
    data_size += sizeof(float) * metadata->max_leaves * metadata->output_dim;
    memset(edata->leaf_data->values, 0, metadata->max_leaves * metadata->output_dim * sizeof(float));
    edata->leaf_data->edge_weights = new float[metadata->max_leaves * metadata->max_depth];
    data_size += sizeof(float) * metadata->max_leaves * metadata->max_depth;
    memset(edata->leaf_data->edge_weights, 0, metadata->max_leaves * metadata->max_depth * sizeof(float));
    
    // Feature data
    edata->feature_data->feature_indices = new int[split_sizes * metadata->max_depth];
    data_size += sizeof(int) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->feature_indices, 0, split_sizes * metadata->max_depth * sizeof(int));
    edata->feature_data->feature_values = new float[split_sizes * metadata->max_depth];
    data_size += sizeof(float) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->feature_values, 0, split_sizes * metadata->max_depth * sizeof(float));
    edata->feature_data->feature_weights = new float[metadata->input_dim];
    data_size += sizeof(float) * metadata->input_dim;
    memset(edata->feature_data->feature_weights, 0, metadata->input_dim * sizeof(float));
    edata->feature_data->is_numerics = new bool[split_sizes * metadata->max_depth];
    data_size += sizeof(bool) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->is_numerics, 0, split_sizes * metadata->max_depth * sizeof(bool));
    edata->feature_data->inequality_directions = new bool[metadata->max_leaves * metadata->max_depth]; 
    data_size += sizeof(bool) * metadata->max_leaves * metadata->max_depth;
    memset(edata->feature_data->inequality_directions, 0, metadata->max_leaves * metadata->max_depth * sizeof(bool));
    edata->feature_data->categorical_values = new char[split_sizes * metadata->max_depth * MAX_CHAR_SIZE];
    data_size += sizeof(char) * split_sizes * metadata->max_depth * MAX_CHAR_SIZE;
    memset(edata->feature_data->categorical_values, 0, split_sizes * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE);
    
    // Multi-objective optimization data
    edata->multi_objective_data->densities = new float[metadata->max_leaves * metadata->n_objs];
    data_size += sizeof(float) * metadata->max_leaves * metadata->n_objs;
    memset(edata->multi_objective_data->densities, 0, metadata->max_leaves * metadata->n_objs * sizeof(float));

    edata->multi_objective_data->lambda_objs = new float[metadata->n_objs];
    data_size += sizeof(float) * metadata->n_objs;
    memset(edata->multi_objective_data->lambda_objs, 0, metadata->n_objs * sizeof(float));

    // Monotonic constraints
    edata->mono_constraints->feature_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memset(edata->mono_constraints->feature_idx, 0, metadata->n_mono_constraints * sizeof(int));
    edata->mono_constraints->output_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memset(edata->mono_constraints->output_idx, 0, metadata->n_mono_constraints * sizeof(int));
    edata->mono_constraints->constraint = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memset(edata->mono_constraints->constraint, 0, metadata->n_mono_constraints * sizeof(int));

    // Feature mappings
    edata->feature_mappings->reverse_num_feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memset(edata->feature_mappings->reverse_num_feature_mapping, 0, metadata->input_dim * sizeof(int));
    edata->feature_mappings->reverse_cat_feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memset(edata->feature_mappings->reverse_cat_feature_mapping, 0, metadata->input_dim * sizeof(int));
    edata->feature_mappings->feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memset(edata->feature_mappings->feature_mapping, 0, metadata->input_dim * sizeof(int));
    edata->feature_mappings->mapping_numerics = new bool[metadata->input_dim];
    data_size += sizeof(bool) * metadata->input_dim;
    memset(edata->feature_mappings->mapping_numerics, 0, metadata->input_dim * sizeof(bool));
    
    edata->alloc_data_size = data_size;
    return edata;
}

ensembleData* ensemble_copy_data_alloc(ensembleMetaData *metadata){
    // same as normal alloc but only allocate memory for existing size 
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        throw std::runtime_error("Error invalid pointer");
    }
    ensembleData *edata = new ensembleData;
    size_t data_size = 0;
    
    // Allocate sub-structs on CPU
    edata->ensemble_info = new ensembleInfo;
    edata->leaf_data = new leafData;
    edata->feature_data = new featureData;
    edata->mono_constraints = new monotonicConstraints;
    edata->mono_constraints->n_constraints = 0;  // Initialize to 0
    edata->feature_mappings = new featureMapping;
    edata->multi_objective_data = new multiObjectiveData;
    
    edata->bias = new float[metadata->output_dim];
    data_size += sizeof(float) * metadata->output_dim;
    memset(edata->bias, 0, metadata->output_dim * sizeof(float));
    int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
#ifdef DEBUG
    edata->n_samples = new int[metadata->n_leaves]; // debugging
    memset(edata->n_samples, 0, metadata->n_leaves * sizeof(int));
    data_size += sizeof(int) * metadata->n_leaves;
#endif
    // Ensemble info
    edata->ensemble_info->tree_indices = new int[metadata->n_trees];
    data_size += sizeof(int) * metadata->n_trees;
    memset(edata->ensemble_info->tree_indices, 0, metadata->n_trees * sizeof(int));
    edata->ensemble_info->depths = new int[split_sizes];
    data_size += sizeof(int) * split_sizes;
    memset(edata->ensemble_info->depths, 0, split_sizes * sizeof(int));
    
    // Leaf data
    edata->leaf_data->values = new float[metadata->n_leaves * metadata->output_dim];
    data_size += sizeof(float) * metadata->n_leaves * metadata->output_dim;
    memset(edata->leaf_data->values, 0, metadata->n_leaves * metadata->output_dim * sizeof(float));
    edata->leaf_data->edge_weights = new float[metadata->n_leaves * metadata->max_depth];
    data_size += sizeof(float) * metadata->n_leaves * metadata->max_depth;
    memset(edata->leaf_data->edge_weights, 0, metadata->n_leaves * metadata->max_depth * sizeof(float));
    
    // Feature data
    edata->feature_data->feature_indices = new int[split_sizes * metadata->max_depth];
    data_size += sizeof(int) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->feature_indices, 0, split_sizes * metadata->max_depth * sizeof(int));
    edata->feature_data->feature_values = new float[split_sizes * metadata->max_depth];
    data_size += sizeof(float) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->feature_values, 0, split_sizes * metadata->max_depth * sizeof(float));
    edata->feature_data->feature_weights = new float[metadata->input_dim];
    data_size += sizeof(float) * metadata->input_dim;
    memset(edata->feature_data->feature_weights, 0, metadata->input_dim * sizeof(float));
    edata->feature_data->is_numerics = new bool[split_sizes * metadata->max_depth];
    data_size += sizeof(bool) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->is_numerics, 0, split_sizes * metadata->max_depth * sizeof(bool));
    edata->feature_data->inequality_directions = new bool[metadata->n_leaves * metadata->max_depth]; 
    data_size += sizeof(bool) * metadata->n_leaves * metadata->max_depth;
    memset(edata->feature_data->inequality_directions, 0, metadata->n_leaves * metadata->max_depth * sizeof(bool));
    edata->feature_data->categorical_values = new char[split_sizes * metadata->max_depth * MAX_CHAR_SIZE];
    data_size += sizeof(char) * split_sizes * metadata->max_depth * MAX_CHAR_SIZE;
    memset(edata->feature_data->categorical_values, 0, split_sizes * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE);
    
    // Multi-objective optimization data
    edata->multi_objective_data->densities = new float[metadata->n_leaves * metadata->n_objs];
    data_size += sizeof(float) * metadata->n_leaves * metadata->n_objs;
    memset(edata->multi_objective_data->densities, 0, metadata->n_leaves * metadata->n_objs * sizeof(float));
    edata->multi_objective_data->lambda_objs = new float[metadata->n_objs];
    data_size += sizeof(float) * metadata->n_objs;
    memset(edata->multi_objective_data->lambda_objs, 0, metadata->n_objs * sizeof(float));

    // Monotonic constraints
    edata->mono_constraints->feature_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memset(edata->mono_constraints->feature_idx, 0, metadata->n_mono_constraints * sizeof(int));
    edata->mono_constraints->output_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memset(edata->mono_constraints->output_idx, 0, metadata->n_mono_constraints * sizeof(int));
    edata->mono_constraints->constraint = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memset(edata->mono_constraints->constraint, 0, metadata->n_mono_constraints * sizeof(int));

    // Feature mappings
    edata->feature_mappings->reverse_num_feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memset(edata->feature_mappings->reverse_num_feature_mapping, 0, metadata->input_dim * sizeof(int));
    edata->feature_mappings->reverse_cat_feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memset(edata->feature_mappings->reverse_cat_feature_mapping, 0, metadata->input_dim * sizeof(int));
    edata->feature_mappings->feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memset(edata->feature_mappings->feature_mapping, 0, metadata->input_dim * sizeof(int));
    edata->feature_mappings->mapping_numerics = new bool[metadata->input_dim];
    data_size += sizeof(bool) * metadata->input_dim;
    memset(edata->feature_mappings->mapping_numerics, 0, metadata->input_dim * sizeof(bool));
    
    edata->alloc_data_size = data_size;
    return edata;
}

ensembleData* copy_ensemble_data(ensembleData *other_edata, ensembleMetaData *metadata){
    if (metadata == nullptr || other_edata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        throw std::runtime_error("Error invalid pointer");
    }
    ensembleData *edata = new ensembleData;
    size_t data_size = 0;
    
    // Allocate sub-structs on CPU
    edata->ensemble_info = new ensembleInfo;
    edata->leaf_data = new leafData;
    edata->feature_data = new featureData;
    edata->mono_constraints = new monotonicConstraints;
    edata->mono_constraints->n_constraints = 0;  // Initialize to 0
    edata->feature_mappings = new featureMapping;
    edata->multi_objective_data = new multiObjectiveData;
    
    edata->bias = new float[metadata->output_dim];
    data_size += sizeof(float) * metadata->output_dim;
    memcpy(edata->bias, other_edata->bias, metadata->output_dim * sizeof(float));
    int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
#ifdef DEBUG
    edata->n_samples = new int[metadata->n_leaves]; // debugging
    memcpy(edata->n_samples, other_edata->n_samples, metadata->n_leaves * sizeof(int));
    data_size += sizeof(float) * metadata->n_leaves;
#endif
    // Ensemble info
    edata->ensemble_info->tree_indices = new int[metadata->n_trees];
    data_size += sizeof(int) * metadata->n_trees;
    memcpy(edata->ensemble_info->tree_indices, other_edata->ensemble_info->tree_indices, metadata->n_trees * sizeof(int));
    edata->ensemble_info->depths = new int[split_sizes];
    data_size += sizeof(int) * split_sizes;
    memcpy(edata->ensemble_info->depths, other_edata->ensemble_info->depths, split_sizes * sizeof(int));
    
    // Leaf data
    edata->leaf_data->values = new float[metadata->n_leaves * metadata->output_dim];
    data_size += sizeof(float) * metadata->n_leaves * metadata->output_dim;
    memcpy(edata->leaf_data->values, other_edata->leaf_data->values, metadata->n_leaves * metadata->output_dim * sizeof(float));
    edata->leaf_data->edge_weights = new float[metadata->n_leaves * metadata->max_depth];
    data_size += sizeof(float) * metadata->n_leaves * metadata->max_depth;
    memcpy(edata->leaf_data->edge_weights, other_edata->leaf_data->edge_weights, metadata->n_leaves * metadata->max_depth * sizeof(float));
    
    // Feature data
    edata->feature_data->feature_indices = new int[split_sizes * metadata->max_depth];
    data_size += sizeof(int) * split_sizes * metadata->max_depth;
    memcpy(edata->feature_data->feature_indices, other_edata->feature_data->feature_indices, split_sizes * metadata->max_depth * sizeof(int));
    edata->feature_data->feature_values = new float[split_sizes * metadata->max_depth];
    data_size += sizeof(float) * split_sizes * metadata->max_depth;
    memcpy(edata->feature_data->feature_values, other_edata->feature_data->feature_values, split_sizes * metadata->max_depth * sizeof(float));
    edata->feature_data->feature_weights = new float[metadata->input_dim];
    data_size += sizeof(float) * metadata->input_dim;
    memcpy(edata->feature_data->feature_weights, other_edata->feature_data->feature_weights, metadata->input_dim * sizeof(float));
    edata->feature_data->is_numerics = new bool[split_sizes * metadata->max_depth];
    data_size += sizeof(bool) * split_sizes * metadata->max_depth;
    memcpy(edata->feature_data->is_numerics, other_edata->feature_data->is_numerics, split_sizes * metadata->max_depth * sizeof(bool));
    edata->feature_data->categorical_values = new char[split_sizes * metadata->max_depth * MAX_CHAR_SIZE];
    data_size += sizeof(char) * split_sizes * metadata->max_depth * MAX_CHAR_SIZE;
    memcpy(edata->feature_data->categorical_values, other_edata->feature_data->categorical_values, split_sizes * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE);
    edata->feature_data->inequality_directions = new bool[metadata->n_leaves * metadata->max_depth]; 
    data_size += sizeof(bool) * metadata->n_leaves * metadata->max_depth;
    memcpy(edata->feature_data->inequality_directions, other_edata->feature_data->inequality_directions, metadata->n_leaves * metadata->max_depth * sizeof(bool));
    
    // Monotonic constraints
    edata->mono_constraints->feature_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memcpy(edata->mono_constraints->feature_idx, other_edata->mono_constraints->feature_idx, metadata->n_mono_constraints * sizeof(int));
    // Multi-objective data
    edata->multi_objective_data->densities = new float[metadata->n_leaves * metadata->n_objs];
    data_size += sizeof(float) * metadata->n_leaves * metadata->n_objs;
    memcpy(edata->multi_objective_data->densities, other_edata->multi_objective_data->densities, metadata->n_leaves * metadata->n_objs * sizeof(float));
    edata->multi_objective_data->lambda_objs = new float[metadata->n_objs];
    data_size += sizeof(float) * metadata->n_objs;
    memcpy(edata->multi_objective_data->lambda_objs, other_edata->multi_objective_data->lambda_objs, metadata->n_objs * sizeof(float));
    edata->mono_constraints->output_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memcpy(edata->mono_constraints->output_idx, other_edata->mono_constraints->output_idx, metadata->n_mono_constraints * sizeof(int));
    edata->mono_constraints->constraint = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memcpy(edata->mono_constraints->constraint, other_edata->mono_constraints->constraint, metadata->n_mono_constraints * sizeof(int));

    // Feature mappings
    edata->feature_mappings->reverse_num_feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memcpy(edata->feature_mappings->reverse_num_feature_mapping, other_edata->feature_mappings->reverse_num_feature_mapping, metadata->input_dim * sizeof(int));
    edata->feature_mappings->reverse_cat_feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memcpy(edata->feature_mappings->reverse_cat_feature_mapping, other_edata->feature_mappings->reverse_cat_feature_mapping, metadata->input_dim * sizeof(int));
    edata->feature_mappings->feature_mapping = new int[metadata->input_dim];
    data_size += sizeof(int) * metadata->input_dim;
    memcpy(edata->feature_mappings->feature_mapping, other_edata->feature_mappings->feature_mapping, metadata->input_dim * sizeof(int));
    edata->feature_mappings->mapping_numerics = new bool[metadata->input_dim];
    data_size += sizeof(bool) * metadata->input_dim;
    memcpy(edata->feature_mappings->mapping_numerics, other_edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool));
    
    metadata->max_trees = metadata->n_trees;
    metadata->max_leaves = metadata->n_leaves;
    edata->alloc_data_size = data_size;
    return edata;
}

ensembleData* copy_compressed_ensemble_data(ensembleData *other_edata, ensembleMetaData *metadata, const int *leaf_indices, const int *tree_indices, const int n_compressed_leaves, const int n_compressed_trees, const int *new_tree_indices){
    
    if (metadata == nullptr || other_edata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        throw std::runtime_error("Error invalid pointer");
    }
    ensembleData *edata = new ensembleData;
    size_t data_size = 0;
    
    // Allocate sub-structs on CPU
    edata->ensemble_info = new ensembleInfo;
    edata->leaf_data = new leafData;
    edata->feature_data = new featureData;
    edata->mono_constraints = new monotonicConstraints;
    edata->mono_constraints->n_constraints = 0;  // Initialize to 0
    edata->feature_mappings = new featureMapping;
    edata->multi_objective_data = new multiObjectiveData;
    
    edata->bias = new float[metadata->output_dim];
    data_size += sizeof(float) * metadata->output_dim;
    memcpy(edata->bias, other_edata->bias, metadata->output_dim * sizeof(float));
    
    // Feature mappings
    edata->feature_mappings->feature_mapping = new int[metadata->input_dim];
    memcpy(edata->feature_mappings->feature_mapping, other_edata->feature_mappings->feature_mapping, metadata->input_dim * sizeof(int));
    data_size += sizeof(int) * metadata->input_dim;
    edata->feature_mappings->reverse_num_feature_mapping = new int[metadata->input_dim];
    memcpy(edata->feature_mappings->reverse_num_feature_mapping, other_edata->feature_mappings->reverse_num_feature_mapping, metadata->input_dim * sizeof(int));
    data_size += sizeof(int) * metadata->input_dim;
    edata->feature_mappings->reverse_cat_feature_mapping = new int[metadata->input_dim];
    memcpy(edata->feature_mappings->reverse_cat_feature_mapping, other_edata->feature_mappings->reverse_cat_feature_mapping, metadata->input_dim * sizeof(int));
    data_size += sizeof(int) * metadata->input_dim;
    edata->feature_mappings->mapping_numerics = new bool[metadata->input_dim];
    memcpy(edata->feature_mappings->mapping_numerics, other_edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool));
    data_size += sizeof(bool) * metadata->input_dim;
    
    // Feature data - feature_weights
    edata->feature_data->feature_weights = new float[metadata->input_dim];
    data_size += sizeof(float) * metadata->input_dim;
    memcpy(edata->feature_data->feature_weights, other_edata->feature_data->feature_weights, metadata->input_dim * sizeof(float));
    
    int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? n_compressed_trees : n_compressed_leaves;
    const int *split_indices = (metadata->grow_policy == OBLIVIOUS) ? tree_indices : leaf_indices;
#ifdef DEBUG
    edata->n_samples = new int[n_compressed_leaves]; // debugging
    data_size += sizeof(int) * n_compressed_leaves;
    memset(edata->n_samples, 0, n_compressed_leaves * sizeof(int));
    selective_copy(n_compressed_leaves, leaf_indices, edata->n_samples, other_edata->n_samples, 1);
#endif
    // Ensemble info
    edata->ensemble_info->tree_indices = new int[n_compressed_trees];
    data_size += sizeof(int) * n_compressed_trees;
    memcpy(edata->ensemble_info->tree_indices, new_tree_indices, n_compressed_trees * sizeof(int));
    edata->ensemble_info->depths = new int[split_sizes];
    data_size += sizeof(int) * split_sizes;
    memset(edata->ensemble_info->depths, 0, split_sizes * sizeof(int));
    selective_copy(split_sizes, split_indices, edata->ensemble_info->depths, other_edata->ensemble_info->depths, 1);
    
    // Leaf data
    edata->leaf_data->values = new float[n_compressed_leaves * metadata->output_dim];
    data_size += sizeof(float) * n_compressed_leaves * metadata->output_dim;
    memset(edata->leaf_data->values, 0, n_compressed_leaves*metadata->output_dim * sizeof(float));
    selective_copy(n_compressed_leaves, leaf_indices, edata->leaf_data->values, other_edata->leaf_data->values, metadata->output_dim);
    edata->leaf_data->edge_weights = new float[n_compressed_leaves * metadata->max_depth];
    data_size += sizeof(float) * n_compressed_leaves * metadata->max_depth;
    memset(edata->leaf_data->edge_weights, 0, n_compressed_leaves * metadata->max_depth * sizeof(float));
    selective_copy(n_compressed_leaves, leaf_indices, edata->leaf_data->edge_weights, other_edata->leaf_data->edge_weights, metadata->max_depth);
    
    // Feature data
    edata->feature_data->feature_indices = new int[split_sizes * metadata->max_depth];
    data_size += sizeof(int) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->feature_indices, 0, split_sizes*metadata->max_depth * sizeof(int));
    selective_copy(split_sizes, split_indices, edata->feature_data->feature_indices, other_edata->feature_data->feature_indices, metadata->max_depth);
    edata->feature_data->feature_values = new float[split_sizes * metadata->max_depth];
    data_size += sizeof(float) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->feature_values, 0, split_sizes * metadata->max_depth * sizeof(float));
    selective_copy(split_sizes, split_indices, edata->feature_data->feature_values, other_edata->feature_data->feature_values, metadata->max_depth);
    edata->feature_data->is_numerics = new bool[split_sizes * metadata->max_depth];
    data_size += sizeof(bool) * split_sizes * metadata->max_depth;
    memset(edata->feature_data->is_numerics, 0, split_sizes * metadata->max_depth * sizeof(bool));
    selective_copy(split_sizes, split_indices, edata->feature_data->is_numerics, other_edata->feature_data->is_numerics, metadata->max_depth);
    edata->feature_data->categorical_values = new char[split_sizes * metadata->max_depth * MAX_CHAR_SIZE];
    data_size += sizeof(char) * split_sizes * metadata->max_depth * MAX_CHAR_SIZE;
    memset(edata->feature_data->categorical_values, 0, split_sizes * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE);
    selective_copy_char(split_sizes, split_indices, edata->feature_data->categorical_values, other_edata->feature_data->categorical_values, metadata->max_depth);
    edata->feature_data->inequality_directions = new bool[n_compressed_leaves * metadata->max_depth]; 
    data_size += sizeof(bool) * n_compressed_leaves * metadata->max_depth;
    memset(edata->feature_data->inequality_directions, 0, n_compressed_leaves * metadata->max_depth * sizeof(bool));
    selective_copy(n_compressed_leaves, leaf_indices, edata->feature_data->inequality_directions, other_edata->feature_data->inequality_directions, metadata->max_depth);
    
    // Monotonic constraints - allocate but keep empty for compressed
    edata->mono_constraints->feature_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memcpy(edata->mono_constraints->feature_idx, other_edata->mono_constraints->feature_idx, metadata->n_mono_constraints * sizeof(int));
    edata->mono_constraints->output_idx = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memcpy(edata->mono_constraints->output_idx, other_edata->mono_constraints->output_idx, metadata->n_mono_constraints * sizeof(int));
    edata->mono_constraints->constraint = new int[metadata->n_mono_constraints];
    data_size += sizeof(int) * metadata->n_mono_constraints;
    memcpy(edata->mono_constraints->constraint, other_edata->mono_constraints->constraint, metadata->n_mono_constraints * sizeof(int));
    
    // Multi-objective data
    edata->multi_objective_data->densities = new float[n_compressed_leaves * metadata->n_objs];
    data_size += sizeof(float) * n_compressed_leaves * metadata->n_objs;
    memset(edata->multi_objective_data->densities, 0, n_compressed_leaves * metadata->n_objs * sizeof(float));
    selective_copy(n_compressed_leaves, leaf_indices, edata->multi_objective_data->densities, other_edata->multi_objective_data->densities, metadata->n_objs);
    edata->multi_objective_data->lambda_objs = new float[metadata->n_objs];
    data_size += sizeof(float) * metadata->n_objs;
    memcpy(edata->multi_objective_data->lambda_objs, other_edata->multi_objective_data->lambda_objs, metadata->n_objs * sizeof(float));
    
    metadata->max_trees = n_compressed_trees;
    metadata->max_leaves = n_compressed_leaves;
    metadata->n_leaves = n_compressed_leaves;
    metadata->n_trees = n_compressed_trees;
    edata->alloc_data_size = data_size;
    return edata;
}

void ensemble_data_dealloc(ensembleData *edata){
    delete[] edata->bias;
#ifdef DEBUG
    delete[] edata->n_samples;
#endif
    // Ensemble info
    delete[] edata->ensemble_info->tree_indices;
    delete[] edata->ensemble_info->depths;
    delete edata->ensemble_info;
    
    // Leaf data
    delete[] edata->leaf_data->values;
    delete[] edata->leaf_data->edge_weights;
    delete edata->leaf_data;
    
    // Feature data
    delete[] edata->feature_data->feature_indices;
    delete[] edata->feature_data->feature_values;
    delete[] edata->feature_data->feature_weights;
    delete[] edata->feature_data->is_numerics;
    delete[] edata->feature_data->categorical_values;
    delete[] edata->feature_data->inequality_directions;
    delete edata->feature_data;
    
    // Feature mappings
    delete[] edata->feature_mappings->reverse_num_feature_mapping;
    delete[] edata->feature_mappings->reverse_cat_feature_mapping;
    delete[] edata->feature_mappings->feature_mapping;
    delete[] edata->feature_mappings->mapping_numerics;
    delete edata->feature_mappings;
    
    // Monotonic constraints
    delete[] edata->mono_constraints->feature_idx;
    delete[] edata->mono_constraints->output_idx;
    delete[] edata->mono_constraints->constraint;
    delete edata->mono_constraints;
    
    // Multi-objective data
    delete[] edata->multi_objective_data->densities;
    delete[] edata->multi_objective_data->lambda_objs;
    delete edata->multi_objective_data;
    
    delete edata;
}


void allocate_ensemble_memory(ensembleMetaData *metadata, ensembleData *edata){
    int leaf_idx = metadata->n_leaves, tree_idx = metadata->n_trees; 
    if ((leaf_idx >= metadata->max_leaves) || (tree_idx >= metadata->max_trees)){
        int new_size_leaves = metadata->n_leaves + metadata->max_leaves_batch;
        int new_tree_size = metadata->n_trees + metadata->max_trees_batch;
        metadata->max_leaves = new_size_leaves;
        metadata->max_trees = new_tree_size;
        ensembleData *new_data = ensemble_data_alloc(metadata);
        memcpy(new_data->bias, edata->bias, metadata->output_dim * sizeof(float));
        memcpy(new_data->feature_data->feature_weights, edata->feature_data->feature_weights, metadata->input_dim * sizeof(float));
#ifdef DEBUG
        memcpy(new_data->n_samples, edata->n_samples, leaf_idx * sizeof(int));
#endif 
        memcpy(new_data->leaf_data->values, edata->leaf_data->values, leaf_idx * metadata->output_dim * sizeof(float));
        memcpy(new_data->ensemble_info->tree_indices, edata->ensemble_info->tree_indices, tree_idx * sizeof(int));
        memcpy(new_data->feature_data->inequality_directions, edata->feature_data->inequality_directions, leaf_idx * metadata->max_depth * sizeof(bool));
        memcpy(new_data->leaf_data->edge_weights, edata->leaf_data->edge_weights, leaf_idx * sizeof(float));
        memcpy(new_data->multi_objective_data->densities, edata->multi_objective_data->densities, leaf_idx * metadata->n_objs * sizeof(float));
        memcpy(new_data->feature_mappings->reverse_cat_feature_mapping, edata->feature_mappings->reverse_cat_feature_mapping, metadata->input_dim * sizeof(int));
        memcpy(new_data->feature_mappings->reverse_num_feature_mapping, edata->feature_mappings->reverse_num_feature_mapping, metadata->input_dim * sizeof(int));
        memcpy(new_data->feature_mappings->feature_mapping, edata->feature_mappings->feature_mapping, metadata->input_dim * sizeof(int));
        // monotonic constraints
        memcpy(new_data->mono_constraints->feature_idx, edata->mono_constraints->feature_idx, metadata->n_mono_constraints * sizeof(int));
        memcpy(new_data->mono_constraints->output_idx, edata->mono_constraints->output_idx, metadata->n_mono_constraints * sizeof(int));
        memcpy(new_data->mono_constraints->constraint, edata->mono_constraints->constraint, metadata->n_mono_constraints * sizeof(int));
        memcpy(new_data->feature_mappings->mapping_numerics, edata->feature_mappings->mapping_numerics, metadata->input_dim * sizeof(bool));
        memcpy(new_data->multi_objective_data->lambda_objs, edata->multi_objective_data->lambda_objs, metadata->n_objs * sizeof(float));
        if (metadata->grow_policy == GREEDY){
            memcpy(new_data->ensemble_info->depths, edata->ensemble_info->depths, leaf_idx * sizeof(int));
            memcpy(new_data->feature_data->feature_indices, edata->feature_data->feature_indices, leaf_idx * metadata->max_depth * sizeof(int));
            memcpy(new_data->feature_data->feature_values, edata->feature_data->feature_values, leaf_idx * metadata->max_depth * sizeof(float));
            memcpy(new_data->feature_data->is_numerics, edata->feature_data->is_numerics, leaf_idx * metadata->max_depth * sizeof(bool));
            memcpy(new_data->feature_data->categorical_values, edata->feature_data->categorical_values, leaf_idx * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE);
        } else {
            memcpy(new_data->ensemble_info->depths, edata->ensemble_info->depths, tree_idx * sizeof(int));
            memcpy(new_data->feature_data->feature_indices, edata->feature_data->feature_indices, tree_idx * metadata->max_depth * sizeof(int));
            memcpy(new_data->feature_data->feature_values, edata->feature_data->feature_values, tree_idx * metadata->max_depth * sizeof(float));
            memcpy(new_data->feature_data->is_numerics, edata->feature_data->is_numerics, tree_idx * metadata->max_depth * sizeof(bool));
            memcpy(new_data->feature_data->categorical_values, edata->feature_data->categorical_values, tree_idx * metadata->max_depth * sizeof(char) * MAX_CHAR_SIZE);
        }
        delete[] edata->bias;
        delete[] edata->feature_data->feature_weights;
#ifdef DEBUG
        delete [] edata->n_samples;
#endif
        delete[] edata->ensemble_info->depths;
        delete[] edata->leaf_data->values;
        // monotonic constraints
        delete[] edata->mono_constraints->feature_idx;
        delete[] edata->mono_constraints->output_idx;
        delete[] edata->mono_constraints->constraint;
        // leaf data
        delete[] edata->feature_data->feature_indices;
        delete[] edata->ensemble_info->tree_indices;
        delete[] edata->feature_data->feature_values;
        delete[] edata->leaf_data->edge_weights;
        delete[] edata->multi_objective_data->densities;
        delete[] edata->feature_data->is_numerics;
        delete[] edata->feature_data->categorical_values;
        delete[] edata->feature_data->inequality_directions; 
        delete[] edata->feature_mappings->reverse_cat_feature_mapping;
        delete[] edata->feature_mappings->feature_mapping;
        delete[] edata->feature_mappings->reverse_num_feature_mapping;
        delete[] edata->feature_mappings->mapping_numerics;
        delete[] edata->multi_objective_data->lambda_objs;

        edata->bias = new_data->bias;
        edata->feature_data->feature_weights = new_data->feature_data->feature_weights;
#ifdef DEBUG
        edata->n_samples = new_data->n_samples;
#endif
        edata->ensemble_info->depths = new_data->ensemble_info->depths;
        edata->ensemble_info->tree_indices = new_data->ensemble_info->tree_indices;
        edata->leaf_data->values = new_data->leaf_data->values;
        edata->feature_data->inequality_directions = new_data->feature_data->inequality_directions;
        edata->feature_data->feature_indices = new_data->feature_data->feature_indices;
        edata->feature_data->feature_values = new_data->feature_data->feature_values;
        edata->leaf_data->edge_weights = new_data->leaf_data->edge_weights;
        edata->multi_objective_data->densities = new_data->multi_objective_data->densities;
        edata->mono_constraints->feature_idx = new_data->mono_constraints->feature_idx;
        edata->mono_constraints->output_idx = new_data->mono_constraints->output_idx;
        edata->mono_constraints->constraint = new_data->mono_constraints->constraint;
        edata->feature_mappings->reverse_cat_feature_mapping = new_data->feature_mappings->reverse_cat_feature_mapping;
        edata->feature_mappings->reverse_num_feature_mapping = new_data->feature_mappings->reverse_num_feature_mapping;
        edata->feature_mappings->feature_mapping = new_data->feature_mappings->feature_mapping;
        edata->feature_mappings->mapping_numerics = new_data->feature_mappings->mapping_numerics;
        edata->feature_data->is_numerics = new_data->feature_data->is_numerics;
        edata->feature_data->categorical_values = new_data->feature_data->categorical_values;
        edata->multi_objective_data->lambda_objs = new_data->multi_objective_data->lambda_objs;
        
        // Delete only the sub-struct containers and ensembleData, not the underlying arrays (they're now owned by edata)
        delete new_data->ensemble_info;
        delete new_data->leaf_data;
        delete new_data->feature_data;
        delete new_data->mono_constraints;
        delete new_data->feature_mappings;
        delete new_data->multi_objective_data;
        delete new_data;
    }
}