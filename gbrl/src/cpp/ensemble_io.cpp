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
 * @file ensemble_io.cpp
 * @brief Implementation of ensemble export, serialization, and tree extract/insert
 */

#include <stdexcept>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

#include "ensemble_io.h"
#include "types.h"
#include "utils.h"
#include "optimizer.h"
#ifdef USE_CUDA
#include "cuda_types.h"
#endif

// ============================================================================
// Tree Data Deallocation
// ============================================================================

/**
 * @brief Deallocate all memory owned by a treeData struct
 *
 * Frees every array pointer inside the struct and then deletes the struct itself.
 */
void tree_data_dealloc(treeData *tdata){
    if (tdata == nullptr) return;
    delete[] tdata->depths;
    delete[] tdata->values;
    delete[] tdata->edge_weights;
    delete[] tdata->feature_indices;
    delete[] tdata->feature_values;
    delete[] tdata->is_numerics;
    delete[] tdata->inequality_directions;
    delete[] tdata->categorical_values;
    delete[] tdata->densities;
    delete tdata;
}

// ============================================================================
// Tree Extraction
// ============================================================================

/**
 * @brief Extract all data for a single tree from the ensemble
 *
 * Allocates a treeData struct and copies out the relevant slices of the
 * ensemble arrays for the requested tree. Works for both GREEDY and OBLIVIOUS
 * grow policies with their different indexing schemes.
 *
 * For GREEDY: split info (feature_indices, feature_values, is_numerics,
 *   categorical_values) is indexed per-leaf (split_count = n_leaves_in_tree).
 * For OBLIVIOUS: split info is indexed per-tree (split_count = 1),
 *   but inequality_directions and edge_weights are still per-leaf.
 */
treeData* get_tree_data(int tree_idx, ensembleMetaData *metadata, ensembleData *edata, deviceType device){
    if (tree_idx < 0 || tree_idx >= metadata->n_trees){
        std::cerr << "Error: tree_idx " << tree_idx << " out of range [0, " << metadata->n_trees << ")" << std::endl;
        throw std::runtime_error("tree_idx out of range");
    }

    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(metadata, edata, nullptr);
    }
#endif
    if (device == cpu)
        edata_cpu = edata;

    // Determine leaf range for this tree
    int leaf_start = edata_cpu->ensemble_info->tree_indices[tree_idx];
    int n_leaves_in_tree;
    if (tree_idx < metadata->n_trees - 1)
        n_leaves_in_tree = edata_cpu->ensemble_info->tree_indices[tree_idx + 1] - leaf_start;
    else
        n_leaves_in_tree = metadata->n_leaves - leaf_start;

    bool is_oblivious = (metadata->grow_policy == OBLIVIOUS);
    int max_depth = metadata->max_depth;
    int output_dim = metadata->output_dim;
    int n_objs = metadata->n_objs;

    // For OBLIVIOUS: split arrays indexed by tree; for GREEDY: by leaf
    int split_count = is_oblivious ? 1 : n_leaves_in_tree;

    treeData *tdata = new treeData;
    tdata->n_leaves = n_leaves_in_tree;
    tdata->output_dim = output_dim;
    tdata->max_depth = max_depth;
    tdata->n_objs = n_objs;
    tdata->split_count = split_count;
    tdata->is_oblivious = is_oblivious;

    // --- Depths ---
    if (is_oblivious){
        tdata->depths = new int[1];
        tdata->depths[0] = edata_cpu->ensemble_info->depths[tree_idx];
        tdata->tree_depth = tdata->depths[0];
    } else {
        tdata->depths = new int[n_leaves_in_tree];
        memcpy(tdata->depths, &edata_cpu->ensemble_info->depths[leaf_start], n_leaves_in_tree * sizeof(int));
        tdata->tree_depth = 0;
        for (int i = 0; i < n_leaves_in_tree; ++i){
            if (tdata->depths[i] > tdata->tree_depth)
                tdata->tree_depth = tdata->depths[i];
        }
    }

    // --- Leaf values: [n_leaves_in_tree * output_dim] ---
    tdata->values = new float[n_leaves_in_tree * output_dim];
    memcpy(tdata->values, &edata_cpu->leaf_data->values[leaf_start * output_dim],
           n_leaves_in_tree * output_dim * sizeof(float));

    // --- Edge weights: [n_leaves_in_tree * max_depth] ---
    tdata->edge_weights = new float[n_leaves_in_tree * max_depth];
    memcpy(tdata->edge_weights, &edata_cpu->leaf_data->edge_weights[leaf_start * max_depth],
           n_leaves_in_tree * max_depth * sizeof(float));

    // --- Inequality directions: [n_leaves_in_tree * max_depth] ---
    tdata->inequality_directions = new bool[n_leaves_in_tree * max_depth];
    memcpy(tdata->inequality_directions, &edata_cpu->feature_data->inequality_directions[leaf_start * max_depth],
           n_leaves_in_tree * max_depth * sizeof(bool));

    // --- Split-indexed arrays ---
    int split_src_start = is_oblivious ? tree_idx : leaf_start;
    tdata->feature_indices = new int[split_count * max_depth];
    memcpy(tdata->feature_indices, &edata_cpu->feature_data->feature_indices[split_src_start * max_depth],
           split_count * max_depth * sizeof(int));

    tdata->feature_values = new float[split_count * max_depth];
    memcpy(tdata->feature_values, &edata_cpu->feature_data->feature_values[split_src_start * max_depth],
           split_count * max_depth * sizeof(float));

    tdata->is_numerics = new bool[split_count * max_depth];
    memcpy(tdata->is_numerics, &edata_cpu->feature_data->is_numerics[split_src_start * max_depth],
           split_count * max_depth * sizeof(bool));

    tdata->categorical_values = new char[split_count * max_depth * MAX_CHAR_SIZE];
    memcpy(tdata->categorical_values, &edata_cpu->feature_data->categorical_values[split_src_start * max_depth * MAX_CHAR_SIZE],
           split_count * max_depth * MAX_CHAR_SIZE * sizeof(char));

    // --- Multi-objective densities: [n_leaves_in_tree * n_objs] ---
    tdata->densities = new float[n_leaves_in_tree * n_objs];
    memcpy(tdata->densities, &edata_cpu->multi_objective_data->densities[leaf_start * n_objs],
           n_leaves_in_tree * n_objs * sizeof(float));

#ifdef USE_CUDA
    if (device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif

    return tdata;
}

// ============================================================================
// Tree Insertion (CPU + GPU)
// ============================================================================

/**
 * @brief Helper: insert tree data into CPU ensemble arrays at the given offsets
 */
static void add_tree_data_cpu(const treeData *tdata, ensembleMetaData *metadata,
                              ensembleData *edata)
{
    bool is_oblivious = (metadata->grow_policy == OBLIVIOUS);
    int n_leaves_in_tree = tdata->n_leaves;
    int max_depth = metadata->max_depth;
    int output_dim = metadata->output_dim;
    int n_objs = metadata->n_objs;

    // Ensure enough capacity
    while (metadata->n_leaves + n_leaves_in_tree > metadata->max_leaves ||
           metadata->n_trees + 1 > metadata->max_trees){
        allocate_ensemble_memory(metadata, edata);
    }

    int leaf_start = metadata->n_leaves;
    int tree_idx = metadata->n_trees;

    edata->ensemble_info->tree_indices[tree_idx] = leaf_start;

    // Depths
    if (is_oblivious){
        edata->ensemble_info->depths[tree_idx] = tdata->depths[0];
    } else {
        memcpy(&edata->ensemble_info->depths[leaf_start], tdata->depths,
               n_leaves_in_tree * sizeof(int));
    }

    // Leaf values
    memcpy(&edata->leaf_data->values[leaf_start * output_dim], tdata->values,
           n_leaves_in_tree * output_dim * sizeof(float));

    // Edge weights
    memcpy(&edata->leaf_data->edge_weights[leaf_start * max_depth], tdata->edge_weights,
           n_leaves_in_tree * max_depth * sizeof(float));

    // Inequality directions
    memcpy(&edata->feature_data->inequality_directions[leaf_start * max_depth], tdata->inequality_directions,
           n_leaves_in_tree * max_depth * sizeof(bool));

    // Split-indexed arrays
    int split_dst_start = is_oblivious ? tree_idx : leaf_start;
    int split_count = tdata->split_count;

    memcpy(&edata->feature_data->feature_indices[split_dst_start * max_depth], tdata->feature_indices,
           split_count * max_depth * sizeof(int));
    memcpy(&edata->feature_data->feature_values[split_dst_start * max_depth], tdata->feature_values,
           split_count * max_depth * sizeof(float));
    memcpy(&edata->feature_data->is_numerics[split_dst_start * max_depth], tdata->is_numerics,
           split_count * max_depth * sizeof(bool));
    memcpy(&edata->feature_data->categorical_values[split_dst_start * max_depth * MAX_CHAR_SIZE], tdata->categorical_values,
           split_count * max_depth * MAX_CHAR_SIZE * sizeof(char));

    // Multi-objective densities
    memcpy(&edata->multi_objective_data->densities[leaf_start * n_objs], tdata->densities,
           n_leaves_in_tree * n_objs * sizeof(float));

    // Update metadata
    metadata->n_leaves += n_leaves_in_tree;
    metadata->n_trees += 1;
    metadata->iteration += 1;
}

void add_tree_data(const treeData *tdata, ensembleMetaData *metadata, ensembleData *edata, deviceType device){
    if (tdata == nullptr){
        throw std::runtime_error("treeData is nullptr");
    }

    bool is_oblivious = (metadata->grow_policy == OBLIVIOUS);
    if (is_oblivious != tdata->is_oblivious){
        throw std::runtime_error("treeData grow policy does not match ensemble grow policy");
    }
    if (tdata->output_dim != metadata->output_dim){
        throw std::runtime_error("treeData output_dim does not match ensemble output_dim");
    }
    if (tdata->max_depth != metadata->max_depth){
        throw std::runtime_error("treeData max_depth does not match ensemble max_depth");
    }

    if (device == cpu){
        add_tree_data_cpu(tdata, metadata, edata);
        return;
    }

#ifdef USE_CUDA
    // GPU path: ensure capacity, then cudaMemcpy tree slices from host treeData
    int n_leaves_in_tree = tdata->n_leaves;
    int max_depth = metadata->max_depth;
    int output_dim = metadata->output_dim;
    int n_objs = metadata->n_objs;

    while (metadata->n_leaves + n_leaves_in_tree > metadata->max_leaves ||
           metadata->n_trees + 1 > metadata->max_trees){
        allocate_ensemble_memory_cuda(metadata, edata);
    }

    int leaf_start = metadata->n_leaves;
    int tree_idx = metadata->n_trees;

    // tree_indices: single int written from host
    cudaMemcpy(&edata->ensemble_info->tree_indices[tree_idx], &leaf_start,
               sizeof(int), cudaMemcpyHostToDevice);

    // Depths
    if (is_oblivious){
        cudaMemcpy(&edata->ensemble_info->depths[tree_idx], tdata->depths,
                   sizeof(int), cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(&edata->ensemble_info->depths[leaf_start], tdata->depths,
                   n_leaves_in_tree * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Leaf values
    cudaMemcpy(&edata->leaf_data->values[leaf_start * output_dim], tdata->values,
               n_leaves_in_tree * output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Edge weights
    cudaMemcpy(&edata->leaf_data->edge_weights[leaf_start * max_depth], tdata->edge_weights,
               n_leaves_in_tree * max_depth * sizeof(float), cudaMemcpyHostToDevice);

    // Inequality directions
    cudaMemcpy(&edata->feature_data->inequality_directions[leaf_start * max_depth], tdata->inequality_directions,
               n_leaves_in_tree * max_depth * sizeof(bool), cudaMemcpyHostToDevice);

    // Split-indexed arrays
    int split_dst_start = is_oblivious ? tree_idx : leaf_start;
    int split_count = tdata->split_count;

    cudaMemcpy(&edata->feature_data->feature_indices[split_dst_start * max_depth], tdata->feature_indices,
               split_count * max_depth * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&edata->feature_data->feature_values[split_dst_start * max_depth], tdata->feature_values,
               split_count * max_depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&edata->feature_data->is_numerics[split_dst_start * max_depth], tdata->is_numerics,
               split_count * max_depth * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(&edata->feature_data->categorical_values[split_dst_start * max_depth * MAX_CHAR_SIZE], tdata->categorical_values,
               split_count * max_depth * MAX_CHAR_SIZE * sizeof(char), cudaMemcpyHostToDevice);

    // Multi-objective densities
    cudaMemcpy(&edata->multi_objective_data->densities[leaf_start * n_objs], tdata->densities,
               n_leaves_in_tree * n_objs * sizeof(float), cudaMemcpyHostToDevice);

    // Update metadata
    metadata->n_leaves += n_leaves_in_tree;
    metadata->n_trees += 1;
    metadata->iteration += 1;
#else
    throw std::runtime_error("add_tree_data: GPU device requested but CUDA not compiled");
#endif
}

// ============================================================================
// Export Data
// ============================================================================

/**
 * @brief Extract simplified export data for inference
 *
 * Copies tree structure (split feature indices and thresholds) and computes
 * optimizer-scaled leaf values suitable for direct inference. The total
 * number of binary split nodes (binary_features) is the sum of tree depths.
 * Leaf values are negated and multiplied by each optimizer's per-tree
 * learning rate to produce final inference-ready predictions.
 */
exportData* get_export_data(ensembleMetaData *metadata, ensembleData *edata, deviceType device, std::vector<Optimizer*> opts){
    exportData *exp_data = new exportData;

    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(metadata, edata, nullptr);
    }
#endif 
    if (device == cpu)
        edata_cpu = edata;
    
    int binary_features = 0;
    for (int i  = 0; i < metadata->n_trees; ++i){
        binary_features += edata_cpu->ensemble_info->depths[i];
    }

    exp_data->input_dim = metadata->input_dim;
    exp_data->output_dim = metadata->output_dim;
    exp_data->n_trees = metadata->n_trees;
    exp_data->n_leaves = metadata->n_leaves;
    exp_data->max_depth = metadata->max_depth;
    exp_data->num_features = metadata->n_num_features;
    exp_data->binary_features = binary_features;

    exp_data->feature_indices = new int[binary_features];
    exp_data->feature_values = new float[binary_features];
    exp_data->leaf_values = new float[metadata->n_leaves * metadata->output_dim];
    exp_data->bias = new float[metadata->output_dim];
    memcpy(exp_data->bias, edata_cpu->bias, metadata->output_dim * sizeof(float));

    for (int i  = 0; i < binary_features; ++i){
        exp_data->feature_indices[i] = edata_cpu->feature_data->feature_indices[i];
        exp_data->feature_values[i] = edata_cpu->feature_data->feature_values[i];
    }

    int tree_idx = 0;
    int limit_leaf_idx = edata_cpu->ensemble_info->tree_indices[tree_idx];
    float value;
    for (int i  = 0; i < metadata->n_leaves; ++i){
        if (i > limit_leaf_idx){
            tree_idx += 1;
            limit_leaf_idx = edata_cpu->ensemble_info->tree_indices[tree_idx];
        }
        int value_idx = i*metadata->output_dim;
        for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
            for (int j=opts[opt_idx]->start_idx; j < opts[opt_idx]->stop_idx; ++j){
                value = -edata_cpu->leaf_data->values[value_idx + j] * opts[opt_idx]->scheduler->get_lr(tree_idx);
                exp_data->leaf_values[value_idx + j] = value;
            }
        }
    }

#ifdef USE_CUDA
    if (device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 

    return exp_data;
}

// ============================================================================
// C-Header Export
// ============================================================================

void export_ensemble_data(std::ofstream& header_file, const std::string& model_name, ensembleData *edata, ensembleMetaData *metadata, deviceType device, std::vector<Optimizer*> opts, exportFormat export_format, exportType export_type, const std::string &prefix)
{
    std::string type_name;
    switch (export_format){
       case EXP_FLOAT:
            type_name = "float";
            break;
        case EXP_FXP8:
            type_name = "int16";
            break;
        case EXP_FXP16:
            type_name = "int32";
            break;
        default:
            std::cerr << "Invalid exportFormat!" << std::endl;
            return; // Exit the function if the format is invalid
    }

    if (export_type == exportType::COMPACT){
        if (metadata->max_depth > 6 || metadata->grow_policy != growPolicy::OBLIVIOUS){
            std::cerr << "Cannot only compact export with max depth <= 6 and oblivious trees" << std::endl;
            return; // Exit the function if the format is invalid
        }
    }
    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(metadata, edata, nullptr);
    }
#endif 
    if (device == cpu)
        edata_cpu = edata;
    
    int binary_splits = 0;
    for (int i  = 0; i < metadata->n_trees; ++i){
        binary_splits += edata_cpu->ensemble_info->depths[i];
    }

    for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
        optimizerAlgo algo = opts[opt_idx]->getAlgo();
        if (algo != SGD){
            std::cerr << "Error. Can only export SGD optimizers" << std::endl;
            header_file.close();
            throw std::runtime_error("Error. Can only export SGD optimizers");
            return;
        }
    }

    header_file << "#ifndef GBRL_MODEL_H\n";
    header_file << "#define GBRL_MODEL_H\n\n";

    
    header_file << "/*\n";

    if (!model_name.empty()) {
           header_file << "###########################\n";
        header_file << "model_name: " << model_name << "\n";
    }
    header_file << "###########################\n";
    header_file << "n_leaves: " << metadata->n_leaves << ", ";
    header_file << "n_trees: " << metadata->n_trees << ", ";
    header_file << "max_trees: " << metadata->max_trees << ", ";
    header_file << "max_leaves: " << metadata->max_leaves << ", ";
    header_file << "max_trees_batch: " << metadata->max_trees_batch << ", ";
    header_file << "max_leaves_batch: " << metadata->max_leaves_batch << ", ";
    header_file << "input_dim: " << metadata->input_dim << ", ";
    header_file << "output_dim: " << metadata->output_dim << ", ";
    header_file << "policy_dim: " << metadata->policy_dim << ", ";
    header_file << "\nmax_depth: " << metadata->max_depth << ", ";
    header_file << "min_data_in_leaf: " << metadata->min_data_in_leaf << ", ";
    header_file << "n_bins: " << metadata->n_bins << ", ";
    header_file << "par_th: " << metadata->par_th << ", ";
    header_file << "cv_beta: " << metadata->cv_beta << ", ";
    header_file << "verbose: " << metadata->verbose << ", ";
    header_file << "batch_size: " << metadata->batch_size << ", ";
    header_file << "lambda_objs: [";
    for (int i  = 0; i < metadata->n_objs; ++i){
        header_file << edata_cpu->multi_objective_data->lambda_objs[i];;
        if (i < metadata->n_objs - 1)
            header_file << ", ";
    }
    header_file << "], n_objs: " << metadata->n_objs << ", ";
    header_file << "use_cv: " << metadata->use_cv;
    header_file << "\nsplit_score_func: " << scoreFuncToString(metadata->split_score_func) << ", ";
    header_file << "generator_type: " << generatorTypeToString(metadata->generator_type) << ", ";
    header_file << "grow_policy: " << growPolicyToString(metadata->grow_policy) << ", ";
    header_file << "n_num_features: " << metadata->n_num_features << ", ";
    header_file << "n_cat_features: " << metadata->n_cat_features << ", ";
    header_file << "iteration: " << metadata->iteration;
    header_file << "alloc_data_size: " << edata->alloc_data_size;
    header_file << "\n*/\n";

    header_file << "#define " << prefix << "N_TREES " << metadata->n_trees << "\n";
    header_file << "#define " << prefix << "N_LEAVES " << metadata->n_leaves << "\n";
    header_file << "#define " << prefix << "BINARY_FEATURES " << binary_splits << "\n";
    header_file << "#define N_INPUTS " << metadata->input_dim << "\n";
    header_file << "#define " << prefix << "N_OUTPUTS " << metadata->output_dim << "\n";
    header_file << "#define " << prefix << "NUM_FEATURES " << metadata->n_num_features  << "\n\n";
    if (metadata->output_dim > 1){
        header_file << "static inline void gbrl_predict(" << type_name << " *results, const " << type_name << " *features){\n\n";
    } else {
        header_file << "static inline " << type_name << " gbrl_predict(const " << type_name << " *features){\n\n";
        header_file << "\t" << type_name << " result = ";
        switch (export_format){
            case EXP_FLOAT:
                header_file << "0.0f;\n";
                break;
            case EXP_FXP8:
                header_file << "0;\n";
                break;
            case EXP_FXP16:
                header_file << "0;\n";
                break;
        }
    }
    
    header_file << "\tunsigned int tree_idx, idx, leaf_ptr, cond_ptr";
    if (metadata->output_dim > 1){
        header_file << ", j";
    }
    if (export_type == exportType::FULL)
        header_file << ", depth, current_depth";
    header_file << ";\n";
    header_file << "\t/* Model data */\n";
    if (export_type == exportType::FULL){
        header_file << "\tconst unsigned int depths[" << prefix << "N_TREES] = {";
        for (int i  = 0; i < metadata->n_trees; ++i){
            header_file << edata_cpu->ensemble_info->depths[i];
            if (i < metadata->n_trees - 1)
                header_file << ", ";
        }
        header_file << "};\n";
    }
    if (metadata->output_dim > 1){
        header_file << "\tconst " << type_name << " bias[" << prefix << "N_OUTPUTS] = {";
        for (int i  = 0; i < metadata->output_dim; ++i){
            switch (export_format){
                case EXP_FLOAT:
                    header_file << edata_cpu->bias[i];
                    break;
                case EXP_FXP8:
                    header_file << float_to_int16(edata_cpu->bias[i]);
                    break;
                case EXP_FXP16:
                    header_file << float_to_int32(edata_cpu->bias[i]);
                    break;
            }
            if (i < metadata->output_dim - 1)
                header_file << ", ";
        }

    } else {
        header_file << "\tconst " << type_name << " bias = ";
        switch (export_format){
            case EXP_FLOAT:
                header_file << edata_cpu->bias[0];
                break;
            case EXP_FXP8:
                header_file << float_to_int16(edata_cpu->bias[0]);
                break;
            case EXP_FXP16:
                header_file << float_to_int32(edata_cpu->bias[0]);
                break;
        }
    }
    header_file << ";\n";

    std::string max_size = (metadata->input_dim < 255) ? "uint8" : "uint16";
    header_file << "\tconst " << max_size <<  " feature_indices[" << prefix << "BINARY_FEATURES] = {";
    for (int i  = 0; i < binary_splits; ++i){
        header_file << edata_cpu->feature_data->feature_indices[i];
        if (i < binary_splits - 1)
            header_file << ", ";
    }
    header_file << "};\n";

    header_file << "\tconst " << type_name << " feature_values[" << prefix << "BINARY_FEATURES] = {";
    for (int i  = 0; i < binary_splits; ++i){
        switch (export_format){
            case EXP_FLOAT:
                header_file << edata_cpu->feature_data->feature_values[i];
                break;
            case EXP_FXP8:
                   header_file << float_to_int16(edata_cpu->feature_data->feature_values[i]);
                break;
            case EXP_FXP16:
                   header_file << float_to_int32(edata_cpu->feature_data->feature_values[i]);
                break;
        }
        if (i < binary_splits - 1)
            header_file << ", ";
    }
    header_file << "};\n";

    header_file << "\tconst " << type_name << " leaf_values[" << prefix << "N_LEAVES*" << prefix << "N_OUTPUTS]  = {";
    int tree_idx = 0;
    int limit_leaf_idx = edata_cpu->ensemble_info->tree_indices[tree_idx];
    float value;
    for (int i  = 0; i < metadata->n_leaves; ++i){
        if (i > limit_leaf_idx){
            tree_idx += 1;
            limit_leaf_idx = edata_cpu->ensemble_info->tree_indices[tree_idx];
        }
        int value_idx = i*metadata->output_dim;
        for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
            for (int j=opts[opt_idx]->start_idx; j < opts[opt_idx]->stop_idx; ++j){
                value = -edata_cpu->leaf_data->values[value_idx + j] * opts[opt_idx]->scheduler->get_lr(tree_idx);
                switch (export_format){
                    case EXP_FLOAT:
                        header_file << value;
                        break;
                    case EXP_FXP8:
                        header_file << float_to_int16(value);
                        break;
                    case EXP_FXP16:
                        header_file << float_to_int32(value);
                        break;
                }
                if ((i < metadata->n_leaves - 1) || (j < metadata->output_dim - 1  && i == metadata->n_leaves - 1))
                    header_file << ", ";
            }
        }
    }
    header_file << "};\n";

    header_file << "\tleaf_ptr = 0;\n";
    header_file << "\tcond_ptr = 0;\n";
    header_file << "\tunsigned char pass;\n";
    header_file << "\tfor (tree_idx = 0; tree_idx < " << prefix << "N_TREES; ++tree_idx)\n";
    header_file << "\t{\n";
    if (export_type == exportType::COMPACT){
        header_file << "\t\tidx = 0;\n";
        for (int depth = 0; depth < metadata->max_depth; ++depth){
            
            header_file << "\t\tpass = (unsigned char)(features[feature_indices[cond_ptr + " << depth << "]] > feature_values[cond_ptr + " << depth << "]);\n";
            header_file << "\t\tidx |= (pass <<  (" << metadata->max_depth << " - 1 - " << depth << "));\n";
        }
    } else {
        header_file << "\t\tcurrent_depth = depths[tree_idx];\n";
        header_file << "\t\tidx = 0;\n";
        header_file << "\t\tfor (depth = 0; depth < current_depth; ++depth){\n";
        header_file << "\t\t\tpass = (unsigned char)(features[feature_indices[cond_ptr + depth]] > feature_values[cond_ptr + depth]);\n";
        header_file << "\t\t\tidx |= (pass <<  (current_depth - 1 - depth));\n";
        header_file << "\t\t}\n";
    }
    
    if (metadata->output_dim > 1){
        header_file << "\t\tfor (j = 0 ; j < " << prefix << "N_OUTPUTS; j++)\n";
        header_file << "\t\t\tresults[j] += leaf_values[(leaf_ptr + idx)*" << prefix << "N_OUTPUTS + j];\n";
    } else {
        header_file << "\t\tresult += leaf_values[leaf_ptr + idx];\n";
    }

    if (export_type == exportType::COMPACT){
        int tmp = 1 << metadata->max_depth;
        header_file << "\t\tleaf_ptr += " << tmp << ";\n";
        header_file << "\t\tcond_ptr += " << metadata->max_depth << ";\n";
    } else {
        header_file << "\t\tleaf_ptr += (1 << current_depth);\n";
        header_file << "\t\tcond_ptr += current_depth;\n";
    }
    header_file << "\t}\n";
    if (metadata->output_dim > 1){
        header_file << "\tfor (j = 0 ; j < " << prefix << "N_OUTPUTS; j++)\n";
        header_file << "\t\tresults[j] += bias[j];\n";
    } else {
        header_file << "\tresult += bias;\n";
        header_file << "\treturn result;\n";
    }
    header_file << "}\n";
    header_file << "#endif\n";

#ifdef USE_CUDA
    if (device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 
}

// ============================================================================
// Binary Serialization
// ============================================================================

void save_ensemble_data(std::ofstream& file, ensembleData *edata, ensembleMetaData *metadata, deviceType device){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        throw std::runtime_error("Error opening file");
    }
    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(metadata, edata, nullptr);
    }
#endif 
    if (device == cpu)
        edata_cpu = edata;
    NULL_CHECK check = edata_cpu->bias != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->bias != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->bias), metadata->output_dim * sizeof(float));
    check = edata_cpu->feature_data->feature_weights != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_data->feature_weights != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_data->feature_weights), metadata->input_dim * sizeof(float));
#ifdef DEBUG
    check = edata_cpu->n_samples != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->n_samples != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->n_samples), metadata->n_leaves * sizeof(int));
#endif 
    check = edata_cpu->ensemble_info->tree_indices != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->ensemble_info->tree_indices != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->ensemble_info->tree_indices), metadata->n_trees * sizeof(int));
    size_t sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees: metadata->n_leaves;
    check = edata_cpu->ensemble_info->depths != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->ensemble_info->depths != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->ensemble_info->depths), sizes * sizeof(int));
    check = edata_cpu->leaf_data->values != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->leaf_data->values != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->leaf_data->values), metadata->n_leaves * metadata->output_dim * sizeof(float));
    check = edata_cpu->feature_data->feature_indices != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_data->feature_indices != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_data->feature_indices), metadata->max_depth * sizes * sizeof(int));
    check = edata_cpu->feature_data->feature_values != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_data->feature_values != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_data->feature_values), metadata->max_depth * sizes * sizeof(float));
    check = edata_cpu->leaf_data->edge_weights != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->leaf_data->edge_weights != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->leaf_data->edge_weights), metadata->max_depth * metadata->n_leaves * sizeof(float));
    check = edata_cpu->multi_objective_data->densities != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->multi_objective_data->densities != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->multi_objective_data->densities), metadata->n_leaves * metadata->n_objs * sizeof(float));
    // monotonic constraints
    check = edata_cpu->mono_constraints->feature_idx != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->mono_constraints->feature_idx != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->mono_constraints->feature_idx), metadata->n_mono_constraints * sizeof(int));
    check = edata_cpu->mono_constraints->output_idx != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->mono_constraints->output_idx != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->mono_constraints->output_idx), metadata->n_mono_constraints * sizeof(int));
    check = edata_cpu->mono_constraints->constraint != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->mono_constraints->constraint != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->mono_constraints->constraint), metadata->n_mono_constraints * sizeof(int));    

    check = edata_cpu->feature_mappings->reverse_num_feature_mapping != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_mappings->reverse_num_feature_mapping != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_mappings->reverse_num_feature_mapping), metadata->input_dim * sizeof(int));
    check = edata_cpu->feature_mappings->reverse_cat_feature_mapping != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_mappings->reverse_cat_feature_mapping != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_mappings->reverse_cat_feature_mapping), metadata->input_dim * sizeof(int));
    check = edata_cpu->feature_mappings->feature_mapping != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_mappings->feature_mapping != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_mappings->feature_mapping), metadata->input_dim * sizeof(int));
    check = edata_cpu->feature_mappings->mapping_numerics != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_mappings->mapping_numerics != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_mappings->mapping_numerics), metadata->input_dim * sizeof(bool));
    check = edata_cpu->feature_data->is_numerics != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_data->is_numerics != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_data->is_numerics), metadata->max_depth * sizes * sizeof(bool));
    check = edata_cpu->feature_data->inequality_directions != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_data->inequality_directions != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_data->inequality_directions), metadata->max_depth * metadata->n_leaves * sizeof(bool));
    check = edata_cpu->feature_data->categorical_values != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_data->categorical_values != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_data->categorical_values), metadata->max_depth * sizes * sizeof(char) * MAX_CHAR_SIZE);
    check = edata_cpu->multi_objective_data->lambda_objs != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->multi_objective_data->lambda_objs != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->multi_objective_data->lambda_objs), metadata->n_objs * sizeof(float));

#ifdef USE_CUDA
    if (device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 
}

ensembleData* load_ensemble_data(std::ifstream& file, ensembleMetaData *metadata){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        throw std::runtime_error("Error opening file");
    }
    ensembleData *edata_cpu = ensemble_data_alloc(metadata);
    NULL_CHECK check;
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->bias), metadata->output_dim * sizeof(float));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_data->feature_weights), metadata->input_dim * sizeof(float));
    } 
#ifdef DEBUG
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->n_samples), metadata->n_leaves * sizeof(int));
    } 
#endif 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->ensemble_info->tree_indices), metadata->n_trees * sizeof(int));
    } 
    size_t sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees: metadata->n_leaves;
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->ensemble_info->depths), sizes * sizeof(int));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->leaf_data->values), metadata->output_dim * metadata->n_leaves * sizeof(float));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_data->feature_indices), metadata->max_depth * sizes * sizeof(int));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_data->feature_values), metadata->max_depth * sizes * sizeof(float));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->leaf_data->edge_weights), metadata->max_depth * metadata->n_leaves * sizeof(float));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->multi_objective_data->densities), metadata->n_leaves * metadata->n_objs * sizeof(float));
    }
    // monotonic constraints
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->mono_constraints->feature_idx), metadata->n_mono_constraints * sizeof(int));
    }
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->mono_constraints->output_idx), metadata->n_mono_constraints * sizeof(int));
    }
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->mono_constraints->constraint), metadata->n_mono_constraints * sizeof(int));
    }

    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_mappings->reverse_num_feature_mapping), metadata->input_dim * sizeof(int));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_mappings->reverse_cat_feature_mapping), metadata->input_dim * sizeof(int));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_mappings->feature_mapping), metadata->input_dim * sizeof(int));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_mappings->mapping_numerics), metadata->input_dim * sizeof(bool));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_data->is_numerics), metadata->max_depth * sizes * sizeof(bool));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_data->inequality_directions), metadata->max_depth * metadata->n_leaves * sizeof(bool));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_data->categorical_values), metadata->max_depth * sizes * sizeof(char) * MAX_CHAR_SIZE);
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->multi_objective_data->lambda_objs), metadata->n_objs * sizeof(float));
    } 
    return edata_cpu;
}
