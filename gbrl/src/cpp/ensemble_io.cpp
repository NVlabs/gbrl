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
#include <memory>
#include <algorithm>
#include <cmath>
#include <limits>
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
 * @brief Grow the ensemble so n_new_leaves leaves and one tree fit.
 *
 * allocate_ensemble_memory{,_cuda} only grow when the ensemble is already full
 * (n_leaves >= max_leaves) and only by one batch, so looping on them never makes
 * progress when a tree straddles the boundary or exceeds a batch.  Both copy
 * existing data by n_leaves/n_trees (not by the old capacity), so we can mark
 * the ensemble full and size the batch to the request.
 */
static void reserve_tree_capacity(ensembleMetaData *metadata, ensembleData *edata,
                                  const int n_new_leaves, deviceType device)
{
    if (metadata->n_leaves + n_new_leaves <= metadata->max_leaves &&
        metadata->n_trees + 1 <= metadata->max_trees)
        return;
    const int saved_leaves_batch = metadata->max_leaves_batch;
    const int saved_trees_batch = metadata->max_trees_batch;
    const int saved_max_leaves = metadata->max_leaves;
    const int saved_max_trees = metadata->max_trees;
    metadata->max_leaves_batch = std::max(saved_leaves_batch, n_new_leaves);
    metadata->max_trees_batch = std::max(saved_trees_batch, 1);
    metadata->max_leaves = metadata->n_leaves;  // force the grow branch
    try {
#ifdef USE_CUDA
        if (device == gpu)
            allocate_ensemble_memory_cuda(metadata, edata);
        else
#endif
            allocate_ensemble_memory(metadata, edata);
    } catch (...) {
        // Storage was not replaced: keep the live capacities and batch sizes.
        metadata->max_leaves = saved_max_leaves;
        metadata->max_trees = saved_max_trees;
        metadata->max_leaves_batch = saved_leaves_batch;
        metadata->max_trees_batch = saved_trees_batch;
        throw;
    }
    (void)device;
    metadata->max_leaves_batch = saved_leaves_batch;
    metadata->max_trees_batch = saved_trees_batch;
}

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
    reserve_tree_capacity(metadata, edata, n_leaves_in_tree, cpu);

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

// Imported trees are checked once, before either backend changes the ensemble.
// This path is not called by tree fitting or prediction.
static void validate_imported_tree(const treeData *tree, const ensembleMetaData *metadata) {
    if (tree->n_leaves <= 0 || tree->tree_depth < 0 || tree->tree_depth > tree->max_depth ||
        tree->split_count != (tree->is_oblivious ? 1 : tree->n_leaves)) {
        throw std::runtime_error("treeData has invalid tree dimensions");
    }
    if (tree->is_oblivious &&
        (tree->tree_depth >= std::numeric_limits<int>::digits ||
         tree->n_leaves != (1 << tree->tree_depth))) {
        throw std::runtime_error("treeData oblivious leaf count must equal 2^tree_depth");
    }
    if (!tree->is_oblivious && tree->max_depth < std::numeric_limits<int>::digits &&
        tree->n_leaves > (1 << tree->max_depth)) {
        throw std::runtime_error("treeData greedy leaf count must not exceed 2^max_depth");
    }
    int actual_depth = 0;
    for (int row = 0; row < tree->split_count; ++row) {
        const int depth = tree->depths[row];
        if (depth < 0 || depth > tree->max_depth) {
            throw std::runtime_error("treeData depths must be within [0, max_depth]");
        }
        actual_depth = std::max(actual_depth, depth);
        for (int d = 0; d < depth; ++d) {
            const size_t idx = static_cast<size_t>(row) * tree->max_depth + d;
            const int count = tree->is_numerics[idx] ? metadata->n_num_features : metadata->n_cat_features;
            if (tree->feature_indices[idx] < 0 || tree->feature_indices[idx] >= count) {
                throw std::runtime_error("treeData active feature index is out of range; initialize "
                                         "the destination feature layout with predict() before add_tree()");
            }
        }
    }
    if (actual_depth != tree->tree_depth) {
        throw std::runtime_error("treeData tree_depth does not match depths");
    }
    for (int leaf = 0; leaf < tree->n_leaves; ++leaf) {
        double sum = 0.0;
        for (int k = 0; k < tree->n_objs; ++k) {
            const float density = tree->densities[static_cast<size_t>(leaf) * tree->n_objs + k];
            if (!std::isfinite(density) || density < 0.0f || density > 1.0f) {
                throw std::runtime_error("treeData densities must be finite and within [0, 1]");
            }
            sum += density;
        }
        if (std::abs(sum - 1.0) > 1e-5) {
            throw std::runtime_error("treeData densities must sum to one per leaf");
        }
    }
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
    // Densities are copied with the ensemble's n_objs stride.
    if (tdata->n_objs != metadata->n_objs){
        throw std::runtime_error("treeData n_objs does not match ensemble n_objs");
    }

    validate_imported_tree(tdata, metadata);

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

    reserve_tree_capacity(metadata, edata, n_leaves_in_tree, gpu);

    int leaf_start = metadata->n_leaves;
    int tree_idx = metadata->n_trees;

    // Every copy writes into the inactive slot past n_leaves/n_trees; publish
    // the tree only if all of them succeed, so a retry rewrites the same slot.
    cudaError_t copy_status = cudaSuccess;
    auto copy = [&copy_status](void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
        if (copy_status != cudaSuccess) return;
        copy_status = cudaMemcpy(dst, src, bytes, kind);
    };

    // tree_indices: single int written from host
    copy(&edata->ensemble_info->tree_indices[tree_idx], &leaf_start,
               sizeof(int), cudaMemcpyHostToDevice);

    // Depths
    if (is_oblivious){
        copy(&edata->ensemble_info->depths[tree_idx], tdata->depths,
                   sizeof(int), cudaMemcpyHostToDevice);
    } else {
        copy(&edata->ensemble_info->depths[leaf_start], tdata->depths,
                   n_leaves_in_tree * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Leaf values
    copy(&edata->leaf_data->values[leaf_start * output_dim], tdata->values,
               n_leaves_in_tree * output_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Edge weights
    copy(&edata->leaf_data->edge_weights[leaf_start * max_depth], tdata->edge_weights,
               n_leaves_in_tree * max_depth * sizeof(float), cudaMemcpyHostToDevice);

    // Inequality directions
    copy(&edata->feature_data->inequality_directions[leaf_start * max_depth], tdata->inequality_directions,
               n_leaves_in_tree * max_depth * sizeof(bool), cudaMemcpyHostToDevice);

    // Split-indexed arrays
    int split_dst_start = is_oblivious ? tree_idx : leaf_start;
    int split_count = tdata->split_count;

    copy(&edata->feature_data->feature_indices[split_dst_start * max_depth], tdata->feature_indices,
               split_count * max_depth * sizeof(int), cudaMemcpyHostToDevice);
    copy(&edata->feature_data->feature_values[split_dst_start * max_depth], tdata->feature_values,
               split_count * max_depth * sizeof(float), cudaMemcpyHostToDevice);
    copy(&edata->feature_data->is_numerics[split_dst_start * max_depth], tdata->is_numerics,
               split_count * max_depth * sizeof(bool), cudaMemcpyHostToDevice);
    copy(&edata->feature_data->categorical_values[split_dst_start * max_depth * MAX_CHAR_SIZE], tdata->categorical_values,
               split_count * max_depth * MAX_CHAR_SIZE * sizeof(char), cudaMemcpyHostToDevice);

    // Multi-objective densities
    copy(&edata->multi_objective_data->densities[leaf_start * n_objs], tdata->densities,
               n_leaves_in_tree * n_objs * sizeof(float), cudaMemcpyHostToDevice);

    if (copy_status != cudaSuccess)
        throw std::runtime_error(std::string("add_tree: failed to copy tree to GPU: ") +
                                 cudaGetErrorString(copy_status));

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
static void validate_export(const ensembleMetaData *metadata, const std::vector<Optimizer*>& opts) {
    if (metadata->grow_policy != OBLIVIOUS || metadata->n_cat_features > 0)
        throw std::runtime_error("Export requires oblivious trees with numerical features only");
    for (auto *opt : opts)
        if (opt->getAlgo() != SGD)
            throw std::runtime_error("Export requires SGD optimizers");
}

exportData* get_export_data(ensembleMetaData *metadata, ensembleData *edata, deviceType device, std::vector<Optimizer*> opts){
    validate_export(metadata, opts);
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

    exp_data->tree_depths = new int[metadata->n_trees];
    exp_data->tree_leaf_counts = new int[metadata->n_trees];
    exp_data->feature_indices = new int[binary_features];
    exp_data->feature_values = new float[binary_features];
    exp_data->leaf_values = new float[metadata->n_leaves * metadata->output_dim]();
    exp_data->bias = new float[metadata->output_dim];
    memcpy(exp_data->bias, edata_cpu->bias, metadata->output_dim * sizeof(float));

    int packed_idx = 0;
    for (int t = 0; t < metadata->n_trees; ++t) {
        exp_data->tree_depths[t] = edata_cpu->ensemble_info->depths[t];
        const int leaf_stop = t + 1 < metadata->n_trees ? edata_cpu->ensemble_info->tree_indices[t + 1] : metadata->n_leaves;
        exp_data->tree_leaf_counts[t] = leaf_stop - edata_cpu->ensemble_info->tree_indices[t];
        for (int d = 0; d < edata_cpu->ensemble_info->depths[t]; ++d) {
            int src = t * metadata->max_depth + d;
            exp_data->feature_indices[packed_idx] = edata_cpu->feature_data->feature_indices[src];
            exp_data->feature_values[packed_idx++] = edata_cpu->feature_data->feature_values[src];
        }
    }

    int tree_idx = 0;
    float value;
    for (int i  = 0; i < metadata->n_leaves; ++i){
        // Advance once leaf i reaches the start of the next tree.
        while (tree_idx + 1 < metadata->n_trees && i >= edata_cpu->ensemble_info->tree_indices[tree_idx + 1])
            tree_idx += 1;
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
    validate_export(metadata, opts);
    if (export_type == exportType::COMPACT && metadata->max_depth > 6)
        throw std::runtime_error("Compact export supports max_depth <= 6; use export_type='full'");
    std::unique_ptr<exportData> packed(get_export_data(metadata, edata, device, opts));
    std::unique_ptr<int[]> packed_depths(packed->tree_depths);
    std::unique_ptr<int[]> packed_counts(packed->tree_leaf_counts);
    std::unique_ptr<int[]> packed_indices(packed->feature_indices);
    std::unique_ptr<float[]> packed_values(packed->feature_values);
    std::unique_ptr<float[]> packed_leaves(packed->leaf_values);
    std::unique_ptr<float[]> packed_bias(packed->bias);
    if (export_type == COMPACT)
        for (int t = 0; t < metadata->n_trees; ++t)
            if (packed->tree_depths[t] != metadata->max_depth)
                throw std::runtime_error("Compact export requires every tree to reach max_depth; use full export");

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

    // GREEDY stores depths and split conditions per leaf; the generated code
    // walks them per tree.
    if (metadata->grow_policy == GREEDY){
        header_file.close();
        throw std::runtime_error("Export is only supported for oblivious trees");
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

    // Write floats (thresholds, leaf values, bias) with round-trip precision so
    // the exported model routes samples exactly like the trained one.
    const std::streamsize saved_precision =
        header_file.precision(std::numeric_limits<float>::max_digits10);
    header_file << "#ifndef " << prefix << "GBRL_MODEL_H\n";
    header_file << "#define " << prefix << "GBRL_MODEL_H\n\n";

    
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
    header_file << "#define " << prefix << "N_INPUTS " << metadata->input_dim << "\n";
    header_file << "#define " << prefix << "N_OUTPUTS " << metadata->output_dim << "\n";
    header_file << "#define " << prefix << "NUM_FEATURES " << metadata->n_num_features  << "\n\n";
    if (metadata->output_dim > 1){
        header_file << "static inline void " << prefix << "gbrl_predict(" << type_name << " *results, const " << type_name << " *features){\n\n";
    } else {
        header_file << "static inline " << type_name << " " << prefix << "gbrl_predict(const " << type_name << " *features){\n\n";
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
        header_file << "}";
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
        header_file << packed->feature_indices[i];
        if (i < binary_splits - 1)
            header_file << ", ";
    }
    header_file << "};\n";

    header_file << "\tconst " << type_name << " feature_values[" << prefix << "BINARY_FEATURES] = {";
    for (int i  = 0; i < binary_splits; ++i){
        switch (export_format){
            case EXP_FLOAT:
                header_file << packed->feature_values[i];
                break;
            case EXP_FXP8:
                   header_file << float_to_int16(packed->feature_values[i]);
                break;
            case EXP_FXP16:
                   header_file << float_to_int32(packed->feature_values[i]);
                break;
        }
        if (i < binary_splits - 1)
            header_file << ", ";
    }
    header_file << "};\n";

    header_file << "\tconst " << type_name << " leaf_values[" << prefix << "N_LEAVES*" << prefix << "N_OUTPUTS]  = {";
    for (int i = 0; i < metadata->n_leaves * metadata->output_dim; ++i) {
        float value = packed->leaf_values[i];
        switch (export_format) {
            case EXP_FLOAT: header_file << value; break;
            case EXP_FXP8: header_file << float_to_int16(value); break;
            case EXP_FXP16: header_file << float_to_int32(value); break;
        }
        if (i + 1 < metadata->n_leaves * metadata->output_dim)
            header_file << ", ";
    }
    header_file << "};\n";

    if (metadata->output_dim > 1) {
        header_file << "\tfor (j = 0; j < " << prefix << "N_OUTPUTS; ++j)\n";
        header_file << "\t\tresults[j] = bias[j];\n";
    }
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
    if (metadata->output_dim == 1){
        header_file << "\tresult += bias;\n";
        header_file << "\treturn result;\n";
    }
    header_file << "}\n";
    header_file << "#endif\n";
    header_file.precision(saved_precision);

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

// Matches the Python-side limit (BaseLearner): leaf counts are computed as
// 1 << depth in int arithmetic.
static constexpr int MAX_SUPPORTED_DEPTH = 30;

struct LegacyMetadata {
    int n_leaves;
    int n_trees;
    int max_trees;
    int max_leaves;
    int max_trees_batch;
    int max_leaves_batch;
    int input_dim;
    int output_dim;
    int policy_dim;
    int max_depth;
    int min_data_in_leaf;
    int n_bins;
    int par_th;
    float cv_beta;
    int verbose;
    int batch_size;
    bool use_cv;
    scoreFunc split_score_func;
    generatorType generator_type;
    growPolicy grow_policy;
    int n_num_features;
    int n_cat_features;
    int iteration;
};

template<typename T>
static void read_scalar(std::ifstream& file, T& value) {
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!file.good())
        throw std::runtime_error("Truncated checkpoint data");
}

void write_ensemble_metadata(std::ofstream& file, const ensembleMetaData& m) {
    file.write(reinterpret_cast<const char*>(&m.n_leaves), sizeof(m.n_leaves));
    file.write(reinterpret_cast<const char*>(&m.n_trees), sizeof(m.n_trees));
    file.write(reinterpret_cast<const char*>(&m.max_trees), sizeof(m.max_trees));
    file.write(reinterpret_cast<const char*>(&m.max_leaves), sizeof(m.max_leaves));
    file.write(reinterpret_cast<const char*>(&m.max_trees_batch), sizeof(m.max_trees_batch));
    file.write(reinterpret_cast<const char*>(&m.max_leaves_batch), sizeof(m.max_leaves_batch));
    file.write(reinterpret_cast<const char*>(&m.input_dim), sizeof(m.input_dim));
    file.write(reinterpret_cast<const char*>(&m.output_dim), sizeof(m.output_dim));
    file.write(reinterpret_cast<const char*>(&m.policy_dim), sizeof(m.policy_dim));
    file.write(reinterpret_cast<const char*>(&m.max_depth), sizeof(m.max_depth));
    file.write(reinterpret_cast<const char*>(&m.min_data_in_leaf), sizeof(m.min_data_in_leaf));
    file.write(reinterpret_cast<const char*>(&m.n_bins), sizeof(m.n_bins));
    file.write(reinterpret_cast<const char*>(&m.par_th), sizeof(m.par_th));
    file.write(reinterpret_cast<const char*>(&m.n_objs), sizeof(m.n_objs));
    file.write(reinterpret_cast<const char*>(&m.verbose), sizeof(m.verbose));
    file.write(reinterpret_cast<const char*>(&m.batch_size), sizeof(m.batch_size));
    file.write(reinterpret_cast<const char*>(&m.n_num_features), sizeof(m.n_num_features));
    file.write(reinterpret_cast<const char*>(&m.n_cat_features), sizeof(m.n_cat_features));
    file.write(reinterpret_cast<const char*>(&m.iteration), sizeof(m.iteration));
    file.write(reinterpret_cast<const char*>(&m.cv_beta), sizeof(m.cv_beta));
    file.write(reinterpret_cast<const char*>(&m.lambda_penalty), sizeof(m.lambda_penalty));
    file.write(reinterpret_cast<const char*>(&m.use_cv), sizeof(m.use_cv));
    file.write(reinterpret_cast<const char*>(&m.split_score_func), sizeof(m.split_score_func));
    file.write(reinterpret_cast<const char*>(&m.generator_type), sizeof(m.generator_type));
    file.write(reinterpret_cast<const char*>(&m.grow_policy), sizeof(m.grow_policy));
    file.write(reinterpret_cast<const char*>(&m.n_mono_constraints), sizeof(m.n_mono_constraints));
}

ensembleMetaData* read_ensemble_metadata(std::ifstream& file, const serializationHeader& header) {
    std::unique_ptr<ensembleMetaData> m(new ensembleMetaData{});
    if (header.minor_version == 1) {
        LegacyMetadata old{};
        file.read(reinterpret_cast<char*>(&old), sizeof(old));
        if (!file.good())
            throw std::runtime_error("Truncated legacy metadata");
        m->n_leaves = old.n_leaves;
        m->n_trees = old.n_trees;
        m->max_trees = old.max_trees;
        m->max_leaves = old.max_leaves;
        m->max_trees_batch = old.max_trees_batch;
        m->max_leaves_batch = old.max_leaves_batch;
        m->input_dim = old.input_dim;
        m->output_dim = old.output_dim;
        m->policy_dim = old.policy_dim;
        m->max_depth = old.max_depth;
        m->min_data_in_leaf = old.min_data_in_leaf;
        m->n_bins = old.n_bins;
        m->par_th = old.par_th;
        m->cv_beta = old.cv_beta;
        m->verbose = old.verbose;
        m->batch_size = old.batch_size;
        m->use_cv = old.use_cv;
        m->split_score_func = old.split_score_func;
        m->generator_type = old.generator_type;
        m->grow_policy = old.grow_policy;
        m->n_num_features = old.n_num_features;
        m->n_cat_features = old.n_cat_features;
        m->iteration = old.iteration;
        if (header.patch_version >= 7)
            read_scalar(file, m->n_mono_constraints);
        m->n_objs = 1;
        m->lambda_penalty = 0.0f;
    } else {
        read_scalar(file, m->n_leaves);
        read_scalar(file, m->n_trees);
        read_scalar(file, m->max_trees);
        read_scalar(file, m->max_leaves);
        read_scalar(file, m->max_trees_batch);
        read_scalar(file, m->max_leaves_batch);
        read_scalar(file, m->input_dim);
        read_scalar(file, m->output_dim);
        read_scalar(file, m->policy_dim);
        read_scalar(file, m->max_depth);
        read_scalar(file, m->min_data_in_leaf);
        read_scalar(file, m->n_bins);
        read_scalar(file, m->par_th);
        read_scalar(file, m->n_objs);
        read_scalar(file, m->verbose);
        read_scalar(file, m->batch_size);
        read_scalar(file, m->n_num_features);
        read_scalar(file, m->n_cat_features);
        read_scalar(file, m->iteration);
        read_scalar(file, m->cv_beta);
        read_scalar(file, m->lambda_penalty);
        read_scalar(file, m->use_cv);
        read_scalar(file, m->split_score_func);
        read_scalar(file, m->generator_type);
        read_scalar(file, m->grow_policy);
        read_scalar(file, m->n_mono_constraints);
    }
    if (m->input_dim <= 0 || m->output_dim <= 0 || m->n_objs <= 0 ||
        m->policy_dim < 0 || m->policy_dim > m->output_dim ||
        m->max_depth < 0 || m->max_depth > MAX_SUPPORTED_DEPTH || m->n_trees < 0 ||
        m->n_leaves < m->n_trees || m->n_mono_constraints < 0 ||
        m->n_num_features < 0 || m->n_cat_features < 0 ||
        m->n_num_features + static_cast<int64_t>(m->n_cat_features) > m->input_dim ||
        m->n_bins <= 0 || m->iteration < 0 ||
        m->grow_policy > OBLIVIOUS || m->split_score_func > Cosine || m->generator_type > Quantile)
        throw std::runtime_error("Invalid checkpoint metadata");
    auto position = file.tellg();
    file.seekg(0, std::ios::end);
    const auto remaining = file.tellg() - position;
    file.seekg(position);
    const uint64_t splits = m->grow_policy == OBLIVIOUS ? m->n_trees : m->n_leaves;
    const long double required = 4.0L * m->output_dim + 17.0L * m->input_dim +
        4.0L * m->n_trees + 4.0L * splits +
        4.0L * m->n_leaves * m->output_dim +
        (9.0L + MAX_CHAR_SIZE) * splits * m->max_depth +
        5.0L * m->n_leaves * m->max_depth + 12.0L * m->n_mono_constraints +
        (header.minor_version == 1 ? 0.0L : 4.0L * m->n_objs * (m->n_leaves + 1.0L));
    if (required > static_cast<long double>(remaining))
        throw std::runtime_error("Checkpoint counts exceed the available data");
    return m.release();
}

ensembleData* load_ensemble_data(std::ifstream& file, ensembleMetaData *metadata, int legacy_patch){
    if (!file.is_open() || file.fail())
        throw std::runtime_error("Error opening checkpoint file");
    std::unique_ptr<ensembleData, decltype(&ensemble_data_dealloc)> owner(
        ensemble_data_alloc(metadata), ensemble_data_dealloc);
    ensembleData *edata_cpu = owner.get();
    auto read_array = [&file](auto *destination, size_t count) {
        NULL_CHECK check;
        read_scalar(file, check);
        if (check != VALID && check != NULL_OPT)
            throw std::runtime_error("Invalid checkpoint array marker");
        if (check == VALID)
            file.read(reinterpret_cast<char*>(destination), count * sizeof(*destination));
        if (!file.good())
            throw std::runtime_error("Truncated checkpoint ensemble data");
    };
    const size_t n_leaves = metadata->n_leaves;
    const size_t n_trees = metadata->n_trees;
    const size_t depth = metadata->max_depth;
    const size_t splits = metadata->grow_policy == OBLIVIOUS ? n_trees : n_leaves;

    read_array(edata_cpu->bias, metadata->output_dim);
    read_array(edata_cpu->feature_data->feature_weights, metadata->input_dim);
#ifdef DEBUG
    read_array(edata_cpu->n_samples, n_leaves);
#endif
    read_array(edata_cpu->ensemble_info->tree_indices, n_trees);
    read_array(edata_cpu->ensemble_info->depths, splits);
    read_array(edata_cpu->leaf_data->values, n_leaves * metadata->output_dim);
    read_array(edata_cpu->feature_data->feature_indices, depth * splits);
    read_array(edata_cpu->feature_data->feature_values, depth * splits);
    read_array(edata_cpu->leaf_data->edge_weights, depth * n_leaves);
    if (legacy_patch < 0) {
        read_array(edata_cpu->multi_objective_data->densities, n_leaves * metadata->n_objs);
    } else {
        std::fill(edata_cpu->multi_objective_data->densities,
                  edata_cpu->multi_objective_data->densities + n_leaves, 1.0f);
        edata_cpu->multi_objective_data->lambda_objs[0] = 1.0f;
    }
    if (legacy_patch < 0 || legacy_patch >= 7) {
        read_array(edata_cpu->mono_constraints->feature_idx, metadata->n_mono_constraints);
        read_array(edata_cpu->mono_constraints->output_idx, metadata->n_mono_constraints);
        read_array(edata_cpu->mono_constraints->constraint, metadata->n_mono_constraints);
    }
    read_array(edata_cpu->feature_mappings->reverse_num_feature_mapping, metadata->input_dim);
    read_array(edata_cpu->feature_mappings->reverse_cat_feature_mapping, metadata->input_dim);
    read_array(edata_cpu->feature_mappings->feature_mapping, metadata->input_dim);
    read_array(edata_cpu->feature_mappings->mapping_numerics, metadata->input_dim);
    read_array(edata_cpu->feature_data->is_numerics, depth * splits);
    read_array(edata_cpu->feature_data->inequality_directions, depth * n_leaves);
    read_array(edata_cpu->feature_data->categorical_values, depth * splits * MAX_CHAR_SIZE);
    if (legacy_patch < 0)
        read_array(edata_cpu->multi_objective_data->lambda_objs, metadata->n_objs);
    return owner.release();
}
