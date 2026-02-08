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
 * @file ensemble_io.h
 * @brief Ensemble data export, serialization, and single-tree extraction/insertion
 *
 * Provides data structures and functions for:
 * - Extracting simplified export data for inference deployment (exportData)
 * - Extracting / inserting individual trees for distributed synchronization (treeData)
 * - Saving / loading full ensemble data to binary files
 * - Exporting ensemble models to C header files
 */

#ifndef ENSEMBLE_IO_H
#define ENSEMBLE_IO_H

#include <vector>
#include <string>
#include <fstream>
#include "types.h"

// Forward declaration – full definition lives in optimizer.h
class Optimizer;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Simplified export data for model deployment and inference
 *
 * Contains the minimal set of arrays needed to perform inference:
 * split feature indices/values, optimizer-scaled leaf values, and bias.
 * Produced by get_export_data() which flattens the full ensemble
 * representation and bakes in the optimizer learning rates.
 */
struct exportData {
    int n_trees;              /**< Number of trees in the ensemble */
    int n_leaves;             /**< Total number of leaves across all trees */
    int input_dim;            /**< Number of input features */
    int output_dim;           /**< Dimensionality of output predictions */
    int max_depth;            /**< Maximum depth of any tree */
    int num_features;         /**< Number of numerical features */
    int binary_features;      /**< Total number of binary split nodes (sum of depths) */

    int *feature_indices;     /**< Split feature index per binary node [binary_features] */
    float *feature_values;    /**< Split threshold per binary node [binary_features] */
    float *leaf_values;       /**< Optimizer-scaled leaf predictions [n_leaves * output_dim] */
    float *bias;              /**< Model bias term [output_dim] */
};

/**
 * @brief Data for a single tree extracted from (or to be inserted into) an ensemble
 *
 * Contains all arrays that define one tree's structure and predictions.
 * For OBLIVIOUS trees: split_count = 1 (one shared set of splits).
 * For GREEDY trees: split_count = n_leaves (each leaf stores its own path).
 *
 * The caller owns all allocated arrays and must free them when done.
 */
struct treeData {
    int n_leaves;              /**< Number of leaves in this tree */
    int tree_depth;            /**< Depth of the tree (for oblivious) or max leaf depth (for greedy) */
    int output_dim;            /**< Dimensionality of leaf values */
    int max_depth;             /**< max_depth parameter of the ensemble */
    int n_objs;                /**< Number of objectives for multi-objective data */
    int split_count;           /**< Number of split-indexed entries: 1 (oblivious) or n_leaves (greedy) */
    bool is_oblivious;         /**< True if tree uses oblivious grow policy */

    int *depths;               /**< Depth(s): [1] for oblivious, [n_leaves] for greedy */
    float *values;             /**< Leaf prediction values [n_leaves * output_dim] */
    float *edge_weights;       /**< Edge weights per leaf path [n_leaves * max_depth] */
    int *feature_indices;      /**< Split feature indices [split_count * max_depth] */
    float *feature_values;     /**< Split thresholds [split_count * max_depth] */
    bool *is_numerics;         /**< Numeric/categorical flag [split_count * max_depth] */
    bool *inequality_directions; /**< Inequality directions [n_leaves * max_depth] */
    char *categorical_values;  /**< Categorical split values [split_count * max_depth * MAX_CHAR_SIZE] */
    float *densities;          /**< Multi-objective densities [n_leaves * n_objs] */
};

// ============================================================================
// Tree Extract / Insert
// ============================================================================

/**
 * @brief Extract data for a single tree from the ensemble
 *
 * Allocates and populates a treeData struct with copies of all arrays
 * belonging to the specified tree. Works for both GREEDY and OBLIVIOUS
 * grow policies. The caller owns all memory and must free it.
 *
 * @param tree_idx Index of the tree to extract (0-based)
 * @param metadata Ensemble metadata
 * @param edata Ensemble data arrays
 * @param device Device where edata resides (GPU data is copied to CPU)
 * @return Pointer to newly allocated treeData; caller must free via tree_data_dealloc
 */
treeData* get_tree_data(int tree_idx, ensembleMetaData *metadata, ensembleData *edata, deviceType device);

/**
 * @brief Insert a tree into the ensemble from a treeData struct
 *
 * Appends the tree described by tdata to the end of the ensemble.
 * Handles memory reallocation if needed. Updates n_trees, n_leaves,
 * and iteration in metadata. The treeData arrays are copied (not moved),
 * so the caller retains ownership of tdata.
 *
 * Supports both CPU and GPU ensembles. For GPU, data is copied directly
 * from host treeData arrays to GPU ensemble arrays via cudaMemcpy.
 *
 * @param tdata Tree data to insert (host memory)
 * @param metadata Ensemble metadata (updated in place)
 * @param edata Ensemble data arrays (updated in place)
 * @param device Device where edata resides
 */
void add_tree_data(const treeData *tdata, ensembleMetaData *metadata, ensembleData *edata, deviceType device);

/**
 * @brief Free all memory in a treeData struct
 *
 * @param tdata Tree data to deallocate (struct itself is also deleted)
 */
void tree_data_dealloc(treeData *tdata);

// ============================================================================
// Export Data
// ============================================================================

/**
 * @brief Extract simplified export data from the ensemble for inference
 *
 * Creates an exportData struct containing the minimal arrays needed for
 * model inference. Flattens tree split information into contiguous arrays
 * and bakes optimizer learning rates into leaf values. If data resides on
 * GPU, a temporary CPU copy is made and freed after extraction.
 *
 * @param metadata Ensemble metadata describing structure
 * @param edata Full ensemble data arrays
 * @param device Device where edata currently resides
 * @param opts Vector of optimizers (used to scale leaf values by learning rate)
 * @return Pointer to newly allocated exportData; caller owns all memory
 */
exportData* get_export_data(ensembleMetaData *metadata, ensembleData *edata, deviceType device, std::vector<Optimizer*> opts);

// ============================================================================
// Serialization / C-Header Export
// ============================================================================

/**
 * @brief Export ensemble model to a C header file
 *
 * Exports the model in a format suitable for deployment in embedded
 * systems or inference-only applications.
 *
 * @param header_file Output header file stream
 * @param model_name Name of the model for code generation
 * @param edata Ensemble data to export
 * @param metadata Ensemble metadata
 * @param device Device where data currently resides
 * @param opts Vector of optimizer configurations
 * @param export_format Numerical format (float, fixed-point, etc.)
 * @param export_type Export style (full or compact)
 * @param prefix String prefix for generated code symbols
 */
void export_ensemble_data(
    std::ofstream& header_file,
    const std::string& model_name,
    ensembleData *edata,
    ensembleMetaData *metadata,
    deviceType device,
    std::vector<Optimizer*> opts,
    exportFormat export_format,
    exportType export_type,
    const std::string &prefix
);

/**
 * @brief Save ensemble data to binary file
 *
 * @param file Output file stream
 * @param edata Ensemble data to save
 * @param metadata Ensemble metadata
 * @param device Device where data currently resides
 */
void save_ensemble_data(
    std::ofstream& file,
    ensembleData *edata,
    ensembleMetaData *metadata,
    deviceType device
);

/**
 * @brief Load ensemble data from binary file
 *
 * @param file Input file stream
 * @param metadata Ensemble metadata (must match saved model)
 * @return Pointer to loaded ensembleData
 */
ensembleData* load_ensemble_data(
    std::ifstream& file,
    ensembleMetaData *metadata
);

#endif // ENSEMBLE_IO_H
