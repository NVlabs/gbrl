//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file shap.h
 * @brief SHAP (SHapley Additive exPlanations) value computation
 * 
 * Implements TreeSHAP algorithm for computing feature attributions
 * in gradient boosted tree ensembles. SHAP values provide model
 * interpretability by explaining individual predictions.
 */

#ifndef SHAP_H
#define SHAP_H

#include "types.h"

/**
 * @brief Data structure for SHAP value computation
 * 
 * Contains all arrays and metadata needed for computing SHAP values
 * using the TreeSHAP algorithm. Stores tree structure, active paths,
 * and intermediate computation results.
 */
struct shapData {
    int n_nodes;                    /**< Number of nodes in current tree */
    const float *base_poly;         /**< Base polynomial coefficients */
    const float *offset_poly;       /**< Offset polynomial coefficients */
    const float *norm_values;       /**< Normalization values */
    float *G;                       /**< Group membership matrix */
    float *C;                       /**< Coverage weights */
    bool *active_nodes;             /**< Flags for active nodes in traversal */
    bool *numerics;                 /**< Flags for numerical vs categorical */
    int *feature_parent_node;       /**< Parent node for each feature */
    int *max_unique_features;       /**< Max unique features per depth */
    int *left_children;             /**< Left child indices */
    int *right_children;            /**< Right child indices */
    int *feature_indices;           /**< Feature used at each node */
    float *feature_values;          /**< Split threshold values */
    float *predictions;             /**< Leaf prediction values */
    float *weights;                 /**< Edge weights */
    char *categorical_values;       /**< Categorical split values */
};

/**
 * @brief Allocate SHAP data structure for a tree
 * 
 * @param metadata Ensemble metadata
 * @param edata Ensemble data
 * @param tree_idx Index of tree to analyze
 * @return Pointer to allocated shapData, caller must free with dealloc_shap_data
 */
shapData* alloc_shap_data(
    const ensembleMetaData *metadata,
    const ensembleData *edata,
    const int tree_idx
);

/**
 * @brief Deallocate SHAP data structure
 * 
 * @param shap_data SHAP data to free
 */
void dealloc_shap_data(shapData *shap_data);

/**
 * @brief Reset SHAP arrays for new computation
 * 
 * Clears intermediate arrays to prepare for computing SHAP values
 * for a new sample.
 * 
 * @param shap_data SHAP data to reset
 * @param metadata Ensemble metadata
 */
void reset_shap_arrays(
    shapData *shap_data,
    const ensembleMetaData *metadata
);

/**
 * @brief Compute SHAP values for dataset samples
 * 
 * Main entry point for SHAP value computation. Computes feature
 * attributions explaining predictions for all samples in dataset.
 * 
 * @param metadata Ensemble metadata
 * @param edata Ensemble data
 * @param shap_data Pre-allocated SHAP computation structure
 * @param dataset Input dataset
 * @param shap_values Output array (n_samples x n_features x output_dim)
 */
void get_shap_values(
    const ensembleMetaData *metadata,
    const ensembleData *edata,
    shapData *shap_data,
    const dataSet *dataset,
    float *shap_values
);

/**
 * @brief Print SHAP data structure for debugging
 * 
 * @param shap_data SHAP data to print
 * @param metadata Ensemble metadata
 */
void print_shap_data(
    const shapData *shap_data,
    const ensembleMetaData *metadata
);

/**
 * @brief Recursive TreeSHAP algorithm for linear trees
 * 
 * Implements the core TreeSHAP recursion for computing exact SHAP
 * values in polynomial time.
 * 
 * @param metadata Ensemble metadata
 * @param edata Ensemble data
 * @param shap_data SHAP computation structure
 * @param dataset Input dataset
 * @param shap_values Output SHAP values (accumulated)
 * @param crnt_node Current node index in recursion
 * @param crnt_depth Current depth in tree
 * @param crnt_feature Current feature being processed
 * @param sample_offset Offset for current sample in dataset
 */
void linear_tree_shap(
    const ensembleMetaData *metadata,
    const ensembleData *edata,
    shapData *shap_data,
    const dataSet *dataset,
    float *shap_values,
    int crnt_node,
    int crnt_depth,
    int crnt_feature,
    const int sample_offset
);

/**
 * @brief Add edge Shapley contribution
 * 
 * Updates SHAP values by adding contribution from traversing an edge.
 * 
 * @param shap_values SHAP values array to update
 * @param e Edge weight
 * @param offset Offset polynomial
 * @param base_poly Base polynomial
 * @param p_e Edge probability
 * @param norm_value Normalization value
 * @param d Depth/dimension
 * @param output_dim Output dimensionality
 */
void add_edge_shapley(
    float *shap_values,
    float *e,
    const float *offset,
    const float *base_poly,
    float p_e,
    const float *norm_value,
    int d,
    int output_dim
);

/**
 * @brief Subtract parent edge Shapley contribution
 * 
 * Updates SHAP values by removing contribution from closest parent
 * edge in the tree path.
 * 
 * @param shap_values SHAP values array to update
 * @param e Edge weight
 * @param offset Offset polynomial
 * @param base_poly Base polynomial
 * @param p_e_ancestor Ancestor edge probability
 * @param norm_value Normalization value
 * @param d Depth/dimension
 * @param output_dim Output dimensionality
 */
void subtract_closest_parent_edge_shapley(
    float *shap_values,
    float *e,
    const float *offset,
    const float *base_poly,
    float p_e_ancestor,
    const float *norm_value,
    int d,
    int output_dim
);

#endif // SHAP_H 