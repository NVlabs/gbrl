//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
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
 * @file compression.h
 * @brief Tree ensemble compression utilities for CPU
 * 
 * Provides static methods for compressing gradient boosted tree ensembles
 * by selecting a subset of trees while maintaining prediction accuracy through
 * learned correction matrices. Supports both oblivious and greedy tree structures.
 */

#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "types.h"
#include "optimizer.h"

/**
 * @brief Static utility class for tree ensemble compression operations
 * 
 * All methods are static and perform compression-related operations on CPU,
 * including matrix representation generation and ensemble compression.
 */
class Compressor {
    public:
        /**
         * @brief Generate matrix representation (A, V) of tree ensemble on CPU
         * 
         * Converts the tree ensemble into matrix form where A is a binary activation
         * matrix (n_samples x n_leaves+1) and V contains scaled leaf values.
         * Supports both parallel and sequential execution with automatic selection.
         * 
         * @param dataset Input dataset containing observations
         * @param edata Ensemble data (tree structure, leaf values)
         * @param metadata Ensemble metadata (dimensions, hyperparameters)
         * @param parallel_predict Whether to enable parallel execution
         * @param matrix Output matrix representation structure (pre-allocated)
         * @param opts Vector of optimizers for scaling leaf values
         */
        static void get_matrix_representation_cpu(dataSet *dataset, const ensembleData *edata, const ensembleMetaData *metadata, const bool parallel_predict, matrixRepresentation *matrix,  std::vector<Optimizer*> opts);
        
        /**
         * @brief Generate representation for greedy (non-oblivious) trees
         * 
         * For each tree, traverses from root to leaf following split conditions
         * to determine which leaf each sample reaches. Sets corresponding A entries.
         * 
         * @param obs Numerical observations (n_samples x n_num_features)
         * @param categorical_obs Categorical observations (flattened)
         * @param sample_idx Index of sample being processed
         * @param edata Ensemble data structure
         * @param metadata Ensemble metadata
         * @param start_tree_idx First tree to process (inclusive)
         * @param stop_tree_idx Last tree to process (exclusive)
         * @param matrix Output matrix representation
         */
        static void get_representation_matrix_over_leaves(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, matrixRepresentation *matrix);
        
        /**
         * @brief Generate representation for oblivious (symmetric) trees
         * 
         * For oblivious trees, all nodes at same depth use identical splits.
         * Computes leaf index directly from binary path representation.
         * 
         * @param obs Numerical observations (n_samples x n_num_features)
         * @param categorical_obs Categorical observations (flattened)
         * @param sample_idx Index of sample being processed
         * @param edata Ensemble data structure
         * @param metadata Ensemble metadata
         * @param start_tree_idx First tree to process (inclusive)
         * @param stop_tree_idx Last tree to process (exclusive)
         * @param matrix Output matrix representation
         */
        static void get_representation_matrix_over_trees(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, matrixRepresentation *matrix);
        
        /**
         * @brief Apply correction matrix W to leaf values during compression
         * 
         * Inverse-scales and subtracts W values from leaf values to maintain
         * prediction accuracy after removing trees. Supports parallel execution.
         * 
         * @param W Correction matrix (n_leaves+1 x output_dim)
         * @param edata Ensemble data containing leaf values to modify
         * @param metadata Ensemble metadata
         * @param opts Vector of optimizers for inverse scaling
         */
        static void add_W_matrix_to_values(const float *W, const ensembleData *edata, const ensembleMetaData *metadata, std::vector<Optimizer*> opts);
        
        /**
         * @brief Extract and scale leaf values into matrix V
         * 
         * Copies leaf values and scales by negative learning rate to populate
         * the V matrix for matrix representation. Supports parallel execution.
         * 
         * @param matrix Matrix representation to populate (modifies V)
         * @param edata Ensemble data containing leaf values
         * @param metadata Ensemble metadata
         * @param opts Vector of optimizers for scaling
         */
        static void get_V(matrixRepresentation *matrix, const ensembleData *edata, const ensembleMetaData *metadata, std::vector<Optimizer*> opts);
        
        /**
         * @brief Compress tree ensemble by applying correction and selecting subset
         * 
         * Applies correction matrix W to leaf values, then creates a new ensemble
         * containing only selected trees/leaves. Original ensemble is deallocated.
         * 
         * @param metadata Ensemble metadata (modified to reflect compression)
         * @param edata Original ensemble data (deallocated after compression)
         * @param opts Vector of optimizers
         * @param n_compressed_leaves Number of leaves in compressed ensemble
         * @param n_compressed_trees Number of trees in compressed ensemble
         * @param leaf_indices Indices of leaves to retain
         * @param tree_indices Indices of trees to retain
         * @param new_tree_indices New starting indices for trees
         * @param W Correction matrix to apply before compression
         * @return Pointer to new compressed ensemble data
         */
        static ensembleData* compress_ensemble(ensembleMetaData *metadata, ensembleData *edata, std::vector<Optimizer*> opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W);
};

#endif 