//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "types.h"
#include "optimizer.h"

class Compressor {
    public:
        static void get_matrix_representation_cpu(dataSet *dataset, const ensembleData *edata, const ensembleMetaData *metadata, const bool parallel_predict, matrixRepresentation *matrix,  std::vector<Optimizer*> opts);
        static void get_representation_matrix_over_leaves(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, matrixRepresentation *matrix);
        static void get_representation_matrix_over_trees(const float *obs, const char *categorical_obs, const int sample_idx, const ensembleData *edata, const ensembleMetaData *metadata, const int start_tree_idx, const int stop_tree_idx, matrixRepresentation *matrix);
        static void add_W_matrix_to_values(const float *W, const ensembleData *edata, const ensembleMetaData *metadata, std::vector<Optimizer*> opts);
        static void get_V(matrixRepresentation *matrix, const ensembleData *edata, const ensembleMetaData *metadata, std::vector<Optimizer*> opts);
        static ensembleData* compress_ensemble(ensembleMetaData *metadata, ensembleData *edata, std::vector<Optimizer*> opts, const int n_compressed_leaves, const int n_compressed_trees, const int *leaf_indices, const int *tree_indices, const int *new_tree_indices, const float *W);
};

#endif 