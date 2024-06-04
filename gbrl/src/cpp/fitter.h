//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef FITTER_H
#define FITTER_H

#include "types.h"
#include "split_candidate_generator.h"
#include "node.h"
#include "optimizer.h"

class Fitter {
    public:
        static void step_cpu(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata);
        static float fit_cpu(dataSet *dataset, const float *targets, ensembleData *edata, ensembleMetaData *metadata, const int iterations, lossType loss_type, std::vector<Optimizer*> opts);
        static int fit_greedy_tree(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const SplitCandidateGenerator &generator);
        static int fit_oblivious_tree(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const SplitCandidateGenerator &generator);
        static void fit_leaves(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata, const int added_leaves);
        static void update_ensemble_per_leaf(ensembleData *edata, ensembleMetaData *metadata, const TreeNode* node);
        static void update_ensemble_per_tree(ensembleData *edata, ensembleMetaData *metadata, std::vector<TreeNode*> nodes, const int n_nodes);
        static void calc_leaf_value(dataSet *data, ensembleData *edata, ensembleMetaData *metadata, const int leaf_idx, const int tree_idx);
        
        static void control_variates(dataSet *dataset, ensembleData *edata, ensembleMetaData *metadata);
};

#endif