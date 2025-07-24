//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
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