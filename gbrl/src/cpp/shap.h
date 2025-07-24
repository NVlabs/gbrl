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
#ifndef SHAP_H
#define SHAP_H

#include "types.h"
struct shapData {
    int n_nodes; 
    const float *base_poly;
    const float *offset_poly;
    const float *norm_values;
    float *G;
    float *C;
    bool *active_nodes;
    bool *numerics;
    int *feature_parent_node; 
    int *max_unique_features;
    int *left_children;
    int *right_children;
    int *feature_indices;
    float *feature_values;
    float *predictions;
    float *weights;
    char *categorical_values;
};

shapData* alloc_shap_data(const ensembleMetaData *metadata, const ensembleData *edata, const int tree_idx);
void dealloc_shap_data(shapData *shap_data);
void reset_shap_arrays(shapData *shap_data, const ensembleMetaData *metadata);
void get_shap_values(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values);
void print_shap_data(const shapData *shap_data, const ensembleMetaData *metadata);
void linear_tree_shap(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values, int crnt_node, int crnt_depth, int crnt_feature, const int sample_offset);
void add_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float p_e, const float *norm_value, int d, int output_dim);
void subtract_closest_parent_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float p_e_ancestor, const float *norm_value, int d, int output_dim);

#endif 