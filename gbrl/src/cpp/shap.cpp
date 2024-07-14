//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <stdexcept>
#include <cstring>

#include "shap.h"
#include "types.h"
#include "math_ops.h"
#include "data_structs.h"
#include "utils.h"



shapData* alloc_shap_data(const ensembleMetaData *metadata, const ensembleData *edata, const int tree_idx){
    int stop_leaf_idx = (tree_idx == metadata->n_trees - 1) ? metadata->n_leaves : edata->tree_indices[tree_idx+1];
    int start_leaf_idx = edata->tree_indices[tree_idx];
    int n_leaves = stop_leaf_idx - start_leaf_idx;

    stack<nodeInfo> node_stack(n_leaves * metadata->max_depth);
    nodeInfo root = {0, -1, 0, false, false};  // Assuming starting from root node
    node_stack.push(root);
    int n_nodes = 0;
    int leaf_idx = start_leaf_idx;

    // Allocate arrays for storing data (adjust sizes as needed)
    int *feature_parent_node = new int[n_leaves * metadata->max_depth];
    int *left_children = new int[n_leaves * metadata->max_depth];
    int *right_children = new int[n_leaves * metadata->max_depth];
    int *feature_indices = new int[n_leaves * metadata->max_depth];
    float *feature_values = new float[n_leaves * metadata->max_depth];
    bool *numerics = new bool[n_leaves * metadata->max_depth];
    float *predictions = new float[n_leaves * metadata->max_depth * metadata->output_dim];
    float *weights = new float[n_leaves * metadata->max_depth];
    char *categorical_values = new char[(n_leaves * metadata->max_depth)*MAX_CHAR_SIZE];
    for (int i = 0; i < n_leaves * metadata->max_depth; ++i){
        left_children[i] = -1;
        right_children[i] = -1;
        feature_indices[i] = -1;
        feature_parent_node[i] = -1;
        weights[i] = 1.0f;
        feature_values[i] = INFINITY;
    }
    int *parents = new int[n_leaves * metadata->max_depth];
    int *max_unique_features = new int[n_leaves * metadata->max_depth];
    
    memset(max_unique_features, 0, sizeof(int) * n_leaves * metadata->max_depth);
    memset(predictions, 0, sizeof(float) * n_leaves * metadata->max_depth * metadata->output_dim);
    // Process the tree using DFS
    while (!node_stack.is_empty()) {
        nodeInfo crnt_node = node_stack.top();
        node_stack.pop();
        crnt_node.idx = n_nodes;
        int idx = (metadata->grow_policy == OBLIVIOUS) ? tree_idx : leaf_idx;

        // Store the parent index for the current node
        parents[n_nodes] = crnt_node.parent_idx;
        if (crnt_node.depth > 0)
            weights[n_nodes] = edata->edge_weights[leaf_idx*metadata->max_depth + crnt_node.depth - 1];
        if (crnt_node.is_left)
            left_children[crnt_node.parent_idx] = crnt_node.idx;
        if (crnt_node.is_right)
            right_children[crnt_node.parent_idx] = crnt_node.idx;
        // If not at leaf, push children nodes onto the stack
        if (crnt_node.depth  < edata->depths[idx]) {
            nodeInfo right_child = {0, n_nodes, crnt_node.depth + 1, false, true};
            
            node_stack.push(right_child);
            nodeInfo left_child = {0, n_nodes, crnt_node.depth + 1, true, false};
            node_stack.push(left_child);
            int feature_idx = edata->feature_indices[idx*metadata->max_depth + crnt_node.depth];
            feature_indices[n_nodes] = feature_idx;
            if (edata->is_numerics[idx*metadata->max_depth + crnt_node.depth])
                feature_values[n_nodes] = edata->feature_values[idx*metadata->max_depth + crnt_node.depth];
            else 
                 memcpy(categorical_values + n_nodes*MAX_CHAR_SIZE, edata->categorical_values + (idx*metadata->max_depth + crnt_node.depth)*MAX_CHAR_SIZE, sizeof(char)*MAX_CHAR_SIZE);
            numerics[n_nodes] = edata->is_numerics[idx*metadata->max_depth + crnt_node.depth];

        } else {
            // Calculate number of unique features at the leaf node
            int n_unique_features = count_distinct(edata->feature_indices + idx * metadata->max_depth, edata->depths[idx]);
            // Backtrack to update max_unique_features array
            int parent_idx = parents[n_nodes];
            if (n_unique_features > max_unique_features[n_nodes])
                max_unique_features[n_nodes] = n_unique_features;
            while (parent_idx >= 0) {
                if (n_unique_features > max_unique_features[parent_idx])
                    max_unique_features[parent_idx] = n_unique_features;
                parent_idx = parents[parent_idx];
            }
            // Increment leaf index and mark children as non-existent (-1)
            float cond_prob = 1.0f;
            for (int d = 0; d < edata->depths[idx]; d++)
                cond_prob *= edata->edge_weights[leaf_idx*metadata->max_depth + d];
            for (int d = 0; d < metadata->output_dim; ++d)
                predictions[n_nodes*metadata->output_dim + d] = edata->values[leaf_idx*metadata->output_dim + d]*cond_prob;
            ++leaf_idx;
        }

        int parent_idx = parents[n_nodes];
        bool found = false;
        if (parent_idx >= 0){
            int grandparent_idx = parents[parent_idx];
            int prev_feature = feature_indices[parent_idx];
            while (grandparent_idx >= 0) {
                if (prev_feature == feature_indices[grandparent_idx]){
                    found = true;
                    break;
                }
                grandparent_idx = parents[grandparent_idx];
            }
        }
        
        if (found){
            feature_parent_node[n_nodes] = parent_idx;
            weights[n_nodes] *= weights[parent_idx];
        }
        
        // Move to the next node
        ++n_nodes;
    }
    
    delete[] parents;   
    
    shapData *shap_data = new shapData;
    shap_data->left_children = left_children;
    shap_data->right_children = right_children;
    shap_data->active_nodes = new bool[n_nodes];
    shap_data->numerics = numerics;
    shap_data->categorical_values = categorical_values;
    shap_data->feature_parent_node = feature_parent_node;
    shap_data->max_unique_features = max_unique_features;
    shap_data->n_nodes = n_nodes;
    shap_data->weights = weights;
    shap_data->feature_indices = feature_indices;
    shap_data->feature_values = feature_values;
    shap_data->predictions = predictions;

    memset(shap_data->active_nodes, 0, sizeof(bool)*shap_data->n_nodes);
    int poly_size = (metadata->max_depth + 1) * metadata->max_depth * metadata->output_dim;
    shap_data->C = init_zero_mat(poly_size);
    shap_data->G = init_zero_mat(poly_size);
    return shap_data;
}

void reset_shap_arrays(shapData *shap_data, const ensembleMetaData *metadata){
    memset(shap_data->active_nodes, 0, sizeof(bool)*shap_data->n_nodes);
    int poly_size = (metadata->max_depth + 1) * metadata->max_depth * metadata->output_dim;
    memset(shap_data->C, 0, sizeof(float)*poly_size);
    memset(shap_data->G, 0, sizeof(float)*poly_size);
    // only set the first row of C
    set_mat_value(shap_data->C, metadata->max_depth * metadata->output_dim, 1.0f, metadata->par_th); 
}

void dealloc_shap_data(shapData *shap_data){
    delete[] shap_data->left_children;
    delete[] shap_data->right_children;
    delete[] shap_data->active_nodes;
    delete[] shap_data->numerics;
    delete[] shap_data->feature_parent_node;
    delete[] shap_data->feature_indices;
    delete[] shap_data->feature_values;
    delete[] shap_data->predictions;
    delete[] shap_data->weights;
    delete[] shap_data->max_unique_features;
    delete[] shap_data->categorical_values;
    delete[] shap_data->C;
    delete[] shap_data->G;
    delete shap_data;
}

void print_shap_data(const shapData *shap_data, const ensembleMetaData *metadata){
    printf("**** shap_data with %d nodes *****\n", shap_data->n_nodes);
    printf("left_children: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%d", shap_data->left_children[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");

    printf("right_children: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%d", shap_data->right_children[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");
    printf("feature_parent_node: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%d", shap_data->feature_parent_node[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");
    printf("max_unique_features: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%d", shap_data->max_unique_features[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");
    printf("feature_indices: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%d", shap_data->feature_indices[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");
    printf("feature_values: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%f", shap_data->feature_values[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");
    printf("weights: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%f", shap_data->weights[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");
    printf("predictions: [");
    for (int i = 0; i < shap_data->n_nodes*metadata->output_dim; i++){
        printf("%f", shap_data->predictions[i]);
        if (i < shap_data->n_nodes*metadata->output_dim - 1)
            printf(", ");
    }
    printf("]\n");

    
}

void linear_tree_shap(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values, int crnt_node, int crnt_depth, int crnt_feature, const int sample_offset){
    /*
    Implementation based on - https://github.com/yupbank/linear_tree_shap
    See Linear TreeShap, Yu et al, 2023, https://arxiv.org/pdf/2209.08192 
    */
    float p_e_ancestor = 0.0f; 
    int feature_parent_node = shap_data->feature_parent_node[crnt_node];
    
    if (feature_parent_node >= 0){
        shap_data->active_nodes[crnt_node] &= shap_data->active_nodes[feature_parent_node];
        shap_data->active_nodes[crnt_node] &= shap_data->weights[crnt_node] > 0.0f;
        // zero sample nodes should not contribute to SHAP value calculation
        if (shap_data->active_nodes[feature_parent_node])
            p_e_ancestor = (shap_data->weights[feature_parent_node] > 0.0f) ? 1.0f / shap_data->weights[feature_parent_node] : 0.0f;
    }
    int col_size = metadata->max_depth * metadata->output_dim;
    float *G_depth = shap_data->G + crnt_depth * col_size;
    float *G_next_depth = shap_data->G + (crnt_depth + 1) * col_size;
    float *C_depth = shap_data->C + crnt_depth * col_size;

    int poly_degree = 0;
    int left = shap_data->left_children[crnt_node];
    int right = shap_data->right_children[crnt_node];
    bool is_leaf = left < 0 && right < 0;
   
    int shap_size = (metadata->n_num_features + metadata->n_cat_features)*metadata->output_dim;

    float p_e = 0.0f;
    if (crnt_feature >= 0){
        // zero sample nodes should not contribute to SHAP value calculation
        if (shap_data->active_nodes[crnt_node]){
            p_e = (shap_data->weights[crnt_node] > 0.0f) ? 1.0f / shap_data->weights[crnt_node]: 0.0f;
        }

        float *C_prev_depth = shap_data->C + (crnt_depth - 1) * col_size;
        _broadcast_mat_elementwise_mult_by_vec_into_mat(C_depth, C_prev_depth, shap_data->base_poly, p_e, metadata->max_depth, metadata->output_dim, metadata->par_th, false);
        if (feature_parent_node >= 0){
            _broadcast_mat_elementwise_div_by_vec(C_depth, shap_data->base_poly, p_e_ancestor, metadata->max_depth, metadata->output_dim, metadata->par_th);
        }

    }

    if (is_leaf){
        _broadcast_mat_elementwise_mult_by_vec_into_mat(G_depth, C_depth, shap_data->predictions + crnt_node * metadata->output_dim, 0.0f, metadata->max_depth, metadata->output_dim, metadata->par_th, true);
    }
    else{
        bool is_greater = (shap_data->numerics[crnt_node]) ? dataset->obs[sample_offset*metadata->n_num_features + shap_data->feature_indices[crnt_node]] > shap_data->feature_values[crnt_node]: strcmp(&dataset->categorical_obs[(sample_offset*metadata->n_cat_features + shap_data->feature_indices[crnt_node]) * MAX_CHAR_SIZE],  shap_data->categorical_values + crnt_node*MAX_CHAR_SIZE) == 0;
        shap_data->active_nodes[right] = (is_greater) ? true : false;
        shap_data->active_nodes[left] = (is_greater) ? false : true;
        linear_tree_shap(metadata, edata, shap_data, dataset, shap_values, left, crnt_depth + 1, shap_data->feature_indices[crnt_node], sample_offset);
        poly_degree = shap_data->max_unique_features[crnt_node] - shap_data->max_unique_features[left];
        _broadcast_mat_elementwise_mult_by_vec(G_next_depth, shap_data->offset_poly + poly_degree * metadata->max_depth, 0.0f, metadata->max_depth, metadata->output_dim, metadata->par_th);
        _copy_mat(G_depth, G_next_depth, col_size, metadata->par_th);
        linear_tree_shap(metadata, edata, shap_data, dataset, shap_values, right, crnt_depth + 1, shap_data->feature_indices[crnt_node], sample_offset);
        poly_degree = shap_data->max_unique_features[crnt_node] - shap_data->max_unique_features[right];
        _broadcast_mat_elementwise_mult_by_vec(G_next_depth, shap_data->offset_poly + poly_degree * metadata->max_depth, 0.0f, metadata->max_depth, metadata->output_dim, metadata->par_th);
        _element_wise_addition(G_depth, G_next_depth, col_size, metadata->par_th);
    }

    if (crnt_feature >= 0){
         if(feature_parent_node >=0 && !shap_data->active_nodes[feature_parent_node]){
            return;	
        }
        const float *norm_value = shap_data->norm_values + shap_data->max_unique_features[crnt_node] * metadata->max_depth;
        const float *offset = shap_data->offset_poly;

        add_edge_shapley(shap_values + sample_offset*shap_size + crnt_feature*metadata->output_dim, G_depth, offset, shap_data->base_poly, p_e, norm_value, shap_data->max_unique_features[crnt_node], metadata->output_dim);
        if (feature_parent_node >= 0){
            poly_degree = shap_data->max_unique_features[feature_parent_node] - shap_data->max_unique_features[crnt_node];
            norm_value = shap_data->norm_values + shap_data->max_unique_features[feature_parent_node] * metadata->max_depth;
            offset = shap_data->offset_poly + poly_degree * metadata->max_depth;
            subtract_closest_parent_edge_shapley(shap_values + sample_offset*shap_size + crnt_feature*metadata->output_dim, G_depth, offset, shap_data->base_poly, p_e_ancestor, norm_value, shap_data->max_unique_features[feature_parent_node], metadata->output_dim);
        }
    }
} 

void get_shap_values(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values){
    for (int sample_idx = 0; sample_idx < dataset->n_samples; ++sample_idx){
        reset_shap_arrays(shap_data, metadata);
        linear_tree_shap(metadata, edata, shap_data, dataset, shap_values, 0, 0, -1, sample_idx);
    }
}

void add_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float p_e, const float *norm_value, int d, int output_dim)
{
    for (int j = 0; j < output_dim; j++){
        float tmp_res = 0.0f;
        for (int i = 0; i < d; i++){
            tmp_res += e[i*output_dim + j] * offset[i] / (base_poly[i] + p_e) * norm_value[i];
        }
        tmp_res /= static_cast<float>(d);
        shap_values[j] += tmp_res*(p_e - 1.0f);
    }
}

void subtract_closest_parent_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float p_e_ancestor, const float *norm_value, int d, int output_dim)
{
    for (int j = 0; j < output_dim; j++){
        float tmp_res = 0.0f;
        for (int i = 0; i < d; i++){
            tmp_res += e[i*output_dim + j] * offset[i] / (base_poly[i] + p_e_ancestor) * norm_value[i];
        }
        tmp_res /= static_cast<float>(d);
        shap_values[j] -= tmp_res*(p_e_ancestor - 1.0f);
    }
}
