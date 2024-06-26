#include <iostream>
#include <stdexcept>
#include <cstring>

#include "shap.h"
#include "types.h"
#include "utils.h"
#include "math_ops.h"
#include "data_structs.h"

shapData* alloc_shap_data(const ensembleMetaData *metadata, const ensembleData *edata, const int tree_idx){
    if (tree_idx < 0 || tree_idx >= metadata->n_trees){
        std::cerr << "ERROR: invalid tree_idx " << tree_idx << " in ensemble with ntrees = " << metadata->n_trees << std::endl;
        throw std::runtime_error("Invalid tree index");
        return nullptr;
    }
    
    int stop_leaf_idx = (tree_idx == metadata->n_trees - 1) ? metadata->n_leaves : edata->tree_indices[tree_idx+1];
    int start_leaf_idx = edata->tree_indices[tree_idx];
    int n_leaves = stop_leaf_idx - start_leaf_idx;

    stack<nodeInfo> node_stack(n_leaves * metadata->max_depth);
    nodeInfo root = {0, -1, 0, false, false};  // Assuming starting from root node
    node_stack.push(root);
    int n_nodes = 0;
    int leaf_idx = 0;

    // Allocate arrays for storing data (adjust sizes as needed)
    int *feature_prev_node = new int[n_leaves * metadata->max_depth];
    int *left_children = new int[n_leaves * metadata->max_depth];
    int *right_children = new int[n_leaves * metadata->max_depth];
    int *features_indices = new int[n_leaves * metadata->max_depth];
    float *feature_values = new float[n_leaves * metadata->max_depth];
    float *weights = new float[n_leaves * metadata->max_depth];
    for (int i = 0; i < n_leaves * metadata->max_depth; ++i){
        left_children[i] = -1;
        right_children[i] = -1;
        features[i] = -1;
        feature_prev_node[i] = -1;
        weights[i] = 1.0f;
        feature_values[i] = INFINITY;
    }
    int *parents = new int[n_leaves * metadata->max_depth];
    int left_ctr = 0;
    int right_ctr = 0;
    int *node_unique_features = new int[n_leaves * metadata->max_depth];
    float weight;
    
    memset(node_unique_features, 0, sizeof(int) * n_leaves * metadata->max_depth);
    // Process the tree using DFS
    while (!node_stack.is_empty()) {
        nodeInfo crnt_node = node_stack.top();
        node_stack.pop();
        crnt_node.idx = n_nodes;

        // Store the parent index for the current node
        parents[n_nodes] = crnt_node.parent_idx;
        weight = edata->edge_weight[]
        if (crnt_node.is_left)
            left_children[crnt_node.parent_idx] = crnt_node.idx;
        if (crnt_node.is_right)
            right_children[crnt_node.parent_idx] = crnt_node.idx;
        // If not at leaf, push children nodes onto the stack
        if (crnt_node.depth  < depths[leaf_idx]) {
            nodeInfo right_child = {0, n_nodes, crnt_node.depth + 1, false, true};
            
            node_stack.push(right_child);
            nodeInfo left_child = {0, n_nodes, crnt_node.depth + 1, true, false};
            node_stack.push(left_child);
            int feature_idx = edata->feature_indices[leaf_idx*metadata->max_depth + crnt_node.depth];
            features_indices[n_nodes] = feature_idx;
            feature_values[n_nodes] = eddata->feature_values[leaf_idx*metadata->max_depth + crnt_node.depth];
            
            int parent_idx = parents[n_nodes];
            bool found = false;
            while (parent_idx >= 0) {
                if (feature_idx ==  features[parent_idx]){
                    found = true;
                    break;
                }
                parent_idx = parents[parent_idx];
            }
            if (found)
                feature_prev_node[n_nodes] = parent_idx;

        } else {
            // Calculate number of unique features at the leaf node
            int n_unique_features = count_distinct(edata->feature_indices + leaf_idx * metadata->max_depth, edata->bias[leaf_idx]);
            // Backtrack to update node_unique_features array
            int parent_idx = parents[n_nodes];
            if (n_unique_features > node_unique_features[n_nodes])
                node_unique_features[n_nodes] = n_unique_features;
            while (parent_idx >= 0) {
                if (n_unique_features > node_unique_features[parent_idx])
                    node_unique_features[parent_idx] = n_unique_features;
                parent_idx = parents[parent_idx];
            }
            // Increment leaf index and mark children as non-existent (-1)
            ++leaf_idx;
        }
        if (crnt_node.parent_idx >= 0){
            float w = edata->edge_weights[leaf_idx*metadata->max_depth + crnt_node.depth];
            // adjust node weight when a feature is used multiple times in a path
            if (feature_prev_node[n_nodes] >= 0)
                w *= weights[feature_prev_node[n_nodes]];
            weights[n_nodes] = w;
        }
        
        // Move to the next node
        ++n_nodes;
    }
    
    delete[] parents;
    
    shapData *shap_data = new shapData;
    shap_data->left_children = left_children;
    shap_data->right_children = right_children;
    shap_data->active_nodes = new bool[n_nodes];
    shap_data->feature_prev_node = feature_prev_node;
    shap_data->node_unique_features = node_unique_features;
    shap_data->n_nodes = n_nodes;
    shap_data->weights = weights;
    shap_data->features_indices = features_indices;
    shap_data->feature_values = feature_values;

    int size = (metadata->max_depth + 1) * metadata->max_depth;
    shap_data->C = new float[size];
    shap_data->E = new float[size];
    for (int i = 0; i < size; ++i)
        shap_data->C[i] = 1.0f;
}

void dealloc_shap_data(shapData *shap_data){
    delete[] shap_data->left_children;
    delete[] shap_data->right_children;
    delete[] shap_data->active_nodes;
    delete[] shap_data->feature_prev_node;
    delete[] shap_data->features_indices;
    delete[] shap_data->features_values;
    delete[] shap_data->weights;
    delete[] shap_data->node_unique_features;
    delete[] shap_data->C;
    delete[] shap_data->E;
    delete shap_data;
}

void shap_inference(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values, int crnt_node, int crnt_depth, int feature, const int sample_offset){

    float temp_shap = 0.0f; 
    int feature_prev_node = shap_data->feature_prev_node[crnt_node];
    
    if (feature_prev_node >= 0){
        shap_data->active_nodes[crnt_node] = shap_data->active_nodes[crnt_node] & shap_data->active_nodes[feature_prev_node];
        if (shap_data->active_nodes[feature_prev_node])
            temp_shap = 1.0f / shap_data->weights[feature_prev_node];
    }
    float *current_e = shap_data->E + crnt_depth * metadata->max_depth;
    float *child_e = shap_data->E + (crnt_depth + 1) * metadata->max_depth;
    float *crnt_c = shap_data->C + crnt_depth * metadata->max_depth;
    float q = 0.0f;
    if (feature >= 0)
    {
        if (shap_data->active_nodes[crnt_node])
        {
            q = 1.0f / shap_data->weights[crnt_node];
        }

        float *prev_c = shap_data->C + (crnt_depth - 1) * metadata->max_depth;
        for (int i = 0; i < metadata->max_depth; i++)
        {
            current_c[i] = prev_c[i] * (Base[i] + q);
        }

        if (feature_prev_node >= 0)
        {
            for (int i = 0; i < metadata->max_depth; i++)
            {
                current_c[i] = current_c[i] / (Base[i] + temp_shap);
            }
        }
    }
    int offset_degree = 0;
    int left = shap_data->children_left[crnt_node];
    int right = shap_data->children_right[crnt_node];
    if (left >= 0)
    {
        shap_data->active_nodes[right] = (dataset->obs[sample_offset + shap_data->features_indices[crnt_node]] > shap_data->feature_values[crnt_node]) ? true : false;
        shap_data->active_nodes[left] = (dataset->obs[sample_offset + shap_data->features_indices[crnt_node]] > shap_data->feature_values[crnt_node]) ? false : true;
        shap_inference(metadata, edata, shap_data, dataset, shap_values, left, crnt_depth + 1, shap_data->feature_indices[crnt_node], sample_offset);
    
        offset_degree = shap_data->node_unique_features[crnt_node] - shap_data->node_unique_features[left];
        _element_wise_multiplication(child_e, shap_data->offset_poly + offset_degree * metadata->max_depth, metadata->max_depth, metadata->par_th);
        _copy_mat(current_e, child_e, metadata->max_depth, metadta->par_th);
        shap_inference(metadata, edata, shap_data, dataset, shap_values, right, crnt_depth + 1, shap_data->feature_indices[crnt_node], sample_offset);
        offset_degree = shap_data->node_unique_features[crnt_node] - shap_data->node_unique_features[right];
        _element_wise_multiplication(child_e, shap_data->offset_poly + offset_degree * metadata->max_depth, metadata->max_depth, metadata->par_th);
        _element_wise_addition(current_e, child_e, metadata->max_depth, metadata->par_th);
    }
    else
    {
        
        times(current_c, current_e, tree.leaf_predictions[node], tree.max_depth);
    }
    if (feature >= 0)
    {
	if(parent >=0 && !activation[parent]){
	    return;	
	}
        const tfloat *normal = Norm + tree.edge_heights[node] * tree.max_depth;
        const tfloat *offset = Offset;
        value[feature] += (q - 1) * psi(current_e, offset, Base, q, normal, tree.edge_heights[node]);
        if (parent >= 0)
        {
            offset_degree = tree.edge_heights[parent] - tree.edge_heights[node];
            const tfloat* normal = Norm + tree.edge_heights[parent] * tree.max_depth;
            const tfloat* offset = Offset + offset_degree * tree.max_depth;
            value[feature] -= (s - 1) * psi(current_e, offset, Base, s, normal, tree.edge_heights[parent]);
        }
    }

} 

void inference_v2(const Tree &tree,
               const tfloat *Base,
               const tfloat *Offset,
               const tfloat *Norm,
               const tfloat *x,
               bool *activation,
               tfloat *value,
               tfloat *C,
               tfloat *E,
               int node = 0,
               int feature = -1,
               int depth = 0)
{
    tfloat s = 0.;
    int parent = tree.parents[node];
    if (parent >= 0)
    {
        activation[node] = activation[node] & activation[parent];
        if (activation[parent])
        {
            s = 1 / tree.weights[parent];
        }
    }

    tfloat *current_e = E + depth * tree.max_depth;
    tfloat *child_e = E + (depth + 1) * tree.max_depth;
    tfloat *current_c = C + depth * tree.max_depth;
    tfloat q = 0.;
    if (feature >= 0)
    {
        if (activation[node])
        {
            q = 1 / tree.weights[node];
        }

        tfloat *prev_c = C + (depth - 1) * tree.max_depth;
        for (int i = 0; i < tree.max_depth; i++)
        {
            current_c[i] = prev_c[i] * (Base[i] + q);
        }

        if (parent >= 0)
        {
            for (int i = 0; i < tree.max_depth; i++)
            {
                current_c[i] = current_c[i] / (Base[i] + s);
            }
        }
    }
    int offset_degree = 0;
    int left = tree.children_left[node];
    int right = tree.children_right[node];
    if (left >= 0)
    {
        if (x[tree.features[node]] <= tree.thresholds[node])
        {
            activation[left] = true;
            activation[right] = false;
        }
        else
        {
            activation[left] = false;
            activation[right] = true;
        }
        inference_v2(tree, Base, Offset, Norm, x, activation, value, C, E, left, tree.features[node], depth + 1);
        offset_degree = tree.edge_heights[node] - tree.edge_heights[left];
        times_broadcast(Offset + offset_degree * tree.max_depth, child_e, tree.max_depth);
        write(child_e, current_e, tree.max_depth);
        inference_v2(tree, Base, Offset, Norm, x, activation, value, C, E, right, tree.features[node], depth + 1);
        offset_degree = tree.edge_heights[node] - tree.edge_heights[right];
        times_broadcast(Offset + offset_degree * tree.max_depth, child_e, tree.max_depth);
        addition(child_e, current_e, tree.max_depth);
    }
    else
    {
        times(current_c, current_e, tree.leaf_predictions[node], tree.max_depth);
    }
    if (feature >= 0)
    {
	if(parent >=0 && !activation[parent]){
	    return;	
	}
        const tfloat *normal = Norm + tree.edge_heights[node] * tree.max_depth;
        const tfloat *offset = Offset;
        value[feature] += (q - 1) * psi(current_e, offset, Base, q, normal, tree.edge_heights[node]);
        if (parent >= 0)
        {
            offset_degree = tree.edge_heights[parent] - tree.edge_heights[node];
            const tfloat* normal = Norm + tree.edge_heights[parent] * tree.max_depth;
            const tfloat* offset = Offset + offset_degree * tree.max_depth;
            value[feature] -= (s - 1) * psi(current_e, offset, Base, s, normal, tree.edge_heights[parent]);
        }
    }
};


inline void linear_tree_shap_v2(const Tree &tree,
                                const tfloat *Base,
                                const tfloat *Offset,
                                const tfloat *Norm,
                                const tfloat* X,
                                int n_row,
                                int n_col,
                                tfloat * out)
{
    int size = (tree.max_depth + 1) * tree.max_depth;
    tfloat *C = new tfloat[size];
    std::fill_n(C, size, 1.);
    tfloat *E = new tfloat[size];
    bool *activation = new bool[tree.num_nodes];
    for (int i = 0; i < n_row; i++)
    {
        const tfloat *x = X + i*n_col;
        tfloat *value = out + i*n_col;
        inference_v2(tree, Base, Offset, Norm, x, activation, value, C, E);
    }
    delete[] C;
    delete[] E;
    delete[] activation;
};
