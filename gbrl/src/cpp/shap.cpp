#include <iostream>
#include <stdexcept>

#include "shap.h"
#include "types.h"
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

    // Run DFS to pre-process tree for SHAP calculation
    stack<nodeInfo> node_stack(n_leaves*metadata->max_depth);
    nodeInfo root = {start_leaf_idx*metadata->max_depth, 0};
    node_stack.push(root);
    int n_nodes = 0;
    int leaf_idx = start_leaf_idx;
    // pre allocate for maximum number of nodes
    int *feature_prev_node = new int[n_leaves*metadata->max_depth]; 
    int *left_children = new int[n_leaves*metadata->max_depth]; 
    int *right_children = new int[n_leaves*metadata->max_depth]; 
    while(!node_stack.is_empty()){
        nodeInfo *crnt_node = node_stack.pop(); 
        int feature_idx = edata->feature_indices[crnt_node->idx];
        bool found = false;
        int path_depth = crnt_node->depth - 1;
        while (path_depth >= 0){
            if (feature_idx == edata->feature_indices[leaf_idx*metadata->max_depth + path_depth]){
                found = true;
                break;
            }
            --path_depth;
        }
        feature_prev_node[n_nodes] = (found) ? leaf_idx*metadata->max_depth + path_depth : -1;
        
        if (crnt_node->depth < edata->depths[leaf_idx]){
            nodeInfo right_child = {(leaf_idx+1)*metadata->max_depth + crnt_node->depth + 1, crnt_node->depth + 1};
            right_children[n_nodes] = right_child.idx;
            node_stack.push(right_child);
            
            nodeInfo left_child = {leaf_idx*metadata->max_depth + crnt_node->depth + 1, crnt_node->depth + 1};
            node_stack.push(left_child);
            left_children[n_nodes] = left_child.idx;
            
        } else {
            ++leaf_idx;
            left_children[n_nodes] = -1;
            right_children[n_nodes] = -1;
        }
        ++n_nodes;
    }
    

    shapData *shap_data = new shapData;

}

void shap_inference(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values){

    float temp_shap = 0.0f; 
    int node_idx = shap_data->path_idx * metadata->max_depth + shap_data->crnt_depth; 
    int parent_idx = (shap_data->crnt_depth > 0) ? node_idx - 1  : -1;
    
    if (parent_idx >= 0){

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
