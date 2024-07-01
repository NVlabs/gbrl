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
    int leaf_idx = 0;

    // Allocate arrays for storing data (adjust sizes as needed)
    int *feature_prev_node = new int[n_leaves * metadata->max_depth];
    int *left_children = new int[n_leaves * metadata->max_depth];
    int *right_children = new int[n_leaves * metadata->max_depth];
    int *feature_indices = new int[n_leaves * metadata->max_depth];
    float *feature_values = new float[n_leaves * metadata->max_depth];
    float *predictions = new float[n_leaves * metadata->max_depth * metadata->output_dim];
    float *weights = new float[n_leaves * metadata->max_depth];
    for (int i = 0; i < n_leaves * metadata->max_depth; ++i){
        left_children[i] = -1;
        right_children[i] = -1;
        feature_indices[i] = -1;
        feature_prev_node[i] = -1;
        weights[i] = 1.0f;
        feature_values[i] = INFINITY;
    }
    int *parents = new int[n_leaves * metadata->max_depth];
    int *node_unique_features = new int[n_leaves * metadata->max_depth];
    
    memset(node_unique_features, 0, sizeof(int) * n_leaves * metadata->max_depth);
    memset(predictions, 0, sizeof(float) * n_leaves * metadata->max_depth * metadata->output_dim);
    // Process the tree using DFS
    while (!node_stack.is_empty()) {
        nodeInfo crnt_node = node_stack.top();
        node_stack.pop();
        crnt_node.idx = n_nodes;

        // Store the parent index for the current node
        parents[n_nodes] = crnt_node.parent_idx;
        weights[n_nodes] = edata->edge_weights[leaf_idx*metadata->max_depth + crnt_node.depth];
        if (crnt_node.is_left)
            left_children[crnt_node.parent_idx] = crnt_node.idx;
        if (crnt_node.is_right)
            right_children[crnt_node.parent_idx] = crnt_node.idx;
        // If not at leaf, push children nodes onto the stack
        if (crnt_node.depth  < edata->depths[leaf_idx]) {
            nodeInfo right_child = {0, n_nodes, crnt_node.depth + 1, false, true};
            
            node_stack.push(right_child);
            nodeInfo left_child = {0, n_nodes, crnt_node.depth + 1, true, false};
            node_stack.push(left_child);
            int feature_idx = edata->feature_indices[leaf_idx*metadata->max_depth + crnt_node.depth];
            feature_indices[n_nodes] = feature_idx;
            feature_values[n_nodes] = edata->feature_values[leaf_idx*metadata->max_depth + crnt_node.depth];
            
            int parent_idx = parents[n_nodes];
            bool found = false;
            while (parent_idx >= 0) {
                if (feature_idx ==  feature_indices[parent_idx]){
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
            for (int d = 0; d < metadata->output_dim; ++d)
                predictions[n_nodes*metadata->output_dim + d] = edata->values[leaf_idx*metadata->max_depth + d];
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
    shap_data->feature_indices = feature_indices;
    shap_data->feature_values = feature_values;
    shap_data->predictions = predictions;

    int poly_size = (metadata->max_depth + 1) * metadata->max_depth * metadata->output_dim;
    float *C = new float[poly_size];
    set_mat_value(C, poly_size, 1.0f, metadata->par_th);
    shap_data->C = C;
    shap_data->E = init_zero_mat(poly_size);
    return shap_data;
}

void dealloc_shap_data(shapData *shap_data){
    delete[] shap_data->left_children;
    delete[] shap_data->right_children;
    delete[] shap_data->active_nodes;
    delete[] shap_data->feature_prev_node;
    delete[] shap_data->feature_indices;
    delete[] shap_data->feature_values;
    delete[] shap_data->predictions;
    delete[] shap_data->weights;
    delete[] shap_data->node_unique_features;
    delete[] shap_data->C;
    delete[] shap_data->E;
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
    printf("feature_prev_node: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%d", shap_data->feature_prev_node[i]);
        if (i < shap_data->n_nodes - 1)
            printf(", ");
    }
    printf("]\n");
    printf("node_unique_features: [");
    for (int i = 0; i < shap_data->n_nodes; i++){
        printf("%d", shap_data->node_unique_features[i]);
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

void shap_inference(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values, int crnt_node, int crnt_depth, int feature, const int sample_offset){
    printf("running shap inference at depth %i for feature %i with sample offset %i\n", crnt_depth, feature, sample_offset);
    float q_parent = 0.0f; 
    int feature_prev_node = shap_data->feature_prev_node[crnt_node];
    
    if (feature_prev_node >= 0){
        shap_data->active_nodes[crnt_node] = shap_data->active_nodes[crnt_node] & shap_data->active_nodes[feature_prev_node];
        if (shap_data->active_nodes[feature_prev_node])
            q_parent = 1.0f / shap_data->weights[feature_prev_node];
    }
    int col_size = metadata->max_depth * metadata->output_dim;
    float *current_e = shap_data->E + crnt_depth * col_size;
    float *child_e = shap_data->E + (crnt_depth + 1) * col_size;
    float *current_c = shap_data->C + crnt_depth * col_size;
    float q = 0.0f;

    if (feature >= 0){
        if (shap_data->active_nodes[crnt_node]){
            q = 1.0f / shap_data->weights[crnt_node];
        }

        float *prev_c = shap_data->C + (crnt_depth - 1) * col_size;
        _broadcast_mat_elementwise_mult_by_vec_into_mat(current_c, prev_c, shap_data->base_poly, q, metadata->max_depth,  metadata->output_dim, metadata->par_th);

        if (feature_prev_node >= 0){
            _broadcast_mat_elementwise_div_by_vec(current_c, shap_data->base_poly, 0.0f, metadata->max_depth, metadata->output_dim, metadata->par_th);
        }
    }
    int offset_degree = 0;
    int left = shap_data->left_children[crnt_node];
    int right = shap_data->right_children[crnt_node];
    if (left >= 0){
        shap_data->active_nodes[right] = (dataset->obs[sample_offset*metadata->n_num_features + shap_data->feature_indices[crnt_node]] > shap_data->feature_values[crnt_node]) ? true : false;
        shap_data->active_nodes[left] = (dataset->obs[sample_offset*metadata->n_num_features + shap_data->feature_indices[crnt_node]] > shap_data->feature_values[crnt_node]) ? false : true;
        printf("crnt_node %d\n", crnt_node);
        shap_inference(metadata, edata, shap_data, dataset, shap_values, left, crnt_depth + 1, shap_data->feature_indices[crnt_node], sample_offset);
    
        offset_degree = shap_data->node_unique_features[crnt_node] - shap_data->node_unique_features[left];
        printf("multiplying left child mult\n");
        _broadcast_mat_elementwise_mult_by_vec(child_e, shap_data->offset_poly + offset_degree * metadata->max_depth, 0.0f, metadata->max_depth, metadata->output_dim, metadata->par_th);
        _copy_mat(current_e, child_e, col_size, metadata->par_th);
        shap_inference(metadata, edata, shap_data, dataset, shap_values, right, crnt_depth + 1, shap_data->feature_indices[crnt_node], sample_offset);
        offset_degree = shap_data->node_unique_features[crnt_node] - shap_data->node_unique_features[right];
        printf("multiplying right child mult\n");
        _broadcast_mat_elementwise_mult_by_vec(child_e, shap_data->offset_poly + offset_degree * metadata->max_depth, 0.0f, metadata->max_depth, metadata->output_dim, metadata->par_th);
        printf("ele addition\n");
        _element_wise_addition(current_e, child_e, col_size, metadata->par_th);
    }
    else 
    {
        printf("broadcast mult\n");
        _broadcast_mat_elementwise_mult_by_vec_into_mat(current_e, current_c, shap_data->predictions + crnt_node * metadata->output_dim, 0.0f, metadata->max_depth, metadata->output_dim, metadata->par_th);
    }
    if (feature >= 0){
        if(feature_prev_node >=0 && !shap_data->active_nodes[feature_prev_node]){
            return;	
        }
            const float *norm_value = shap_data->norm_values + shap_data->node_unique_features[crnt_node] * col_size;
            const float *offset = shap_data->offset_poly;
            add_edge_shapley(shap_values + sample_offset*(metadata->n_num_features + metadata->n_cat_features)*metadata->output_dim + feature*metadata->output_dim, current_e, offset, shap_data->base_poly, q, norm_value, shap_data->node_unique_features[crnt_node], metadata->output_dim);
            if (feature_prev_node >= 0)
            {
                offset_degree = shap_data->node_unique_features[feature_prev_node] - shap_data->node_unique_features[crnt_node];
                norm_value = shap_data->norm_values + shap_data->node_unique_features[feature_prev_node] * metadata->max_depth;
                offset = shap_data->offset_poly + offset_degree * metadata->max_depth;
                subtract_closest_parent_edge_shapley(shap_values + sample_offset*(metadata->n_num_features + metadata->n_cat_features)*metadata->output_dim + feature*metadata->output_dim, current_e, offset, shap_data->base_poly, q_parent, norm_value, shap_data->node_unique_features[feature_prev_node], metadata->output_dim);
            }
    }

} 

void get_shap_values(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values){
    int n_sample_threads = calculate_num_threads(dataset->n_samples, metadata->par_th);
    // sample parallelization
    if (n_sample_threads > 1) {
        int samples_per_thread = dataset->n_samples / n_sample_threads;
        omp_set_num_threads(n_sample_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * samples_per_thread;
            int end_idx = (thread_id == n_sample_threads - 1) ? dataset->n_samples : start_idx + samples_per_thread;
            for (int sample_idx = start_idx; sample_idx < end_idx; ++sample_idx) {
                shap_inference(metadata, edata, shap_data, dataset, shap_values, 0, 0, -1, sample_idx);
            }
        }
    // no parallelization
    } else{ 
        for (int sample_idx = 0; sample_idx < dataset->n_samples; ++sample_idx){
            shap_inference(metadata, edata, shap_data, dataset, shap_values, 0, 0, -1, sample_idx);
        }
    }

}

void add_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float q, const float *norm_value, int d, int output_dim)
{
    float *res = new float[output_dim];
    for (int j = 0; j < output_dim; j++){
        res[j] = 0.0f;
        for (int i = 0; i < d; i++){
            res[j] += e[i*output_dim + j] * offset[i] / (base_poly[i] + q) * norm_value[i];
        }
        res[j] /= static_cast<float>(d);
        shap_values[j] += res[j]*(q - 1.0f);
    }
    delete[] res;
}

void subtract_closest_parent_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float q_parent, const float *norm_value, int d, int output_dim)
{
    float *res = new float[output_dim];
    for (int j = 0; j < output_dim; j++){
        res[j] = 0.0f;
        for (int i = 0; i < d; i++){
            res[j] += e[i*output_dim + j] * offset[i] / (base_poly[i] + q_parent) * norm_value[i];
        }
        res[j] /= static_cast<float>(d);
        shap_values[j] -= res[j]*(q_parent - 1.0f);
    }
    delete[] res;
}

// tfloat psi(tfloat *e, const tfloat *offset, const tfloat *Base, tfloat q, const tfloat *n, int d)
// {
//     tfloat res = 0.;
//     for (int i = 0; i < d; i++)
//     {
//         res += e[i] * offset[i] / (Base[i] + q) * n[i];
//     }
//     return res / d;
// }


// void inference_v2(const Tree &tree,
//                const tfloat *Base,
//                const tfloat *Offset,
//                const tfloat *Norm,
//                const tfloat *x,
//                bool *activation,
//                tfloat *value,
//                tfloat *C,
//                tfloat *E,
//                int node = 0,
//                int feature = -1,
//                int depth = 0)
// {
//     tfloat s = 0.;
//     int parent = tree.parents[node];
//     if (parent >= 0)
//     {
//         activation[node] = activation[node] & activation[parent];
//         if (activation[parent])
//         {
//             s = 1 / tree.weights[parent];
//         }
//     }

//     tfloat *current_e = E + depth * tree.max_depth;
//     tfloat *child_e = E + (depth + 1) * tree.max_depth;
//     tfloat *current_c = C + depth * tree.max_depth;
//     tfloat q = 0.;
//     if (feature >= 0)
//     {
//         if (activation[node])
//         {
//             q = 1 / tree.weights[node];
//         }

//         tfloat *prev_c = C + (depth - 1) * tree.max_depth;
//         for (int i = 0; i < tree.max_depth; i++)
//         {
//             current_c[i] = prev_c[i] * (Base[i] + q);
//         }

//         if (parent >= 0)
//         {
//             for (int i = 0; i < tree.max_depth; i++)
//             {
//                 current_c[i] = current_c[i] / (Base[i] + s);
//             }
//         }
//     }
//     int offset_degree = 0;
//     int left = tree.children_left[node];
//     int right = tree.children_right[node];
//     if (left >= 0)
//     {
//         if (x[tree.features[node]] <= tree.thresholds[node])
//         {
//             activation[left] = true;
//             activation[right] = false;
//         }
//         else
//         {
//             activation[left] = false;
//             activation[right] = true;
//         }
//         inference_v2(tree, Base, Offset, Norm, x, activation, value, C, E, left, tree.features[node], depth + 1);
//         offset_degree = tree.edge_heights[node] - tree.edge_heights[left];
//         times_broadcast(Offset + offset_degree * tree.max_depth, child_e, tree.max_depth);
//         write(child_e, current_e, tree.max_depth);
//         inference_v2(tree, Base, Offset, Norm, x, activation, value, C, E, right, tree.features[node], depth + 1);
//         offset_degree = tree.edge_heights[node] - tree.edge_heights[right];
//         times_broadcast(Offset + offset_degree * tree.max_depth, child_e, tree.max_depth);
//         addition(child_e, current_e, tree.max_depth);
//     }
//     else
//     {
//         times(current_c, current_e, tree.leaf_predictions[node], tree.max_depth);
//     }
//     if (feature >= 0)
//     {
// 	if(parent >=0 && !activation[parent]){
// 	    return;	
// 	}
//         const tfloat *normal = Norm + tree.edge_heights[node] * tree.max_depth;
//         const tfloat *offset = Offset;
//         value[feature] += (q - 1) * psi(current_e, offset, Base, q, normal, tree.edge_heights[node]);
//         if (parent >= 0)
//         {
//             offset_degree = tree.edge_heights[parent] - tree.edge_heights[node];
//             const tfloat* normal = Norm + tree.edge_heights[parent] * tree.max_depth;
//             const tfloat* offset = Offset + offset_degree * tree.max_depth;
//             value[feature] -= (s - 1) * psi(current_e, offset, Base, s, normal, tree.edge_heights[parent]);
//         }
//     }
// };


// void times(const tfloat *input, tfloat *output, tfloat scalar, int size)
// {
//     for (int i = 0; i < size; i++)
//     {
//         output[i] = input[i] * scalar;
//     }
// };

// void times_broadcast(const tfloat *input, tfloat *output, int size)
// {
//     for (int i = 0; i < size; i++)
//     {
//         output[i] *= input[i];
//     }
// };


// inline void linear_tree_shap_v2(const Tree &tree,
//                                 const tfloat *Base,
//                                 const tfloat *Offset,
//                                 const tfloat *Norm,
//                                 const tfloat* X,
//                                 int n_row,
//                                 int n_col,
//                                 tfloat * out)
// {
//     int size = (tree.max_depth + 1) * tree.max_depth;
//     tfloat *C = new tfloat[size];
//     std::fill_n(C, size, 1.);
//     tfloat *E = new tfloat[size];
//     bool *activation = new bool[tree.num_nodes];
//     for (int i = 0; i < n_row; i++)
//     {
//         const tfloat *x = X + i*n_col;
//         tfloat *value = out + i*n_col;
//         inference_v2(tree, Base, Offset, Norm, x, activation, value, C, E);
//     }
//     delete[] C;
//     delete[] E;
//     delete[] activation;
// };
