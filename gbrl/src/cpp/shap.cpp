#include "shap.h"
#include "types.h"


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
