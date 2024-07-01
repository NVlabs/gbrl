#ifndef SHAP_H
#define SHAP_H

#include "types.h"
struct shapData {
    int n_nodes; 
    const float *base_poly;
    const float *offset_poly;
    const float *norm_values;
    float *E;
    float *C;
    bool *active_nodes;
    int *feature_prev_node; 
    int *node_unique_features;
    int *left_children;
    int *right_children;
    int *feature_indices;
    float *feature_values;
    float *predictions;
    float *weights;
};

shapData* alloc_shap_data(const ensembleMetaData *metadata, const ensembleData *edata, const int tree_idx);
void dealloc_shap_data(shapData *shap_data);
void get_shap_values(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values);
void print_shap_data(const shapData *shap_data, const ensembleMetaData *metadata);
void shap_inference(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values, int crnt_node, int crnt_depth, int feature, const int sample_offset);
void add_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float q, const float *norm_value, int d, int output_dim);
void subtract_closest_parent_edge_shapley(float *shap_values, float *e, const float *offset, const float *base_poly, float q_parent, const float *norm_value, int d, int output_dim);

#endif 