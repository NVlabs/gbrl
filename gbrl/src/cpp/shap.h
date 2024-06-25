#ifndef SHAP_H
#define SHAP_H

struct shapData {
    int n_nodes;
    int path_idx; // leaf idx
    int last_path_idx;  
    int feature_idx; 
    int crnt_depth; 
    const float *base_poly;
    const float *offset_poly;
    const float *norm_values;
    bool *active_nodes;
    int *feature_prev_node; 
    int *left_children;
    int *right_children;
};

shapData* alloc_shap_data(const ensembleMetaData *metadata, const ensembleData *edata, const int tree_idx);

void shap_inference(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values);

#endif 