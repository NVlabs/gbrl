#ifndef SHAP_H
#define SHAP_H

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
    int *features;
    float *feature_values;
    float *weights;
};

shapData* alloc_shap_data(const ensembleMetaData *metadata, const ensembleData *edata, const int tree_idx);
void dealloc_shap_data(shapData *shap_data);

void shap_inference(const ensembleMetaData *metadata, const ensembleData *edata, shapData *shap_data, const dataSet *dataset, float *shap_values);

#endif 