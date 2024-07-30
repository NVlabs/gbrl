//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef TYPES_H
#define TYPES_H

#include <memory>
#include <vector>
#include <string>
#include <cstdint>

#define INITAL_MAX_TREES 50000 //50k
#define TREES_BATCH  25000 // 100 K
#define MAX_CHAR_SIZE 128

class Optimizer;
struct splitCondition {
    int feature_idx;
    float feature_value;
    bool inequality_direction;
    float edge_weight;
    char *categorical_value;
};

struct splitCandidate {
    int feature_idx;
    float feature_value;
    char *categorical_value;
};

struct categoryInfo {
    float total_grad_norm;
    int cat_count;
    int feature_idx;
    std::string feature_name;
};

enum schedulerFunc : uint8_t {
    Const,
    Linear
};

enum optimizerAlgo : uint8_t {
    SGD,
    Adam
};

enum deviceType : uint8_t {
    cpu,
    gpu,
    unspecified
};

enum generatorType: uint8_t {
    Uniform, 
    Quantile
};

enum lossType : uint8_t {
    MultiRMSE,
};

enum scoreFunc: uint8_t {
    L2, 
    Cosine
};

enum NULL_CHECK : uint8_t {
    NULL_OPT,
    VALID
};

enum growPolicy : uint8_t {
    GREEDY,
    OBLIVIOUS
};

struct optimizerConfig {
    std::string algo;
    std::string scheduler_func;
    float init_lr;
    float stop_lr;
    int start_idx;
    int stop_idx;
    int T;
    float beta_1;
    float beta_2;
    float eps;
};

using FloatVector = std::vector<float>;
using IntVector = std::vector<int>;
using BoolVector = std::vector<bool>;

struct ensembleMetaData {
    int n_leaves;
    int n_trees;
    int max_trees;
    int max_leaves;
    int max_trees_batch; // maximum number of trees to add in a batch
    int max_leaves_batch; // maximum number of leaves to add in a batch
    int output_dim;
    int max_depth;
    int min_data_in_leaf;
    int n_bins;
    int par_th;
    float cv_beta;
    int verbose;
    int batch_size;
    bool use_cv;
    scoreFunc split_score_func;
    generatorType generator_type;
    growPolicy grow_policy;
    int n_num_features;
    int n_cat_features;
    int iteration;
};

struct dataSet {
    const float *obs;
    const char *categorical_obs;
    float *grads;
    const float *feature_weights;
    const float *build_grads;
    const float *norm_grads;
    int n_samples;
};

struct ensembleData {
    float *bias;
#ifdef DEBUG
    int *n_samples; // debugging
#endif 
    int *tree_indices;
    int *depths;
    float *values;
    // leaf data
    int* feature_indices;
    float* feature_values;
    float *edge_weights;
    bool* is_numerics;
    bool* inequality_directions; 
    char* categorical_values;  
};

struct serializationHeader {
    uint16_t major_version;
    uint16_t minor_version;
    uint16_t patch_version;
    uint64_t reserved1 = 0;
    uint32_t reserved2 = 0;
};

struct nodeInfo {
    int idx; // relative idx for current tree
    int parent_idx;  // relative idx for current tree
    int depth;
    bool is_left;
    bool is_right;
};

scoreFunc stringToScoreFunc(std::string str);
generatorType stringTogeneratorType(std::string str);
growPolicy stringTogrowPolicy(std::string str);
lossType stringTolossType(std::string str);
deviceType stringTodeviceType(std::string str);
optimizerAlgo stringToAlgoType(std::string str);
schedulerFunc stringToSchedulerType(std::string str);

std::string scoreFuncToString(scoreFunc func);
std::string generatorTypeToString(generatorType type);
std::string growPolicyToString(growPolicy type);
std::string lossTypeToString(lossType type);
std::string deviceTypeToString(deviceType type);
std::string algoTypeToString(optimizerAlgo algo);
std::string schedulerTypeToString(schedulerFunc func);

ensembleMetaData* ensemble_metadata_alloc(int max_trees, int max_leaves, int max_trees_batch, int max_leaves_batch, int output_dim, int max_depth, int min_data_in_leaf, int n_bins, int par_th, float cv_beta, int verbose, int batch_size, bool use_cv, scoreFunc split_score_func, generatorType generator_type, growPolicy grow_policy);
ensembleData* ensemble_data_alloc(ensembleMetaData *metadata);
ensembleData* ensemble_copy_data_alloc(ensembleMetaData *metadata);
ensembleData* copy_ensemble_data(ensembleData *other_edata, ensembleMetaData *metadata);
void ensemble_data_dealloc(ensembleData *edata);
void save_ensemble_data(std::ofstream& file, ensembleData *edata, ensembleMetaData *metadata, deviceType device);
void export_ensemble_data(std::ofstream& header_file, const std::string& model_name, ensembleData *edata, ensembleMetaData *metadata, deviceType device, std::vector<Optimizer*> opts);
ensembleData* load_ensemble_data(std::ifstream& file, ensembleMetaData *metadata);
void allocate_ensemble_memory(ensembleMetaData *metadata, ensembleData *edata);
#endif 