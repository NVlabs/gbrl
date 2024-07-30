//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef GBRL_H
#define GBRL_H

#include <string>
#include <tuple>

#include "node.h"
#include "optimizer.h"
#include "loss.h"
#include "split_candidate_generator.h"
#include "types.h"

#ifdef USE_CUDA
#include "cuda_types.h"
#endif


class GBRL {
    public:
        GBRL(int output_dim, int max_depth, int min_data_in_leaf, 
             int n_bins, int par_th, float cv_beta, scoreFunc split_score_func, generatorType generator_type, bool use_control_variates, 
             int batch_size, growPolicy grow_policy, int verbose, deviceType device);
        GBRL(int output_dim, int max_depth, int min_data_in_leaf, 
             int n_bins, int par_th, float cv_beta, std::string split_score_func, std::string generator_type, bool use_control_variates, 
             int batch_size, std::string grow_policy, int verbose, std::string device);
        GBRL(const std::string& filename);
        GBRL(GBRL& other);
        ~GBRL();
        float* tree_shap(const int tree_idx, const float *obs, const char *categorical_obs, const int n_samples, float *norm, float *base_poly, float *offset);
        float* ensemble_shap(const float *obs, const char *categorical_obs, const int n_samples, float *norm, float *base_poly, float *offset);
        static bool cuda_available();
        void to_device(deviceType device);
        std::string get_device();
        int saveToFile(const std::string& filename);
        int exportModel(const std::string& filename, const std::string& modelname);
        int loadFromFile(const std::string& filename);

        void step(const float *obs, const char *categorical_obs, float *grads, const float *feature_weights, const int n_samples, const int n_num_features, const int n_cat_features);
#ifdef USE_CUDA
        void _step_gpu(dataSet *dataset);
        float _fit_gpu(dataSet *dataset, float *targets, const int n_iterations);
#endif
        float fit(float *obs, char *categorical_obs, float *targets, const float *feature_weights, int iterations, const int n_samples, const int n_num_features, const int n_cat_features, bool shuffle = true, std::string _loss_type = "MultiRMSE");
        void set_bias(float *bias, const int output_dim);
        float* get_bias();
        float* predict(const float *obs, const char *categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features, int start_tree_idx, int stop_tree_idx);
        void predict(const float *obs, const char *categorical_obs, float *start_preds, const int n_samples, const int n_num_features, const int n_cat_features, int start_tree_idx = 0, int stop_tree_idx = 0);
        
        float* get_scheduler_lrs();

        int get_num_trees();
        int get_iteration();

        void set_optimizer(optimizerAlgo algo, schedulerFunc scheduler_func, float init_lr, int start_idx, int stop_idx, float stop_lr, int T, float beta_1, float beta_2, float eps, float shrinkage);

        void print_tree(int tree_idx);
        void plot_tree(int tree_idx, const std::string &filename);

        ensembleData *edata;
        ensembleMetaData *metadata;
        serializationHeader sheader;
        std::vector<Optimizer*> opts;
        deviceType device = unspecified;
        bool parallel_predict = true;
#ifdef USE_CUDA
        SGDOptimizerGPU** cuda_opt = nullptr;
        int n_cuda_opts;
#endif 
};


#ifdef USE_CUDA
bool valid_device();
#endif


#endif 