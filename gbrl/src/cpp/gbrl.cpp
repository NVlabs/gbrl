
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <omp.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <numeric>
#include <unordered_set> 
#include <unordered_map>
#include <algorithm> 
#include <random>
#include <cmath>
#include <stdexcept>

#include <cstring>

#ifdef USE_GRAPHVIZ
#include <gvc.h>
#include <cgraph.h>
#include <cstring>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda_predictor.h"
#include "cuda_fitter.h"
#include "cuda_preprocess.h"
#include "cuda_loss.h"
#include "cuda_types.h"
#include "cuda_utils.h"
#endif

#include "gbrl.h"
#include "split_candidate_generator.h"
#include "optimizer.h"
#include "fitter.h"
#include "predictor.h"
#include "loss.h"
#include "utils.h"
#include "shap.h"
#include "math_ops.h"



GBRL::GBRL(int output_dim, int max_depth, int min_data_in_leaf, 
           int n_bins, int par_th, float cv_beta, scoreFunc split_score_func,
           generatorType generator_type, bool use_cv, int batch_size, growPolicy grow_policy, int verbose, 
           deviceType device){
    this->metadata = ensemble_metadata_alloc(INITAL_MAX_TREES, INITAL_MAX_TREES * (1 << max_depth), TREES_BATCH, TREES_BATCH * (1 << max_depth), output_dim, max_depth, min_data_in_leaf, n_bins, par_th, cv_beta, verbose, batch_size, use_cv, split_score_func, generator_type, grow_policy);
    this->sheader = create_header();
#ifdef USE_CUDA
    if (device == gpu){
        this->metadata->max_trees = INITAL_MAX_GPU_TREES;
        this->metadata->max_leaves = INITAL_MAX_GPU_TREES * (1 << max_depth);
        this->metadata->max_trees_batch = GPU_TREES_BATCH;
        this->metadata->max_leaves_batch = GPU_TREES_BATCH * (1 << max_depth);
    }
#endif
    this->to_device(device);
        
}

GBRL::GBRL(int output_dim, int max_depth, int min_data_in_leaf, 
           int n_bins, int par_th, float cv_beta, std::string split_score_func,
           std::string generator_type, bool use_cv, int batch_size, 
           std::string grow_policy, int verbose, std::string device){
    this->metadata = ensemble_metadata_alloc(INITAL_MAX_TREES, INITAL_MAX_TREES * (1 << max_depth), TREES_BATCH, TREES_BATCH * (1 << max_depth), output_dim, max_depth, min_data_in_leaf, n_bins, par_th, cv_beta, verbose, batch_size, use_cv, stringToScoreFunc(split_score_func), stringTogeneratorType(generator_type), stringTogrowPolicy(grow_policy));
    this->sheader = create_header();
#ifdef USE_CUDA
    if (stringTodeviceType(device) == gpu){
        this->metadata->max_trees = INITAL_MAX_GPU_TREES;
        this->metadata->max_leaves = INITAL_MAX_GPU_TREES * (1 << max_depth);
        this->metadata->max_trees_batch = GPU_TREES_BATCH;
        this->metadata->max_leaves_batch = GPU_TREES_BATCH * (1 << max_depth);
    }
#endif
    this->to_device(stringTodeviceType(device));
    

}

GBRL::GBRL(const std::string& filename){
    int status = this->loadFromFile(filename);
    if (status != 0){
        std::cerr << "Error loading . " <<  filename  << std::endl; 
        throw std::runtime_error("File load error");
    }
}

GBRL::GBRL(GBRL& other):  
           opts(), parallel_predict(other.parallel_predict){
        this->metadata = new ensembleMetaData;
        memcpy(this->metadata, other.metadata, sizeof(ensembleMetaData));
        this->device= other.device;
#ifdef USE_CUDA
        if (this->device == gpu)
            this->edata = ensemble_data_copy_gpu_gpu(this->metadata, other.edata);
#endif
        if (this->device == cpu)
            this->edata = copy_ensemble_data(other.edata, this->metadata);

        for(size_t i = 0; i < other.opts.size(); ++i) {
            optimizerAlgo algo = other.opts[i]->getAlgo();
            if (algo == Adam){
                AdamOptimizer* adamOpt = dynamic_cast<AdamOptimizer*>(other.opts[i]);
                this->opts.push_back(new AdamOptimizer(*adamOpt));
            } else if (algo == SGD){
                SGDOptimizer* SGDOpt = dynamic_cast<SGDOptimizer*>(other.opts[i]);
                this->opts.push_back(new SGDOptimizer(*SGDOpt));
            }
        }
}

GBRL::~GBRL() {
    for (size_t i = 0; i < this->opts.size(); i++)
        delete this->opts[i];

#ifdef USE_CUDA
    if (this->device == gpu){
        ensemble_data_dealloc_cuda(this->edata);
        freeSGDOptimizer(this->cuda_opt, this->n_cuda_opts);
    }
#endif
    if (this->device == cpu)
        ensemble_data_dealloc(this->edata);
    this->edata = nullptr;
    delete this->metadata;
    this->metadata = nullptr;
}



void GBRL::to_device(deviceType device){
    if (device == this->device){
        std::cout << "GBRL device is already " << deviceTypeToString(device) << std::endl;
        return;
    }
#ifndef USE_CUDA
    if (device == gpu)
        std::cerr << "GBRL was not compiled for GPU. Using cpu device" << std::endl;
    this->edata = ensemble_data_alloc(this->metadata);
    this->device = cpu;
#else
    if (device == gpu){
        bool is_valid = valid_device();
        if (!is_valid){
            std::cerr << "No GPU device found. Using cpu device" << std::endl;
            device = cpu;
        }
    }
    if (this->device == unspecified){
        if (device == cpu){
            this->edata = ensemble_data_alloc(this->metadata);
            this->device = cpu;
        } else {
            this->edata = ensemble_data_alloc_cuda(this->metadata);
            this->device = gpu;
        }
    } else if (this->device == cpu && device == gpu){
        ensembleData* edata_gpu = ensemble_data_copy_cpu_gpu(this->metadata, this->edata);
        ensemble_data_dealloc(this->edata);
        this->edata = edata_gpu;
        this->device = gpu;
    } else {
        ensembleData* edata_cpu = ensemble_data_copy_gpu_cpu(this->metadata, this->edata);
        ensemble_data_dealloc_cuda(this->edata);
        this->edata = edata_cpu;
        this->device = cpu;
    }
    if (this->device == gpu && this->metadata->use_cv){
        std::cout << "Cannot use control variates with GPU. Setting use_cv to False." << std::endl;
        this->metadata->use_cv = false;
    }
#endif 
    std::cout << "Setting GBRL device to " << deviceTypeToString(device) << std::endl;
}

void GBRL::set_bias(float *bias, const int output_dim){
    if (output_dim != this->metadata->output_dim)
    {
        std::cerr << "Given bias vector has different dimensions than expect. " << " Given: " << output_dim << " expected: " << this->metadata->output_dim << std::endl; 
        throw std::runtime_error("Incompatible dimensions");
        return;
    }
#ifdef USE_CUDA
    if (this->device == gpu)
        cudaMemcpy(this->edata->bias, bias, sizeof(float)*this->metadata->output_dim, cudaMemcpyHostToDevice);
#endif
    if (this->device == cpu)
        memcpy(this->edata->bias, bias, sizeof(float)*this->metadata->output_dim);
}

float* GBRL::get_bias(){
    // returns a copy. must deallocated new float pointer!
#ifdef USE_CUDA
    if (this->device == gpu){
        float *bias = new float[this->metadata->output_dim];
        cudaMemcpy(bias, this->edata->bias, sizeof(float)*this->metadata->output_dim, cudaMemcpyDeviceToHost);
        return bias;
    }
#endif 
    if (this->device == cpu)
        return copy_mat(this->edata->bias, this->metadata->output_dim, this->metadata->par_th);
    return nullptr;
}


float* GBRL::predict(const float *obs, const char *categorical_obs, const int n_samples, const int n_num_features, const int n_cat_features, int start_tree_idx, int stop_tree_idx){
    float *preds = init_zero_mat(n_samples*this->metadata->output_dim);
    for (size_t optIdx = 0; optIdx < this->opts.size(); ++optIdx){
        this->opts[optIdx]->set_memory(n_samples ,this->metadata->output_dim);
    }
    if (this->metadata->iteration == 0){
        this->metadata->n_num_features = n_num_features;
        this->metadata->n_cat_features = n_cat_features;
    }
    if (n_num_features != metadata->n_num_features || n_cat_features != metadata->n_cat_features){
        delete[] preds;
        std::cerr << "Error. Cannot use ensemble with this dataset. Excepted input with " << metadata->n_num_features << " numerical features followed by " << metadata->n_cat_features << " categorical features, but received " << n_num_features << " numerical features and " << n_cat_features << " categorical features.";
        throw std::runtime_error("Incompatible dataset");
    }

    dataSet dataset{obs, categorical_obs, nullptr, nullptr, nullptr, nullptr, n_samples};

    // int n_trees = this->get_num_trees();
#ifdef USE_CUDA
    if (this->device == gpu){
        if (this->cuda_opt == nullptr){
            this->cuda_opt = deepCopySGDOptimizerVectorToGPU(this->opts);
            this->n_cuda_opts = static_cast<int>(this->opts.size());
        }
        predict_cuda(&dataset, preds, this->metadata, this->edata, this->cuda_opt, this->n_cuda_opts, start_tree_idx, stop_tree_idx);
    
    }
#endif
    if (this->device == cpu)
        Predictor::predict_cpu(&dataset, preds, this->edata, this->metadata, start_tree_idx, stop_tree_idx, this->parallel_predict, this->opts);
    
    return preds;

}

void GBRL::predict(const float *obs, const char *categorical_obs, float *start_preds, const int n_samples, const int n_num_features, const int n_cat_features, int start_tree_idx, int stop_tree_idx){
    for (size_t optIdx = 0; optIdx < this->opts.size(); ++optIdx){
        this->opts[optIdx]->set_memory(n_samples ,this->metadata->output_dim);
    }
    if (this->metadata->iteration == 0){
        this->metadata->n_num_features = n_num_features;
        this->metadata->n_cat_features = n_cat_features;
    }
    if (n_num_features != metadata->n_num_features || n_cat_features != metadata->n_cat_features){
        std::cerr << "Error. Cannot use ensemble with this dataset. Excepted input with " << metadata->n_num_features << " numerical features followed by " << metadata->n_cat_features << " categorical features, but received " << n_num_features << " numerical features and " << n_cat_features << " categorical features.";
        throw std::runtime_error("Incompatible dataset");
        return;
    }
    dataSet dataset{obs, categorical_obs, nullptr, nullptr, nullptr, nullptr, n_samples};

    // int n_trees = this->get_num_trees();
#ifdef USE_CUDA
    if (this->device == gpu){
        if (this->cuda_opt == nullptr){
            this->cuda_opt = deepCopySGDOptimizerVectorToGPU(this->opts);
            this->n_cuda_opts = static_cast<int>(this->opts.size());
        }
        predict_cuda(&dataset, start_preds, this->metadata, this->edata, this->cuda_opt, this->n_cuda_opts, start_tree_idx, stop_tree_idx);
    }
#endif
    if (this->device == cpu)
        Predictor::predict_cpu(&dataset, start_preds, this->edata, this->metadata, start_tree_idx, stop_tree_idx, this->parallel_predict, this->opts);

}

int GBRL::get_num_trees(){
    return this->metadata->n_trees;
}

int GBRL::get_iteration(){
    return this->metadata->iteration;
}

std::string GBRL::get_device(){
    return deviceTypeToString(this->device);
}

void GBRL::set_optimizer(optimizerAlgo algo, schedulerFunc scheduler_func, float init_lr, 
                        int start_idx, int stop_idx, 
                        float stop_lr, int T, 
                        float beta_1, float beta_2, 
                        float eps = 1.0e-8, float shrinkage = 1.0e-5){
    if (this->opts.size() >= static_cast<size_t>(this->metadata->output_dim)){
        std::cerr << "Already set " << this->opts.size() << " optimizers. This is the limit." << std::endl;
        throw std::runtime_error("Optimizer Limit Reached");
        return;
    }
    if (start_idx >= stop_idx){
        std::cerr << "start idx " << start_idx << " is not < " << stop_idx << "! Start idx must be smaller than stop idx" << std::endl;
        throw std::runtime_error("invalid index ranges");
        return; 
    }
    if (start_idx < 0 || stop_idx <= 0 || start_idx >= this->metadata->output_dim || stop_idx > this->metadata->output_dim){
        std::cerr << "Invalid start index: "  << start_idx << " or stop index: " << stop_idx << " in range: [0, " << this->metadata->output_dim  <<  "]" << std::endl;
        throw std::runtime_error("invalid index ranges");
        return; 
    }

    (void)shrinkage;

    Optimizer *opt;
    if (algo == Adam){
#ifdef USE_CUDA
        if (this->device == gpu){
            std::cerr << "The Adam optimizer has cpu support only." << std::endl;
            throw std::runtime_error("Incompatible GPU optimizer");
            return;
        }
#endif
        if (scheduler_func == Const){
            opt = new AdamOptimizer(scheduler_func, init_lr, beta_1, beta_2, eps);
        } else if (scheduler_func == Linear){
            opt = new AdamOptimizer(scheduler_func, init_lr, stop_lr, T,  beta_1, beta_2, eps);
        } else {
            std::cerr << "Unrecognized scheduler func." << std::endl;
            opt = nullptr;
            throw std::runtime_error("Unrecognized scheduler func");
            return;
        }
        this->parallel_predict = false;
    } else if (algo == SGD){
        if (scheduler_func == Const){
            opt = new SGDOptimizer(scheduler_func, init_lr);
        } else if (scheduler_func == Linear){
            opt = new SGDOptimizer(scheduler_func, init_lr, stop_lr, T);
        } else {
            std::cerr << "Unrecognized scheduler func." << std::endl;
            opt = nullptr;
            throw std::runtime_error("Unrecognized scheduler func");
            return;
        }
    } else {
            std::cerr << "Unrecognized optimizer algo" << std::endl;
            opt = nullptr;
            throw std::runtime_error("Unrecognized optimizer algo");
            return;
        }
     opt->set_indices(start_idx, stop_idx);
#ifdef DEBUG
            std::cout << "Setting optimizer " << this->opts.size() + 1 << " out of a maximum of " << this->metadata->output_dim << std::endl;
#endif
    this->opts.push_back(opt);
}

float* GBRL::get_scheduler_lrs(){
    if (this->opts.size() == 0){
        std::cerr << "No optimizers found." << std::endl;
        throw std::runtime_error("No optimizers found");
        return nullptr;
    }
    float *lrs = init_zero_mat(static_cast<int>(this->opts.size()));
    int T = this->metadata->n_trees;
    for (size_t i = 0; i < this->opts.size(); ++i){
        lrs[i] = this->opts[i]->scheduler->get_lr(T);
    }
    return lrs;
}

bool GBRL::cuda_available(){
#ifdef USE_CUDA
    return true;
#endif 
    return false;
}

#ifdef USE_CUDA
void GBRL::_step_gpu(dataSet *dataset){
    const int output_dim = this->metadata->output_dim;
    const int n_bins = this->metadata->n_bins;
    const int n_num_features = this->metadata->n_num_features, n_cat_features = this->metadata->n_cat_features;
    const int n_samples = dataset->n_samples;
    cudaError_t err;

    size_t obs_size = sizeof(float)*n_num_features*n_samples;
    size_t cat_obs_size = sizeof(char)*n_cat_features*n_samples*MAX_CHAR_SIZE;
    size_t grads_size = sizeof(float)*output_dim*n_samples;
    size_t feature_size = sizeof(float)*(n_num_features + n_cat_features);
    size_t grads_norm_size = sizeof(float)*n_samples;

    size_t cand_indices_size =  sizeof(int)*n_bins*(n_num_features + n_cat_features);
    size_t cand_float_size =  sizeof(float)*n_bins*(n_num_features + n_cat_features);
    size_t cand_cat_size =  sizeof(char)*n_bins*(n_num_features + n_cat_features)*MAX_CHAR_SIZE;
    size_t cand_numerical_size =  sizeof(bool)*n_bins*(n_num_features + n_cat_features);

    size_t alloc_size = obs_size*2 + cat_obs_size + feature_size +  grads_size*2 + grads_norm_size + cand_indices_size + cand_float_size + cand_cat_size + cand_numerical_size;
    char *device_memory_block; 

    err = cudaMalloc((void**)&device_memory_block, alloc_size);
    if (err != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA allocate step_gpu data error: " << cudaGetErrorString(err)
                << " when trying to allocate " << ((alloc_size) / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        return;
    }
    cudaMemset(device_memory_block, 0, alloc_size);
    size_t trace = 0;
    float *gpu_obs = (float*)(device_memory_block + trace);
    trace += obs_size;
    float *trans_obs = (float*)(device_memory_block + trace);
    trace += obs_size;
    float *gpu_feature_weights = (float*)(device_memory_block + trace);
    trace += feature_size;
    float *gpu_build_grads = (float*)(device_memory_block + trace);
    trace += grads_size;
    float *gpu_grads = (float*)(device_memory_block + trace);
    trace += grads_size;
    float *gpu_grads_norm = (float*)(device_memory_block + trace);
    trace += grads_norm_size;
    int *candidate_indices = (int*)(device_memory_block + trace);
    trace += cand_indices_size;
    float *candidate_values = (float*)(device_memory_block + trace);
    trace += cand_float_size;
    bool *candidate_numerical = (bool*)(device_memory_block + trace);
    trace += cand_numerical_size;
    char *gpu_categorical_obs = (char*)(device_memory_block + trace);
    trace += cat_obs_size;
    char *candidate_categories = (char*)(device_memory_block + trace);
    
    cudaMemcpy(gpu_obs, dataset->obs, sizeof(float)*n_num_features*n_samples, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_categorical_obs, dataset->categorical_obs, sizeof(char)*n_cat_features*n_samples*MAX_CHAR_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_build_grads, dataset->grads, sizeof(float)*output_dim*n_samples, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_grads, dataset->grads, sizeof(float)*output_dim*n_samples, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_feature_weights, dataset->feature_weights, feature_size, cudaMemcpyHostToDevice);
    transpose_matrix(gpu_obs, trans_obs, n_num_features, n_samples);
    preprocess_matrices(gpu_build_grads, gpu_grads_norm, n_samples, output_dim, this->metadata->split_score_func);

    int n_candidates = process_candidates_cuda(gpu_obs, dataset->categorical_obs, gpu_grads_norm, candidate_indices, candidate_values, candidate_categories, candidate_numerical, n_samples, n_num_features, n_cat_features, n_bins, this->metadata->generator_type);
    dataSet cuda_dataset{trans_obs, gpu_categorical_obs, gpu_grads, gpu_feature_weights, gpu_build_grads, gpu_grads_norm, n_samples};
    candidatesData candidata{n_candidates, candidate_indices, candidate_values, candidate_numerical, candidate_categories};
    splitDataGPU *split_data = allocate_split_data(this->metadata, candidata.n_candidates);  
    if (this->metadata->grow_policy == GREEDY)
        fit_tree_greedy_cuda(&cuda_dataset, this->edata, this->metadata, &candidata, split_data);
    else
        fit_tree_oblivious_cuda(&cuda_dataset, this->edata, this->metadata, &candidata, split_data);
    cudaFree(split_data->split_scores);
    delete split_data;
    cudaFree(device_memory_block);
    
    ++this->metadata->iteration;
}

float GBRL::_fit_gpu(dataSet *dataset, float *targets, const int n_iterations){
    const int output_dim = this->metadata->output_dim;
    const int n_bins = this->metadata->n_bins;
    const int n_num_features = this->metadata->n_num_features, n_cat_features = this->metadata->n_cat_features;
    const int n_samples = dataset->n_samples;
    cudaError_t err;

    int n_blocks, threads_per_block;
    get_grid_dimensions(n_samples*output_dim, n_blocks, threads_per_block);

    size_t obs_size = sizeof(float)*n_num_features*n_samples;
    size_t cat_obs_size = sizeof(char)*n_cat_features*n_samples*MAX_CHAR_SIZE;
    size_t grads_size = sizeof(float)*output_dim*n_samples;
    size_t grads_norm_size = sizeof(float)*n_samples;
    size_t feature_size = sizeof(float)*(n_num_features + n_cat_features);

    size_t cand_indices_size =  sizeof(int)*n_bins*(n_num_features + n_cat_features);
    size_t cand_float_size =  sizeof(float)*n_bins*(n_num_features + n_cat_features);
    size_t cand_cat_size =  sizeof(char)*n_bins*(n_num_features + n_cat_features)*MAX_CHAR_SIZE;
    size_t cand_numerical_size =  sizeof(bool)*n_bins*(n_num_features + n_cat_features);
    size_t result_tmp_size = sizeof(float)*n_blocks;
    size_t alloc_size = obs_size*2 + cat_obs_size + feature_size + grads_size*4 + grads_norm_size + cand_indices_size + cand_float_size + cand_cat_size + cand_numerical_size + result_tmp_size;
    char *device_memory_block; 

    if (this->cuda_opt == nullptr){
        this->cuda_opt = deepCopySGDOptimizerVectorToGPU(this->opts);
        this->n_cuda_opts = static_cast<int>(this->opts.size());
    }

    err = cudaMalloc((void**)&device_memory_block, alloc_size);
    if (err != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "CUDA allocate fit_gpu_sl data error: " << cudaGetErrorString(err)
                << " when trying to allocate " << ((alloc_size) / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Free memory: " << (free_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        std::cerr << "Total memory: " << (total_mem / (1024.0 * 1024.0)) << " MB."
                << std::endl;
        throw std::runtime_error("GPU allocation error");
        return -INFINITY;
    }
    cudaMemset(device_memory_block, 0, alloc_size);
    size_t trace = 0;
    float *gpu_obs = (float*)(device_memory_block + trace);
    trace += obs_size;
    float *trans_obs = (float*)(device_memory_block + trace);
    trace += obs_size;
    float *gpu_feature_weights = (float*)(device_memory_block + trace);
    trace += feature_size;
    float *gpu_build_grads = (float*)(device_memory_block + trace);
    trace += grads_size;
    float *gpu_grads = (float*)(device_memory_block + trace);
    trace += grads_size;
    float *gpu_targets = (float*)(device_memory_block + trace);
    trace += grads_size;
    float *gpu_preds = (float*)(device_memory_block + trace);
    trace += grads_size;
    float *gpu_grads_norm = (float*)(device_memory_block + trace);
    trace += grads_norm_size;
    float *result_tmp = (float*)(device_memory_block + trace);
    trace += result_tmp_size;
    int *candidate_indices = (int*)(device_memory_block + trace);
    trace += cand_indices_size;
    float *candidate_values = (float*)(device_memory_block + trace);
    trace += cand_float_size;
    bool *candidate_numerical = (bool*)(device_memory_block + trace);
    trace += cand_numerical_size;
    char *gpu_categorical_obs = (char*)(device_memory_block + trace);
    trace += cat_obs_size;
    char *candidate_categories = (char*)(device_memory_block + trace);

    cudaMemcpy(gpu_obs, dataset->obs, sizeof(float)*n_num_features*n_samples, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_targets, targets, sizeof(float)*output_dim*n_samples, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_feature_weights, dataset->feature_weights, feature_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_categorical_obs, dataset->categorical_obs, sizeof(char)*n_cat_features*n_samples*MAX_CHAR_SIZE, cudaMemcpyHostToDevice);
    transpose_matrix(gpu_obs, trans_obs, n_num_features, n_samples);
    
    dataSet cuda_dataset{gpu_obs, gpu_categorical_obs, gpu_grads, gpu_feature_weights, gpu_build_grads, gpu_grads_norm, n_samples};
    predict_cuda_no_host(&cuda_dataset, gpu_preds, this->metadata, this->edata, this->cuda_opt, this->n_cuda_opts, 0, 0, true);

    MultiRMSEGrad(gpu_preds, gpu_targets, gpu_grads,  output_dim, n_samples, n_blocks, threads_per_block);
    cudaMemcpy(gpu_build_grads, gpu_grads, sizeof(float)*output_dim*n_samples, cudaMemcpyDeviceToDevice);

    preprocess_matrices(gpu_build_grads, gpu_grads_norm, n_samples, output_dim, this->metadata->split_score_func);

    int n_candidates = process_candidates_cuda(gpu_obs, gpu_categorical_obs, gpu_grads_norm, candidate_indices, candidate_values, candidate_categories, candidate_numerical, n_samples, n_num_features, n_cat_features, n_bins, this->metadata->generator_type);

    candidatesData candidata{n_candidates, candidate_indices, candidate_values, candidate_numerical, candidate_categories};
    splitDataGPU *split_data = allocate_split_data(this->metadata, candidata.n_candidates);  
    for (int i = 0 ; i < n_iterations; ++i){
       cuda_dataset.grads = gpu_grads;
       cuda_dataset.obs = trans_obs;
       cuda_dataset.build_grads = gpu_build_grads;
       cuda_dataset.norm_grads = gpu_grads_norm;
        if (this->metadata->grow_policy == GREEDY)
            fit_tree_greedy_cuda(&cuda_dataset, this->edata, this->metadata, &candidata, split_data);
        else
            fit_tree_oblivious_cuda(&cuda_dataset, this->edata, this->metadata, &candidata, split_data);

        ++this->metadata->iteration;

        cuda_dataset.obs = gpu_obs;
        predict_cuda_no_host(&cuda_dataset, gpu_preds, this->metadata, this->edata, this->cuda_opt, this->n_cuda_opts, i, 0, false);
        cudaMemset(gpu_grads, 0, grads_size);
        if (this->metadata->verbose == 0)
            MultiRMSEGrad(gpu_preds, gpu_targets, gpu_grads, output_dim, n_samples, n_blocks, threads_per_block);
        else{
            cudaMemset(result_tmp, 0, result_tmp_size);
            float loss = MultiRMSEGradandLoss(gpu_preds, gpu_targets, gpu_grads, result_tmp, output_dim, n_samples, n_blocks, threads_per_block);
            std::cout << "Boosting iteration: " << this->metadata->iteration << " - MultiRMSE Loss: " << loss << std::endl;
        }
        cudaMemcpy(gpu_build_grads, gpu_grads, grads_size, cudaMemcpyDeviceToDevice);
        
        preprocess_matrices(gpu_build_grads, gpu_grads_norm, n_samples, output_dim, this->metadata->split_score_func);
        
    }
    cudaFree(split_data->split_scores);
    delete split_data;
    cudaMemset(gpu_preds, 0, grads_size);
    cuda_dataset.obs = gpu_obs;
    predict_cuda_no_host(&cuda_dataset, gpu_preds, this->metadata, this->edata, this->cuda_opt, this->n_cuda_opts, 0, 0, true);
    cudaMemset(gpu_grads, 0, grads_size);
    cudaMemset(result_tmp, 0, result_tmp_size);
    float loss = MultiRMSEGradandLoss(gpu_preds, gpu_targets, gpu_grads, result_tmp, output_dim, n_samples, n_blocks, threads_per_block);

    cudaFree(device_memory_block);
    return loss;
}

#endif
void GBRL::step(const float *obs, const char *categorical_obs, float *grads, const float *feature_weights, const int n_samples, const int n_num_features, const int n_cat_features){
    if (this->metadata->iteration == 0){
        this->metadata->n_num_features = n_num_features;
        this->metadata->n_cat_features = n_cat_features;
    }
    if (n_num_features != metadata->n_num_features || n_cat_features != metadata->n_cat_features){
        std::cerr << "Error. Cannot use ensemble with this dataset. Excepted input with " << metadata->n_num_features << " numerical features followed by " << metadata->n_cat_features << " categorical features, but received " << n_num_features << " numerical features and " << n_cat_features << " categorical features.";
        throw std::runtime_error("Incompatible dataset");
        return;
    }
    dataSet dataset{obs, categorical_obs, grads, feature_weights, nullptr, nullptr, n_samples};
#ifdef USE_CUDA
    if (this->device == gpu)
        this->_step_gpu(&dataset);
#endif
    if (this->device == cpu)
        Fitter::step_cpu(&dataset, this->edata, this->metadata);
}

float GBRL::fit(float *obs, char *categorical_obs, float *targets, const float *feature_weights, int iterations, const int n_samples, const int n_num_features, const int n_cat_features, bool shuffle, std::string _loss_type){
    lossType loss_type = stringTolossType(_loss_type);
    int output_dim = metadata->output_dim;
    if (this->metadata->iteration == 0){
        this->metadata->n_num_features = n_num_features;
        this->metadata->n_cat_features = n_cat_features;
    }

    if (n_num_features != metadata->n_num_features || n_cat_features != metadata->n_cat_features){
        std::cerr << "Error. Cannot use ensemble with this dataset. Excepted input with " << metadata->n_num_features << " numerical features followed by " << metadata->n_cat_features << " categorical features, but received " << n_num_features << " numerical features and " << n_cat_features << " categorical features.";
        throw std::runtime_error("Incompatible dataset");
        return -INFINITY;
    }

    for (auto& algo:  this->opts){
        if (algo->getAlgo() == Adam){
            std::cerr << "Adam optimizer not supported in fit function. Use SGD" << std::endl;
            throw std::runtime_error("Unsupported optimizer");
            return 0.0;
        }    
    }

    float* training_obs, *training_targets;
    char *training_cat_obs;
    if (shuffle){
        std::random_device rd;
        std::mt19937 g(rd());
        // Generate indices
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), g);

        // Allocate memory for shuffled data
        training_obs = new float[n_samples * n_num_features];
        training_cat_obs = new char[n_samples * n_cat_features * MAX_CHAR_SIZE];
        training_targets = new float[n_samples * output_dim];

        // Apply shuffled indices
        for (int i = 0; i < n_samples; ++i) {
            if (n_num_features > output_dim){
                for (int j = 0; j < output_dim; ++j) {
                    training_obs[i * n_num_features + j] = obs[indices[i] * n_num_features + j];
                    training_targets[i * output_dim + j] = targets[indices[i] * output_dim + j];
                }
                for (int j = output_dim; j < n_num_features; ++j){
                    training_obs[i * n_num_features + j] = obs[indices[i] * n_num_features + j];
                }
            } else {
                for (int j = 0; j < n_num_features; ++j) {
                    training_obs[i * n_num_features + j] = obs[indices[i] * n_num_features + j];
                    training_targets[i * output_dim + j] = targets[indices[i] * output_dim + j];
                }
                for (int j = n_num_features; j < output_dim; ++j){
                    training_targets[i * output_dim + j] = targets[indices[i] * output_dim + j];
                }
            }
            for (int k = 0; k < n_cat_features; ++k){
                training_cat_obs[(i * n_cat_features + k) * MAX_CHAR_SIZE] = categorical_obs[(indices[i] * n_cat_features + k) * MAX_CHAR_SIZE];
            }
        }
    } else {
        training_obs = obs;
        training_cat_obs = categorical_obs;
        training_targets = targets;
    }

    float *bias = calculate_mean(training_targets, n_samples, output_dim, metadata->par_th);
    this->set_bias(bias, this->metadata->output_dim);
    dataSet dataset{training_obs, training_cat_obs, nullptr, feature_weights, nullptr, nullptr, n_samples};

    float full_loss = -INFINITY;
#ifdef USE_CUDA
    if (this->device == gpu)
        full_loss = this->_fit_gpu(&dataset, training_targets, iterations);
#endif
    if (this->device == cpu){
       full_loss = Fitter::fit_cpu(&dataset, training_targets, this->edata, this->metadata, iterations, loss_type, this->opts);
    }

    if (shuffle){
        delete[] training_obs;
        delete[] training_cat_obs;
        delete[] training_targets;
    }

    return full_loss;   
}

int GBRL::exportModel(const std::string& filename, const std::string& modelname){
    std::ofstream header_file(filename, std::ios::binary);
    if (!header_file.is_open() || header_file.fail()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("File opening error");
        return -1;
    }
    if (this->metadata->grow_policy != OBLIVIOUS) {
        std::cerr << "Export is supported only for Oblivious trees." << std::endl;
        header_file.close();
        throw std::runtime_error("Export is supported only for Oblivious trees.");
        return -1;
    }
    export_ensemble_data(header_file, modelname, this->edata, this->metadata, this->device, this->opts);
    if (!header_file.good()) {
        std::cerr << "Error occurred at writing time." << std::endl;
        throw std::runtime_error("Writing to file error");
        return -1;
    }

    header_file.close();
    return 0;
}

int GBRL::saveToFile(const std::string& filename){
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("File opening error");
        return -1;
    }

    write_header(file, this->sheader);
    file.write(reinterpret_cast<const char*>(this->metadata), sizeof(ensembleMetaData));
    char byte = static_cast<char>(this->parallel_predict);
    file.write(&byte, sizeof(byte));
    // properly save bool values
    byte = static_cast<char>(this->metadata->use_cv);
    file.write(&byte, sizeof(byte));

    save_ensemble_data(file, this->edata, this->metadata, this->device);

    int num_opts = static_cast<int>(this->opts.size());
    file.write(reinterpret_cast<char*>(&num_opts), sizeof(int));

    for (int i = 0; i < num_opts; ++i){
        int status = this->opts[i]->saveToFile(file);
        if (status != 0){
            std::cerr << "Could not save optimizers: " << i << " exited with status: " << status << std::endl;
            throw std::runtime_error("Optimizer save error");
            return status;
        }
    }

    if (!file.good()) {
        std::cerr << "Error occurred at writing time." << std::endl;
        throw std::runtime_error("Writing to file error");
        return -1;
    }

    file.close();
    return 0;  // success
}

int GBRL::loadFromFile(const std::string& filename){
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        throw std::runtime_error("Error opening file");
        return -1;
    }
    serializationHeader crnt_sheader = create_header();
    std::cout << "Current GBRL Library ";
    display_header(crnt_sheader);
    this->sheader = read_header(file);
    std::cout << "Reading GBRL Library ";
    display_header(this->sheader);
    this->metadata = new ensembleMetaData;
    file.read(reinterpret_cast<char*>(this->metadata), sizeof(ensembleMetaData));

    char byte;
    file.read(&byte, sizeof(byte));
    this->parallel_predict = static_cast<bool>(byte);
    file.read(&byte, sizeof(byte));
    this->metadata->use_cv = static_cast<bool>(byte);

    if (!file.good()) {
        std::cerr << "Error occurred while reading the file." << std::endl;
        throw std::runtime_error("Reading file error");
        return -1;
    }

    this->edata = load_ensemble_data(file, this->metadata);

    for (size_t i = 0; i < this->opts.size(); i++)
        delete this->opts[i];
    this->opts.clear();

    int num_opts;
    file.read(reinterpret_cast<char*>(&num_opts), sizeof(int));
    // Similar procedure for loading optimizers
    for (int i = 0; i < num_opts; ++i) {  // Adjust as needed
        Optimizer* opt = Optimizer::loadFromFile(file);  // Adjust as needed
        if (!opt) {
            std::cerr << "Error loading optimizer " << i << std::endl;
            delete opt;
            throw std::runtime_error("Optimizer load error");
            return -1;
        }
        this->opts.push_back(opt);
    }
    file.close();


    if (file.fail()) {
        std::cerr << "Error occurred at file closing time." << std::endl;
        throw std::runtime_error("File closing error");
        return -1;
    }

    this->device = cpu;
    std::cout << "######## Loaded GBRL model ########" << std::endl;
    std::cout << "output_dim: " << this->metadata->output_dim;
    std::cout << " max_depth: " << this->metadata->max_depth << " min_data_in_leaf: " << this->metadata->min_data_in_leaf << std::endl;
    std::cout << "generator_type: " << generatorTypeToString(this->metadata->generator_type) << " n_bins: " << this->metadata->n_bins;
    std::cout << " cv_beta: " << this->metadata->cv_beta << " split_score_func: " << scoreFuncToString(this->metadata->split_score_func) << std::endl;
    std::cout << "grow_policy: " << growPolicyToString(this->metadata->grow_policy);
    std::cout << " verbose: " << this->metadata->verbose << " device: "<< deviceTypeToString(this->device);
    std::cout << " use_cv: " << this->metadata->use_cv << " batch_size: " << this->metadata->batch_size << std::endl;
    std::cout << "Loaded: " << this->metadata->n_leaves << " leaves from " << this->metadata->n_trees << " trees" <<  std::endl;
    std::cout << "Model has: " << num_opts << " optimizers " <<  std::endl;
    
    return 0;  // Success
}

float* GBRL::tree_shap(const int tree_idx, const float *obs, const char *categorical_obs, const int n_samples, float *norm, float *base_poly, float *offset){
    valid_tree_idx(tree_idx, this->metadata);
ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (this->device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(this->metadata, this->edata);
    }
#endif 
    if (this->device == cpu)
        edata_cpu = this->edata;
    shapData* shap_data = alloc_shap_data(this->metadata, edata_cpu, tree_idx);
    shap_data->offset_poly = offset;
    shap_data->base_poly = base_poly;
    shap_data->norm_values = norm;
    float *shap_values = init_zero_mat((this->metadata->n_num_features + this->metadata->n_cat_features)*this->metadata->output_dim * n_samples);
    dataSet dataset{obs, categorical_obs, nullptr, nullptr, nullptr, nullptr, n_samples};
    // print_shap_data(shap_data, this->metadata);
    get_shap_values(this->metadata, edata_cpu, shap_data, &dataset, shap_values);
    dealloc_shap_data(shap_data);
#ifdef USE_CUDA
    if (this->device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 
    return shap_values;
}

float* GBRL::ensemble_shap(const float *obs, const char *categorical_obs, const int n_samples, float *norm, float *base_poly, float *offset){
    valid_tree_idx(0, this->metadata);
    float *shap_values = init_zero_mat((this->metadata->n_num_features + this->metadata->n_cat_features)*this->metadata->output_dim * n_samples);
    dataSet dataset{obs, categorical_obs, nullptr, nullptr, nullptr, nullptr, n_samples};
    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (this->device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(this->metadata, this->edata);
    }
#endif 
    if (this->device == cpu)
        edata_cpu = this->edata;

    for (int tree_idx = 0; tree_idx < this->metadata->n_trees; ++tree_idx){
        shapData* shap_data = alloc_shap_data(this->metadata, edata_cpu, tree_idx);
        shap_data->offset_poly = offset;
        shap_data->base_poly = base_poly;
        shap_data->norm_values = norm;
        get_shap_values(this->metadata, edata_cpu, shap_data, &dataset, shap_values);
        dealloc_shap_data(shap_data);
    }
#ifdef USE_CUDA
    if (this->device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 
   
    return shap_values;
}

void GBRL::print_tree(int tree_idx){
    valid_tree_idx(tree_idx, this->metadata);
    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (this->device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(this->metadata, this->edata);
    }
#endif 
    if (this->device == cpu)
        edata_cpu = this->edata;

    int n_trees = this->metadata->n_trees;
    int stop_leaf_idx = (tree_idx == n_trees - 1) ? this->metadata->n_leaves : edata_cpu->tree_indices[tree_idx+1];
    int n_leaves = stop_leaf_idx - edata_cpu->tree_indices[tree_idx];

    std::cout << growPolicyToString(this->metadata->grow_policy) <<" DecisionTree idx: " << tree_idx;
    std::cout <<  " output_dim: " << this->metadata->output_dim << " n_bins: " << this->metadata->n_bins;
    std::cout <<  " min_data_in_leaf: " << this->metadata->min_data_in_leaf << " par_th: " << this->metadata->par_th << " max_depth: " << this->metadata->max_depth << std::endl;
    std::cout << "Leaf Nodes: " << n_leaves << std::endl;
    int ctr = 0;
    for (int leaf_idx = edata_cpu->tree_indices[tree_idx]; leaf_idx < stop_leaf_idx; ++leaf_idx){
        print_leaf(leaf_idx, ctr, tree_idx, edata_cpu, this->metadata);
        ctr++;
    }
    std::cout << "******************" << std::endl;
#ifdef USE_CUDA
    if (this->device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 
}

#ifdef USE_CUDA
bool valid_device(){
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess) {
        std::cout << "CUDA error when querying device count: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    if (device_count == 0)
        return false; 
    return true;
}
#endif

void GBRL::plot_tree(int tree_idx, const std::string &filename){
#ifdef USE_GRAPHVIZ
    ensembleData *edata_cpu = this->edata; 
#ifdef USE_CUDA
    if (this->device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(this->metadata, this->edata);
    }
#endif 
    valid_tree_idx(tree_idx, this->metadata);
    std::cout << "Plotting tree: " << tree_idx << " to filename: " << filename << ".png" << std::endl;
    
    GVC_t *gvc;
    Agraph_t *g;

    gvc = gvContext();
    g = agopen((char*)"g", Agdirected, NULL);
    char buffer[1024];
    Agnode_t *currentNode, *parentNode = nullptr;
    Agedge_t *edge;
    std::unordered_map<int, Agnode_t*> nodesMap;
    std::unordered_set<std::string> edgesSet;

    int n_trees = this->metadata->n_trees;
    int stop_leaf_idx = tree_idx == n_trees - 1 ? this->metadata->n_leaves  : edata_cpu->tree_indices[tree_idx+1];
    
    for (int leaf_idx = edata_cpu->tree_indices[tree_idx]; leaf_idx < stop_leaf_idx; ++leaf_idx){
        int nodeIndex = 0, parentIdx = 0; 
        int idx = (this->metadata->grow_policy == OBLIVIOUS) ? tree_idx : leaf_idx;
        int depth = edata_cpu->depths[idx];
        int cond_idx = idx * this->metadata->max_depth;
        // Process the root node
        int feature_idx = edata_cpu->feature_indices[cond_idx];
        float feature_value = edata_cpu->feature_values[cond_idx];
        char *categorical_value = edata_cpu->categorical_values + cond_idx * MAX_CHAR_SIZE;
        bool inequality_direction  = edata_cpu->inequality_directions[leaf_idx*this->metadata->max_depth];
        float edge_weight  = edata_cpu->edge_weights[leaf_idx*this->metadata->max_depth];
        bool is_numeric  = edata_cpu->is_numerics[cond_idx];
        
        if (nodesMap.find(nodeIndex) == nodesMap.end()) {  // Check if the root node already exists
            std::strcpy(buffer, std::to_string(nodeIndex).c_str());
            parentNode = agnode(g, buffer, true);
            std::string nodeLabel = (is_numeric) ? std::to_string(feature_idx) + ", value > " + std::to_string(feature_value) : std::to_string(feature_idx + this->metadata->n_num_features) + ", value == " + std::string(categorical_value);
            std::strcpy(buffer, nodeLabel.c_str());
            agsafeset(parentNode, (char*)"label", buffer, (char*)"");
            nodesMap[nodeIndex] = parentNode;
            
        } else {
            parentNode = nodesMap[nodeIndex];
        }

        std::vector<bool> path;
        for (int i = 1; i < depth; ++i){
            feature_idx = edata_cpu->feature_indices[cond_idx + i];
            feature_value = edata_cpu->feature_values[cond_idx + i];
            char *categorical_value = edata_cpu->categorical_values + (cond_idx + i)*MAX_CHAR_SIZE;
            is_numeric = edata_cpu->is_numerics[cond_idx + i];
            path.push_back(inequality_direction);
            nodeIndex = binaryToDecimal(path);
        
            if (nodesMap.find(nodeIndex) == nodesMap.end()) {
                std::strcpy(buffer, std::to_string(nodeIndex).c_str());
                currentNode = agnode(g, buffer, true);
                std::stringstream nodeLabel;
                nodeLabel << std::fixed << std::setprecision(3);
                if (is_numeric) {
                    nodeLabel << feature_idx << ", value > " << feature_value;
                } else {
                    nodeLabel << feature_idx + this->metadata->n_num_features << ", value == " << categorical_value;
                }
                std::strcpy(buffer, nodeLabel.str().c_str());

                agsafeset(currentNode, (char*)"label", buffer, (char*)"");
                nodesMap[nodeIndex] = currentNode;
            } else {
                currentNode = nodesMap[nodeIndex];
            }

            std::stringstream edgeLabel;
            edgeLabel << (inequality_direction ? "Yes\nweight: " : "No\nweight: ") << std::fixed << std::setprecision(3) << edge_weight;
            std::string edgeKey = std::to_string(parentIdx) + "->" + std::to_string(nodeIndex) + " " + edgeLabel.str();

            if (edgesSet.find(edgeKey) == edgesSet.end()) {
                std::strcpy(buffer, edgeLabel.str().c_str());
                edge = agedge(g, parentNode, currentNode, buffer, true);
                agsafeset(edge, (char*)"label", buffer, (char*)"");
                edgesSet.insert(edgeKey);  
            }

            parentNode = currentNode;
            parentIdx = nodeIndex;
            inequality_direction = edata_cpu->inequality_directions[leaf_idx*this->metadata->max_depth + i];
            edge_weight = edata_cpu->edge_weights[leaf_idx*this->metadata->max_depth + i];

        }
    
        std::string leafLabel = "val = " + VectoString(edata_cpu->values + leaf_idx*this->metadata->output_dim, this->metadata->output_dim);
        std::string uniqueLeafLabel = leafLabel + "_idx_" + std::to_string(leaf_idx); 
        std::strcpy(buffer, uniqueLeafLabel.c_str());
        currentNode = agnode(g, buffer, true);
        edge = agedge(g, parentNode, currentNode, NULL, true);

        agsafeset(currentNode, (char*)"label", buffer, (char*)"");
        agsafeset(currentNode, (char*)"shape", (char*)"box", (char*)"");
        agsafeset(currentNode, (char*)"color", (char*)"red", (char*)"");
        std::strcpy(buffer, leafLabel.c_str());  // Setting the displayed label
        agset(currentNode, (char*)"label", buffer);
        std::stringstream edgeLabel;
        edgeLabel << (edata_cpu->inequality_directions[leaf_idx * this->metadata->max_depth + depth - 1] ? "Yes\nweight: " : "No\nweight: ")
                  << std::fixed << std::setprecision(3) << edata_cpu->edge_weights[leaf_idx * this->metadata->max_depth + depth - 1];
        std::strcpy(buffer, edgeLabel.str().c_str());
        agsafeset(edge, (char*)"label", buffer, (char*)"");  // Fixing edge label
    }

    gvLayout(gvc, g, "dot");

    gvRenderFilename(gvc, g, "png", (filename + ".png").c_str());
    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);

#ifdef USE_CUDA
    if (this->device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif
#else
    (void)tree_idx;
    (void)filename;
    throw std::runtime_error("GBRL compiled without Graphviz! Cannot plot model");
#endif 
}



