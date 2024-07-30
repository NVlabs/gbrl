//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <stdexcept>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "types.h"
#include "optimizer.h"
#ifdef USE_CUDA
#include "cuda_types.h"
#endif


scoreFunc stringToScoreFunc(std::string str) {
    if (str == "L2" || str == "l2") return scoreFunc::L2;
    if (str == "Cosine" || str == "cosine") return scoreFunc::Cosine;
    throw std::runtime_error("Invalid score function! Options are: Cosine/L2");
    return scoreFunc::L2;
}

generatorType stringTogeneratorType(std::string str) {
    if (str == "uniform" || str == "Uniform") return generatorType::Uniform;
    if (str == "quantile" || str == "Quantile") return generatorType::Quantile;
    throw std::runtime_error("Invalid generator function! Options are: Uniform/Quantile");
    return generatorType::Uniform;
}

growPolicy stringTogrowPolicy(std::string str) {
    if (str == "oblivious" || str == "Oblivious") return growPolicy::OBLIVIOUS;
    if (str == "greedy" || str == "Greedy") return growPolicy::GREEDY;
    throw std::runtime_error("Invalid generator function! Options are: Greedy/Oblivious");
    return growPolicy::GREEDY;
}

lossType stringTolossType(std::string str) {
    if (str == "MultiRMSE") return lossType::MultiRMSE;
    throw std::runtime_error("Invalid loss function! Options are: MultiRMSE");
    return lossType::MultiRMSE;
}

deviceType stringTodeviceType(std::string str) {
    if (str == "cpu") return deviceType::cpu;
    if (str == "cuda" || str == "gpu") return deviceType::gpu;
    throw std::runtime_error("Invalid device! Options are: cpu/cuda");
}

optimizerAlgo stringToAlgoType(std::string str) {
    if (str == "Adam" || str == "adam") return optimizerAlgo::Adam;
    if (str == "SGD" || str == "sgd") return optimizerAlgo::SGD;
    throw std::runtime_error("Invalid Optimizer Algorithm! Options are: SGD/Adam");
    return optimizerAlgo::SGD;
}

schedulerFunc stringToSchedulerType(std::string str) {
    if (str == "Const" || str == "const") return schedulerFunc::Const;
    if (str == "Linear" || str == "linear") return schedulerFunc::Linear;
    throw std::runtime_error("Invalid Scheduler type! Options are: Linear/Const");
    return schedulerFunc::Const;
}

std::string scoreFuncToString(scoreFunc func) {
    switch (func) {
        case scoreFunc::L2:
            return "L2";
        case scoreFunc::Cosine:
            return "Cosine";
        default:
            throw std::runtime_error("Invalid score function.");
    }
}

std::string generatorTypeToString(generatorType type) {
    switch (type) {
        case generatorType::Uniform:
            return "Uniform";
        case generatorType::Quantile:
            return "Quantile";
        default:
            throw std::runtime_error("Invalid generator type.");
    }
}

std::string growPolicyToString(growPolicy type) {
    switch (type) {
        case growPolicy::OBLIVIOUS:
            return "Oblivous";
        case growPolicy::GREEDY:
            return "Greedy";
        default:
            throw std::runtime_error("Invalid generator type.");
    }
}


std::string lossTypeToString(lossType type) {
    switch (type) {
        case lossType::MultiRMSE:
            return "MultiRMSE";
        default:
            throw std::runtime_error("Invalid loss type.");
    }
}

std::string deviceTypeToString(deviceType type) {
    switch (type) {
        case deviceType::cpu:
            return "cpu";
        case deviceType::gpu:
            return "cuda"; // Assuming 'cuda' is the preferred string for GPU.
        default:
            throw std::runtime_error("Invalid device type.");
    }
}

std::string algoTypeToString(optimizerAlgo algo) {
    switch (algo) {
        case optimizerAlgo::Adam:
            return "Adam";
        case optimizerAlgo::SGD:
            return "SGD";
        default:
            throw std::runtime_error("Invalid optimizer algorithm.");
    }
}

std::string schedulerTypeToString(schedulerFunc func) {
    switch (func) {
        case schedulerFunc::Const:
            return "Const";
        case schedulerFunc::Linear:
            return "Linear";
        default:
            throw std::runtime_error("Invalid scheduler type.");
    }
}

ensembleMetaData* ensemble_metadata_alloc(int max_trees, int max_leaves, int max_trees_batch, int max_leaves_batch, int output_dim, int max_depth, int min_data_in_leaf, int n_bins, int par_th, float cv_beta, int verbose, int batch_size, bool use_cv, scoreFunc split_score_func, generatorType generator_type, growPolicy grow_policy){
    ensembleMetaData *metadata = new ensembleMetaData;
    metadata->output_dim = output_dim; 
    metadata->max_depth = max_depth; 
    metadata->min_data_in_leaf = min_data_in_leaf; 
    metadata->par_th = par_th; 
    metadata->n_bins = n_bins; 
    metadata->n_leaves = 0;
    metadata->n_trees = 0; 
    metadata->verbose = verbose; 
    metadata->batch_size = batch_size; 
    metadata->grow_policy = grow_policy;
    metadata->generator_type = generator_type;
    metadata->cv_beta = cv_beta;
    metadata->split_score_func = split_score_func;
    metadata->use_cv = use_cv;
    metadata->max_trees = max_trees;
    metadata->max_leaves = max_leaves; 
    metadata->max_trees_batch = max_trees_batch; 
    metadata->max_leaves_batch = max_leaves_batch;
    metadata->n_num_features = 0;
    metadata->n_cat_features = 0;
    metadata->iteration = 0;
    return metadata;
}

ensembleData* ensemble_data_alloc(ensembleMetaData *metadata){
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        throw std::runtime_error("Error invalid pointer");
        return nullptr;
    }
    edata->bias = new float[metadata->output_dim];
    memset(edata->bias, 0, metadata->output_dim * sizeof(float));
    int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->max_trees : metadata->max_leaves;
#ifdef DEBUG
    edata->n_samples = new int[metadata->max_leaves]; // debugging
    memset(edata->n_samples, 0, metadata->max_leaves * sizeof(int));
#endif
    edata->tree_indices = new int[metadata->max_trees];
    memset(edata->tree_indices, 0, metadata->max_trees * sizeof(int));
    edata->depths = new int[split_sizes];
    memset(edata->depths, 0, split_sizes * sizeof(int));
    edata->values = new float[metadata->max_leaves*metadata->output_dim];
    memset(edata->values, 0, metadata->max_leaves*metadata->output_dim * sizeof(float));
    // leaf data
    edata->feature_indices = new int[split_sizes*metadata->max_depth];
    memset(edata->feature_indices, 0, split_sizes*metadata->max_depth * sizeof(int));
    edata->feature_values = new float[split_sizes*metadata->max_depth];
    memset(edata->feature_values, 0, split_sizes*metadata->max_depth * sizeof(float));
    edata->edge_weights = new float[metadata->max_leaves*metadata->max_depth];
    memset(edata->edge_weights, 0, metadata->max_leaves*metadata->max_depth * sizeof(float));
    edata->is_numerics = new bool[split_sizes*metadata->max_depth];
    memset(edata->is_numerics, 0, split_sizes*metadata->max_depth * sizeof(bool));
    edata->inequality_directions = new bool[metadata->max_leaves*metadata->max_depth]; 
    memset(edata->inequality_directions, 0, metadata->max_leaves*metadata->max_depth * sizeof(bool));
    edata->categorical_values = new char[split_sizes*metadata->max_depth*MAX_CHAR_SIZE];
    memset(edata->categorical_values, 0, split_sizes*metadata->max_depth * sizeof(char)*MAX_CHAR_SIZE);
    return edata;
}

ensembleData* ensemble_copy_data_alloc(ensembleMetaData *metadata){
    // same as normal alloc but only allocate memory for existing size 
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        throw std::runtime_error("Error invalid pointer");
        return nullptr;
    }
    edata->bias = new float[metadata->output_dim];
    memset(edata->bias, 0, metadata->output_dim * sizeof(float));
    int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
#ifdef DEBUG
    edata->n_samples = new int[metadata->n_leaves]; // debugging
    memset(edata->n_samples, 0, metadata->n_leaves * sizeof(int));
#endif
    edata->tree_indices = new int[metadata->n_trees];
    memset(edata->tree_indices, 0, metadata->n_trees * sizeof(int));
    edata->depths = new int[split_sizes];
    memset(edata->depths, 0, split_sizes * sizeof(int));
    edata->values = new float[metadata->n_leaves*metadata->output_dim];
    memset(edata->values, 0, metadata->n_leaves*metadata->output_dim * sizeof(float));
    // leaf data
    edata->feature_indices = new int[split_sizes*metadata->max_depth];
    memset(edata->feature_indices, 0, split_sizes*metadata->max_depth * sizeof(int));
    edata->feature_values = new float[split_sizes*metadata->max_depth];
    memset(edata->feature_values, 0, split_sizes*metadata->max_depth * sizeof(float));
    edata->edge_weights = new float[metadata->n_leaves*metadata->max_depth];
    memset(edata->edge_weights, 0, metadata->n_leaves*metadata->max_depth * sizeof(float));
    edata->is_numerics = new bool[split_sizes*metadata->max_depth];
    memset(edata->is_numerics, 0, split_sizes*metadata->max_depth * sizeof(bool));
    edata->inequality_directions = new bool[metadata->n_leaves*metadata->max_depth]; 
    memset(edata->inequality_directions, 0, metadata->n_leaves*metadata->max_depth * sizeof(bool));
    edata->categorical_values = new char[split_sizes*metadata->max_depth*MAX_CHAR_SIZE];
    memset(edata->categorical_values, 0, split_sizes*metadata->max_depth * sizeof(char)*MAX_CHAR_SIZE);
    return edata;
}

ensembleData* copy_ensemble_data(ensembleData *other_edata, ensembleMetaData *metadata){
    ensembleData *edata = new ensembleData;
    if (metadata == nullptr || other_edata == nullptr){
        std::cerr << "Error metadata is nullptr cannot allocate ensembleData." << std::endl;
        throw std::runtime_error("Error invalid pointer");
        return nullptr;
    }
    edata->bias = new float[metadata->output_dim];
    memcpy(edata->bias, other_edata->bias, metadata->output_dim * sizeof(float));
    int split_sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees : metadata->n_leaves;
#ifdef DEBUG
    edata->n_samples = new int[metadata->n_leaves]; // debugging
    memcpy(edata->n_samples, other_edata->n_samples, metadata->n_leaves * sizeof(int));
#endif
    edata->tree_indices = new int[metadata->n_trees];
    memcpy(edata->tree_indices, other_edata->tree_indices, metadata->n_trees * sizeof(int));
    edata->depths = new int[split_sizes];
    memcpy(edata->depths, other_edata->depths, split_sizes * sizeof(int));
    edata->values = new float[metadata->n_leaves*metadata->output_dim];
    memcpy(edata->values, other_edata->values, metadata->n_leaves*metadata->output_dim * sizeof(float));
    // leaf data
    edata->feature_indices = new int[split_sizes*metadata->max_depth];
    memcpy(edata->feature_indices, other_edata->feature_indices, split_sizes*metadata->max_depth * sizeof(int));
    edata->feature_values = new float[split_sizes*metadata->max_depth];
    memcpy(edata->feature_values, other_edata->feature_values, split_sizes*metadata->max_depth * sizeof(float));
    edata->edge_weights = new float[metadata->n_leaves*metadata->max_depth];
    memcpy(edata->edge_weights, other_edata->edge_weights, metadata->n_leaves*metadata->max_depth * sizeof(float));
    edata->is_numerics = new bool[split_sizes*metadata->max_depth];
    memcpy(edata->is_numerics, other_edata->is_numerics, split_sizes*metadata->max_depth * sizeof(bool));
    edata->categorical_values = new char[split_sizes*metadata->max_depth*MAX_CHAR_SIZE];
    memcpy(edata->categorical_values, other_edata->categorical_values, split_sizes*metadata->max_depth * sizeof(char)*MAX_CHAR_SIZE);
    edata->inequality_directions = new bool[metadata->n_leaves*metadata->max_depth]; 
    memcpy(edata->inequality_directions, other_edata->inequality_directions, metadata->n_leaves*metadata->max_depth * sizeof(bool));
    metadata->max_trees = metadata->n_trees;
    metadata->max_leaves = metadata->n_leaves;
    return edata;
}

void ensemble_data_dealloc(ensembleData *edata){
    delete[] edata->bias;
#ifdef DEBUG
    delete[] edata->n_samples;
#endif
    delete[] edata->depths;
    delete[] edata->values;
    delete[] edata->feature_indices;
    delete[] edata->tree_indices;
    delete[] edata->feature_values;
    delete[] edata->edge_weights;
    delete[] edata->is_numerics;
    delete[] edata->categorical_values;
    delete[] edata->inequality_directions; 
    delete edata;
}

void export_ensemble_data(std::ofstream& header_file, const std::string& model_name, ensembleData *edata, ensembleMetaData *metadata, deviceType device, std::vector<Optimizer*> opts)
{
    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(metadata, edata);
    }
#endif 
    if (device == cpu)
        edata_cpu = edata;
    
    int binary_splits = 0;
    for (int i  = 0; i < metadata->n_trees; ++i){
        binary_splits += edata_cpu->depths[i];
    }

    for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
        optimizerAlgo algo = opts[opt_idx]->getAlgo();
        if (algo != SGD){
            std::cerr << "Error. Can only export SGD optimizers" << std::endl;
            header_file.close();
            throw std::runtime_error("Error. Can only export SGD optimizers");
            return;
        }
    }

    header_file << "#ifndef GBRL_MODEL_H\n";
    header_file << "#define GBRL_MODEL_H\n\n";

    
    header_file << "/*\n";

    if (!model_name.empty()) {
           header_file << "###########################\n";
        header_file << "model_name: " << model_name << "\n";
    }
    header_file << "###########################\n";
    header_file << "n_leaves: " << metadata->n_leaves << ", ";
    header_file << "n_trees: " << metadata->n_trees << ", ";
    header_file << "max_trees: " << metadata->max_trees << ", ";
    header_file << "max_leaves: " << metadata->max_leaves << ", ";
    header_file << "max_trees_batch: " << metadata->max_trees_batch << ", ";
    header_file << "max_leaves_batch: " << metadata->max_leaves_batch << ", ";
    header_file << "output_dim: " << metadata->output_dim << ", ";
    header_file << "\nmax_depth: " << metadata->max_depth << ", ";
    header_file << "min_data_in_leaf: " << metadata->min_data_in_leaf << ", ";
    header_file << "n_bins: " << metadata->n_bins << ", ";
    header_file << "par_th: " << metadata->par_th << ", ";
    header_file << "cv_beta: " << metadata->cv_beta << ", ";
    header_file << "verbose: " << metadata->verbose << ", ";
    header_file << "batch_size: " << metadata->batch_size << ", ";
    header_file << "use_cv: " << metadata->use_cv;
    header_file << "\nsplit_score_func: " << scoreFuncToString(metadata->split_score_func) << ", ";
    header_file << "generator_type: " << generatorTypeToString(metadata->generator_type) << ", ";
    header_file << "grow_policy: " << growPolicyToString(metadata->grow_policy) << ", ";
    header_file << "n_num_features: " << metadata->n_num_features << ", ";
    header_file << "n_cat_features: " << metadata->n_cat_features << ", ";
    header_file << "iteration: " << metadata->iteration;
    header_file << "\n*/\n";

    header_file << "#define N_TREES " << metadata->n_trees << "\n";
    header_file << "#define N_LEAVES " << metadata->n_leaves << "\n";
    header_file << "#define BINARY_FEATURES " << binary_splits << "\n";
    header_file << "#define N_OUTPUTS " << metadata->output_dim << "\n";
    header_file << "#define N_FEATURES " << metadata->n_num_features  << "\n\n";

    header_file << "static inline void gbrl_predict(float *results, const float *features){\n\n";
    header_file << "\tunsigned int j, tree_idx, depth, current_depth, idx, leaf_ptr, cond_ptr;\n";
    header_file << "\t/* Model data */\n";
    header_file << "\tconst unsigned int depths[N_TREES] = {";
    for (int i  = 0; i < metadata->n_trees; ++i){
        header_file << edata_cpu->depths[i];
        if (i < metadata->n_trees - 1)
            header_file << ", ";
    }
    header_file << "};\n";
    header_file << "\tconst float bias[N_OUTPUTS] = {";
    for (int i  = 0; i < metadata->output_dim; ++i){
        header_file << edata_cpu->bias[i];
        if (i < metadata->output_dim - 1)
            header_file << ", ";
    }
    header_file << "};\n";
    header_file << "\tconst unsigned int feature_indices[BINARY_FEATURES] = {";
    for (int i  = 0; i < binary_splits; ++i){
        header_file << edata_cpu->feature_indices[i];
        if (i < binary_splits - 1)
            header_file << ", ";
    }
    header_file << "};\n";
    header_file << "\tconst float feature_values[BINARY_FEATURES] = {";
    for (int i  = 0; i < binary_splits; ++i){
        header_file << edata_cpu->feature_values[i];
        if (i < binary_splits - 1)
            header_file << ", ";
    }
    header_file << "};\n";
    header_file << "\tconst float leaf_values[N_LEAVES*N_OUTPUTS] = {";
    int tree_idx = 0;
    int limit_leaf_idx = edata_cpu->tree_indices[tree_idx];
    float value;
    for (int i  = 0; i < metadata->n_leaves; ++i){
        if (i > limit_leaf_idx){
            tree_idx += 1;
            limit_leaf_idx = edata_cpu->tree_indices[tree_idx];
        }
        int value_idx = i*metadata->output_dim;
        for (size_t opt_idx = 0; opt_idx < opts.size(); ++opt_idx){
            for (int j=opts[opt_idx]->start_idx; j < opts[opt_idx]->stop_idx; ++j){
                value = -edata_cpu->values[value_idx + j] * opts[opt_idx]->scheduler->get_lr(tree_idx);
                header_file << value;
                if ((i < metadata->n_leaves - 1) || (j < metadata->output_dim - 1  && i == metadata->n_leaves - 1))
                    header_file << ", ";
            }
        }
    }
    header_file << "};\n";

    header_file << "\tleaf_ptr = 0;\n";
    header_file << "\tcond_ptr = 0;\n";
    header_file << "\tunsigned char pass;\n";
    header_file << "\tfor (tree_idx = 0; tree_idx < N_TREES; ++tree_idx)\n";
    header_file << "\t{\n";
    header_file << "\t\tcurrent_depth = depths[tree_idx];\n";
    header_file << "\t\tidx = 0;\n";
    header_file << "\t\tfor (depth = 0; depth < current_depth; ++depth){\n";
    header_file << "\t\t\tpass = (unsigned char)(features[feature_indices[cond_ptr + depth]] > feature_values[cond_ptr + depth]);\n";
    header_file << "\t\t\tidx |= (pass <<  (current_depth - 1 - depth));\n";
    header_file << "\t\t}\n";
    header_file << "\t\tfor (j = 0 ; j < N_OUTPUTS; j++)\n";
    header_file << "\t\t\tresults[j] += leaf_values[(leaf_ptr + idx)*N_OUTPUTS + j];\n";
    header_file << "\t\tleaf_ptr += (1 << current_depth);\n";
    header_file << "\t\tcond_ptr += current_depth;\n";
    header_file << "\t}\n";
    header_file << "\tfor (j = 0 ; j < N_OUTPUTS; j++)\n";
    header_file << "\t\tresults[j] += bias[j];\n";
    header_file << "}\n";
    header_file << "#endif\n";

#ifdef USE_CUDA
    if (device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 
}

void save_ensemble_data(std::ofstream& file, ensembleData *edata, ensembleMetaData *metadata, deviceType device){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        throw std::runtime_error("Error opening file");
    }
    ensembleData *edata_cpu = nullptr;
#ifdef USE_CUDA
    if (device == gpu){
        edata_cpu = ensemble_data_copy_gpu_cpu(metadata, edata);
    }
#endif 
    if (device == cpu)
        edata_cpu = edata;
    NULL_CHECK check = edata_cpu->bias != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->bias != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->bias), metadata->output_dim * sizeof(float));
#ifdef DEBUG
    check = edata_cpu->n_samples != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->n_samples != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->n_samples), metadata->n_leaves * sizeof(int));
#endif 
    check = edata_cpu->tree_indices != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->tree_indices != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->tree_indices), metadata->n_trees * sizeof(int));
    size_t sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees: metadata->n_leaves;
    check = edata_cpu->depths != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->depths != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->depths), sizes * sizeof(int));
    check = edata_cpu->values != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->values != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->values), metadata->n_leaves * metadata->output_dim * sizeof(float));
    check = edata_cpu->feature_indices != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_indices != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_indices), metadata->max_depth * sizes * sizeof(int));
    check = edata_cpu->feature_values != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->feature_values != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->feature_values), metadata->max_depth * sizes * sizeof(float));
    check = edata_cpu->edge_weights != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->edge_weights != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->edge_weights), metadata->max_depth * metadata->n_leaves * sizeof(float));
    check = edata_cpu->is_numerics != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->is_numerics != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->is_numerics), metadata->max_depth * sizes * sizeof(bool));
    check = edata_cpu->inequality_directions != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->inequality_directions != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->inequality_directions), metadata->max_depth * metadata->n_leaves * sizeof(bool));
    check = edata_cpu->categorical_values != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (edata_cpu->categorical_values != nullptr)
        file.write(reinterpret_cast<char*>(edata_cpu->categorical_values), metadata->max_depth * sizes * sizeof(char) * MAX_CHAR_SIZE);

#ifdef USE_CUDA
    if (device == gpu){
        ensemble_data_dealloc(edata_cpu);
    }
#endif 
}

ensembleData* load_ensemble_data(std::ifstream& file, ensembleMetaData *metadata){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        throw std::runtime_error("Error opening file");
    }
    ensembleData *edata_cpu = ensemble_data_alloc(metadata);
    NULL_CHECK check;
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->bias), metadata->output_dim * sizeof(float));
    } 
#ifdef DEBUG
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->n_samples), metadata->n_leaves * sizeof(int));
    } 
#endif 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->tree_indices), metadata->n_trees * sizeof(int));
    } 
    size_t sizes = (metadata->grow_policy == OBLIVIOUS) ? metadata->n_trees: metadata->n_leaves;
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->depths), sizes * sizeof(int));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->values), metadata->output_dim * metadata->n_leaves *sizeof(float));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_indices), metadata->max_depth * sizes *sizeof(int));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->feature_values), metadata->max_depth * sizes *sizeof(float));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
       if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->edge_weights), metadata->max_depth * metadata->n_leaves *sizeof(float));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->is_numerics), metadata->max_depth * sizes *sizeof(bool));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->inequality_directions), metadata->max_depth * metadata->n_leaves *sizeof(bool));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        file.read(reinterpret_cast<char*>(edata_cpu->categorical_values), metadata->max_depth * sizes *sizeof(char) * MAX_CHAR_SIZE);
    } 
    return edata_cpu;
}


void allocate_ensemble_memory(ensembleMetaData *metadata, ensembleData *edata){
    int leaf_idx = metadata->n_leaves, tree_idx = metadata->n_trees; 
    if ((leaf_idx >= metadata->max_leaves) || (tree_idx >= metadata->max_trees)){
        int new_size_leaves = metadata->n_leaves + metadata->max_leaves_batch;
        int new_tree_size = metadata->n_trees + metadata->max_trees_batch;
        metadata->max_leaves = new_size_leaves;
        metadata->max_trees = new_tree_size;
        ensembleData *new_data = ensemble_data_alloc(metadata);
        memcpy(new_data->bias, edata->bias, metadata->output_dim*sizeof(float));
#ifdef DEBUG
        memcpy(new_data->n_samples, edata->n_samples, leaf_idx*sizeof(int));
#endif 
        memcpy(new_data->values, edata->values, leaf_idx*metadata->output_dim*sizeof(float));
        memcpy(new_data->tree_indices, edata->tree_indices, tree_idx*sizeof(int));
        memcpy(new_data->inequality_directions, edata->inequality_directions, leaf_idx*metadata->max_depth*sizeof(bool));
        memcpy(new_data->edge_weights, edata->edge_weights, leaf_idx*metadata->max_depth*sizeof(float));
        if (metadata->grow_policy == GREEDY){
            memcpy(new_data->depths, edata->depths, leaf_idx*sizeof(int));
            memcpy(new_data->feature_indices, edata->feature_indices, leaf_idx*metadata->max_depth*sizeof(int));
            memcpy(new_data->feature_values, edata->feature_values, leaf_idx*metadata->max_depth*sizeof(float));
            memcpy(new_data->is_numerics, edata->is_numerics, leaf_idx*metadata->max_depth*sizeof(bool));
            memcpy(new_data->categorical_values, edata->categorical_values, leaf_idx*metadata->max_depth*sizeof(char)*MAX_CHAR_SIZE);
        } else {
            memcpy(new_data->depths, edata->depths, tree_idx*sizeof(int));
            memcpy(new_data->feature_indices, edata->feature_indices, tree_idx*metadata->max_depth*sizeof(int));
            memcpy(new_data->feature_values, edata->feature_values, tree_idx*metadata->max_depth*sizeof(float));
            memcpy(new_data->is_numerics, edata->is_numerics, tree_idx*metadata->max_depth*sizeof(bool));
            memcpy(new_data->categorical_values, edata->categorical_values, tree_idx*metadata->max_depth*sizeof(char)*MAX_CHAR_SIZE);
        }
        delete[] edata->bias;
#ifdef DEBUG
        delete [] edata->n_samples;
#endif
        delete[] edata->depths;
        delete[] edata->values;
        // leaf data
        delete[] edata->feature_indices;
        delete[] edata->tree_indices;
        delete[] edata->feature_values;
        delete[] edata->edge_weights;
        delete[] edata->is_numerics;
        delete[] edata->categorical_values;
        delete[] edata->inequality_directions; 

        edata->bias = new_data->bias;
#ifdef DEBUG
        edata->n_samples = new_data->n_samples;
#endif
        edata->depths = new_data->depths;
        edata->tree_indices = new_data->tree_indices;
        edata->values = new_data->values;
        edata->inequality_directions = new_data->inequality_directions;
        edata->feature_indices = new_data->feature_indices;
        edata->feature_values = new_data->feature_values;
        edata->edge_weights = new_data->edge_weights;
        edata->is_numerics = new_data->is_numerics;
        edata->categorical_values = new_data->categorical_values;
        delete new_data;
    }
}