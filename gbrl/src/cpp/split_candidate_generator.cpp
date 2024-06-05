//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <omp.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <set>
#include <tuple>
#include <string>
#include <cstring>

#include "split_candidate_generator.h"
#include "utils.h"
#include "math_ops.h"
#include "types.h"

SplitCandidateGenerator::SplitCandidateGenerator(const int n_samples, const int n_num_features, const int n_cat_features, const int n_bins, const int par_th, const generatorType &generator_type):
    n_samples(n_samples), n_num_features(n_num_features), n_cat_features(n_cat_features), n_bins(n_bins), par_th(par_th), generator_type(generator_type){
    // printf("allocating enough for %d\n", n_bins*(n_num_features + n_cat_features));
    this->split_candidates = new splitCandidate[n_bins*(n_num_features + n_cat_features)];
    for (int i = 0; i < n_bins*(n_num_features + n_cat_features); ++i)
        this->split_candidates[i].categorical_value = nullptr;
}

SplitCandidateGenerator::~SplitCandidateGenerator(){
    for (int i = 0; i < this->n_candidates; ++i){
        if (this->split_candidates[i].categorical_value != nullptr){
            delete[] this->split_candidates[i].categorical_value;
            this->split_candidates[i].categorical_value = nullptr;
        }
    }

    delete[] this->split_candidates;
    this->split_candidates = nullptr;
}

void SplitCandidateGenerator::generateNumericalSplitCandidates(const float *obs, int* const* sorted_indices){
    if (this->generator_type == Uniform)
        uniformSplitCandidates(obs);
    else
        quantileSplitCandidates(obs, sorted_indices);
}

void SplitCandidateGenerator::uniformSplitCandidates(const float *obs){
    float *max_vec = calculate_max(obs, this->n_samples, this->n_num_features, this->par_th);
    float *min_vec = calculate_min(obs, this->n_samples, this->n_num_features, this->par_th);
    #pragma omp parallel for
    for (int feature_idx = 0; feature_idx < this->n_num_features; ++feature_idx) {
        float step = (max_vec[feature_idx] - min_vec[feature_idx]) / static_cast<float>(this->n_bins);

        for (int bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
            float feature_value = min_vec[feature_idx] + static_cast<float>(bin_idx) * step;
            this->split_candidates[feature_idx*this->n_bins + bin_idx].feature_idx = feature_idx;
            this->split_candidates[feature_idx*this->n_bins + bin_idx].feature_value = feature_value;
            this->split_candidates[feature_idx*this->n_bins + bin_idx].categorical_value = nullptr;
        }
    }
    this->n_candidates = this->n_bins*this->n_num_features;
    delete[] max_vec;
    delete[] min_vec;
}


void SplitCandidateGenerator::quantileSplitCandidates(const float *obs, int* const* sorted_indices){
    int n_candidates = 0;
    int n_threads = calculate_num_threads(this->n_num_features, this->par_th);
    if (n_threads > 1){
        omp_set_num_threads(n_threads);
        std::vector<splitCandidate *> thread_split_candidates(n_threads); 
        std::vector<int> thread_n_candidates(n_threads, 0); 
        std::vector<FloatVector> thread_quantiles(n_threads); 
        int batch_size = this->n_num_features / n_threads;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * batch_size;
            int end_idx = (thread_id == n_threads - 1) ? this->n_num_features : start_idx + batch_size;
            thread_quantiles[thread_id] = FloatVector(this->n_bins, 0.0f);
            thread_split_candidates[thread_id] = new splitCandidate[(end_idx - start_idx)*this->n_bins];
            for (int feature_idx = start_idx; feature_idx < end_idx; ++feature_idx) {
                // Pass sorted views to computeQuantiles
                thread_n_candidates[thread_id] = computeQuantiles(obs, thread_quantiles[thread_id], sorted_indices[feature_idx], feature_idx, thread_split_candidates[thread_id], thread_n_candidates[thread_id]);
            }
        }

        for (int i = 0; i < n_threads; ++i){
            for (int j = 0; j < thread_n_candidates[i]; ++j){
                this->split_candidates[n_candidates + j] = thread_split_candidates[i][j];
            }
            n_candidates += thread_n_candidates[i];
            delete[] thread_split_candidates[i];
        }
    } else{
        FloatVector quantiles = FloatVector(this->n_bins, 0.0f);
        for (int feature_idx = 0; feature_idx < this->n_num_features; ++feature_idx) {
            n_candidates = computeQuantiles(obs, quantiles, sorted_indices[feature_idx], feature_idx, this->split_candidates, n_candidates);
        }  
    }    
    this->n_candidates = n_candidates;
}

void SplitCandidateGenerator::processCategoricalCandidates(const char *categorical_obs, const float *grad_norms){
    std::unordered_map<std::string, categoryInfo> unique_cats;
    for (int feature_idx = 0; feature_idx < this->n_cat_features; ++feature_idx){
        for (int sample_idx=0; sample_idx < this->n_samples; ++sample_idx){
            std::string feature_name = std::string(categorical_obs + (sample_idx * this->n_cat_features + feature_idx)*MAX_CHAR_SIZE, MAX_CHAR_SIZE);
            std::string cat = feature_name + "_" + std::to_string(feature_idx);
            
            unique_cats[cat].total_grad_norm += grad_norms[sample_idx];
            unique_cats[cat].cat_count += 1;
            unique_cats[cat].feature_idx = feature_idx;
            unique_cats[cat].feature_name = feature_name;
        }
    }

    std::vector<std::pair<std::string, float>> cat_vec;
    for (const auto& pair : unique_cats) {
        const auto& category = pair.first;
        const auto& info = pair.second;
        // Now use 'category' and 'info' as needed

        float avg_grad = info.total_grad_norm / info.cat_count;
        cat_vec.emplace_back(category, avg_grad);
    }
    
    int n_unique = static_cast<int>(cat_vec.size());
    if (n_unique > this->n_cat_features*this->n_bins){
        /// sort according to descending order of grad_norms
        std::sort(cat_vec.begin(), cat_vec.end(), 
        [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second > b.second;
        });

        n_unique = this->n_cat_features*this->n_bins;
    }
    
    int n_candidates = this->n_candidates;
    for (int i = 0; i < n_unique; ++i){
        // convert each unique element's string back to char* and copy exactly MAX_CHAR_SIZE of it to the correct position in categorical value
        categoryInfo cat_info = unique_cats[cat_vec[i].first];
        // printf("n_candidates %d/%d\n", n_candidates, n_unique);
        this->split_candidates[n_candidates].feature_idx = cat_info.feature_idx;
        this->split_candidates[n_candidates].feature_value = INFINITY;
        this->split_candidates[n_candidates].categorical_value = new char[MAX_CHAR_SIZE]; 
        memcpy(this->split_candidates[n_candidates].categorical_value, cat_info.feature_name.c_str(), sizeof(char)*MAX_CHAR_SIZE);
        n_candidates++;
    }
    this->n_candidates = n_candidates;
}



int processCategoricalCandidates_func(const char *categorical_obs, const float *grad_norms, const int n_samples, const int n_cat_features, const int n_bins, int* feature_inds, float *feature_values, char* category_values, bool* numerics){
    std::unordered_map<std::string, categoryInfo> unique_cats;
    for (int feature_idx = 0; feature_idx < n_cat_features; ++feature_idx){
        for (int sample_idx=0; sample_idx < n_samples; ++sample_idx){
            std::string feature_name = std::string(categorical_obs + (sample_idx * n_cat_features + feature_idx)*MAX_CHAR_SIZE, MAX_CHAR_SIZE);
            std::string cat = feature_name + "_" + std::to_string(feature_idx);
            
            unique_cats[cat].total_grad_norm += grad_norms[sample_idx];
            unique_cats[cat].cat_count += 1;
            unique_cats[cat].feature_idx = feature_idx;
            unique_cats[cat].feature_name = feature_name;
        }
    }

    std::vector<std::pair<std::string, float>> cat_vec;
    for (const auto& pair : unique_cats) {
        const auto& category = pair.first;
        const auto& info = pair.second;
        // Now use 'category' and 'info' as needed

        float avg_grad = info.total_grad_norm / info.cat_count;
        cat_vec.emplace_back(category, avg_grad);
    }
    
    int n_unique = static_cast<int>(cat_vec.size());
    if (n_unique > n_cat_features*n_bins){
        /// sort according to descending order of grad_norms
        std::sort(cat_vec.begin(), cat_vec.end(), 
        [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
            return a.second > b.second;
        });

        n_unique = n_cat_features*n_bins;
    }
    
    int n_candidates = 0;
    for (int i = 0; i < n_unique; ++i){
        // convert each unique element's string back to char* and copy exactly MAX_CHAR_SIZE of it to the correct position in categorical value
        categoryInfo cat_info = unique_cats[cat_vec[i].first];
        // printf("n_candidates %d/%d\n", n_candidates, n_unique);
        feature_inds[n_candidates] = cat_info.feature_idx;
        feature_values[n_candidates] = INFINITY;
        numerics[n_candidates] = false;
        memcpy(category_values + n_candidates*MAX_CHAR_SIZE, cat_info.feature_name.c_str(), sizeof(char)*MAX_CHAR_SIZE);
        n_candidates++;
    }
    return n_candidates;
}

int SplitCandidateGenerator::computeQuantiles(const float *obs, FloatVector &quantiles, const int *sorted_feature_indices, const int feature_idx, splitCandidate *split_candidates, int n_candidates){
    
    int cumulative_count = 0;
    int actual_bins = this->n_bins + 1;

    int samples_per_bin = this->n_samples / actual_bins; 
    int remainder = this->n_samples % actual_bins;
    IntVector bin_counts(actual_bins, samples_per_bin); 

    // Distribute the remainder to the bin counts in a round-robin fashion
    while (remainder > 0){
        for (int i = 0 ; i < actual_bins; i++){
            bin_counts[i] += 1;
            remainder -= 1;
            if (remainder == 0)
                break;
        }
    }

    for (int i = 0; i < this->n_bins; ++i) {
        // Update the cumulative count for the current iteration
        cumulative_count += bin_counts[i];
        const int split_point = sorted_feature_indices[cumulative_count - 1];
        // Set quantile based on ideal distribution
        quantiles[i] = obs[split_point*this->n_num_features + feature_idx];
        if (n_candidates == 0 || (split_candidates[n_candidates - 1].feature_idx == feature_idx && split_candidates[n_candidates- 1].feature_value != quantiles[i]) || (split_candidates[n_candidates - 1].feature_idx != feature_idx )){
            split_candidates[n_candidates].feature_idx = feature_idx;
            split_candidates[n_candidates].feature_value = quantiles[i];
            split_candidates[n_candidates].categorical_value = nullptr;
            ++n_candidates;
        }
    }                      
    return n_candidates;          
}


std::ostream& operator<<(std::ostream& os, const splitCandidate& obj){
    os << "splitCandidate feature_idx: " << obj.feature_idx;
    if (obj.categorical_value != nullptr){
        os << " == " << std::string(obj.categorical_value) << std::endl; ;
    } else {
        os << " > " << obj.feature_value << std::endl; ;
    }
    return os;
}

float scoreCosine(const int *indices, const int n_samples, const float *grads, const float *grads_norm_raw, const int n_cols){
    float *mean = new float[n_cols]; 
    float n_samples_f = static_cast<float>(n_samples);
    float n_samples_recip = 1.0f / n_samples_f;
#ifndef _MSC_VER    
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        mean[d] = 0;
    }
    float squared_norms = 0.0f;
    for (int i = 0; i < n_samples; ++i){
        int idx = indices[i];
        int row = idx * n_cols;
#ifndef _MSC_VER
    #pragma omp simd
#endif
        for (int d = 0; d < n_cols; ++d){
            mean[d] += grads[row + d];
        }
        squared_norms += grads_norm_raw[idx];
    }

#ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        mean[d] *= n_samples_recip;
    }

    float cosine = cosine_dist(indices, grads, mean, n_samples, n_cols, squared_norms);
    delete[] mean;
    return cosine;
}


float scoreL2(const int *indices, const int n_samples, const float *grads, const int n_cols){
    float *mean = new float[n_cols]; 
    float n_samples_f = static_cast<float>(n_samples);
    float n_samples_recip = 1.0f / n_samples_f;

#ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        mean[d] = 0.0f;
    }
#ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int i = 0; i < n_samples * n_cols; ++i){
        int row = i / n_cols;
        int col = i % n_cols;
        mean[col] += grads[indices[row]*n_cols + col];
    }
    
#ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int d = 0; d < n_cols; ++d){
        mean[d] *= n_samples_recip;
    }
    float mean_norm = squared_norm(mean, n_cols);
    delete[] mean;
    return mean_norm*n_samples_f;

}
