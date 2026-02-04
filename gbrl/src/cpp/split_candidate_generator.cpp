//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//////////////////////////////////////////////////////////////////////////////
/**
 * @file split_candidate_generator.cpp
 * @brief Implementation of split candidate generation for tree building
 */

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
    int _n_candidates = 0;
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
                this->split_candidates[_n_candidates + j] = thread_split_candidates[i][j];
            }
            _n_candidates += thread_n_candidates[i];
            delete[] thread_split_candidates[i];
        }
    } else{
        FloatVector quantiles = FloatVector(this->n_bins, 0.0f);
        for (int feature_idx = 0; feature_idx < this->n_num_features; ++feature_idx) {
            _n_candidates = computeQuantiles(obs, quantiles, sorted_indices[feature_idx], feature_idx, this->split_candidates, _n_candidates);
        }  
    }    
    this->n_candidates = _n_candidates;
}


void SplitCandidateGenerator::processCategoricalCandidates(const char *categorical_obs) {
    // 1. Count Frequencies using optimized Map
    // Key: Pair(Feature Index, Category String) -> categoryInfo
    std::unordered_map<std::pair<int, std::string>, categoryInfo, PairHash> unique_cats;

    for (int feature_idx = 0; feature_idx < this->n_cat_features; ++feature_idx) {
        for (int sample_idx = 0; sample_idx < this->n_samples; ++sample_idx) {
            
            // Efficiently grab the string

            const char* start_ptr = categorical_obs + (sample_idx * n_cat_features + feature_idx) * MAX_CHAR_SIZE;
            size_t safe_len = strnlen(start_ptr, MAX_CHAR_SIZE);
            std::pair<int, std::string> key = {feature_idx, std::string(start_ptr, safe_len)};

            categoryInfo& info = unique_cats[key];
            
            // Initialize on first sight
            if (info.cat_count == 0) {
                info.feature_idx = feature_idx;
                info.feature_name = key.second; 
            }
            
            // Increment Count (Frequency)
            info.cat_count++;
        }
    }

    // 2. Flatten to Vector for Sorting
    std::vector<categoryInfo> cat_vec;
    cat_vec.reserve(unique_cats.size());
    
    for (auto& pair : unique_cats) {
        cat_vec.push_back(std::move(pair.second));
    }
    
    // 3. Prune by Frequency
    int n_unique = static_cast<int>(cat_vec.size());
    int limit = this->n_cat_features * this->n_bins;

    if (n_unique > limit) {
        // Sort DESCENDING by Frequency (cat_count)
        std::sort(cat_vec.begin(), cat_vec.end(), 
            [](const categoryInfo& a, const categoryInfo& b) {
                return a.cat_count > b.cat_count; 
            });

        n_unique = limit;
    }
    
    // 4. Generate Split Candidates
    int _n_candidates = this->n_candidates;
    
    for (int i = 0; i < n_unique; ++i) {
        const categoryInfo& cat_info = cat_vec[i];
        
        this->split_candidates[_n_candidates].feature_idx = cat_info.feature_idx;
        this->split_candidates[_n_candidates].feature_value = INFINITY;
        
        // Allocate and Copy String safely
        this->split_candidates[_n_candidates].categorical_value = new char[MAX_CHAR_SIZE]; 
        memset(this->split_candidates[_n_candidates].categorical_value, 0, MAX_CHAR_SIZE);
        strncpy(this->split_candidates[_n_candidates].categorical_value, cat_info.feature_name.c_str(), MAX_CHAR_SIZE - 1);
        
        _n_candidates++;
    }
    
    this->n_candidates = _n_candidates;
}


int processCategoricalCandidates_func(
    const char *categorical_obs, 
    // const float *grad_norms,   <-- REMOVED
    const int n_samples, 
    const int n_cat_features, 
    const int n_bins, 
    int* feature_inds, 
    float *feature_values, 
    char* category_values, 
    bool* numerics)
{
    // Map Key: Pair(Feature Index, Category String) -> Stats
    // Using a pair key prevents us from allocating heap memory for "cat_name_idx" string concatenation
    std::unordered_map<std::pair<int, std::string>, CategoryStats, PairHash> unique_cats;

    // 1. COUNT FREQUENCIES
    // ---------------------
    for (int feature_idx = 0; feature_idx < n_cat_features; ++feature_idx) {
        for (int sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            
            // Construct string from fixed-width char buffer
            const char* start_ptr = categorical_obs + (sample_idx * n_cat_features + feature_idx) * MAX_CHAR_SIZE;
            size_t safe_len = strnlen(start_ptr, MAX_CHAR_SIZE);
            // Key construction
            std::pair<int, std::string> key = {feature_idx, std::string(start_ptr, safe_len)};
            
            // Update Stats
            CategoryStats& stats = unique_cats[key];
            if (stats.count == 0) {
                stats.feature_idx = feature_idx;
                stats.value = key.second; 
            }
            stats.count++;
        }
    }

    // 2. FLATTEN TO VECTOR
    // ---------------------
    std::vector<CategoryStats> cat_vec;
    cat_vec.reserve(unique_cats.size());
    
    for (auto& pair : unique_cats) {
        cat_vec.push_back(std::move(pair.second));
    }
    
    int n_unique = static_cast<int>(cat_vec.size());
    int limit = n_cat_features * n_bins;

    // 3. PRUNE BY FREQUENCY (If needed)
    // ---------------------
    if (n_unique > limit) {
        // Sort DESCENDING by Count (Frequency)
        // Keep the most frequent categories
        std::sort(cat_vec.begin(), cat_vec.end(), 
            [](const CategoryStats& a, const CategoryStats& b) {
                return a.count > b.count; 
            });

        n_unique = limit;
    }
    
    // 4. OUTPUT CANDIDATES
    // ---------------------
    int n_candidates = 0;
    for (int i = 0; i < n_unique; ++i) {
        feature_inds[n_candidates] = cat_vec[i].feature_idx;
        feature_values[n_candidates] = INFINITY;
        numerics[n_candidates] = false;
        
        // Safe Copy
        // Ensure we don't overflow MAX_CHAR_SIZE
        memset(category_values + n_candidates * MAX_CHAR_SIZE, 0, MAX_CHAR_SIZE);
        strncpy(category_values + n_candidates * MAX_CHAR_SIZE, cat_vec[i].value.c_str(), MAX_CHAR_SIZE - 1);
        
        n_candidates++;
    }
    
    return n_candidates;
}


int SplitCandidateGenerator::computeQuantiles(const float *obs, FloatVector &quantiles, const int *sorted_feature_indices, const int feature_idx, splitCandidate *_split_candidates, int _n_candidates){
    
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
        if (n_candidates == 0 || (_split_candidates[_n_candidates - 1].feature_idx == feature_idx && _split_candidates[_n_candidates- 1].feature_value != quantiles[i]) || (_split_candidates[_n_candidates - 1].feature_idx != feature_idx )){
            _split_candidates[_n_candidates].feature_idx = feature_idx;
            _split_candidates[_n_candidates].feature_value = quantiles[i];
            _split_candidates[_n_candidates].categorical_value = nullptr;
            ++_n_candidates;
        }
    }                      
    return _n_candidates;          
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

float scoreCosine(const int *indices, const int n_samples, const float *grads, const int n_cols){
    float *mean = new float[n_cols]; 
    float n_samples_f = static_cast<float>(n_samples);
    float n_samples_recip = 1.0f / n_samples_f;

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d){
        mean[d] = 0;
    }

    for (int i = 0; i < n_samples; ++i){
        int idx = indices[i];
        int row = idx * n_cols;

        #pragma omp simd
        for (int d = 0; d < n_cols; ++d){
            mean[d] += grads[row + d];
        }
    }

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d){
        mean[d] *= n_samples_recip;
    }

    float cosine = cosine_dist(indices, grads, mean, n_samples, n_cols);
    delete[] mean;
    return cosine;
}


float scoreL2(const int *indices, const int n_samples, const float *grads, const int n_cols){
    float *mean = new float[n_cols]; 
    float n_samples_f = static_cast<float>(n_samples);
    float n_samples_recip = 1.0f / n_samples_f;


    #pragma omp simd
    for (int d = 0; d < n_cols; ++d){
        mean[d] = 0.0f;
    }

    #pragma omp simd
    for (int i = 0; i < n_samples * n_cols; ++i){
        int row = i / n_cols;
        int col = i % n_cols;
        mean[col] += grads[indices[row]*n_cols + col];
    }
    

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d){
        mean[d] *= n_samples_recip;
    }
    float mean_norm = squared_norm(mean, n_cols);
    delete[] mean;
    return mean_norm*n_samples_f;

}
