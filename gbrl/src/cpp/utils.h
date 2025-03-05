//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <cmath>  // For std::round
#include <cstdint> // For int16_t
#include <algorithm> // For std::clamp
#include <tuple>
#include <functional>

#include "types.h"

std::string VectoString(const float* vec, const int vec_size);
int binaryToDecimal(const BoolVector& binaryPath);

inline int calculate_num_threads(int total_elements, int min_elements_per_thread) {
    int max_threads = omp_get_max_threads();
    int n_threads = total_elements / min_elements_per_thread;
    
    if (n_threads > total_elements)
        n_threads = total_elements;

    if (n_threads <= 1) {
        return 1; // At least one thread
    } else if (n_threads > max_threads) {
        return max_threads; // Do not exceed the maximum available threads
    } else {
        return n_threads; // Number of threads based on workload
    }
}

template<typename T>
int count_distinct(T *arr, int n);

template <typename T>
void selective_copy(const int num_indices, const int* indices, T* dest, const T* src, const int elements_dim);

void selective_copy_char(const int num_indices, const int* indices, char* dest, const char* src, const int elements_dim);

inline void valid_tree_idx(const int tree_idx, const ensembleMetaData* metadata){
    if (tree_idx < 0 || tree_idx > metadata->n_trees){
        std::cerr << "ERROR: invalid tree_idx " << tree_idx << " in ensemble with ntrees = " << metadata->n_trees <<std::endl;
        throw std::runtime_error("Invalid tree index");
    }
}

inline void valid_tree_range(const int start_tree_idx, const int stop_tree_idx, const ensembleMetaData* metadata){
    if (start_tree_idx < 0 || stop_tree_idx > metadata->n_trees || start_tree_idx >= stop_tree_idx){
        std::cerr << "ERROR: invalid tree range [" <<  start_tree_idx << ", " << stop_tree_idx << "] in ensemble with ntrees = " << metadata->n_trees <<std::endl;
        throw std::runtime_error("Invalid tree range");
    }
}


template<typename T>
inline T* reallocate_and_copy(T* original, size_t new_n_elements, size_t old_n_elements) {
    // Allocate new memory block with the given size
    T* new_block = new T[new_n_elements];
    // Copy data from the original to the new block
    memcpy(new_block, original, old_n_elements * sizeof(T));
    // Delete the original block
    delete[] original;
    // Return the new memory block pointer
    return new_block;
}

void write_header(std::ofstream& file, const serializationHeader& header);
serializationHeader create_header();
serializationHeader read_header(std::ifstream& file);

void display_header(serializationHeader header);

// Function to convert float to int16
inline int16_t float_to_int16(float value) {
    // Multiply by 256 (2^8) and round to nearest integer
    float scaled_value = value * 256.0f;
    if (scaled_value < static_cast<float>(INT16_MIN))
        scaled_value = static_cast<float>(INT16_MIN);

    if (scaled_value > static_cast<float>(INT16_MAX))
        scaled_value = static_cast<float>(INT16_MAX);
    int16_t result = static_cast<int16_t>(std::round(scaled_value));
    return result;
}

inline int32_t float_to_int32(float value) {
    // Multiply by 256 (2^16) and round to nearest integer
    float scaled_value = value * 65536.0f;
    if (scaled_value < static_cast<float>(INT32_MIN))
        scaled_value = static_cast<float>(INT32_MIN);

    if (scaled_value > static_cast<float>(INT32_MAX))
        scaled_value = static_cast<float>(INT32_MAX);
    int32_t result = static_cast<int32_t>(std::round(scaled_value));
    return result;
}

struct tuple_hash {
    std::size_t operator()(const std::tuple<int, bool>& t) const {
        return std::hash<int>()(std::get<0>(t)) ^ (std::hash<bool>()(std::get<1>(t)) << 1);
    }
};

#endif 