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

#include "types.h"

std::string VectoString(const float* vec, const int vec_size);
int binaryToDecimal(const BoolVector& binaryPath);

inline int calculate_num_threads(int total_elements, int min_elements_per_thread) {
    int max_threads = omp_get_max_threads();
    int n_threads = total_elements / min_elements_per_thread;
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

inline void valid_tree_idx(const int tree_idx, const ensembleMetaData* metadata){
    if (tree_idx < 0 || tree_idx >= metadata->n_trees){
        std::cerr << "ERROR: invalid tree_idx " << tree_idx << " in ensemble with ntrees = " << metadata->n_trees <<std::endl;
        throw std::runtime_error("Invalid tree index");
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
#endif 