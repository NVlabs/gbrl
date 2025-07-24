//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
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
#endif 