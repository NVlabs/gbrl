//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
/**
 * @file utils.cpp
 * @brief Implementation of utility functions for validation, serialization, and type conversion
 * 
 * Provides helper functions for tree index validation, fixed-point number conversion,
 * and serialization header management for model persistence.
 */

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "utils.h"
#include "config.h"

std::string VectoString(const float* vec, const int vec_size) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    if (vec_size > 1)
        oss << "[";
        
    for (int i = 0; i < vec_size; ++i) {
        oss << vec[i];
        if (i < vec_size - 1)
            oss << ", ";
    }
    
    if (vec_size > 1)
        oss << "]";
        
    return oss.str();
}

int binaryToDecimal(const BoolVector& binaryPath) {
    int decimal = 0;
    int i = static_cast<int>(binaryPath.size()) - 1;
    size_t j = 0;
    
    while (i >= 0) {
        decimal += binaryPath[i] * (1 << j);
        j++;
        i--;
    }
    
    return decimal + (1 << binaryPath.size()) - 1;
}

void write_header(std::ofstream& file, const serializationHeader& header) {
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file.good()) {
        throw std::runtime_error("Failed to write header to file");
    }
}

serializationHeader create_header() {
    serializationHeader header;
    header.major_version = MAJOR_VERSION;
    header.minor_version = MINOR_VERSION;
    header.patch_version = PATCH_VERSION;
    return header;
}

serializationHeader read_header(std::ifstream& file) {
    serializationHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good()) {
        throw std::runtime_error("Failed to read header from file");
    }
    return header;
}

void display_header(serializationHeader header) {
    std::cout << "Version " << header.major_version << "."
              << header.minor_version << "." << header.patch_version 
              << std::endl;
}

template<typename T>
int count_distinct(T *arr, int n) {
    /*
     * Count the number of distinct elements in an unsorted array
     * Time complexity: O(n^2)
     * Space complexity: O(1)
     */
    int res = 1;

    // Pick all elements one by one
    for (int i = 1; i < n; i++) {
        int j = 0;
        for (j = 0; j < i; j++) {
            if (arr[i] == arr[j])
                break;
        }

        if (i == j)
            res++;
    }
    return res;
}

// Explicit template instantiations
template int count_distinct<int>(int* arr, int n);
template int count_distinct<float>(float* arr, int n);

/**
 * @brief Selectively copy elements from source to destination array
 * 
 * Copies elements from src to dest based on provided indices. Each element
 * has elements_dim components that are copied together. Parallelized with OpenMP.
 * 
 * @tparam T Element type (float, int, bool)
 * @param num_indices Number of elements to copy
 * @param indices Array of source indices to copy from
 * @param dest Destination array
 * @param src Source array
 * @param elements_dim Dimensionality of each element (number of components)
 */
template <typename T>
void selective_copy(const int num_indices, const int* indices, T* dest, const T* src, const int elements_dim){
    #pragma omp parallel for
    for (int i = 0; i < num_indices; ++i) {
        int start_idx = indices[i];
        for (int j = 0; j < elements_dim; ++j) {
            dest[i*elements_dim + j] = src[start_idx*elements_dim + j];
        }
    }
}

template void selective_copy<float>(const int num_indices, const int* indices, float* dest, const float* src, const int elements_dim);
template void selective_copy<int>(const int num_indices, const int* indices, int* dest, const int* src, const int elements_dim);
template void selective_copy<bool>(const int num_indices, const int* indices, bool* dest, const bool* src, const int elements_dim);

/**
 * @brief Selectively copy character strings from source to destination
 * 
 * Specialized version of selective_copy for character arrays with fixed-size
 * strings (MAX_CHAR_SIZE bytes each). Uses memcpy for efficient copying.
 * Parallelized with OpenMP.
 * 
 * @param num_indices Number of strings to copy
 * @param indices Array of source indices to copy from
 * @param dest Destination character array
 * @param src Source character array
 * @param elements_dim Number of strings per indexed element
 */
void selective_copy_char(const int num_indices, const int* indices, char* dest, const char* src, const int elements_dim){
    #pragma omp parallel for
    for (int i = 0; i < num_indices; ++i) {
        memcpy(dest + (i*elements_dim)*MAX_CHAR_SIZE, src + (indices[i] * elements_dim) * MAX_CHAR_SIZE, sizeof(char)*MAX_CHAR_SIZE);
    }
}