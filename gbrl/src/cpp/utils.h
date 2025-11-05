//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file utils.h
 * @brief Utility functions and helper methods for GBRL library
 * 
 * Contains various utility functions for string conversion, validation,
 * memory management, numerical conversions, and other common operations.
 */

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <tuple>
#include <functional>

#include "types.h"

/**
 * @brief Convert float array to formatted string representation
 * 
 * @param vec Pointer to float array
 * @param vec_size Number of elements in the array
 * @return String representation with fixed precision
 */
std::string VectoString(const float* vec, const int vec_size);

/**
 * @brief Convert binary path (represented as bool vector) to decimal index
 * 
 * Used for converting tree traversal paths to leaf indices in oblivious trees.
 * 
 * @param binaryPath Vector of boolean values representing binary path
 * @return Decimal representation of the binary path
 */
int binaryToDecimal(const BoolVector& binaryPath);

/**
 * @brief Calculate optimal number of threads for parallel execution
 * 
 * Determines the number of threads to use based on workload size and
 * available hardware threads, ensuring efficient parallelization.
 * 
 * @param total_elements Total number of elements to process
 * @param min_elements_per_thread Minimum elements each thread should handle
 * @return Optimal number of threads to use
 */
inline int calculate_num_threads(
    int total_elements,
    int min_elements_per_thread
) {
    int max_threads = omp_get_max_threads();
    int n_threads = total_elements / min_elements_per_thread;
    
    if (n_threads > total_elements)
        n_threads = total_elements;

    if (n_threads <= 1) {
        return 1;
    } else if (n_threads > max_threads) {
        return max_threads;
    } else {
        return n_threads;
    }
}

/**
 * @brief Count number of distinct elements in an array
 * 
 * @tparam T Element type
 * @param arr Array to examine
 * @param n Number of elements
 * @return Count of unique elements
 */
template<typename T>
int count_distinct(T *arr, int n);

/**
 * @brief Validate tree index is within valid range
 * 
 * @param tree_idx Index to validate
 * @param metadata Ensemble metadata containing valid tree count
 * @throws std::runtime_error if index is out of range
 */
inline void valid_tree_idx(const int tree_idx, const ensembleMetaData* metadata) {
    if (tree_idx < 0 || tree_idx > metadata->n_trees) {
        std::cerr << "ERROR: invalid tree_idx " << tree_idx 
                  << " in ensemble with n_trees = " << metadata->n_trees
                  << std::endl;
        throw std::runtime_error("Invalid tree index");
    }
}

/**
 * @brief Validate tree range is valid
 * 
 * @param start_tree_idx Starting tree index (inclusive)
 * @param stop_tree_idx Stopping tree index (exclusive)
 * @param metadata Ensemble metadata containing valid tree count
 * @throws std::runtime_error if range is invalid
 */
inline void valid_tree_range(
    const int start_tree_idx,
    const int stop_tree_idx,
    const ensembleMetaData* metadata
) {
    if (start_tree_idx < 0 || 
        stop_tree_idx > metadata->n_trees || 
        start_tree_idx >= stop_tree_idx) {
        std::cerr << "ERROR: invalid tree range [" << start_tree_idx 
                  << ", " << stop_tree_idx << "] in ensemble with n_trees = "
                  << metadata->n_trees << std::endl;
        throw std::runtime_error("Invalid tree range");
    }
}

/**
 * @brief Reallocate memory block and copy existing data
 * 
 * Allocates a new memory block of the specified size, copies data from
 * the original block, and frees the original memory.
 * 
 * @tparam T Element type
 * @param original Pointer to original memory block
 * @param new_n_elements Number of elements in new block
 * @param old_n_elements Number of elements in original block
 * @return Pointer to new memory block
 */
template<typename T>
inline T* reallocate_and_copy(
    T* original,
    size_t new_n_elements,
    size_t old_n_elements
) {
    // Allocate new memory block with the given size
    T* new_block = new T[new_n_elements];
    
    // Copy data from the original to the new block
    memcpy(new_block, original, old_n_elements * sizeof(T));
    
    // Delete the original block
    delete[] original;
    
    // Return the new memory block pointer
    return new_block;
}

/**
 * @brief Write serialization header to file
 * 
 * @param file Output file stream
 * @param header Header structure to write
 */
void write_header(std::ofstream& file, const serializationHeader& header);

/**
 * @brief Create a new serialization header with current version
 * 
 * @return Header populated with version information
 */
serializationHeader create_header();

/**
 * @brief Read serialization header from file
 * 
 * @param file Input file stream
 * @return Header structure read from file
 */
serializationHeader read_header(std::ifstream& file);

/**
 * @brief Display header version information to console
 * 
 * @param header Header to display
 */
void display_header(serializationHeader header);

/**
 * @brief Convert float to 16-bit fixed-point representation
 * 
 * Scales the float value by 2^8 and converts to int16_t, clamping
 * to valid range to prevent overflow.
 * 
 * @param value Float value to convert
 * @return 16-bit fixed-point representation
 */
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

/**
 * @brief Convert float to 32-bit fixed-point representation
 * 
 * Scales the float value by 2^16 and converts to int32_t, clamping
 * to valid range to prevent overflow.
 * 
 * @param value Float value to convert
 * @return 32-bit fixed-point representation
 */
inline int32_t float_to_int32(float value) {
    // Multiply by 65536 (2^16) and round to nearest integer
    float scaled_value = value * 65536.0f;
    
    if (scaled_value < static_cast<float>(INT32_MIN))
        scaled_value = static_cast<float>(INT32_MIN);

    if (scaled_value > static_cast<float>(INT32_MAX))
        scaled_value = static_cast<float>(INT32_MAX);
        
    int32_t result = static_cast<int32_t>(std::round(scaled_value));
    return result;
}

/**
 * @brief Hash function for tuple of int and bool
 * 
 * Used for creating hash maps with composite keys.
 */
struct tuple_hash {
    std::size_t operator()(const std::tuple<int, bool>& t) const {
        return std::hash<int>()(std::get<0>(t)) ^ 
               (std::hash<bool>()(std::get<1>(t)) << 1);
    }
};

#endif // UTILS_H 