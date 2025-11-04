//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file math_ops.h
 * @brief Mathematical operations for matrix and vector computations
 * 
 * Provides optimized mathematical operations used in gradient boosting,
 * including matrix operations, statistical computations, and scoring functions.
 * Many operations are parallelized for efficiency.
 */

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <cmath>
#include <cstring>

// ============================================================================
// Matrix and Vector Operations
// ============================================================================

/**
 * @brief Add vector to each row of a matrix in-place
 * 
 * @param mat Matrix to modify (n_samples x n_cols)
 * @param vec Vector to add (n_cols)
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void add_vec_to_mat(
    float *mat,
    const float *vec,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Subtract vector from each row of a matrix in-place
 * 
 * @param mat Matrix to modify (n_samples x n_cols)
 * @param vec Vector to subtract (n_cols)
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void subtract_vec_from_mat(
    float *mat,
    float *vec,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Calculate column-wise mean of a matrix
 * 
 * @param mat Input matrix (n_samples x n_cols)
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @return Pointer to mean vector (n_cols), caller must free
 */
float* calculate_mean(
    const float *mat,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Calculate column-wise variance of a matrix
 * 
 * @param mat Input matrix (n_samples x n_cols)
 * @param mean Mean vector (n_cols)
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @return Pointer to variance vector (n_cols), caller must free
 */
float* calculate_var(
    const float *mat,
    const float *mean,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Multiply matrix by scalar in-place
 * 
 * @param mat Matrix to modify
 * @param scalar Scalar multiplier
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void _multiply_mat_by_scalar(
    float *mat,
    float scalar,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Element-wise multiply right matrix by vector, store in left matrix
 * 
 * @param lmat Left/output matrix
 * @param rmat Right/input matrix
 * @param vec Vector for broadcasting
 * @param scalar Additional scalar multiplier
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @param col_wise If true, broadcast column-wise; else row-wise
 */
void _broadcast_mat_elementwise_mult_by_vec_into_mat(
    float *lmat,
    const float *rmat,
    const float *vec,
    const float scalar,
    const int n_samples,
    const int n_cols,
    const int par_th,
    const bool col_wise
);

/**
 * @brief Element-wise multiply matrix by broadcasted vector in-place
 * 
 * @param mat Matrix to modify
 * @param vec Vector for broadcasting
 * @param scalar Additional scalar multiplier
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void _broadcast_mat_elementwise_mult_by_vec(
    float *mat,
    const float *vec,
    const float scalar,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Element-wise divide matrix by broadcasted vector in-place
 * 
 * @param mat Matrix to modify
 * @param vec Vector for broadcasting
 * @param scalar Additional scalar multiplier
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void _broadcast_mat_elementwise_div_by_vec(
    float *mat,
    const float *vec,
    const float scalar,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Calculate variance and center matrix (subtract mean) in-place
 * 
 * @param mat Matrix to center (modified in-place)
 * @param mean Mean vector
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @return Pointer to variance vector, caller must free
 */
float* calculate_var_and_center(
    float *mat,
    const float *mean,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Calculate standard deviation and center matrix in-place
 * 
 * @param mat Matrix to center (modified in-place)
 * @param mean Mean vector
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @return Pointer to std deviation vector, caller must free
 */
float* calculate_std_and_center(
    float *mat,
    const float *mean,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Create a copy of a matrix
 * 
 * @param mat Source matrix
 * @param size Total number of elements
 * @param par_th Parallelization threshold
 * @return Pointer to copied matrix, caller must free
 */
float* copy_mat(
    const float *mat,
    const int size,
    const int par_th
);

/**
 * @brief Copy matrix from right to left in-place
 * 
 * @param mat_l Destination matrix
 * @param mat_r Source matrix
 * @param size Total number of elements
 * @param par_th Parallelization threshold
 */
void _copy_mat(
    float *mat_l,
    const float *mat_r,
    const int size,
    const int par_th
);

/**
 * @brief Calculate row-wise covariance between two matrices
 * 
 * @param mat_l Left matrix
 * @param mat_r Right matrix
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @return Pointer to covariance vector, caller must free
 */
float* calculate_row_covariance(
    const float *mat_l,
    const float *mat_r,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Element-wise addition of two matrices in-place
 * 
 * @param mat_l Left matrix (modified to store result)
 * @param mat_r Right matrix
 * @param size Total number of elements
 * @param par_th Parallelization threshold
 */
void _element_wise_addition(
    float *mat_l,
    const float *mat_r,
    const int size,
    const int par_th
);

/**
 * @brief Element-wise multiplication of two matrices in-place
 * 
 * @param mat_l Left matrix (modified to store result)
 * @param mat_r Right matrix
 * @param size Total number of elements
 * @param par_th Parallelization threshold
 */
void _element_wise_multiplication(
    float *mat_l,
    const float *mat_r,
    const int size,
    const int par_th
);

/**
 * @brief Element-wise division of two matrices
 * 
 * @param mat_l Left matrix (numerator)
 * @param mat_r Right matrix (denominator)
 * @param size Total number of elements
 * @param par_th Parallelization threshold
 * @return Pointer to result matrix, caller must free
 */
float* element_wise_division(
    const float *mat_l,
    const float *mat_r,
    const int size,
    const int par_th
);

/**
 * @brief Multiply matrix by vector and subtract from result
 * 
 * @param result Result array to subtract from
 * @param mat Input matrix
 * @param vec Input vector
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void multiply_mat_by_vec_subtract_result(
    float *result,
    const float *mat,
    const float *vec,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Divide matrix by vector in-place (broadcasting)
 * 
 * @param mat Matrix to modify
 * @param vec Divisor vector
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void divide_mat_by_vec_inplace(
    float *mat,
    const float *vec,
    const int n_samples,
    const int n_cols,
    const int par_th
);
/**
 * @brief Initialize a zero matrix
 * 
 * @param size Total number of elements
 * @return Pointer to zero-initialized matrix, caller must free
 */
inline float* init_zero_mat(const int size) {
    float *mat = new float[size];
    memset(mat, 0, sizeof(float) * size);
    return mat;
}

/**
 * @brief Set all elements of matrix to a specific value
 * 
 * @param mat Matrix to modify
 * @param size Total number of elements
 * @param value Value to set
 * @param par_th Parallelization threshold
 */
void set_mat_value(
    float *mat,
    const int size,
    const float value,
    const int par_th
);

/**
 * @brief Calculate column-wise maximum values
 * 
 * @param mat Input matrix
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @return Pointer to max vector (n_cols), caller must free
 */
float* calculate_max(
    const float *mat,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Calculate column-wise minimum values
 * 
 * @param mat Input matrix
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 * @return Pointer to min vector (n_cols), caller must free
 */
float* calculate_min(
    const float *mat,
    const int n_samples,
    const int n_cols,
    const int par_th
);

/**
 * @brief Calculate row-wise squared norms
 * 
 * @param norm Output array for norms
 * @param mat Input matrix
 * @param n_samples Number of rows
 * @param n_cols Number of columns
 * @param par_th Parallelization threshold
 */
void calculate_squared_norm(
    float *norm,
    const float *mat,
    const int n_samples,
    const int n_cols,
    const int par_th
);

// ============================================================================
// Inline Mathematical Functions
// ============================================================================

/**
 * @brief Compute dot product sum for indexed samples
 * 
 * Computes sum of dot products between gradients at specified indices
 * and a vector, using SIMD vectorization.
 * 
 * @param indices Sample indices to include
 * @param grads Gradient matrix
 * @param vec Vector to dot with
 * @param n_samples Number of samples
 * @param n_cols Number of columns
 * @return Sum of dot products
 */
inline float mat_vec_dot_sum(
    const int *indices,
    const float *grads,
    const float *vec,
    const int n_samples,
    const int n_cols
) {
    float sum = 0.0f;

    for (int row = 0; row < n_samples; row++) {
        #pragma omp simd
        for (int col = 0; col < n_cols; ++col) {
            sum += grads[indices[row] * n_cols + col] * vec[col];
        }
    }

    return sum;
}

/**
 * @brief Calculate L2 norm of a vector
 * 
 * @param vec Input vector
 * @param n_samples Number of elements
 * @return L2 norm (Euclidean length)
 */
inline float norm(const float *vec, const int n_samples) {
    float sum = 0.0f;

    #pragma omp simd
    for (int n = 0; n < n_samples; ++n) {
        sum += (vec[n] * vec[n]);
    }
    
    return sqrtf(sum);
}

/**
 * @brief Calculate squared L2 norm of a vector
 * 
 * @param vec Input vector
 * @param n_samples Number of elements
 * @return Squared L2 norm
 */
inline float squared_norm(const float *vec, const int n_samples) {
    float sum = 0.0f;

    #pragma omp simd
    for (int n = 0; n < n_samples; ++n) {
        sum += (vec[n] * vec[n]);
    }
    
    return sum;
}

/**
 * @brief Calculate cosine distance between indexed gradients and mean vector
 * 
 * Computes normalized cosine similarity between gradients at specified
 * indices and a mean vector.
 * 
 * @param indices Sample indices
 * @param raw_grads Gradient matrix
 * @param mean Mean vector
 * @param n_samples Number of samples
 * @param n_cols Number of columns
 * @return Cosine distance score
 */
inline float cosine_dist(
    const int *indices,
    const float *raw_grads,
    const float *mean,
    const int n_samples,
    const int n_cols
) {
    if (n_samples == 0)
        return 0.0f;
        
    float n_samples_f = static_cast<float>(n_samples);
    float sum_dot_product = mat_vec_dot_sum(indices, raw_grads, mean, n_samples, n_cols);
    float mean_norm = squared_norm(mean, n_cols);
    float denominator = mean_norm * n_samples_f;
    
    if (denominator == 0.0f) {
        return 0.0f;
    }
    
    return (sum_dot_product / sqrt(denominator));
}

/**
 * @brief Calculate cosine score for a binary split
 * 
 * Computes combined cosine score for both sides of a split, measuring
 * how well the split separates the gradient directions.
 * 
 * @param true_indices Indices going to true/left child
 * @param false_indices Indices going to false/right child
 * @param raw_grads Gradient matrix
 * @param true_mean Mean vector for true child
 * @param false_mean Mean vector for false child
 * @param true_n_samples Number of samples in true child
 * @param false_n_samples Number of samples in false child
 * @param n_cols Number of columns
 * @return Combined cosine score
 */
inline float cosine_score(
    const int *true_indices,
    const int *false_indices,
    const float *raw_grads,
    const float *true_mean,
    const float *false_mean,
    const int true_n_samples,
    const int false_n_samples,
    const int n_cols
) {
    float true_numerator = 0.0f, false_numerator = 0.0f;
    float true_n_samples_f = static_cast<float>(true_n_samples);
    float false_n_samples_f = static_cast<float>(false_n_samples);
    
    if (true_n_samples > 0)
        true_numerator = mat_vec_dot_sum(
            true_indices, raw_grads, true_mean, true_n_samples, n_cols
        );
        
    if (false_n_samples > 0)
        false_numerator = mat_vec_dot_sum(
            false_indices, raw_grads, false_mean, false_n_samples, n_cols
        );

    float true_mean_norm = squared_norm(true_mean, n_cols);
    float false_mean_norm = squared_norm(false_mean, n_cols);
    float true_denominator = true_mean_norm * true_n_samples_f;
    float false_denominator = false_mean_norm * false_n_samples_f;

    float numerator = true_numerator + false_numerator;
    float denominator = true_denominator + false_denominator;

    if (denominator == 0.0f) {
        return 0.0f;
    }
    
    return (numerator / sqrtf(denominator));
}

#endif // MATH_OPS_H 