//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <cmath>
#include <cstring>

void add_vec_to_mat(float *mat, const float *vec, const int n_samples, const int n_cols, const int par_th);
void subtract_vec_from_mat(float *mat, float *vec, const int n_samples, const int n_cols, const int par_th);
float* calculate_mean(const float *mat, const int n_samples, const int n_cols, const int par_th);
float* calculate_var(const float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th);
void _multiply_mat_by_scalar(float *mat, float scalar, const int n_samples, const int n_cols, const int par_th);
void _broadcast_mat_elementwise_mult_by_vec_into_mat(float *lmat, const float *rmat, const float *vec, const float scalar, const int n_samples, const int n_cols, const int par_th, const bool col_wise);
void _broadcast_mat_elementwise_mult_by_vec(float *mat, const float *vec, const float scalar, const int n_samples, const int n_cols, const int par_th);
void _broadcast_mat_elementwise_div_by_vec(float *mat, const float *vec, const float scalar, const int n_samples, const int n_cols, const int par_th);
float* calculate_var_and_center(float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th);
float* calculate_std_and_center(float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th);
float* copy_mat(const float *mat, const int size, const int par_th);
void _copy_mat(float *mat_l, const float *mat_r, const int size, const int par_th);
float* calculate_row_covariance(const float *mat_l, const float *mat_r, const int n_samples, const int n_cols, const int par_th);
void _element_wise_addition(float *mat_l, const float *mat_r, const int size, const int par_th);
void _element_wise_multiplication(float *mat_l, const float *mat_r, const int size, const int par_th);
float* element_wise_division(const float *mat_l, const float *mat_r, const int size, const int par_th);
void multiply_mat_by_vec_subtract_result(float *result, const float *mat, const float *vec, const int n_samples, const int n_cols, const int par_th);
void divide_mat_by_vec_inplace(float *mat, const float *vec, const int n_samples, const int n_cols, const int par_th);\

inline float* init_zero_mat(const int size){
    float *mat = new float[size];
    memset(mat, 0, sizeof(float)*size);
    return mat;
}

void set_mat_value(float *mat, const int size, const float value, const int par_th);
float* calculate_max(const float *mat, const int n_samples, const int n_cols, const int par_th);
float* calculate_min(const float *mat, const int n_samples, const int n_cols, const int par_th);
void calculate_squared_norm(float *norm, const float *mat, const int n_samples, const int n_cols, const int par_th);

inline float mat_vec_dot_sum(const int *indices, const float *grads, const float *vec, const int n_samples, const int n_cols){
    float sum = 0.0f;

    for (int row = 0; row < n_samples; row++){
        #pragma omp simd
        for (int col = 0; col < n_cols; ++col){
            sum += grads[indices[row]*n_cols + col] * vec[col];
        }
    }

    return sum;
}

inline float norm(const float *vec, const int n_samples){
    float sum = 0.0f;

    #pragma omp simd
    for (int n = 0; n < n_samples; ++n){
        sum += (vec[n]*vec[n]);
    }
    return sqrtf(sum);
}

inline float squared_norm(const float *vec, const int n_samples){
    float sum = 0.0f;

    #pragma omp simd
    for (int n = 0; n < n_samples; ++n){
        sum += (vec[n]*vec[n]);
    }
    return sum;
}

inline float cosine_dist(const int *indices, const float *raw_grads, const float *mean, const int n_samples, const int n_cols){
    if (n_samples == 0)
        return 0.0f;
    float n_samples_f = static_cast<float>(n_samples);
    float sum_dot_product = mat_vec_dot_sum(indices, raw_grads, mean, n_samples, n_cols);
    float mean_norm = squared_norm(mean, n_cols);
    float denominator = mean_norm * n_samples_f;
    if (denominator == 0.0f) {
        return 0.0f;
    }
    return (sum_dot_product / sqrt(denominator)) ;
}

inline float cosine_score(const int *true_indices, const int *false_indices,  const float *raw_grads, const float *true_mean, const float *false_mean, const int true_n_samples, const int false_n_samples, const int n_cols){
    float true_numerator = 0.0f, false_numerator = 0.0f;
    float true_n_samples_f = static_cast<float>(true_n_samples), false_n_samples_f = static_cast<float>(false_n_samples);
    if (true_n_samples > 0)
        true_numerator = mat_vec_dot_sum(true_indices, raw_grads, true_mean, true_n_samples, n_cols);
    if (false_n_samples > 0)
        false_numerator = mat_vec_dot_sum(false_indices, raw_grads, false_mean, false_n_samples, n_cols);

    float true_mean_norm = squared_norm(true_mean, n_cols);
    float false_mean_norm = squared_norm(false_mean, n_cols);
    float true_denominator = true_mean_norm * true_n_samples_f;
    float false_denominator = false_mean_norm * false_n_samples_f;

    float numerator = true_numerator + false_numerator;
    float denominator = true_denominator + false_denominator;

    if (denominator == 0.0f) {
        return 0.0f;
    }
    return (numerator / sqrtf(denominator)) ;
}

#endif 