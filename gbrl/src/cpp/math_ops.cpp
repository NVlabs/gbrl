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
 * @file math_ops.cpp
 * @brief Implementation of optimized mathematical operations and matrix functions
 */

#include <omp.h>
#include <iostream>
#include <cmath>

#include "math_ops.h"
#include "utils.h"


void add_vec_to_mat(float *mat, const float *vec, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = n_elements / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                mat[i] += vec[col]; 
            }
        }
    } else {

        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            mat[i] += vec[col]; 
        }
    }
}

void multiply_mat_by_vec_subtract_result(float *result, const float *mat, const float *vec, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = (n_elements) / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                result[i] -= (mat[i]*vec[col]); 
            }
        }
    } else {

        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            result[i] -= (mat[i]*vec[col]); 
            
        }
    }
}

void divide_mat_by_vec_inplace(float *mat, const float *vec, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = (n_elements) / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                mat[i] /= (vec[col] + 1e-8f); 
            }
        }
    } else {

        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            mat[i] /= (vec[col] + 1e-8f);    
        }
    }
}

void subtract_vec_from_mat(float *mat, float *vec, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples*n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        omp_set_num_threads(n_threads);
        int elements_per_thread = n_elements / n_threads;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                mat[i] -= vec[col]; 
            }
        }
    } else {

        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            mat[i] -= vec[col];   
        }
    }
}


void _multiply_mat_by_scalar(float *mat, float scalar, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = (n_elements) / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                mat[i] *= scalar; 
            }
        }
    } else {

        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            mat[i] *= scalar;  
        }
    }
}

void _broadcast_mat_elementwise_mult_by_vec_into_mat(float *lmat, const float *rmat, const float *vec, const float scalar, const int n_samples, const int n_cols, const int par_th, const bool col_wise){
    /* Broadcastig element-wise multiplication of an NxM matrix by an Nx1 vector and storing the result in another matrix.
       Each row of the matrix is multiplied by */
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = (n_elements) / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int vec_idx = (col_wise) ? i % n_cols : i / n_cols;
                lmat[i] = rmat[i]*(vec[vec_idx] + scalar); 
            }
        }
    } else {

        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int vec_idx = (col_wise) ? i % n_cols : i / n_cols;
            lmat[i] = rmat[i]*(vec[vec_idx] + scalar); 
        }
    }
}


void _broadcast_mat_elementwise_mult_by_vec(float *mat, const float *vec, const float scalar, const int n_samples, const int n_cols, const int par_th){
    /* Broadcastig element-wise multiplication of an NxM matrix by an Nx1 vector.
       Each row of the matrix is multiplied by the same vector element*/
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = (n_elements) / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int row = i / n_cols;
                mat[i] *= (vec[row] + scalar); 
            }
        }
    } else {

        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int row = i / n_cols;
            mat[i] *= (vec[row] + scalar); 
        }
    }
}


void _broadcast_mat_elementwise_div_by_vec(float *mat, const float *vec, const float scalar, const int n_samples, const int n_cols, const int par_th){
    /* Broadcastig element-wise multiplication of an NxM matrix by an Nx1 vector.
       Each row of the matrix is multiplied by the same vector element*/
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = (n_elements) / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;

            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int row = i / n_cols;
                mat[i] /= (vec[row] + scalar); 
            }
        }
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int row = i / n_cols;
            mat[i] /= (vec[row] + scalar); 
        }
    }
}



// ---------------------------------------------------------------------------
// Per-column reductions over a row-major (n_samples x n_cols) matrix.
//
// `omp simd` is only used on loops whose iterations are independent:
//  - n_cols >= COL_SIMD_MIN: rows outer, columns inner (each lane owns a column);
//  - narrower matrices: REDUCE_LANES independent partial results per column,
//    vectorized across the lanes and combined at the end.
// No `simd reduction` clause is used (MSVC's experimental OpenMP ignores it).
// Threads split the rows using the team actually granted by OpenMP.
// ---------------------------------------------------------------------------
static constexpr int REDUCE_LANES = 8;
static constexpr int COL_SIMD_MIN = 8;

// acc[d] = op(acc[d], term(r, d)) over rows [row_start, row_end).
template <typename Term, typename Op>
static void reduce_rows(int row_start, int row_end, int n_cols, float *acc,
                        float identity, Term term, Op op) {
    if (n_cols >= COL_SIMD_MIN) {
        for (int r = row_start; r < row_end; ++r) {
            #pragma omp simd
            for (int d = 0; d < n_cols; ++d)
                acc[d] = op(acc[d], term(r, d));
        }
        return;
    }
    float lanes[COL_SIMD_MIN * REDUCE_LANES];
    for (int i = 0; i < n_cols * REDUCE_LANES; ++i)
        lanes[i] = identity;
    int r = row_start;
    for (; r + REDUCE_LANES <= row_end; r += REDUCE_LANES) {
        for (int d = 0; d < n_cols; ++d) {
            float *lane = lanes + d * REDUCE_LANES;
            #pragma omp simd
            for (int l = 0; l < REDUCE_LANES; ++l)
                lane[l] = op(lane[l], term(r + l, d));
        }
    }
    for (; r < row_end; ++r)
        for (int d = 0; d < n_cols; ++d)
            lanes[d * REDUCE_LANES] = op(lanes[d * REDUCE_LANES], term(r, d));
    for (int d = 0; d < n_cols; ++d)
        for (int l = 0; l < REDUCE_LANES; ++l)
            acc[d] = op(acc[d], lanes[d * REDUCE_LANES + l]);
}

// out[d] = op over all rows of term(r, d), starting from identity.
template <typename Term, typename Op>
static void column_reduce(int n_samples, int n_cols, int par_th, float *out,
                          float identity, Term term, Op op) {
    for (int d = 0; d < n_cols; ++d)
        out[d] = identity;
    const int n_threads = calculate_num_threads(n_samples * n_cols, par_th);
    if (n_threads <= 1) {
        reduce_rows(0, n_samples, n_cols, out, identity, term, op);
        return;
    }
    float *partial = new float[n_threads * n_cols];
    for (int i = 0; i < n_threads * n_cols; ++i)
        partial[i] = identity;
    #pragma omp parallel num_threads(n_threads)
    {
        const int thread_id = omp_get_thread_num();
        const int team_size = omp_get_num_threads();
        const int rows_per_thread = n_samples / team_size;
        const int start = thread_id * rows_per_thread;
        const int end = (thread_id == team_size - 1) ? n_samples : start + rows_per_thread;
        reduce_rows(start, end, n_cols, partial + thread_id * n_cols, identity, term, op);
    }
    for (int t = 0; t < n_threads; ++t)
        for (int d = 0; d < n_cols; ++d)
            out[d] = op(out[d], partial[t * n_cols + d]);
    delete[] partial;
}

struct SumOp { float operator()(float a, float b) const { return a + b; } };
struct MaxOp { float operator()(float a, float b) const { return a > b ? a : b; } };
struct MinOp { float operator()(float a, float b) const { return a < b ? a : b; } };


float* calculate_mean(const float *mat, const int n_samples, const int n_cols, const int par_th){
    float *mean = new float[n_cols];
    column_reduce(n_samples, n_cols, par_th, mean, 0.0f,
                  [mat, n_cols](int r, int d) { return mat[r * n_cols + d]; }, SumOp());
    const float n_samples_recip = 1.0f / static_cast<float>(n_samples);
#pragma omp simd
    for (int d = 0; d < n_cols; ++d)
        mean[d] *= n_samples_recip;
    return mean;
}

float* calculate_var(const float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th){
    float *var = new float[n_cols];
    column_reduce(n_samples, n_cols, par_th, var, 0.0f,
                  [mat, mean, n_cols](int r, int d) {
                      const float value = mat[r * n_cols + d] - mean[d];
                      return value * value;
                  }, SumOp());
    const float n_samples_recip = 1.0f / (static_cast<float>(n_samples) - 1.0f);
#pragma omp simd
    for (int d = 0; d < n_cols; ++d)
        var[d] *= n_samples_recip;
    return var;
}


float* calculate_row_covariance(const float *mat_l, const float *mat_r, const int n_samples, const int n_cols, const int par_th){
    // assumes both matrices are centered
    float *cov = new float[n_cols];
    column_reduce(n_samples, n_cols, par_th, cov, 0.0f,
                  [mat_l, mat_r, n_cols](int r, int d) {
                      return mat_l[r * n_cols + d] * mat_r[r * n_cols + d];
                  }, SumOp());
    const float n_samples_recip = 1.0f / (static_cast<float>(n_samples) - 1.0f);
#pragma omp simd
    for (int d = 0; d < n_cols; ++d)
        cov[d] *= n_samples_recip;
    return cov;
}

float* calculate_var_and_center(float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th){
    float *var = new float[n_cols];
    // Each (r, d) element is centered exactly once, so the in-place write is
    // independent across iterations.
    column_reduce(n_samples, n_cols, par_th, var, 0.0f,
                  [mat, mean, n_cols](int r, int d) {
                      const float value = mat[r * n_cols + d] - mean[d];
                      mat[r * n_cols + d] = value;
                      return value * value;
                  }, SumOp());
    const float n_samples_recip = 1.0f / (static_cast<float>(n_samples) - 1.0f);
#pragma omp simd
    for (int d = 0; d < n_cols; ++d)
        var[d] *= n_samples_recip;
    return var;
}

float* calculate_std_and_center(float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th){
    float *var = calculate_var_and_center(mat, mean, n_samples, n_cols, par_th);
#pragma omp simd
    for (int d = 0; d < n_cols; ++d)
        var[d] = sqrtf(var[d]);
    return var;
}

float* copy_mat(const float *mat, const int size, const int par_th){
    float *copied_mat = new float[size];
     int n_threads = calculate_num_threads(size, par_th);
     if (n_threads > 1){
        int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#pragma omp simd
            for (int i = start_idx; i < end_idx; ++i)
                copied_mat[i] = mat[i];
        }
     } else {
        for (int i = 0; i < size; ++i)
            copied_mat[i] = mat[i];
    }
    return copied_mat;
}

void _copy_mat(float *mat_l, const float *mat_r, const int size, const int par_th){
     int n_threads = calculate_num_threads(size, par_th);
     if (n_threads > 1){
        int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#pragma omp simd
            for (int i = start_idx; i < end_idx; ++i)
                mat_l[i] = mat_r[i];
        }
     } else {
        for (int i = 0; i < size; ++i)
            mat_l[i] = mat_r[i];
    }
}

void _element_wise_addition(float *mat_l, const float *mat_r, const int size, const int par_th){
     int n_threads = calculate_num_threads(size, par_th);
     if (n_threads > 1){
        int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#pragma omp simd
            for (int i = start_idx; i < end_idx; ++i)
                mat_l[i] += mat_r[i];
        }
     } else {
        for (int i = 0; i < size; ++i)
            mat_l[i] += mat_r[i];
    }
}

void _element_wise_multiplication(float *mat_l, const float *mat_r, const int size, const int par_th){
     int n_threads = calculate_num_threads(size, par_th);
     if (n_threads > 1){
        int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#pragma omp simd
            for (int i = start_idx; i < end_idx; ++i)
                mat_l[i] *= mat_r[i];
        }
     } else {
        for (int i = 0; i < size; ++i)
            mat_l[i] *= mat_r[i];
    }
}

float* element_wise_division(const float *mat_l, const float *mat_r, const int size, const int par_th){
    float *result = new float[size];
     int n_threads = calculate_num_threads(size, par_th);
     if (n_threads > 1){
        int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#pragma omp simd
            for (int i = start_idx; i < end_idx; ++i)
                result[i] = mat_l[i] / (mat_r[i] + + 1e-8f);
        }
     } else {
        for (int i = 0; i < size; ++i)
            result[i] = mat_l[i] / (mat_r[i] + + 1e-8f);;
    }
    return result;
}

void set_mat_value(float *mat, const int size, const float value, const int par_th){
    int n_threads = calculate_num_threads(size, par_th);
     if (n_threads > 1){
        int elements_per_thread = size / n_threads;
        omp_set_num_threads(n_threads);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? size : start_idx + elements_per_thread;
#pragma omp simd
            for (int i = start_idx; i < end_idx; ++i)
                mat[i] = value;
        }
     } else {
#pragma omp simd
        for (int i = 0; i < size; ++i)
            mat[i] = value;
    }
}

float* calculate_max(const float *mat, const int n_samples, const int n_cols, const int par_th){
    float *max = new float[n_cols];
    column_reduce(n_samples, n_cols, par_th, max, -INFINITY,
                  [mat, n_cols](int r, int d) { return mat[r * n_cols + d]; }, MaxOp());
    return max;
}

float* calculate_min(const float *mat, const int n_samples, const int n_cols, const int par_th){
    float *min = new float[n_cols];
    column_reduce(n_samples, n_cols, par_th, min, INFINITY,
                  [mat, n_cols](int r, int d) { return mat[r * n_cols + d]; }, MinOp());
    return min;
}

void calculate_squared_norm(float *norm, const float *mat, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        omp_set_num_threads(n_threads);
        int elements_per_thread = (n_elements) / n_threads;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int row = i / n_cols;
                norm[row] += mat[i]*mat[i];
            }
        }
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int row = i / n_cols;
            norm[row] += mat[i]*mat[i];
        }
    }
}