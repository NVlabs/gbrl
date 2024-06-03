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


void multiply_mat_by_scalar(float *mat, float scalar, const int n_samples, const int n_cols, const int par_th){
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

float* calculate_mean(const float *mat, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    float *mean = new float[n_cols];
    float n_samples_recip = 1.0f / static_cast<float>(n_samples);
    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) 
        mean[d] = 0.0f;
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        omp_set_num_threads(n_threads);
        int elements_per_thread = (n_elements) / n_threads;
        float *thread_mean = new float[n_threads*n_cols];
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) 
            thread_mean[d] = 0.0f;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                thread_mean[thread_id * n_cols + col] += mat[i];
            }
        }
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) {
            int col = d % n_cols;
            mean[col] += thread_mean[d];
        }
        delete[] thread_mean;
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            mean[col] += mat[i]; 
        }
    }

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) 
        mean[d] *= n_samples_recip;

    return mean;
}

float* calculate_var(const float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    float n_samples_recip = 1.0f / (static_cast<float>(n_samples) - 1.0f);
    float value;

    float *var = new float[n_cols];
    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) {
        var[d] = 0;
    }

    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = n_elements / n_threads;
        omp_set_num_threads(n_threads);
        float *thread_var = new float[n_threads*n_cols];
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d){
            thread_var[d] = 0;
        }
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                value = mat[i] - mean[col];
                thread_var[thread_id * n_cols + col] += (value * value);
            }
        }
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) {
            int col = d % n_cols;
            var[col] += thread_var[d];
        }
        delete[] thread_var;
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            value = mat[i] - mean[col];
            var[col] += (value * value);  
        }
    }

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) {
        var[d] *= n_samples_recip;
    }
    return var;
}


float* calculate_row_covariance(const float *mat_l, const float *mat_r, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    // assumes both matrices are centered
    float *cov = new float[n_cols];
    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) 
        cov[d] = 0.0f;

    float n_samples_recip = 1.0f / (static_cast<float>(n_samples) - 1.0f);

    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = n_elements / n_threads;
        omp_set_num_threads(n_threads);
        float *thread_cov = new float[n_threads*n_cols];
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) 
            thread_cov[d] = 0.0f;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                thread_cov[thread_id * n_cols + col] += (mat_l[i] * mat_r[i]);
            }
        }
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) {
            int col = d % n_cols;
            cov[col] += thread_cov[d];
        }
        delete[] thread_cov;
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            cov[col] += ((mat_l[i] * mat_r[i]));  
        }
    }

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) 
        cov[d] *= n_samples_recip;

    return cov;
}

float* calculate_var_and_center(float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    float *var = new float[n_cols];
    float n_samples_recip = 1.0f / (static_cast<float>(n_samples) - 1.0f);
    float value;

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) 
        var[d] = 0;

    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = n_elements / n_threads;
        omp_set_num_threads(n_threads);
        float *thread_var = new float[n_threads*n_cols];
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) 
            thread_var[d] = 0.0f;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                value = mat[i] - mean[col];
                thread_var[thread_id * n_cols + col] += (value * value);
                mat[i] -= mean[col]; 
            }
        }
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) {
            int col = d % n_cols;
            var[col] += thread_var[d];
        }
        delete[] thread_var;
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            value = mat[i] - mean[col];
            var[col] += (value * value);  
            mat[i] -= mean[col]; 
        }
    }

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) 
        var[d] *= n_samples_recip;

    return var;
}

float* calculate_std_and_center(float *mat, const float *mean, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    float *var = new float[n_cols];
    float n_samples_recip = 1.0f / (static_cast<float>(n_samples) - 1.0f);
    float value;

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d) 
        var[d] = 0;

    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        int elements_per_thread = n_elements / n_threads;
        omp_set_num_threads(n_threads);
        float *thread_var = new float[n_threads*n_cols];
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) 
            thread_var[d] = 0;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                value = mat[i] - mean[col];
                thread_var[thread_id * n_cols + col] += (value * value);
                mat[i] -= mean[col]; 
            }
        }
        #pragma omp simd
        for (int d = 0; d < n_threads*n_cols; ++d) {
            int col = d % n_cols;
            var[col] += thread_var[d];
        }
        delete[] thread_var;
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            value = mat[i] - mean[col];
            var[col] += (value * value);  
            mat[i] -= mean[col]; 
        }
    }

    #pragma omp simd
    for (int d = 0; d < n_cols; ++d)
        var[d] = sqrtf(var[d] * n_samples_recip);

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

void set_zero_mat(float *mat, const int size, const int par_th){
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
                mat[i] = 0.0f;
        }
     } else {
        #pragma omp simd
        for (int i = 0; i < size; ++i)
            mat[i] = 0.0f;
    }
}


float* init_zero_mat(const int size, const int par_th){
    float *mat = new float[size];
    set_zero_mat(mat, size, par_th);
    return mat;
}


float* calculate_max(const float *mat, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    float *max = new float[n_cols];
    for (int d = 0; d < n_cols; ++d) 
        max[d] = -INFINITY;
    
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        omp_set_num_threads(n_threads);
        int elements_per_thread = (n_elements) / n_threads;
        float *thread_max = new float[n_threads * n_cols];
        for (int d = 0; d < n_threads * n_cols; ++d) 
            thread_max[d] = -INFINITY;

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                thread_max[thread_id * n_cols + col] = mat[i] > thread_max[thread_id * n_cols + col] ? mat[i] : thread_max[thread_id * n_cols + col] ;
            }
        }

        #pragma omp simd
        for (int d = 0; d < n_threads * n_cols; ++d) {
            int col = d % n_cols;
            max[col] = max[col] > thread_max[d] ? max[col] : thread_max[d];
        }
        delete[] thread_max;
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            max[col] = max[col] > mat[i] ? max[col] : mat[i];
        }
    }

    return max;
}

float* calculate_min(const float *mat, const int n_samples, const int n_cols, const int par_th){
    int n_elements = n_samples * n_cols;
    float *min = new float[n_cols];
    for (int d = 0; d < n_cols; ++d) 
        min[d] = INFINITY;
    
    int n_threads = calculate_num_threads(n_elements, par_th);
    if (n_threads > 1){
        omp_set_num_threads(n_threads);
        int elements_per_thread = (n_elements) / n_threads;
        float *thread_min = new float[n_threads * n_cols];
        for (int d = 0; d < n_threads * n_cols; ++d) 
            thread_min[d] = INFINITY;

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int start_idx = thread_id * elements_per_thread;
            int end_idx = (thread_id == n_threads - 1) ? n_elements : start_idx + elements_per_thread;
            #pragma omp simd
            for (int i = start_idx; i < end_idx; ++i) {
                int col = i % n_cols;
                thread_min[thread_id * n_cols + col] = mat[i] < thread_min[thread_id * n_cols + col] ? mat[i] : thread_min[thread_id * n_cols + col] ;
            }
        }

        #pragma omp simd
        for (int d = 0; d < n_threads * n_cols; ++d) {
            int col = d % n_cols;
            min[col] = min[col] < thread_min[d] ? min[col] : thread_min[d];
        }
        delete[] thread_min;
    } else {
        #pragma omp simd
        for (int i = 0; i < n_elements; ++i) {
            int col = i % n_cols;
            min[col] = min[col] < mat[i] ? min[col] : mat[i];
        }
    }

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