//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_FEATURE_CONSTRAINTS_H
#define CUDA_FEATURE_CONSTRAINTS_H

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

#include "feature_constraints.h"
#include "cuda_types.h"

#ifdef __cplusplus
extern "C" {
#endif
thresholdConstraints* allocate_threshold_constraints_cuda(const int n_cons);
void deallocate_threshold_constraints_cuda(thresholdConstraints *th_cons);
thresholdConstraints* copy_threshold_constraints_cpu_gpu(const thresholdConstraints *th_cons, const int n_cons);
thresholdConstraints* copy_threshold_constraints_gpu_cpu(const thresholdConstraints *th_cons, const int n_cons);
thresholdConstraints* copy_threshold_constraints_gpu_gpu(const thresholdConstraints *th_cons, const int n_cons);
void add_threshold_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, bool inequality_direction, bool is_numeric, 
                    float constraint_value);
hierarchyConstraints* allocate_hierarchy_constraints_cuda(const int n_hr_cons, const int n_hr_features);
void deallocate_hierarchy_constraints_cuda(hierarchyConstraints *hr_cons);
hierarchyConstraints* copy_hierarchy_constraints_cpu_gpu(const hierarchyConstraints *hr_const, const int n_hr_cons, const int n_hr_features);
hierarchyConstraints* copy_hierarchy_constraints_gpu_cpu(const hierarchyConstraints *hr_cons, const int n_hr_cons, const int n_hr_features);
hierarchyConstraints* copy_hierarchy_constraints_gpu_gpu(const hierarchyConstraints *hr_cons, const int n_hr_cons, const int n_hr_features);
void add_hierarchy_constraint_cuda(featureConstraints *constraints, int feature_idx,  bool is_numeric, int* dependent_features, int n_features);
outputConstraints* allocate_output_constraints_cuda(const int n_out_cons, const int output_dim);
void deallocate_output_constraints_cuda(outputConstraints *out_cons);
outputConstraints* copy_output_constraints_cpu_gpu(const outputConstraints *out_cons, const int n_out_cons, const int output_dim);
outputConstraints* copy_output_constraints_gpu_cpu(const outputConstraints *out_cons, const int n_out_cons, const int output_dim);
outputConstraints* copy_output_constraints_gpu_gpu(const outputConstraints *out_cons, const int n_out_cons, const int output_dim);
void add_output_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
    const char *categorical_value, bool inequality_direction, bool is_numeric, float* output_values, int output_dim);
void add_feature_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
        const char * categorical_value, constraintType const_type, int* dependent_features,
        int n_features, float constraint_value, bool inequality_direction,
        bool is_numeric, int output_dim, float *output_values);
featureConstraints* copy_feature_constraint_gpu_cpu(const featureConstraints *constraints, const int output_dim);
featureConstraints* copy_feature_constraint_cpu_gpu(const featureConstraints *constraints, const int output_dim);
featureConstraints* copy_feature_constraint_gpu_gpu(const featureConstraints* constraints, const int output_dim);
void deallocate_constraints_cuda(featureConstraints* constraints);
#ifdef __CUDACC__  // This macro is defined by NVCC
__global__ void reset_satisfied_kernel(thresholdConstraints *th_cons);
__global__ void check_hierarchical_candidates(const n_hr_cons, const TreeNodeGPU* __restrict__ node, const int* __restrict__ dep_count, const int* __restrict__ feature_indices, const int* __restrict__ dep_features, const int* __restrict__ candidate_indices, float* __restrict__ split_scores, const int n_candidates);
#endif

#ifdef __cplusplus
}
#endif

#endif