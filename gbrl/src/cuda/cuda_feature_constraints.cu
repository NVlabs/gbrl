
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "feature_constraints.h"
#include "cuda_utils.h"
#include "cuda_feature_constraints.h"
#include "cuda_types.h"


thresholdConstraints* allocate_threshold_constraints_cuda(const int n_cons){
    cudaError_t error;

    thresholdConstraints *th_cons = new thresholdConstraints;
    th_cons->feature_indices = nullptr;
    th_cons->feature_values = nullptr;
    th_cons->constraint_values = nullptr;
    th_cons->is_numerics = nullptr;
    th_cons->inequality_directions = nullptr;
    th_cons->satisfied = nullptr;
    th_cons->categorical_values = nullptr;

    char* device_memory_block;
    size_t alloc_size = sizeof(int) * n_cons
                      + sizeof(float) * n_cons * 2
                      + sizeof(bool) * n_cons * 3
                      + sizeof(char) * MAX_CHAR_SIZE * n_cons;
    error = allocateCudaMemory((void**)&device_memory_block, alloc_size, "when trying to allocate data for threshold constraints memory block");
    if (error != cudaSuccess) {
        delete th_cons;
        return nullptr;
    }
    cudaMemset(device_memory_block, 0, alloc_size);
    size_t trace = 0;
    th_cons->feature_indices = (int*)device_memory_block;
    trace += sizeof(int) * n_cons;
    th_cons->feature_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_cons;
    th_cons->constraint_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_cons;
    th_cons->is_numerics = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_cons;
    th_cons->inequality_directions = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_cons;
    th_cons->satisfied = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_cons;
    th_cons->categorical_values = (char*)(device_memory_block + trace);

    return th_cons;
}

void deallocate_threshold_constraints_cuda(thresholdConstraints *th_cons){
    if (th_cons != nullptr && th_cons->feature_indices != nullptr)
        cudaFree(th_cons->feature_indices);
    delete th_cons;
}

thresholdConstraints* copy_threshold_constraints_cpu_gpu(const thresholdConstraints *th_cons, const int n_cons){
    thresholdConstraints* new_th_cons = allocate_threshold_constraints_cuda(n_cons); 

    cudaMemcpy(new_th_cons.feature_indices, th_cons->feature_indices, sizeof(int) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons.feature_values, th_cons->feature_values, sizeof(float) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons.constraint_values, th_cons->constraint_values, sizeof(float) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons.is_numerics, th_cons->is_numerics, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons.satisfied, th_cons->satisfied, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons.inequality_directions, th_cons->inequality_directions, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons.categorical_values, th_cons->categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyHostToDevice);

    return new_th_cons;
}

thresholdConstraints* copy_threshold_constraints_gpu_cpu(const thresholdConstraints *th_cons, const int n_cons){

    thresholdConstraints *new_th_cons = allocate_threshold_constraints(n_cons); 
    cudaMemcpy(new_th_cons->feature_indices, th_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_th_cons->feature_values, th_cons.feature_values, sizeof(float) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_th_cons->constraint_values, th_cons.constraint_values, sizeof(float) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_th_cons->is_numerics, th_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_th_cons->satisfied, th_cons.satisfied, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_th_cons->inequality_directions, th_cons.inequality_directions, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_th_cons->categorical_values, th_cons.categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToHost);

    return new_th_cons;
}

thresholdConstraints* copy_threshold_constraints_gpu_gpu(const thresholdConstraints *th_cons, const int n_cons){

    thresholdConstraints *new_th_cons = allocate_threshold_constraints_cuda(n_cons); 
    cudaMemcpy(new_th_cons.feature_indices, th_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_th_cons.feature_values, th_cons.feature_values, sizeof(float) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_th_cons.constraint_values, th_cons.constraint_values, sizeof(float) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_th_cons.is_numerics, th_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_th_cons.satisfied, th_cons.satisfied, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_th_cons.inequality_directions, th_cons.inequality_directions, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_th_cons.categorical_values, th_cons.categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);

    return new_th_cons;
}

void add_threshold_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, bool inequality_direction, bool is_numeric, 
                    float constraint_value){
    thresholdConstraints* new_th_cons = nullptr, *old_th_cons = nullptr;
    int n_th_cons = constraints->n_th_cons + 1;
    new_th_cons = allocate_threshold_constraints_cuda(n_th_cons);
    if (constraints->th_cons != nullptr){
        cudaMemcpy(new_th_cons->feature_indices, th_cons.feature_indices, sizeof(int) * constraints->n_th_cons, cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_th_cons->feature_values, th_cons.feature_values, sizeof(float) * constraints->n_th_cons, cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_th_cons->constraint_values, th_cons.constraint_values, sizeof(float) * constraints->n_th_cons, cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_th_cons->is_numerics, th_cons.is_numerics, sizeof(bool) * constraints->n_th_cons, cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_th_cons->satisfied, th_cons.satisfied, sizeof(bool) * constraints->n_th_cons, cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_th_cons->inequality_directions, th_cons.inequality_directions, sizeof(bool) * constraints->n_th_cons, cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_th_cons->categorical_values, th_cons.categorical_values, sizeof(char) * constraints->n_th_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);
    
        deallocate_threshold_constraints_cuda(constraints->th_cons);
    }
    cudaMemcpy(new_th_cons->feature_indices + constraints->n_th_cons, &feature_idx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons->feature_values + constraints->n_th_cons, &feature_values, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons->constraint_values + constraints->n_th_cons, &constraint_value, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons->is_numerics + constraints->n_th_cons, &is_numeric, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(new_th_cons->inequality_directions + constraints->n_th_cons, &inequality_directions, sizeof(bool), cudaMemcpyHostToDevice);
    bool satisfied = false;
    cudaMemcpy(new_th_cons->satisfied + constraints->n_th_cons, &satisfied, sizeof(bool), cudaMemcpyHostToDevice);
    if (categorical_value != nullptr) 
        cudaMemcpy(new_th_cons->categorical_values + constraints->n_th_cons*MAX_CHAR_SIZE, categorical_value, sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyHostToDevice);

    constraints->th_cons = new_th_cons;
    constraints->n_th_cons = n_th_cons;
}

hierarchyConstraints* allocate_hierarchy_constraints_cuda(const int n_hr_cons, const int n_hr_features){
    cudaError_t error;

    hierarchyConstraints *hr_cons = new hierarchyConstraints;
    constraints->feature_indices = nullptr;
    constraints->dep_count = nullptr;
    constraints->dep_features = nullptr;
    constraints->is_numerics = nullptr;

    char* device_memory_block;
    size_t alloc_size = sizeof(int) * n_hr_cons * 2
                      + sizeof(int) * n_hr_features
                      + sizeof(bool) * n_hr_cons;
    error = allocateCudaMemory((void**)&device_memory_block, alloc_size, "when trying to allocate data for hierarchy constraints memory block");
    if (error != cudaSuccess) {
        cudaFree(device_constraints);
        return nullptr;
    }
    cudaMemset(device_memory_block, 0, alloc_size);

    size_t trace = 0;
    hr_cons->feature_indices = (int*)device_memory_block;
    trace += sizeof(int) * n_hr_cons;
    hr_cons->dep_count = (int*)(device_memory_block + trace);
    trace += sizeof(int) * n_hr_cons;
    hr_cons->dep_features = (int*)(device_memory_block + trace);
    trace += sizeof(int) * n_hr_features;
    hr_cons->is_numerics = (bool*)(device_memory_block + trace);

    return hr_cons;
}

void deallocate_hierarchy_constraints_cuda(hierarchyConstraints *hr_cons){
    if (hr_cons != nullptr && hr_cons->feature_indices != nullptr)
        cudaFree(hr_cons->feature_indices);
    delete hr_cons;
}

hierarchyConstraints* copy_hierarchy_constraints_cpu_gpu(const hierarchyConstraints *hr_const, const int n_hr_cons, const int n_hr_features){

    hierarchyConstraints* new_hr_cons = allocate_hierarchy_constraints_cuda(n_hr_cons, n_hr_features); 

    cudaMemcpy(new_hr_cons.feature_indices, hr_const->feature_indices, sizeof(int) * n_hr_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_hr_cons.dep_count, hr_const->dep_count, sizeof(int) * n_hr_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_hr_cons.dep_features, hr_const->dep_features, sizeof(int) * n_hr_features, cudaMemcpyHostToDevice);
    cudaMemcpy(new_hr_cons.is_numerics, hr_const->is_numerics, sizeof(bool) * n_hr_cons, cudaMemcpyHostToDevice);

    return new_hr_cons;
}

hierarchyConstraints* copy_hierarchy_constraints_gpu_cpu(const hierarchyConstraints *hr_cons, const int n_hr_cons, const int n_hr_features){
    hierarchyConstraints *new_hr_cons = allocate_hierarchy_constraints(n_hr_cons, n_hr_features); 
    cudaMemcpy(new_hr_cons->feature_indices, hr_cons->feature_indices, sizeof(int) * n_hr_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_hr_cons->dep_count, hr_cons->dep_count, sizeof(int) * n_hr_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_hr_cons->dep_features, hr_cons->dep_features, sizeof(int) * n_hr_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_hr_cons->is_numerics, hr_cons->is_numerics, sizeof(bool) * n_hr_cons, cudaMemcpyDeviceToHost);
   
    return new_hr_cons;
}

hierarchyConstraints* copy_hierarchy_constraints_gpu_gpu(const hierarchyConstraints *hr_cons, const int n_hr_cons, const int n_hr_features){

    hierarchyConstraints *new_hr_cons = allocate_hierarchy_constraints_cuda(n_hr_cons, n_hr_features); 
    cudaMemcpy(new_hr_cons->feature_indices, hr_cons->feature_indices, sizeof(int) * n_hr_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_hr_cons->dep_count, hr_cons->dep_count, sizeof(int) * n_hr_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_hr_cons->dep_features, hr_cons->dep_features, sizeof(int) * n_hr_features, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_hr_cons->is_numerics, hr_cons->is_numerics, sizeof(bool) * n_hr_cons, cudaMemcpyDeviceToDevice);

    return new_hr_cons;
}

void add_hierarchy_constraint_cuda(featureConstraints *constraints, int feature_idx,  bool is_numeric, int* dependent_features, int n_features){
    int n_hr_cons = constraints->n_hr_cons + 1;
    int total_n_features = n_features + constraints->n_hr_features;

    hierarchyConstraints *new_hr_cons = allocate_hierarchy_constraints(n_hr_cons, total_n_features);
    if (constraints->hr_cons != nullptr){
        cudaMemcpy(new_hr_cons->feature_indices, constraints->hr_cons->feature_indices, (n_hr_cons-1)*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_hr_cons->dep_count, constraints->hr_cons->dep_count, (n_hr_cons-1)*sizeof(int)), cudaMemcpyDeviceToDevice;
        cudaMemcpy(new_hr_cons->dep_features, constraints->hr_cons->dep_features, constraints->n_hr_features*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_hr_cons->is_numerics, constraints->hr_cons->is_numerics, (n_hr_cons-1)*sizeof(bool), cudaMemcpyDeviceToDevice);
        
        deallocate_hierarchy_constraints_cuda(constraints->hr_cons);
    }

    cudaMemcpy(new_hr_cons->feature_indices + constraints->n_hr_cons, &feature_idx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(new_hr_cons->dep_count + constraints->n_hr_cons, &dep_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(new_hr_cons->dep_features + constraints->n_hr_cons, dependent_features, sizeof(int)*n_features, cudaMemcpyHostToDevice);
    cudaMemcpy(new_hr_cons->is_numeric + constraints->n_hr_cons, &is_numeric, sizeof(bool), cudaMemcpyHostToDevice);

    constraints->n_hr_cons = n_hr_cons;
    constraints->n_hr_features = total_n_features;
}

outputConstraints* allocate_output_constraints_cuda(const int n_out_cons, const int output_dim){
    cudaError_t error;

    outputConstraints *out_cons = new outputConstraints;
    constraints->feature_indices = nullptr;
    constraints->feature_values = nullptr;
    constraints->output_values = nullptr;
    constraints->is_numerics = nullptr;
    constraints->inequality_directions = nullptr;
    constraints->categorical_values = nullptr;

    char* device_memory_block;
    size_t alloc_size = sizeof(int) * n_out_cons
                      + sizeof(float) * n_out_cons * (1 + output_dim)
                      + sizeof(bool) * n_out_cons * 2
                      + sizeof(char) * MAX_CHAR_SIZE * n_out_cons;
    error = allocateCudaMemory((void**)&device_memory_block, alloc_size, "when trying to allocate data for output constraints memory block");
    if (error != cudaSuccess) {
        cudaFree(device_constraints);
        return nullptr;
    }

    cudaMemset(device_memory_block, 0, alloc_size);
    size_t trace = 0;
    out_cons->feature_indices = (int*)device_memory_block;
    trace += sizeof(int) * n_out_cons;
    out_cons->feature_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_out_cons;
    out_cons->output_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_out_cons * output_dim;
    out_cons->is_numerics = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_out_cons;
    out_cons->inequality_directions = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_out_cons;
    out_cons->categorical_values = (char*)(device_memory_block + trace);

    return out_cons;
}

void deallocate_output_constraints_cuda(outputConstraints *out_cons){
    if (out_cons != nullptr && out_cons->feature_indices != nullptr)
        cudaFree(out_cons->feature_indices);
    delete out_cons;
}

outputConstraints* copy_output_constraints_cpu_gpu(const outputConstraints *out_cons, const int n_out_cons, const int output_dim){
    outputConstraints* new_out_cons = allocate_output_constraints_cuda(n_out_cons, output_dim); 

    cudaMemcpy(new_out_cons->feature_indices, out_cons->feature_indices, sizeof(int) * n_out_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->feature_values, out_cons->feature_values, sizeof(float) * n_out_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->output_values, out_cons->output_values, sizeof(float) * n_out_cons * output_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->is_numerics, out_cons->is_numerics, sizeof(bool) * n_out_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->inequality_directions, out_cons->inequality_directions, sizeof(bool) * n_out_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->categorical_values, out_cons->categorical_values, sizeof(char) * n_out_cons * MAX_CHAR_SIZE, cudaMemcpyHostToDevice);

    return new_out_cons;
}

outputConstraints* copy_output_constraints_gpu_cpu(const outputConstraints *out_cons, const int n_out_cons, const int output_dim){

    outputConstraints *new_out_cons = allocate_output_constraints(n_out_cons, output_dim); 
    cudaMemcpy(new_out_cons->feature_indices, out_cons->feature_indices, sizeof(int) * n_out_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_out_cons->feature_values, out_cons->feature_values, sizeof(float) * n_out_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_out_cons->output_values, out_cons->output_values, sizeof(float) * n_out_cons * output_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_out_cons->is_numerics, out_cons->is_numerics, sizeof(bool) * n_out_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_out_cons->inequality_directions, out_cons->inequality_directions, sizeof(bool) * n_out_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_out_cons->categorical_values, out_cons->categorical_values, sizeof(char) * n_out_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToHost);

    return new_out_cons;
}

outputConstraints* copy_output_constraints_gpu_gpu(const outputConstraints *out_cons, const int n_out_cons, const int output_dim){

    outputConstraints *new_out_cons = allocate_output_constraints_cuda(n_out_cons, output_dim); 
    cudaMemcpy(new_out_cons->feature_indices, out_cons->feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_out_cons->feature_values, out_cons->feature_values, sizeof(float) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_out_cons->output_values, out_cons->output_values, sizeof(float) * n_cons * output_dim, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_out_cons->is_numerics, out_cons->is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_out_cons->inequality_directions, out_cons->inequality_directions, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_out_cons->categorical_values, out_cons->categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);

    return new_out_cons;
}

void add_output_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
    const char *categorical_value, bool inequality_direction, bool is_numeric, float* output_values, int output_dim){

    outputConstraints* new_out_cons = nullptr;
    int n_out_cons = constraints->n_out_cons + 1;
    new_out_cons = allocate_output_constraints(n_out_cons, output_dim);
    if (constraints->out_cons != nullptr){
        cudaMemcpy(new_out_cons->feature_indices, constraints->out_cons->feature_indices, (n_out_cons-1)*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_out_cons->inequality_directions, constraints->out_cons->inequality_directions, (n_out_cons-1)*sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_out_cons->is_numerics, constraints->out_cons->is_numerics, (n_out_cons-1)*sizeof(bool), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_out_cons->output_values, constraints->out_cons->output_values, (n_out_cons-1)*sizeof(float) * output_dim, cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_out_cons->feature_values, constraints->out_cons->feature_values, (n_out_cons-1)*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(new_out_cons->categorical_values, constraints->out_cons->categorical_values, (n_out_cons-1)*sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);
        
        deallocate_output_constraints_cuda(constraints->out_cons);
    }
    cudaMemcpy(new_out_cons->feature_indices + constraints->n_out_cons, &feature_idx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->feature_values + constraints->n_out_cons, &feature_values, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->output_values + constraints->n_out_cons*output_dim, output_values, sizeof(float) * output_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->is_numerics + constraints->n_out_cons, &is_numeric, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(new_out_cons->inequality_directions + constraints->n_out_cons, &inequality_directions, sizeof(bool), cudaMemcpyHostToDevice);
    if (categorical_value != nullptr) 
        cudaMemcpy(new_th_cons->categorical_values + constraints->n_th_cons*MAX_CHAR_SIZE, categorical_value, sizeof(char)*MAX_CHAR_SIZE, cudaMemcpyHostToDevice);

    constraints->out_cons = new_out_cons;
    constraints->n_out_cons = n_out_cons;
}

void add_feature_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
    const char * categorical_value, constraintType const_type, int* dependent_features,
    int n_features, float constraint_value, bool inequality_direction,
    bool is_numeric, int output_dim, float *output_values){

    if (const_type == THRESHOLD){
        add_threshold_constraint_cuda(constraints, feature_idx, feature_value, categorical_value, inequality_direction, is_numeric, constraint_value);
    } else if (const_type == HIERARCHY){
        add_hierarchy_constraint_cuda(constraints, feature_idx, is_numeric, dependent_features, n_features);
    } else {
        add_output_constraint_cuda(constraints, feature_idx, feature_value, categorical_value, inequality_direction, is_numeric, output_values, output_dim);
    }
    constraints->n_cons += 1;
}

featureConstraints* copy_feature_constraint_cpu_gpu(const featureConstraints *constraints, const int output_dim){
    featureConstraints* new_constraints = init_constraints();
    if (constraints->th_cons != nullptr)
        new_constraints->th_cons = copy_threshold_constraints_cpu_gpu(constraints->th_cons, constraints->n_th_cons);
    if (constraints->hr_cons != nullptr)
        new_constraints->hr_cons = copy_hierarchy_constraints_cpu_gpu(constraints->hr_cons, constraints->n_hr_cons, constraints->n_hr_features);
    if (constraints->out_cons != nullptr)
        new_constraints->out_cons = copy_output_constraints_cpu_gpu(constraints->out_cons, constraints->n_out_cons, output_dim);

    new_constraints->n_cons = constraints->n_cons;
    new_constraints->n_th_cons = constraints->n_th_cons;
    new_constraints->n_hr_cons = constraints->n_hr_cons;
    new_constraints->n_hr_features = constraints->n_hr_features;
    new_constraints->n_out_cons = constraints->n_out_cons;

    return new_constraints;

}

featureConstraints* copy_feature_constraint_gpu_cpu(const featureConstraints *constraints, const int output_dim){
    featureConstraints* new_constraints = init_constraints();

    if (constraints->th_cons != nullptr)
        new_constraints->th_cons = copy_threshold_constraints_gpu_cpu(constraints->th_cons, constraints->n_th_cons);
    if (constraints->hr_cons != nullptr)
        new_constraints->hr_cons = copy_hierarchy_constraints_gpu_cpu(constraints->hr_cons, constraints->n_hr_cons, constraints->n_hr_features);
    if (constraints->out_cons != nullptr)
        new_constraints->out_cons = copy_output_constraints_gpu_cpu(constraints->out_cons, constraints->n_out_cons, output_dim);

    new_constraints->n_cons = constraints->n_cons;
    new_constraints->n_th_cons = constraints->n_th_cons;
    new_constraints->n_hr_cons = constraints->n_hr_cons;
    new_constraints->n_hr_features = constraints->n_hr_features;
    new_constraints->n_out_cons = constraints->n_out_cons;

    return new_constraints;
}

featureConstraints* copy_feature_constraint_gpu_gpu(const featureConstraints* constraints, const int output_dim){
    if (constraints == nullptr)
        return nullptr;
    featureConstraints* new_constraints = init_constraints();
    if (constraints->th_cons != nullptr)
        new_constraints->th_cons = copy_threshold_constraints_gpu_gpu(constraints->th_cons, constraints->n_th_cons);
    if (constraints->hr_cons != nullptr)
        new_constraints->hr_cons = copy_hierarchy_constraints_gpu_gpu(constraints->hr_cons, constraints->n_hr_cons, constraints->n_hr_features);
    if (constraints->out_cons != nullptr)
        new_constraints->out_cons = copy_output_constraints_gpu_gpu(constraints->out_cons, constraints->n_out_cons, output_dim);
    
    new_constraints->n_cons = constraints->n_cons;
    new_constraints->n_th_cons = constraints->n_th_cons;
    new_constraints->n_hr_cons = constraints->n_hr_cons;
    new_constraints->n_hr_features = constraints->n_hr_features;
    new_constraints->n_out_cons = constraints->n_out_cons;

    return new_constraints;
}

void deallocate_constraints_cuda(featureConstraints* constraints){
    if (constraints == nullptr)
        return;

    if (constraints->th_cons != nullptr)
        deallocate_threshold_constraints_cuda(constraints->th_cons);
    constraints->th_cons = nullptr;

    if (constraints->hr_cons != nullptr)
        deallocate_hierarchy_constraints_cuda(constraints->hr_cons);
    constraints->hr_cons = nullptr;
    if (constraints->out_cons != nullptr)
        deallocate_output_constraints_cuda(constraints->out_cons);
    constraints->out_cons = nullptr;
    
    delete constraints;
}

__global__ void reset_satisfied_kernel(thresholdConstraints *th_cons) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < th_cons->n_cons) {
        th_cons->satisfied[idx] = false;  // Set all values to false (0)
    }
}

__global__ void check_hierarchical_candidates(const n_hr_cons, const TreeNodeGPU* __restrict__ node, const int* __restrict__ dep_count, const int* __restrict__ feature_indices, const int* __restrict__ dep_features, const int* __restrict__ candidate_indices, float* __restrict__ split_scores, const int n_candidates) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cumsum = 0;

    if (idx >= n_candidates) return;

    int candidate_feature = __ldg(candidate_indices + idx);
    int node_depth = __ldg(&node->depth);
    bool found = false;

    for (int i = 0; i < n_hr_cons; ++i){
        if (feature_indices[i] == candidate_feature){
            if (node_depth == 0){
                split_scores[idx] = -INFINITY;
                return;
            }
            for (int k = 0; k < dep_count[i]; ++k){
                int dep_feature = __ldg(dep_features + cumsum + k);
                found = false;
                for (int j = 0; j < node_depth; ++j){
                    int node_feature = __ldg(&node->feature_indices[j]);
                    if (node_feature == dep_feature){
                        found = true;
                        break;
                    }
                }
                if (!found){
                    split_scores[idx] = -INFINITY;
                    return;
                }
            }
        }
        cumsum += dep_count[i];
    } 
}