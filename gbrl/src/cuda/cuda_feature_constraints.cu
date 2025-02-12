
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


thresholdConstraints* allocate_threshold_constraints_cuda(int n_constraints){
    cudaError_t error;

    thresholdConstraints constraints;
    constraints.feature_indices = nullptr;
    constraints.feature_values = nullptr;
    constraints.constraint_values = nullptr;
    constraints.is_numerics = nullptr;
    constraints.inequality_directions = nullptr;
    constraints.satisfied = nullptr;
    constraints.categorical_values = nullptr;
    constraints.n_cons = n_constraints;

    thresholdConstraints *device_constraints;
    error = allocateCudaMemory((void**)&device_constraints, sizeof(thresholdConstraints), "when trying to allocate thresholdConstraints");
    if (error != cudaSuccess) {
        return nullptr;
    }

    char* device_memory_block;
    size_t alloc_size = sizeof(int) * n_constraints
                      + sizeof(float) * n_constraints * 2
                      + sizeof(bool) * n_constraints * 3
                      + sizeof(char) * MAX_CHAR_SIZE * n_constraints;
    error = allocateCudaMemory((void**)&device_memory_block, alloc_size, "when trying to allocate data for threshold constraints memory block");
    if (error != cudaSuccess) {
        cudaFree(device_constraints);
        return nullptr;
    }
    cudaMemset(device_memory_block, 0, alloc_size);
    size_t trace = 0;
    constraints.feature_indices = (int*)device_memory_block;
    trace += sizeof(int) * n_constraints;
    constraints.feature_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_constraints;
    constraints.constraint_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_constraints;
    constraints.is_numerics = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_constraints;
    constraints.inequality_directions = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_constraints;
    constraints.satisfied = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_constraints;
    constraints.categorical_values = (char*)(device_memory_block + trace);

    cudaMemcpy(device_constraints, &constraints, sizeof(thresholdConstraints), cudaMemcpyHostToDevice);
    return device_constraints;
}

void deallocate_threshold_constraints_cuda(thresholdConstraints *th_cons){
    thresholdConstraints h_device_th_cons;
    cudaMemcpy(&h_device_th_cons, th_cons, sizeof(thresholdConstraints), cudaMemcpyDeviceToHost);
    if (h_device_th_cons.feature_indices != nullptr)
        cudaFree(h_device_th_cons.feature_indices);
    cudaFree(th_cons);
}

thresholdConstraints* copy_threshold_constraints_cpu_gpu(const thresholdConstraints *original_cons){
    int n_cons = original_cons->n_cons;
    thresholdConstraints* device_th_cons = allocate_threshold_constraints_cuda(n_cons); 
    thresholdConstraints h_device_th_cons;
    cudaMemcpy(&h_device_th_cons, device_th_cons, sizeof(thresholdConstraints), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_device_th_cons.feature_indices, original_cons->feature_indices, sizeof(int) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_th_cons.feature_values, original_cons->feature_values, sizeof(float) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_th_cons.constraint_values, original_cons->constraint_values, sizeof(float) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_th_cons.is_numerics, original_cons->is_numerics, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_th_cons.satisfied, original_cons->satisfied, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_th_cons.inequality_directions, original_cons->inequality_directions, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_th_cons.categorical_values, original_cons->categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyHostToDevice);

    cudaMemcpy(device_th_cons, &h_device_th_cons, sizeof(thresholdConstraints), cudaMemcpyHostToDevice);
    return device_th_cons;
}

thresholdConstraints* copy_threshold_constraints_gpu_cpu(const thresholdConstraints *device_th_cons){
    thresholdConstraints h_device_th_cons;
    cudaMemcpy(&h_device_th_cons, device_th_cons, sizeof(thresholdConstraints), cudaMemcpyDeviceToHost);
    int n_cons = h_device_th_cons.n_cons;

    thresholdConstraints *host_th_cons = allocate_threshold_constraints(n_cons); 
    cudaMemcpy(host_th_cons->feature_indices, h_device_th_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_th_cons->feature_values, h_device_th_cons.feature_values, sizeof(float) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_th_cons->constraint_values, h_device_th_cons.constraint_values, sizeof(float) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_th_cons->is_numerics, h_device_th_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_th_cons->satisfied, h_device_th_cons.satisfied, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_th_cons->inequality_directions, h_device_th_cons.inequality_directions, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_th_cons->categorical_values, h_device_th_cons.categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToHost);

    return host_th_cons;
}

thresholdConstraints* copy_threshold_constraints_gpu_gpu(const thresholdConstraints *device_th_cons){
    thresholdConstraints h_device_th_cons, host_th_cons;
    cudaMemcpy(&h_device_th_cons, device_th_cons, sizeof(thresholdConstraints), cudaMemcpyDeviceToHost);
    int n_cons = h_device_th_cons.n_cons;

    thresholdConstraints *device_th_cons_new = allocate_threshold_constraints_cuda(n_cons); 
    cudaMemcpy(&host_th_cons, device_th_cons_new, sizeof(thresholdConstraints), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_th_cons.feature_indices, h_device_th_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_th_cons.feature_values, h_device_th_cons.feature_values, sizeof(float) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_th_cons.constraint_values, h_device_th_cons.constraint_values, sizeof(float) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_th_cons.is_numerics, h_device_th_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_th_cons.satisfied, h_device_th_cons.satisfied, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_th_cons.inequality_directions, h_device_th_cons.inequality_directions, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_th_cons.categorical_values, h_device_th_cons.categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);

    cudaMemcpy(device_th_cons_new, &host_th_cons, sizeof(thresholdConstraints), cudaMemcpyHostToDevice);

    return device_th_cons_new;
}

void add_threshold_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, bool inequality_direction, bool is_numeric, 
                    float constraint_value){
    thresholdConstraints* host_th_cons = nullptr, *old_host_th_cons = nullptr;
    int n_cons = 1;
    if (constraints->th_cons != nullptr){
        old_host_th_cons = copy_threshold_constraints_gpu_cpu(constraints->th_cons);
        n_cons = old_host_th_cons->n_cons + 1;
    }
    host_th_cons = allocate_threshold_constraints(n_cons);
    if (constraints->th_cons != nullptr){
        memcpy(host_th_cons->feature_indices, old_host_th_cons->feature_indices, (n_cons-1)*sizeof(int));
        memcpy(host_th_cons->inequality_directions, old_host_th_cons->inequality_directions, (n_cons-1)*sizeof(bool));
        memcpy(host_th_cons->is_numerics, old_host_th_cons->is_numerics, (n_cons-1)*sizeof(bool));
        memcpy(host_th_cons->constraint_values, old_host_th_cons->constraint_values, (n_cons-1)*sizeof(float));
        memcpy(host_th_cons->feature_values, old_host_th_cons->feature_values, (n_cons-1)*sizeof(float));
        memcpy(host_th_cons->categorical_values, old_host_th_cons->categorical_values, (n_cons-1)*sizeof(char)*MAX_CHAR_SIZE);
        
        deallocate_threshold_constraints_cuda(constraints->th_cons);
        deallocate_threshold_constraints(old_host_th_cons);
    }
    host_th_cons->feature_indices[n_cons-1] = feature_idx;
    host_th_cons->feature_values[n_cons-1] = feature_value;
    host_th_cons->constraint_values[n_cons-1] = constraint_value;
    host_th_cons->inequality_directions[n_cons-1] = inequality_direction;
    if (categorical_value != nullptr) 
        memcpy(host_th_cons->categorical_values + (n_cons-1)*MAX_CHAR_SIZE, categorical_value, sizeof(char)*MAX_CHAR_SIZE);
    host_th_cons->is_numerics[n_cons-1] = is_numeric;

    constraints->th_cons = copy_threshold_constraints_cpu_gpu(host_th_cons);
    deallocate_threshold_constraints(host_th_cons);
}

hierarchyConstraints* allocate_hierarchy_constraints_cuda(int n_constraints, int n_features){
    cudaError_t error;

    hierarchyConstraints constraints;
    constraints.feature_indices = nullptr;
    constraints.dep_count = nullptr;
    constraints.dep_features = nullptr;
    constraints.is_numerics = nullptr;
    constraints.n_cons = n_constraints;
    constraints.n_features = n_features;

    hierarchyConstraints *device_constraints;
    error = allocateCudaMemory((void**)&device_constraints, sizeof(hierarchyConstraints), "when trying to allocate hierarchyConstraints");
    if (error != cudaSuccess) {
        return nullptr;
    }

    char* device_memory_block;
    size_t alloc_size = sizeof(int) * n_constraints * 2
                      + sizeof(int) * n_features
                      + sizeof(bool) * n_constraints;
    error = allocateCudaMemory((void**)&device_memory_block, alloc_size, "when trying to allocate data for hierarchy constraints memory block");
    if (error != cudaSuccess) {
        cudaFree(device_constraints);
        return nullptr;
    }
    cudaMemset(device_memory_block, 0, alloc_size);

    size_t trace = 0;
    constraints.feature_indices = (int*)device_memory_block;
    trace += sizeof(int) * n_constraints;
    constraints.dep_count = (int*)(device_memory_block + trace);
    trace += sizeof(int) * n_constraints;
    constraints.dep_features = (int*)(device_memory_block + trace);
    trace += sizeof(int) * n_features;
    constraints.is_numerics = (bool*)(device_memory_block + trace);

    cudaMemcpy(device_constraints, &constraints, sizeof(hierarchyConstraints), cudaMemcpyHostToDevice);
    return device_constraints;
}

void deallocate_hierarchy_constraints_cuda(hierarchyConstraints *hr_cons){
    hierarchyConstraints h_device_hr_cons;
    cudaMemcpy(&h_device_hr_cons, hr_cons, sizeof(hierarchyConstraints), cudaMemcpyDeviceToHost);
    if (h_device_hr_cons.feature_indices != nullptr)
        cudaFree(h_device_hr_cons.feature_indices);
    cudaFree(hr_cons);
}


hierarchyConstraints* copy_hierarchy_constraints_cpu_gpu(const hierarchyConstraints *original_cons){
    int n_cons = original_cons->n_cons;
    int n_features = original_cons->n_features;
    hierarchyConstraints* device_hr_cons = allocate_hierarchy_constraints_cuda(n_cons, n_features); 
    hierarchyConstraints h_device_hr_cons;
    cudaMemcpy(&h_device_hr_cons, device_hr_cons, sizeof(hierarchyConstraints), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_device_hr_cons.feature_indices, original_cons->feature_indices, sizeof(int) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_hr_cons.dep_count, original_cons->dep_count, sizeof(int) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_hr_cons.dep_features, original_cons->dep_features, sizeof(int) * n_features, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_hr_cons.is_numerics, original_cons->is_numerics, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);

    cudaMemcpy(device_hr_cons, &h_device_hr_cons, sizeof(hierarchyConstraints), cudaMemcpyHostToDevice);
    return device_hr_cons;
}

hierarchyConstraints* copy_hierarchy_constraints_gpu_cpu(const hierarchyConstraints *device_hr_cons){
    hierarchyConstraints h_device_hr_cons;
    cudaMemcpy(&h_device_hr_cons, device_hr_cons, sizeof(hierarchyConstraints), cudaMemcpyDeviceToHost);
    int n_cons = h_device_hr_cons.n_cons;
    int n_features = h_device_hr_cons.n_features;

    hierarchyConstraints *host_hr_cons = allocate_hierarchy_constraints(n_cons, n_features); 
    cudaMemcpy(host_hr_cons->feature_indices, h_device_hr_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_hr_cons->dep_count, h_device_hr_cons.dep_count, sizeof(int) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_hr_cons->dep_features, h_device_hr_cons.dep_features, sizeof(int) * n_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_hr_cons->is_numerics, h_device_hr_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
   
    return host_hr_cons;
}

hierarchyConstraints* copy_hierarchy_constraints_gpu_gpu(const hierarchyConstraints *device_hr_cons){
    hierarchyConstraints h_device_hr_cons, host_hr_cons;
    cudaMemcpy(&h_device_hr_cons, device_hr_cons, sizeof(hierarchyConstraints), cudaMemcpyDeviceToHost);
    int n_cons = h_device_hr_cons.n_cons;
    int n_features = h_device_hr_cons.n_features;

    hierarchyConstraints *new_device_hr_cons = allocate_hierarchy_constraints_cuda(n_cons, n_features); 
    cudaMemcpy(&host_hr_cons, new_device_hr_cons, sizeof(hierarchyConstraints), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_hr_cons.feature_indices, h_device_hr_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_hr_cons.dep_count, h_device_hr_cons.dep_count, sizeof(int) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_hr_cons.dep_features, h_device_hr_cons.dep_features, sizeof(int) * n_features, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_hr_cons.is_numerics, h_device_hr_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);

    cudaMemcpy(new_device_hr_cons, &host_hr_cons, sizeof(hierarchyConstraints), cudaMemcpyHostToDevice);
   
    return new_device_hr_cons;
}

void add_hierarchy_constraint_cuda(featureConstraints *constraints, int feature_idx,  bool is_numeric, int* dependent_features, int n_features){
    hierarchyConstraints* host_hr_cons = nullptr, *old_host_hr_cons = nullptr;
    int n_cons = 1;
    int total_n_features = n_features, old_n_features = 0;
    if (constraints->hr_cons != nullptr){
        old_host_hr_cons = copy_hierarchy_constraints_gpu_cpu(constraints->hr_cons);
        n_cons = old_host_hr_cons->n_cons + 1;
        old_n_features = old_host_hr_cons->n_features;
    }
    total_n_features += old_n_features;
    host_hr_cons = allocate_hierarchy_constraints(n_cons, total_n_features);
    if (constraints->hr_cons != nullptr){
        memcpy(host_hr_cons->feature_indices, old_host_hr_cons->feature_indices, (n_cons-1)*sizeof(int));
        memcpy(host_hr_cons->dep_count, old_host_hr_cons->dep_count, (n_cons-1)*sizeof(int));
        memcpy(host_hr_cons->dep_features, old_host_hr_cons->dep_features, old_n_features*sizeof(int));
        memcpy(host_hr_cons->is_numerics, old_host_hr_cons->is_numerics, (n_cons-1)*sizeof(bool));
        
        deallocate_hierarchy_constraints_cuda(constraints->hr_cons);
        deallocate_hierarchy_constraints(old_host_hr_cons);
    }
    host_hr_cons->feature_indices[n_cons-1] = feature_idx;
    host_hr_cons->dep_count[n_cons-1] = n_features;
    memcpy(host_hr_cons->dep_features + old_n_features, dependent_features, sizeof(int)*n_features);
    host_hr_cons->is_numerics[n_cons-1] = is_numeric;

    constraints->hr_cons = copy_hierarchy_constraints_cpu_gpu(host_hr_cons);
    deallocate_hierarchy_constraints(host_hr_cons);
}

outputConstraints* allocate_output_constraints_cuda(int n_constraints, int output_dim){
    cudaError_t error;

    outputConstraints constraints;
    constraints.feature_indices = nullptr;
    constraints.feature_values = nullptr;
    constraints.output_values = nullptr;
    constraints.is_numerics = nullptr;
    constraints.inequality_directions = nullptr;
    constraints.categorical_values = nullptr;
    constraints.n_cons = n_constraints;

    outputConstraints *device_constraints;
    error = allocateCudaMemory((void**)&device_constraints, sizeof(outputConstraints), "when trying to allocate outputConstraints");
    if (error != cudaSuccess) {
        return nullptr;
    }

    char* device_memory_block;
    size_t alloc_size = sizeof(int) * n_constraints
                      + sizeof(float) * n_constraints * (1 + output_dim)
                      + sizeof(bool) * n_constraints * 2
                      + sizeof(char) * MAX_CHAR_SIZE * n_constraints;
    error = allocateCudaMemory((void**)&device_memory_block, alloc_size, "when trying to allocate data for output constraints memory block");
    if (error != cudaSuccess) {
        cudaFree(device_constraints);
        return nullptr;
    }

    cudaMemset(device_memory_block, 0, alloc_size);
    size_t trace = 0;
    constraints.feature_indices = (int*)device_memory_block;
    trace += sizeof(int) * n_constraints;
    constraints.feature_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_constraints;
    constraints.output_values = (float*)(device_memory_block + trace);
    trace += sizeof(float) * n_constraints * output_dim;
    constraints.is_numerics = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_constraints;
    constraints.inequality_directions = (bool*)(device_memory_block + trace);
    trace += sizeof(bool) * n_constraints;
    constraints.categorical_values = (char*)(device_memory_block + trace);

    cudaMemcpy(device_constraints, &constraints, sizeof(outputConstraints), cudaMemcpyHostToDevice);
    return device_constraints;
}

void deallocate_output_constraints_cuda(outputConstraints *out_cons){
    outputConstraints h_device_out_cons;
    cudaMemcpy(&h_device_out_cons, out_cons, sizeof(outputConstraints), cudaMemcpyDeviceToHost);
    if (h_device_out_cons.feature_indices != nullptr)
        cudaFree(h_device_out_cons.feature_indices);
    cudaFree(out_cons);
}

outputConstraints* copy_output_constraints_cpu_gpu(const outputConstraints *original_cons, const int output_dim){
    int n_cons = original_cons->n_cons;
    outputConstraints* device_out_cons = allocate_output_constraints_cuda(n_cons, output_dim); 
    outputConstraints h_device_out_cons;
    cudaMemcpy(&h_device_out_cons, device_out_cons, sizeof(outputConstraints), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_device_out_cons.feature_indices, original_cons->feature_indices, sizeof(int) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_out_cons.feature_values, original_cons->feature_values, sizeof(float) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_out_cons.output_values, original_cons->output_values, sizeof(float) * n_cons * output_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_out_cons.is_numerics, original_cons->is_numerics, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_out_cons.inequality_directions, original_cons->inequality_directions, sizeof(bool) * n_cons, cudaMemcpyHostToDevice);
    cudaMemcpy(h_device_out_cons.categorical_values, original_cons->categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyHostToDevice);

    cudaMemcpy(device_out_cons, &h_device_out_cons, sizeof(outputConstraints), cudaMemcpyHostToDevice);
    return device_out_cons;
}

outputConstraints* copy_output_constraints_gpu_cpu(const outputConstraints *device_out_cons, const int output_dim){
    outputConstraints h_device_out_cons;
    cudaMemcpy(&h_device_out_cons, device_out_cons, sizeof(outputConstraints), cudaMemcpyDeviceToHost);
    int n_cons = h_device_out_cons.n_cons;

    outputConstraints *host_out_cons = allocate_output_constraints(n_cons, output_dim); 
    cudaMemcpy(host_out_cons->feature_indices, h_device_out_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_cons->feature_values, h_device_out_cons.feature_values, sizeof(float) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_cons->output_values, h_device_out_cons.output_values, sizeof(float) * n_cons * output_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_cons->is_numerics, h_device_out_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_cons->inequality_directions, h_device_out_cons.inequality_directions, sizeof(bool) * n_cons, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_cons->categorical_values, h_device_out_cons.categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToHost);

    return host_out_cons;
}

outputConstraints* copy_output_constraints_gpu_gpu(const outputConstraints *device_out_cons, const int output_dim){
    outputConstraints h_device_out_cons, host_out_cons;
    cudaMemcpy(&h_device_out_cons, device_out_cons, sizeof(outputConstraints), cudaMemcpyDeviceToHost);
    int n_cons = h_device_out_cons.n_cons;

    outputConstraints *new_device_out_cons = allocate_output_constraints_cuda(n_cons, output_dim); 
    cudaMemcpy(&host_out_cons, new_device_out_cons, sizeof(outputConstraints), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_out_cons.feature_indices, h_device_out_cons.feature_indices, sizeof(int) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_out_cons.feature_values, h_device_out_cons.feature_values, sizeof(float) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_out_cons.output_values, h_device_out_cons.output_values, sizeof(float) * n_cons * output_dim, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_out_cons.is_numerics, h_device_out_cons.is_numerics, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_out_cons.inequality_directions, h_device_out_cons.inequality_directions, sizeof(bool) * n_cons, cudaMemcpyDeviceToDevice);
    cudaMemcpy(host_out_cons.categorical_values, h_device_out_cons.categorical_values, sizeof(char) * n_cons * MAX_CHAR_SIZE, cudaMemcpyDeviceToDevice);

    cudaMemcpy(new_device_out_cons, &host_out_cons, sizeof(outputConstraints), cudaMemcpyHostToDevice);
    return new_device_out_cons;
}

void add_output_constraint_cuda(featureConstraints *constraints, int feature_idx, float feature_value,
    const char *categorical_value, bool inequality_direction, bool is_numeric, float* output_values, int output_dim){

    outputConstraints* host_out_cons = nullptr, *old_host_out_cons = nullptr;
    int n_cons = 1;
    if (constraints->out_cons != nullptr){
        old_host_out_cons = copy_output_constraints_gpu_cpu(constraints->out_cons, output_dim);
        n_cons = old_host_out_cons->n_cons + 1;
    }
    host_out_cons = allocate_output_constraints(n_cons, output_dim);
    if (constraints->out_cons != nullptr){
        memcpy(host_out_cons->feature_indices, old_host_out_cons->feature_indices, (n_cons-1)*sizeof(int));
        memcpy(host_out_cons->inequality_directions, old_host_out_cons->inequality_directions, (n_cons-1)*sizeof(bool));
        memcpy(host_out_cons->is_numerics, old_host_out_cons->is_numerics, (n_cons-1)*sizeof(bool));
        memcpy(host_out_cons->output_values, old_host_out_cons->output_values, (n_cons-1)*sizeof(float) * output_dim);
        memcpy(host_out_cons->feature_values, old_host_out_cons->feature_values, (n_cons-1)*sizeof(float));
        memcpy(host_out_cons->categorical_values, old_host_out_cons->categorical_values, (n_cons-1)*sizeof(char)*MAX_CHAR_SIZE);
        
        deallocate_output_constraints_cuda(constraints->out_cons);
        deallocate_output_constraints(old_host_out_cons);
    }
    host_out_cons->feature_indices[n_cons-1] = feature_idx;
    host_out_cons->feature_values[n_cons-1] = feature_value;
    memcpy(host_out_cons->output_values + (n_cons-1)*output_dim, output_values, sizeof(float)*output_dim);
    host_out_cons->inequality_directions[n_cons-1] = inequality_direction;
    if (categorical_value != nullptr) 
        memcpy(host_out_cons->categorical_values + (n_cons-1)*MAX_CHAR_SIZE, categorical_value, sizeof(char)*MAX_CHAR_SIZE);
    host_out_cons->is_numerics[n_cons-1] = is_numeric;

    constraints->out_cons = copy_output_constraints_cpu_gpu(host_out_cons, output_dim);
    deallocate_output_constraints(host_out_cons);
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
    featureConstraints* host_constraints = init_constraints();
    if (constraints->th_cons != nullptr)
        host_constraints->th_cons = copy_threshold_constraints_cpu_gpu(constraints->th_cons);
    if (constraints->hr_cons != nullptr)
        host_constraints->hr_cons = copy_hierarchy_constraints_cpu_gpu(constraints->hr_cons);
    if (constraints->out_cons != nullptr)
        host_constraints->out_cons = copy_output_constraints_cpu_gpu(constraints->out_cons, output_dim);

    host_constraints->n_cons = constraints->n_cons;

    return host_constraints;

}

featureConstraints* copy_feature_constraint_gpu_cpu(const featureConstraints *constraints, const int output_dim){
    featureConstraints* host_constraints = init_constraints();

    if (constraints->th_cons != nullptr)
        host_constraints->th_cons = copy_threshold_constraints_gpu_cpu(constraints->th_cons);
    if (constraints->hr_cons != nullptr)
        host_constraints->hr_cons = copy_hierarchy_constraints_gpu_cpu(constraints->hr_cons);
    if (constraints->out_cons != nullptr)
        host_constraints->out_cons = copy_output_constraints_gpu_cpu(constraints->out_cons, output_dim);

    host_constraints->n_cons = constraints->n_cons;

    return host_constraints;
}

featureConstraints* copy_feature_constraint_gpu_gpu(const featureConstraints* constraints, const int output_dim){
    if (constraints == nullptr)
        return nullptr;
    featureConstraints* new_cons = init_constraints();
    if (constraints->th_cons != nullptr)
        new_cons->th_cons = copy_threshold_constraints_gpu_gpu(constraints->th_cons);
    if (constraints->hr_cons != nullptr)
        new_cons->hr_cons = copy_hierarchy_constraints_gpu_gpu(constraints->hr_cons);
    if (constraints->out_cons != nullptr)
        new_cons->out_cons = copy_output_constraints_gpu_gpu(constraints->out_cons, output_dim);
    new_cons->n_cons = constraints->n_cons;

    return new_cons;
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