//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

#ifndef FEATURE_CONSTRAINTS_H
#define FEATURE_CONSTRAINTS_H

#include <cstdint>
#include <string>

#include "types.h"

enum constraintType : uint8_t {
    THRESHOLD, // works on feature weight
    HIERARCHY, // works on feature split order
    OUTPUT, // works on action
};

struct thresholdConstraints {
    int*   feature_indices;
    float* feature_values;
    float* constraint_values;
    bool*  is_numerics;
    bool*  inequality_directions;
    bool*  satisfied;
    char*  categorical_values;
};

struct hierarchyConstraints {
    int*   feature_indices; // indices of constrained features
    int*   dep_count; // number of dependent features per constraint
    int*   dep_features; // dependent features per constraint
    bool*  is_numerics;
};

struct outputConstraints {
    int*   feature_indices;
    float* feature_values;
    float* output_values;
    bool*  is_numerics;
    bool*  inequality_directions;
    char*  categorical_values;
};

struct featureConstraints {
    thresholdConstraints*   th_cons;
    hierarchyConstraints*   hr_cons;
    outputConstraints*      out_cons;
    int n_cons;
    int n_th_cons;
    int n_hr_cons;
    int n_hr_features;
    int n_out_cons;
};


featureConstraints* init_constraints();
featureConstraints* copy_feature_constraint(const featureConstraints* th_cons, const int output_dim);
thresholdConstraints* allocate_threshold_constraints(const int n_th_cons);
void deallocate_threshold_constraints(thresholdConstraints* th_cons);
hierarchyConstraints* allocate_hierarchy_constraints(const int n_hr_cons, const int n_hr_features);
void deallocate_hierarchy_constraints(hierarchyConstraints* hr_cons);
outputConstraints* allocate_output_constraints(const int n_out_cons, int output_dim);
void deallocate_output_constraints(outputConstraints* out_cons);

float get_threshold_score(float crnt_score, thresholdConstraints* th_cons, const splitCandidate &split_candidate, const int n_th_cons, bool *th_cons_satisfied);
void add_feature_constraint(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, constraintType const_type, int* dependent_features,
                    int n_features, float constraint_value, bool inequality_direction, 
                    bool is_numeric, int output_dim, float *output_values);

void add_threshold_constraint(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, bool inequality_direction, bool is_numeric, float constraint_value);
thresholdConstraints* copy_threshold_constraints(const thresholdConstraints* th_cons, const int n_th_cons);
void add_hierarchy_constraint(featureConstraints *constraints, int feature_idx,  bool is_numeric, int* dependent_features, int n_features);  
hierarchyConstraints* copy_hierarchy_constraints(const hierarchyConstraints* hr_cons, const int n_hr_cons, const int n_hr_features);
void add_output_constraint(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, bool inequality_direction, bool is_numeric, float* output_values, int output_dim); 
outputConstraints* copy_output_constraints(const outputConstraints* out_cons, const int n_out_cons, const int output_dim);     
void deallocate_constraints(featureConstraints* constraints);

void print_threshold_constraints(const thresholdConstraints *th_cons, const int n_th_cons);
void print_hierarchy_constraints(const hierarchyConstraints *hr_cons, const int n_hr_cons, const int n_hr_features);
void print_output_constraints(const outputConstraints *out_cons, const int n_out_cons, const int output_dim);

void save_constraints(std::ofstream& file, featureConstraints *constraints, int output_dim);
featureConstraints* load_constraints(std::ifstream& file, int output_dim);

constraintType stringToconstraintType(std::string str);
#endif 