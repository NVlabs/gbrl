//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>

#include "feature_constraints.h"
#include "types.h"


featureConstraints* init_constraints(){
    featureConstraints *constraints = new featureConstraints;
    constraints->th_cons = nullptr;
    constraints->hr_cons = nullptr;
    constraints->out_cons = nullptr;
    constraints->n_cons = 0;
    constraints->n_th_cons = 0;
    constraints->n_hr_cons = 0;
    constraints->n_hr_features = 0;
    constraints->n_out_cons = 0;

    return constraints;
}

featureConstraints* copy_feature_constraint(const featureConstraints* constraints, const int output_dim){
    if (constraints == nullptr)
        return nullptr;
    featureConstraints* new_cons = init_constraints();
    if (constraints->th_cons != nullptr)
        new_cons->th_cons = copy_threshold_constraints(constraints->th_cons, constraints->n_th_cons);
    if (constraints->hr_cons != nullptr)
        new_cons->hr_cons = copy_hierarchy_constraints(constraints->hr_cons, constraints->n_hr_cons, constraints->n_hr_features);
    if (constraints->out_cons != nullptr)
        new_cons->out_cons = copy_output_constraints(constraints->out_cons, constraints->n_out_cons, output_dim);
    new_cons->n_cons = constraints->n_cons;
    new_cons->n_th_cons = constraints->n_th_cons;
    new_cons->n_hr_cons = constraints->n_hr_cons;
    new_cons->n_hr_features = constraints->n_hr_features;
    new_cons->n_out_cons = constraints->n_out_cons;

    return new_cons;
}

thresholdConstraints* allocate_threshold_constraints(const int n_th_cons){
    thresholdConstraints *th_cons = new thresholdConstraints;
    th_cons->feature_indices = new int[n_th_cons];
    th_cons->feature_values = new float[n_th_cons];
    th_cons->constraint_values = new float[n_th_cons];
    th_cons->is_numerics = new bool[n_th_cons];
    th_cons->inequality_directions = new bool[n_th_cons];
    th_cons->categorical_values = new char[n_th_cons*MAX_CHAR_SIZE];

    return th_cons;
}

void deallocate_threshold_constraints(thresholdConstraints* th_cons){
    delete[] th_cons->feature_indices;
    delete[] th_cons->feature_values;
    delete[] th_cons->constraint_values;
    delete[] th_cons->is_numerics;
    delete[] th_cons->inequality_directions;
    delete[] th_cons->categorical_values;
    delete th_cons;
}

hierarchyConstraints* allocate_hierarchy_constraints(const int n_hr_cons, const int n_hr_features){
    hierarchyConstraints *hr_cons = new hierarchyConstraints;
    hr_cons->feature_indices = new int[n_hr_cons];
    hr_cons->dep_count = new int[n_hr_cons];
    hr_cons->dep_features = new int[n_hr_features];
    hr_cons->is_numerics = new bool[n_hr_cons];

    return hr_cons;
}

float get_threshold_score(float crnt_score, thresholdConstraints* th_cons, splitCandidate &split_candidate, const int n_th_cons, bool *th_cons_satisfied){
    if (!th_cons) return crnt_score;
    bool numeric_cand = split_candidate.categorical_value == nullptr;
    for (int i = 0; i < n_th_cons; ++i){
        if (th_cons_satisfied[i] || split_candidate.feature_idx != th_cons->feature_indices[i] || th_cons->is_numerics[i] != numeric_cand) 
            continue; // Skip already satisfied constraints and irrelevant indices

        bool numerical_cond = th_cons->is_numerics[i] && 
                                (th_cons->inequality_directions[i] == (split_candidate.feature_value > th_cons->feature_values[i]));
        bool categorical_cond = !th_cons->is_numerics[i] && (th_cons->inequality_directions[i] == (strcmp(th_cons->categorical_values + i* MAX_CHAR_SIZE, split_candidate.categorical_value) == 0));
        if (numerical_cond || categorical_cond){
            crnt_score *= th_cons->constraint_values[i];
        }
    }
    return crnt_score;
}

void deallocate_hierarchy_constraints(hierarchyConstraints* hr_cons){
    delete[] hr_cons->feature_indices;
    delete[] hr_cons->dep_count;
    delete[] hr_cons->dep_features;
    delete[] hr_cons->is_numerics;
    delete hr_cons;
}

outputConstraints* allocate_output_constraints(const int n_out_cons, const int output_dim){
    outputConstraints *out_cons = new outputConstraints;
    out_cons->feature_indices = new int[n_out_cons];
    out_cons->feature_values = new float[n_out_cons];
    out_cons->output_values = new float[n_out_cons*output_dim];
    out_cons->is_numerics = new bool[n_out_cons];
    out_cons->inequality_directions = new bool[n_out_cons];
    out_cons->categorical_values = new char[n_out_cons*MAX_CHAR_SIZE];

    return out_cons;
}

void deallocate_output_constraints(outputConstraints* out_cons){
    delete[] out_cons->feature_indices;
    delete[] out_cons->feature_values;
    delete[] out_cons->output_values;
    delete[] out_cons->is_numerics;
    delete[] out_cons->inequality_directions;
    delete[] out_cons->categorical_values;
    delete out_cons;
}

void add_feature_constraint(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char * categorical_value, constraintType const_type, int* dependent_features,
                    int n_features, float constraint_value, bool inequality_direction,
                    bool is_numeric, int output_dim, float *output_values){
    
    if (const_type == THRESHOLD){
        add_threshold_constraint(constraints, feature_idx, feature_value, categorical_value, inequality_direction, is_numeric, constraint_value);
    } else if (const_type == HIERARCHY){
        add_hierarchy_constraint(constraints, feature_idx, is_numeric, dependent_features, n_features);
    } else {
        add_output_constraint(constraints, feature_idx, feature_value, categorical_value, inequality_direction, is_numeric, output_values, output_dim);
    }
    constraints->n_cons += 1;
}


thresholdConstraints* copy_threshold_constraints(const thresholdConstraints* th_cons, const int n_th_cons){
    if (th_cons == nullptr)
        return nullptr;

    thresholdConstraints *new_th_cons = allocate_threshold_constraints(n_th_cons);
    memcpy(new_th_cons->feature_indices, th_cons->feature_indices, n_th_cons*sizeof(int));
    memcpy(new_th_cons->inequality_directions, th_cons->inequality_directions, n_th_cons*sizeof(bool));
    memcpy(new_th_cons->is_numerics, th_cons->is_numerics,n_th_cons*sizeof(bool));
    memcpy(new_th_cons->constraint_values, th_cons->constraint_values, n_th_cons*sizeof(float));
    memcpy(new_th_cons->feature_values, th_cons->feature_values, n_th_cons*sizeof(float));
    memcpy(new_th_cons->categorical_values, th_cons->categorical_values, n_th_cons*sizeof(char)*MAX_CHAR_SIZE);

    return new_th_cons;
}

void add_threshold_constraint(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, bool inequality_direction, bool is_numeric, 
                    float constraint_value){

    int n_th_cons = constraints->n_th_cons + 1;
    thresholdConstraints *new_th_cons = allocate_threshold_constraints(n_th_cons);
    if (constraints->th_cons != nullptr){ 
        memcpy(new_th_cons->feature_indices, constraints->th_cons->feature_indices, (n_th_cons-1)*sizeof(int));
        memcpy(new_th_cons->inequality_directions, constraints->th_cons->inequality_directions, (n_th_cons-1)*sizeof(bool));
        memcpy(new_th_cons->is_numerics, constraints->th_cons->is_numerics, (n_th_cons-1)*sizeof(bool));
        memcpy(new_th_cons->constraint_values, constraints->th_cons->constraint_values, (n_th_cons-1)*sizeof(float));
        memcpy(new_th_cons->feature_values, constraints->th_cons->feature_values, (n_th_cons-1)*sizeof(float));
        memcpy(new_th_cons->categorical_values, constraints->th_cons->categorical_values, (n_th_cons-1)*sizeof(char)*MAX_CHAR_SIZE);
        
        deallocate_threshold_constraints(constraints->th_cons);
    }
    new_th_cons->feature_indices[n_th_cons-1] = feature_idx;
    new_th_cons->feature_values[n_th_cons-1] = feature_value;
    new_th_cons->constraint_values[n_th_cons-1] = constraint_value;
    new_th_cons->inequality_directions[n_th_cons-1] = inequality_direction;
    if (categorical_value != nullptr) 
        memcpy(new_th_cons->categorical_values + (n_th_cons-1)*MAX_CHAR_SIZE, categorical_value, sizeof(char)*MAX_CHAR_SIZE);
    new_th_cons->is_numerics[n_th_cons-1] = is_numeric;
    constraints->th_cons = new_th_cons;
    constraints->n_th_cons = n_th_cons;
}

void add_hierarchy_constraint(featureConstraints *constraints, int feature_idx,  bool is_numeric, int* dependent_features, int n_features){
    
    int n_hr_cons = constraints->n_hr_cons + 1;
    int old_n_features = constraints->n_hr_features;
    int total_n_features = old_n_features + n_features;
    hierarchyConstraints *new_hr_cons = allocate_hierarchy_constraints(n_hr_cons, total_n_features);
    if (constraints->hr_cons != nullptr){ 
        memcpy(new_hr_cons->feature_indices, constraints->hr_cons->feature_indices, (n_hr_cons-1)*sizeof(int));
        memcpy(new_hr_cons->dep_count, constraints->hr_cons->dep_count, (n_hr_cons-1)*sizeof(int));
        memcpy(new_hr_cons->dep_features, constraints->hr_cons->dep_features, old_n_features*sizeof(int));
        memcpy(new_hr_cons->is_numerics, constraints->hr_cons->is_numerics, (n_hr_cons-1)*sizeof(bool));
        
        deallocate_hierarchy_constraints(constraints->hr_cons);
    }
    new_hr_cons->feature_indices[n_hr_cons-1] = feature_idx;
    new_hr_cons->dep_count[n_hr_cons-1] = n_features;
    new_hr_cons->is_numerics[n_hr_cons-1] = is_numeric;
    memcpy(new_hr_cons->dep_features + old_n_features, dependent_features, sizeof(int)*n_features);
    
    constraints->hr_cons = new_hr_cons;
    constraints->n_hr_cons = n_hr_cons;
    constraints->n_hr_features = total_n_features;
}

hierarchyConstraints* copy_hierarchy_constraints(const hierarchyConstraints* hr_cons, const int n_hr_cons, const int n_hr_features){
    if (hr_cons == nullptr)
        return nullptr;

    hierarchyConstraints *new_hr_cons = allocate_hierarchy_constraints(n_hr_cons, n_hr_features);
    memcpy(new_hr_cons->feature_indices, hr_cons->feature_indices, n_hr_cons*sizeof(int));
    memcpy(new_hr_cons->dep_count, hr_cons->dep_count, n_hr_cons*sizeof(int));
    memcpy(new_hr_cons->is_numerics, hr_cons->is_numerics,n_hr_cons*sizeof(bool));
    memcpy(new_hr_cons->dep_features, hr_cons->dep_features, n_hr_features*sizeof(int));

    return new_hr_cons;
}

void add_output_constraint(featureConstraints *constraints, int feature_idx, float feature_value,
                    const char *categorical_value, bool inequality_direction, bool is_numeric, float* output_values, int output_dim){
    
    int n_out_cons = constraints->n_out_cons + 1;
    outputConstraints *new_out_cons = allocate_output_constraints(n_out_cons, output_dim);
    if (constraints->out_cons != nullptr){ 
        memcpy(new_out_cons->feature_indices, constraints->out_cons->feature_indices, (n_out_cons-1)*sizeof(int));
        memcpy(new_out_cons->inequality_directions, constraints->out_cons->inequality_directions, (n_out_cons-1)*sizeof(bool));
        memcpy(new_out_cons->is_numerics, constraints->out_cons->is_numerics, (n_out_cons-1)*sizeof(bool));
        memcpy(new_out_cons->output_values, constraints->out_cons->output_values, (n_out_cons-1)*sizeof(float)*output_dim);
        memcpy(new_out_cons->feature_values, constraints->out_cons->feature_values, (n_out_cons-1)*sizeof(float));
        memcpy(new_out_cons->categorical_values, constraints->out_cons->categorical_values, (n_out_cons-1)*sizeof(char)*MAX_CHAR_SIZE);
        
        deallocate_output_constraints(constraints->out_cons);
    }
    new_out_cons->feature_indices[n_out_cons-1] = feature_idx;
    new_out_cons->feature_values[n_out_cons-1] = feature_value;
    memcpy(new_out_cons->output_values + (n_out_cons-1)*output_dim, output_values, sizeof(float)*output_dim);
    new_out_cons->inequality_directions[n_out_cons-1] = inequality_direction;
    if (categorical_value != nullptr)
        memcpy(new_out_cons->categorical_values + (n_out_cons-1)*MAX_CHAR_SIZE, categorical_value, sizeof(char)*MAX_CHAR_SIZE);
    new_out_cons->is_numerics[n_out_cons-1] = is_numeric;
    constraints->out_cons = new_out_cons;
    constraints->n_out_cons = n_out_cons;
}

outputConstraints* copy_output_constraints(const outputConstraints* out_cons, const int n_out_cons, const int output_dim){
    if (out_cons == nullptr)
        return nullptr;

    outputConstraints *new_out_cons = allocate_output_constraints(n_out_cons, output_dim);
    memcpy(new_out_cons->feature_indices, out_cons->feature_indices, n_out_cons*sizeof(int));
    memcpy(new_out_cons->inequality_directions, out_cons->inequality_directions, n_out_cons*sizeof(bool));
    memcpy(new_out_cons->is_numerics, out_cons->is_numerics,n_out_cons*sizeof(bool));
    memcpy(new_out_cons->output_values, out_cons->output_values, n_out_cons*sizeof(float)*output_dim);
    memcpy(new_out_cons->feature_values, out_cons->feature_values, n_out_cons*sizeof(float));
    memcpy(new_out_cons->categorical_values, out_cons->categorical_values, n_out_cons*sizeof(char)*MAX_CHAR_SIZE);

    return new_out_cons;
}

void deallocate_constraints(featureConstraints* constraints){
    if (constraints == nullptr)
        return;
    if (constraints->th_cons != nullptr)
        deallocate_threshold_constraints(constraints->th_cons);
    constraints->th_cons = nullptr;

    if (constraints->hr_cons != nullptr)
        deallocate_hierarchy_constraints(constraints->hr_cons);
    constraints->hr_cons = nullptr;
    if (constraints->out_cons != nullptr)
        deallocate_output_constraints(constraints->out_cons);
    constraints->out_cons = nullptr;
    
    delete constraints;
}

void print_threshold_constraints(const thresholdConstraints * th_cons, const int n_th_cons){
    std::cout << "#### " << n_th_cons  << " threshold constraints #######" << std::endl;
    for (int i = 0; i < n_th_cons; ++i){
        std::cout << "constraint: " << i + 1 << " - mapped feature_idx: " << th_cons->feature_indices[i];
        std::cout << " constraint value: " << th_cons->constraint_values[i];
        if (th_cons->is_numerics[i]){
            std::cout << " numerical constraint - feature_value";
            if (th_cons->inequality_directions[i])
                std::cout << " > ";
            else 
                std::cout << " <= ";
            std::cout << th_cons->feature_values[i] << std::endl;
        } else {
            std::cout << " categorcal constraint - feature_value";
            if (th_cons->inequality_directions[i])
                std::cout << " == ";
            else 
                std::cout << " != ";
            for (int j = 0; j < MAX_CHAR_SIZE; ++j)
                std::cout << th_cons->categorical_values[i*MAX_CHAR_SIZE + j];
            std::cout << std::endl;
        }

    }
}

void save_constraints(std::ofstream& file, featureConstraints *constraints, int output_dim){
    NULL_CHECK check = constraints->th_cons != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (constraints->th_cons != nullptr){
        int n_th_cons = constraints->n_th_cons;
        file.write(reinterpret_cast<char*>(&n_th_cons), sizeof(int));
        file.write(reinterpret_cast<char*>(constraints->th_cons->feature_indices), n_th_cons * sizeof(int));
        file.write(reinterpret_cast<char*>(constraints->th_cons->feature_values), n_th_cons * sizeof(float));
        file.write(reinterpret_cast<char*>(constraints->th_cons->constraint_values), n_th_cons * sizeof(float));
        file.write(reinterpret_cast<char*>(constraints->th_cons->is_numerics), n_th_cons * sizeof(bool));
        file.write(reinterpret_cast<char*>(constraints->th_cons->inequality_directions), n_th_cons * sizeof(bool));
        file.write(reinterpret_cast<char*>(constraints->th_cons->categorical_values), n_th_cons * sizeof(char) * MAX_CHAR_SIZE);
    }
    check = constraints->hr_cons != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (constraints->hr_cons != nullptr){
        int n_hr_cons = constraints->n_hr_cons;
        int n_hr_features = constraints->n_hr_features;
        file.write(reinterpret_cast<char*>(&n_hr_cons), sizeof(int));
        file.write(reinterpret_cast<char*>(&n_hr_features), sizeof(int));
        file.write(reinterpret_cast<char*>(constraints->hr_cons->feature_indices), n_hr_cons * sizeof(int));
        file.write(reinterpret_cast<char*>(constraints->hr_cons->dep_count), n_hr_cons * sizeof(int));
        file.write(reinterpret_cast<char*>(constraints->hr_cons->dep_features), n_hr_features * sizeof(float));
        file.write(reinterpret_cast<char*>(constraints->hr_cons->is_numerics), n_hr_cons * sizeof(bool));
    }
    check = constraints->out_cons != nullptr ? VALID : NULL_OPT;
    file.write(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (constraints->out_cons != nullptr){
        int n_out_cons = constraints->n_out_cons;
        file.write(reinterpret_cast<char*>(&n_out_cons), sizeof(int));
        file.write(reinterpret_cast<char*>(constraints->out_cons->feature_indices), n_out_cons * sizeof(int));
        file.write(reinterpret_cast<char*>(constraints->out_cons->feature_values), n_out_cons * sizeof(float));
        file.write(reinterpret_cast<char*>(constraints->out_cons->output_values), n_out_cons * sizeof(float) * output_dim);
        file.write(reinterpret_cast<char*>(constraints->out_cons->is_numerics), n_out_cons * sizeof(bool));
        file.write(reinterpret_cast<char*>(constraints->out_cons->inequality_directions), n_out_cons * sizeof(bool));
        file.write(reinterpret_cast<char*>(constraints->out_cons->categorical_values), n_out_cons * sizeof(char) * MAX_CHAR_SIZE);
    }
    int n_cons = constraints->n_cons;
    file.write(reinterpret_cast<char*>(&n_cons), sizeof(int));
}

featureConstraints* load_constraints(std::ifstream& file, int output_dim){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        throw std::runtime_error("Error opening file");
    }
    featureConstraints *constraints = init_constraints();
    thresholdConstraints *th_cons = nullptr;
    hierarchyConstraints *hr_cons = nullptr;
    outputConstraints *out_cons = nullptr;
    NULL_CHECK check;
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        int n_th_cons;
        file.read(reinterpret_cast<char*>(&n_th_cons), sizeof(int));
        constraints->n_th_cons = n_th_cons;
        th_cons = allocate_threshold_constraints(n_th_cons);
        file.read(reinterpret_cast<char*>(th_cons->feature_indices), n_th_cons * sizeof(int));
        file.read(reinterpret_cast<char*>(th_cons->feature_values), n_th_cons * sizeof(float));
        file.read(reinterpret_cast<char*>(th_cons->constraint_values), n_th_cons * sizeof(float));
        file.read(reinterpret_cast<char*>(th_cons->is_numerics), n_th_cons * sizeof(bool));
        file.read(reinterpret_cast<char*>(th_cons->inequality_directions), n_th_cons * sizeof(bool));
        file.read(reinterpret_cast<char*>(th_cons->categorical_values), n_th_cons * sizeof(char) * MAX_CHAR_SIZE);
    } 
    
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        int n_hr_cons;
        file.read(reinterpret_cast<char*>(&n_hr_cons), sizeof(int));
        constraints->n_hr_cons = n_hr_cons;
        int n_hr_features;
        file.read(reinterpret_cast<char*>(&n_hr_features), sizeof(int));
        constraints->n_hr_features = n_hr_features;
        hr_cons = allocate_hierarchy_constraints(n_hr_cons, n_hr_features);
        file.read(reinterpret_cast<char*>(hr_cons->feature_indices), n_hr_cons * sizeof(int));
        file.read(reinterpret_cast<char*>(hr_cons->dep_count), n_hr_cons * sizeof(int));
        file.read(reinterpret_cast<char*>(hr_cons->dep_features), n_hr_features * sizeof(float));
        file.read(reinterpret_cast<char*>(hr_cons->is_numerics), n_hr_cons * sizeof(bool));
    } 
    file.read(reinterpret_cast<char*>(&check), sizeof(NULL_CHECK));
    if (check == VALID) {
        int n_out_cons;
        file.read(reinterpret_cast<char*>(&n_out_cons), sizeof(int));
        constraints->n_out_cons = n_out_cons;
        out_cons = allocate_output_constraints(n_out_cons, output_dim);
        file.read(reinterpret_cast<char*>(out_cons->feature_indices), n_out_cons * sizeof(int));
        file.read(reinterpret_cast<char*>(out_cons->feature_values), n_out_cons * sizeof(float));
        file.read(reinterpret_cast<char*>(out_cons->output_values), n_out_cons * sizeof(float) * output_dim);
        file.read(reinterpret_cast<char*>(out_cons->is_numerics), n_out_cons * sizeof(bool));
        file.read(reinterpret_cast<char*>(out_cons->inequality_directions), n_out_cons * sizeof(bool));
        file.read(reinterpret_cast<char*>(out_cons->categorical_values), n_out_cons * sizeof(char) * MAX_CHAR_SIZE);
    } 

    constraints->th_cons = th_cons;
    constraints->hr_cons = hr_cons;
    constraints->out_cons = out_cons;

    int n_cons;
    file.read(reinterpret_cast<char*>(&n_cons), sizeof(int));
    constraints->n_cons = n_cons;

    return constraints;
}

void print_hierarchy_constraints(const hierarchyConstraints *hr_cons, const int n_hr_cons, const int n_hr_features){
    std::cout << "#### " << n_hr_cons << " hierarchy constraints #######" << std::endl;
    int cumsum = 0;
    for (int i = 0; i < n_hr_cons; ++i){
        std::cout << "constraint: " << i + 1 << " - mapped feature_idx: " << hr_cons->feature_indices[i];
        if (hr_cons->is_numerics[i]){
            std::cout << " numerical constraint with ";
        } else {
            std::cout << " categorical constraint with ";
        }
        std::cout << hr_cons->dep_count[i] << " dependent features [";
        for (int j = 0; j < hr_cons->dep_count[i]; ++j){
            std::cout << hr_cons->dep_features[cumsum + j];
            if (j < hr_cons->dep_count[i] - 1)
                std::cout << ",";
        }
        std::cout << "]" << std::endl;
        cumsum += hr_cons->dep_count[i];
    }
    std::cout << "total features: " << n_hr_features << std::endl;
}

void print_output_constraints(const outputConstraints *out_cons, const int n_out_cons, const int output_dim){
    std::cout << "#### " << n_out_cons << " output constraints #######" << std::endl;
    for (int i = 0; i < n_out_cons; ++i){
        std::cout << "constraint: " << i + 1 << " - mapped feature_idx: " << out_cons->feature_indices[i];
        if (out_cons->is_numerics[i]){
            std::cout << " numerical constraint - feature_value";
            if (out_cons->inequality_directions[i])
                std::cout << " > ";
            else 
                std::cout << " <= ";
            std::cout << out_cons->feature_values[i];
        } else {
            std::cout << " categorcal constraint - feature_value";
            if (out_cons->inequality_directions[i])
                std::cout << " == ";
            else 
                std::cout << " != ";
            for (int j = 0; j < MAX_CHAR_SIZE; ++j)
                std::cout << out_cons->categorical_values[i*MAX_CHAR_SIZE + j];
        }
        std::cout << " output values: [";
        for (int j = 0; j < output_dim; ++j){
            std::cout << out_cons->output_values[i*output_dim + j];
            if (j < output_dim - 1)
                std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
}

constraintType stringToconstraintType(std::string str) {
    if (str == "threshold") return constraintType::THRESHOLD;
    if (str == "hierarchy") return constraintType::HIERARCHY;
    if (str == "output") return constraintType::OUTPUT;
    throw std::runtime_error("Invalid exportFormat! Options are: threshold/hierarchy/output");
}