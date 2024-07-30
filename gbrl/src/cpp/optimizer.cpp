//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <iostream>
#include <fstream>

#include "optimizer.h"
#include "scheduler.h"
#include "math_ops.h"

Optimizer::Optimizer(){}


Optimizer::Optimizer(schedulerFunc schedule_func, float init_lr){
    if (schedule_func == Const)
        this->scheduler = new ConstScheduler(init_lr);
}

Optimizer::Optimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T){
    if (schedule_func == Linear)
        this->scheduler =  new LinearScheduler(init_lr, stop_lr, T);
}

Optimizer::Optimizer(const Optimizer& other):
    start_idx(other.start_idx), stop_idx(other.stop_idx){
    schedulerFunc sched_type = other.scheduler->getType();
    switch (sched_type) {
        case Const: {
            ConstScheduler *const_sched = dynamic_cast<ConstScheduler *>(other.scheduler);
            if (const_sched) {
                this->scheduler = new ConstScheduler(*const_sched);
            } else {
                std::cerr << "Failed to dynamic_cast to ConstScheduler." << std::endl;
            }
            break;
        }
        case Linear:{
        LinearScheduler *lin_sched = dynamic_cast<LinearScheduler *>(other.scheduler);
            if (lin_sched) {
                this->scheduler = new LinearScheduler(*lin_sched);
            } else {
                std::cerr << "Failed to dynamic_cast to LinearScheduler." << std::endl;
            }
            break;
        }
        default: {
            std::cerr << "Unknown scheduler type." << std::endl;
            break;
        }
    }
}

void Optimizer::setAlgo(optimizerAlgo algo){
    this->algo = algo;
}

optimizerAlgo Optimizer::getAlgo() const {
    return this->algo;
}


void Optimizer::set_indices(int start_idx, int stop_idx){
    this->start_idx = start_idx;
    this->stop_idx = stop_idx;
}

Optimizer* Optimizer::loadFromFile(std::ifstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return nullptr;
    }

    optimizerAlgo algo;
    file.read(reinterpret_cast<char*>(&algo), sizeof(optimizerAlgo));
    switch (algo) {
        case SGD:
            return SGDOptimizer::loadFromFile(file);
        case Adam:
            return AdamOptimizer::loadFromFile(file);
        default:
            std::cerr << "Unknown Optimizer algo." << std::endl;
            return nullptr;  // Or handle the error as appropriate
    }
}

SGDOptimizer::SGDOptimizer(): Optimizer(){
    optimizerAlgo algo = SGD;
    this->setAlgo(algo);
}
SGDOptimizer::SGDOptimizer(schedulerFunc schedule_func, float init_lr): Optimizer(schedule_func, init_lr){
    optimizerAlgo algo = SGD;
    this->setAlgo(algo);
}
SGDOptimizer::SGDOptimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T): Optimizer(schedule_func, init_lr, stop_lr, T){
    optimizerAlgo algo = SGD;
    this->setAlgo(algo);
}

void SGDOptimizer::step(float *theta, const float *raw_grad_theta, int t, int sample_idx){
    int start_idx = this->start_idx, stop_idx = this->stop_idx;
    float lr = this->scheduler->get_lr(t);
#ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int i = start_idx; i < stop_idx; i++){
        theta[sample_idx + i] -= lr * raw_grad_theta[i];
    }
}

int SGDOptimizer::saveToFile(std::ofstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return -1;
    }
    optimizerAlgo algo = SGD;
    file.write(reinterpret_cast<char*>(&algo), sizeof(optimizerAlgo));
    file.write(reinterpret_cast<char*>(&this->start_idx), sizeof(int));
    file.write(reinterpret_cast<char*>(&this->stop_idx), sizeof(int));
    this->scheduler->saveToFile(file);
    return 0;
}

SGDOptimizer* SGDOptimizer::loadFromFile(std::ifstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return nullptr;
    }

    int start_idx, count;
    file.read(reinterpret_cast<char*>(&start_idx), sizeof(int));
    file.read(reinterpret_cast<char*>(&count), sizeof(int));
    Scheduler *sched = Scheduler::loadFromFile(file);
    SGDOptimizer* opt = new SGDOptimizer();
    opt->scheduler = sched;
    opt->set_indices(start_idx, count);
    return opt;
}

SGDOptimizer::SGDOptimizer(const SGDOptimizer& other): Optimizer(other){}

SGDOptimizer::~SGDOptimizer(){
    delete this->scheduler;
}

optimizerConfig* SGDOptimizer::getConfig() {
    optimizerConfig *conf = new optimizerConfig;
    conf->algo = algoTypeToString(this->getAlgo());
    schedulerFunc sched_func = this->scheduler->getType();
    conf->scheduler_func = schedulerTypeToString(sched_func);
    conf->init_lr = this->scheduler->init_lr;
    if (sched_func == Linear){
        LinearScheduler* linearScheduler = dynamic_cast<LinearScheduler*>(this->scheduler);
        conf->stop_lr = linearScheduler->stop_lr;
        conf->T = linearScheduler->T;
    } else {
        conf->stop_lr = 0.0f;
        conf->T = 10000;
    }
    conf->beta_1 = 0.99f;
    conf->beta_2 = 0.999f;
    conf->start_idx = this->start_idx;
    conf->stop_idx = this->stop_idx;
    conf->eps = 1e-8f;
    return conf;
}


void SGDOptimizer::set_memory(const int n_samples, const int output_dim) {
    (void)n_samples;
    (void)output_dim;
}

AdamOptimizer::AdamOptimizer(float beta_1, float beta_2, float eps = 1.0e-8): Optimizer(), beta_1(beta_1), beta_2(beta_2), eps(eps){
    optimizerAlgo algo = Adam;
    this->setAlgo(algo);
}
AdamOptimizer::AdamOptimizer(schedulerFunc schedule_func, float init_lr, float beta_1, float beta_2, float eps = 1.0e-8): Optimizer(schedule_func, init_lr), beta_1(beta_1), beta_2(beta_2), eps(eps){
    optimizerAlgo algo = Adam;
    this->setAlgo(algo);
}
AdamOptimizer::AdamOptimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T, float beta_1, float beta_2, float eps = 1.0e-8): Optimizer(schedule_func, init_lr, stop_lr, T), beta_1(beta_1), beta_2(beta_2), eps(eps){
    optimizerAlgo algo = Adam;
    this->setAlgo(algo);
}

AdamOptimizer::AdamOptimizer(const AdamOptimizer& other): Optimizer(other), 
beta_1(other.beta_1), beta_2(other.beta_2), eps(other.eps){

}

optimizerConfig* AdamOptimizer::getConfig() {
    optimizerConfig *conf = new optimizerConfig;
    conf->algo = algoTypeToString(this->getAlgo());
    schedulerFunc sched_func = this->scheduler->getType();
    conf->scheduler_func = schedulerTypeToString(sched_func);
    conf->init_lr = this->scheduler->init_lr;
    if (sched_func == Linear){
        LinearScheduler* linearScheduler = dynamic_cast<LinearScheduler*>(this->scheduler);
        conf->stop_lr = linearScheduler->stop_lr;
        conf->T = linearScheduler->T;
    } else {
        conf->stop_lr = 0.0;
        conf->T = 10000;
    }
    conf->beta_1 = this->beta_1;
    conf->beta_2 = this->beta_2;
    conf->start_idx = this->start_idx;
    conf->stop_idx = this->stop_idx;
    conf->eps = this->eps;
    return conf;
}

int AdamOptimizer::saveToFile(std::ofstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return -1;
    }
    optimizerAlgo algo = Adam;
    file.write(reinterpret_cast<char*>(&algo), sizeof(optimizerAlgo));
    file.write(reinterpret_cast<char*>(&this->start_idx), sizeof(int));
    file.write(reinterpret_cast<char*>(&this->stop_idx), sizeof(int));
    file.write(reinterpret_cast<char*>(&this->beta_1), sizeof(float));
    file.write(reinterpret_cast<char*>(&this->beta_2), sizeof(float));
    file.write(reinterpret_cast<char*>(&this->eps), sizeof(float));
    this->scheduler->saveToFile(file);
    return 0;
}

AdamOptimizer* AdamOptimizer::loadFromFile(std::ifstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return nullptr;
    }

    float beta_1, beta_2, eps;
    int start_idx, count;
    file.read(reinterpret_cast<char*>(&start_idx), sizeof(int));
    file.read(reinterpret_cast<char*>(&count), sizeof(int));
    file.read(reinterpret_cast<char*>(&beta_1), sizeof(float));
    file.read(reinterpret_cast<char*>(&beta_2), sizeof(float));
    file.read(reinterpret_cast<char*>(&eps), sizeof(float));
    AdamOptimizer* opt = new AdamOptimizer(beta_1, beta_2, eps);
    Scheduler *sched = Scheduler::loadFromFile(file);
    opt->scheduler = sched;
    opt->set_indices(start_idx, count);
    return opt;
}


void AdamOptimizer::step(float *theta, const float *raw_grad_theta, int t, int sample_idx){
    if (this->m == nullptr|| this->v == nullptr){
        std::cerr << "Trying to use step without initializing memory." << std::endl;
    }

    float lr = this->scheduler->get_lr(t);
    float t_float = static_cast<float>(t) + 1;
    start_idx = this->start_idx, stop_idx = this->stop_idx;
    float *raw_m = this->m, *raw_v = this->v;
    float alpha = lr*sqrt(1 - pow(this->beta_2, t_float)) / (1 - pow(this->beta_1, t_float));

#ifndef _MSC_VER
    #pragma omp simd
#endif
    for (int i = start_idx; i < stop_idx; ++i){
        int index = sample_idx + i;
        raw_m[index] *= this->beta_1; 
        raw_v[index] *= this->beta_2; 
        raw_m[index] += raw_grad_theta[i]*(1.0f - this->beta_1);
        raw_v[index] += (raw_grad_theta[i] * raw_grad_theta[i])*(1.0f - this->beta_2);
        float m_val = raw_m[index];
        float v_val = sqrt(raw_v[index]);
        theta[index] -= (alpha * m_val) / (v_val + this->eps);
    }
}

void AdamOptimizer::set_memory(const int n_samples, const int output_dim) {
    if (this->m != nullptr){
        delete[] this->m;
        this->m = nullptr;
    }
    
    if (this->v != nullptr){
        delete[] this->v;
        this->v = nullptr;
    }

    int size = n_samples*output_dim;
    this->m = init_zero_mat(size);
    this->v = init_zero_mat(size);
}

AdamOptimizer::~AdamOptimizer(){
    delete this->scheduler;
    if (this->m != nullptr)
        delete[] this->m;
    if (this->v != nullptr)
        delete[] this->v;
    
    this->m = nullptr;
    this->v = nullptr;
}