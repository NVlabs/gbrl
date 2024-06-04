//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <memory>
#include <fstream>
#include <iostream>

#include "scheduler.h"

Scheduler::Scheduler(float init_lr): init_lr(init_lr) {}

schedulerFunc Scheduler::getType() const { 
    return this->type;
}

Scheduler::Scheduler(const Scheduler& other):
    init_lr(other.init_lr)
{
    schedulerFunc sched_type = other.getType();
    this->setType(sched_type);

}

void Scheduler::setType(schedulerFunc type){
    this->type = type;
}

Scheduler* Scheduler::loadFromFile(std::ifstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return nullptr;
    }

    schedulerFunc sched;
    file.read(reinterpret_cast<char*>(&sched), sizeof(schedulerFunc));
    switch (sched) {
        case Const:
            return ConstScheduler::loadFromFile(file);
        case Linear:
            return LinearScheduler::loadFromFile(file);
        default:
            std::cerr << "Unknown scheduler type." << std::endl;
            return nullptr;  // Or handle the error as appropriate
    }
}

LinearScheduler::LinearScheduler(float init_lr, float stop_lr, int T): Scheduler(init_lr), stop_lr(stop_lr), T(T) {
    schedulerFunc f = Linear;
    this->setType(f);
}

LinearScheduler::LinearScheduler(const LinearScheduler& other): Scheduler(other), stop_lr(other.stop_lr), T(other.T){}

int LinearScheduler::saveToFile(std::ofstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return -1;
    }
    schedulerFunc type = this->getType();
    file.write(reinterpret_cast<char*>(&type), sizeof(schedulerFunc));
    file.write(reinterpret_cast<char*>(&this->init_lr), sizeof(float));
    file.write(reinterpret_cast<char*>(&this->stop_lr), sizeof(float));
    file.write(reinterpret_cast<char*>(&this->T), sizeof(int));
    return 0;
}

LinearScheduler* LinearScheduler::loadFromFile(std::ifstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return nullptr;
    }

    float init_lr, stop_lr;
    int T;
    file.read(reinterpret_cast<char*>(&init_lr), sizeof(float));
    file.read(reinterpret_cast<char*>(&stop_lr), sizeof(float));
    file.read(reinterpret_cast<char*>(&T), sizeof(int));
    return new LinearScheduler(init_lr, stop_lr, T);
}


ConstScheduler::ConstScheduler(float init_lr): Scheduler(init_lr) {
    schedulerFunc f = Const;
    this->setType(f);
}

ConstScheduler::ConstScheduler(const ConstScheduler& other): Scheduler(other){}

int ConstScheduler::saveToFile(std::ofstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return -1;
    }
    schedulerFunc type = this->getType();
    file.write(reinterpret_cast<char*>(&type), sizeof(schedulerFunc));
    file.write(reinterpret_cast<char*>(&this->init_lr), sizeof(float));
    return 0;
}

ConstScheduler* ConstScheduler::loadFromFile(std::ifstream& file){
    if (!file.is_open() || file.fail()) {
        std::cerr << "Error file is not open for writing: " << std::endl;
        return nullptr;
    }

    float init_lr;
    file.read(reinterpret_cast<char*>(&init_lr), sizeof(float));
    return new ConstScheduler(init_lr);
}
