//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
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

void Scheduler::setType(schedulerFunc _type){
    this->type = _type;
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
    schedulerFunc _type = this->getType();
    file.write(reinterpret_cast<char*>(&_type), sizeof(schedulerFunc));
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
    schedulerFunc _type = this->getType();
    file.write(reinterpret_cast<char*>(&_type), sizeof(schedulerFunc));
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
