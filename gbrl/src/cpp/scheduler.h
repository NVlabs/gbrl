//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <iostream>
#include <cstdint>

#include "types.h"

class Scheduler {
    public: 
        Scheduler(float init_lr);
        Scheduler(const Scheduler& other);
        virtual ~Scheduler() = default; // Declare virtual destructor
        virtual float get_lr(int t) = 0; 
        virtual int saveToFile(std::ofstream& file) = 0;
        schedulerFunc getType() const ;
        void setType(schedulerFunc type);
        static Scheduler *loadFromFile(std::ifstream& file);

        float init_lr;

    private:
        schedulerFunc type;
};

class LinearScheduler: public Scheduler {
    public:
        LinearScheduler(float init_lr, float stop_lr, int T);
        LinearScheduler(const LinearScheduler& other);
        inline float get_lr(int t) override {
            float T_ = static_cast<float>(this->T), t_ = static_cast<float>(t) + 1;
            float progress_remaining = (T_ - t_) / T_;
            float lr = this->init_lr + (1.0f - progress_remaining) * (this->stop_lr - this->init_lr);
            if (lr < this->stop_lr)
                return this->stop_lr;
            return lr;
        }
        
        int saveToFile(std::ofstream& file) override;
        static LinearScheduler* loadFromFile(std::ifstream& file);

        float stop_lr;
        int T;
};

class ConstScheduler: public Scheduler {
    public:
        ConstScheduler(float init_lr);
        ConstScheduler(const ConstScheduler& other);
        inline float get_lr(int t) override {
            (void)t;
            return this->init_lr;
        }

        int saveToFile(std::ofstream& file) override;
        static ConstScheduler* loadFromFile(std::ifstream& file);
};

#endif 