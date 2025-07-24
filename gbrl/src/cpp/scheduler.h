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
        void setType(schedulerFunc _type);
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