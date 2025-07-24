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
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory> 

#include "types.h"
#include "scheduler.h"

class Optimizer {
    public: 
        Optimizer();
        virtual ~Optimizer() {}  // Declare virtual destructor
        Optimizer(schedulerFunc schedule_func, float init_lr);
        Optimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T);
        Optimizer(const Optimizer& other);
        virtual void step(float *theta, const float *raw_grad_theta, int t, int sample_idx) = 0;
        virtual optimizerConfig* getConfig() = 0;
        virtual int saveToFile(std::ofstream& file) = 0;
        static Optimizer* loadFromFile(std::ifstream& file);
        virtual void set_memory(const int n_samples, const int output_dim) = 0 ;
        void setAlgo(optimizerAlgo _algo);
        optimizerAlgo getAlgo() const ;
        void set_indices(int _start_idx, int _stop_idx);

        Scheduler *scheduler;
        int start_idx = 0;
        int stop_idx = 0;
    private:
        optimizerAlgo algo;
};

class SGDOptimizer: public Optimizer {
    public:
        SGDOptimizer();
        ~SGDOptimizer();
        SGDOptimizer(schedulerFunc schedule_func, float init_lr);
        SGDOptimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T);
        SGDOptimizer(const SGDOptimizer& other);
        optimizerConfig* getConfig() override;
        
        void step(float *theta, const float *raw_grad_theta, int t, int sample_idx) override;
        void set_memory(const int n_samples, const int output_dim) override; 
        int saveToFile(std::ofstream& file) override;
        static SGDOptimizer* loadFromFile(std::ifstream& file);
};

class AdamOptimizer: public Optimizer {
    public:
        AdamOptimizer(float beta_1, float beta_2, float eps);
        ~AdamOptimizer();
        AdamOptimizer(schedulerFunc schedule_func, float init_lr, float beta_1, float beta_2, float eps);
        AdamOptimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T, float beta_1, float beta_2, float eps);
        AdamOptimizer(const AdamOptimizer& other);
        optimizerConfig* getConfig() override;

        void step(float *theta, const float *raw_grad_theta, int t, int sample_idx) override;
        void set_memory(const int n_samples, const int output_dim) override; 
        int saveToFile(std::ofstream& file) override;
        static AdamOptimizer* loadFromFile(std::ifstream& file);

        float beta_1 = 0.9f;
        float beta_2 = 0.99f;
        float eps = 1e-8f; 
        float* m = nullptr;
        float* v = nullptr;
};



#endif 