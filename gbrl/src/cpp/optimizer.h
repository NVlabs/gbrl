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
        virtual void set_memory(const int n_samples, const int output_dim, const int par_th) = 0 ;
        void setAlgo(optimizerAlgo algo);
        optimizerAlgo getAlgo() const ;
        void set_indices(int start_idx, int end_idx);

        Scheduler *scheduler;
        int start_idx = 0;
        int end_idx = 0;
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
        void set_memory(const int n_samples, const int output_dim, const int par_th) override; 
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
        void set_memory(const int n_samples, const int output_dim, const int par_th) override; 
        int saveToFile(std::ofstream& file) override;
        static AdamOptimizer* loadFromFile(std::ifstream& file);

        float beta_1 = 0.9f;
        float beta_2 = 0.99f;
        float eps = 1e-8f; 
        float* m = nullptr;
        float* v = nullptr;
};



#endif 