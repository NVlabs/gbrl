//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2026, NVIDIA Corporation. All rights reserved.
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

/**
 * @file optimizer.h
 * @brief Optimization algorithms for gradient descent
 * 
 * Provides optimizer implementations including SGD and Adam for updating
 * model parameters during gradient boosting training.
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>

#include "types.h"
#include "scheduler.h"

/**
 * @brief Base class for gradient descent optimizers
 * 
 * Abstract base class defining the interface for optimization algorithms.
 * Derived classes implement specific update rules (SGD, Adam, etc.).
 */
class Optimizer {
    public:
        /**
         * @brief Default constructor
         */
        Optimizer();
        
        /**
         * @brief Virtual destructor
         */
        virtual ~Optimizer() {}
        
        /**
         * @brief Construct optimizer with constant learning rate
         * 
         * @param schedule_func Scheduler type
         * @param init_lr Initial learning rate
         */
        Optimizer(schedulerFunc schedule_func, float init_lr);
        
        /**
         * @brief Construct optimizer with learning rate decay
         * 
         * @param schedule_func Scheduler type
         * @param init_lr Initial learning rate
         * @param stop_lr Final learning rate
         * @param T Total number of iterations
         */
        Optimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T);
        
        /**
         * @brief Copy constructor
         * @param other Optimizer to copy from
         */
        Optimizer(const Optimizer& other);
        
        /**
         * @brief Pure virtual: Perform optimization step on parameters
         * 
         * @param theta Parameters to update (modified in-place)
         * @param raw_grad_theta Gradient values
         * @param t Current iteration/timestep
         * @param sample_idx Sample index for per-sample state
         */
        virtual void step(float *theta, const float *raw_grad_theta, int t, int sample_idx) = 0;
        
        /**
         * @brief Copy and scale values by negative learning rate
         * 
         * Copies values from source to destination, scaling by -lr(t).
         * Used for matrix representation generation (V = -lr * leaf_values).
         * Only operates on indices [start_idx, stop_idx) for this optimizer.
         * 
         * @param scaled_grad_theta Destination array for scaled values
         * @param raw_grad_theta Source values to copy and scale
         * @param t Current iteration for learning rate lookup
         */
        void copy_and_scale(float *scaled_grad_theta, const float *raw_grad_theta, int t);
        
        /**
         * @brief Add inverse-scaled values to destination
         * 
         * Subtracts source values divided by learning rate from destination.
         * Used for compression correction: dest -= src / lr(t).
         * Only operates on indices [start_idx, stop_idx) for this optimizer.
         * 
         * @param raw_grad_theta Destination array (modified in-place)
         * @param scaled_grad_theta Source values to inverse-scale and subtract
         * @param t Current iteration for learning rate lookup
         */
        void add_scaled(float *raw_grad_theta, const float *scaled_grad_theta, int t);
        
        /**
         * @brief Get optimizer configuration for serialization
         * @return Pointer to config struct, caller must delete
         */
        virtual optimizerConfig* getConfig() = 0;
        
        /**
         * @brief Save optimizer state to binary file
         * 
         * @param file Output file stream
         * @return Number of bytes written
         */
        virtual int saveToFile(std::ofstream& file) = 0;
        
        /**
         * @brief Load optimizer from binary file
         * 
         * @param file Input file stream
         * @return Pointer to loaded optimizer, caller must delete
         */
        static Optimizer* loadFromFile(std::ifstream& file);
        
        /**
         * @brief Allocate memory for optimizer state
         * 
         * @param n_samples Number of samples
         * @param output_dim Output dimensionality
         */
        virtual void set_memory(const int n_samples, const int output_dim) = 0;
        
        /**
         * @brief Set optimizer algorithm type
         * @param _algo Algorithm enum value
         */
        void setAlgo(optimizerAlgo _algo);
        
        /**
         * @brief Get optimizer algorithm type
         * @return Algorithm enum value
         */
        optimizerAlgo getAlgo() const;
        
        /**
         * @brief Set range of tree indices this optimizer applies to
         * 
         * @param _start_idx Starting tree index
         * @param _stop_idx Stopping tree index (exclusive)
         */
        void set_indices(int _start_idx, int _stop_idx);

        Scheduler *scheduler;       /**< Learning rate scheduler */
        int start_idx = 0;          /**< Start tree index */
        int stop_idx = 0;           /**< Stop tree index */
        
    private:
        optimizerAlgo algo;         /**< Algorithm type */
};

/**
 * @brief Stochastic Gradient Descent optimizer
 * 
 * Implements basic SGD with momentum and learning rate scheduling.
 */
class SGDOptimizer : public Optimizer {
    public:
        /**
         * @brief Default constructor
         */
        SGDOptimizer();
        
        /**
         * @brief Destructor
         */
        ~SGDOptimizer();
        
        /**
         * @brief Construct SGD with constant learning rate
         * 
         * @param schedule_func Scheduler type
         * @param init_lr Initial learning rate
         */
        SGDOptimizer(schedulerFunc schedule_func, float init_lr);
        
        /**
         * @brief Construct SGD with learning rate decay
         * 
         * @param schedule_func Scheduler type
         * @param init_lr Initial learning rate
         * @param stop_lr Final learning rate
         * @param T Total iterations
         */
        SGDOptimizer(schedulerFunc schedule_func, float init_lr, float stop_lr, int T);
        
        /**
         * @brief Copy constructor
         * @param other Optimizer to copy from
         */
        SGDOptimizer(const SGDOptimizer& other);
        
        /**
         * @brief Get optimizer configuration
         * @return Pointer to config struct, caller must delete
         */
        optimizerConfig* getConfig() override;
        
        /**
         * @brief Perform SGD update step
         * 
         * @param theta Parameters to update
         * @param raw_grad_theta Gradient
         * @param t Iteration number
         * @param sample_idx Sample index
         */
        void step(
            float *theta,
            const float *raw_grad_theta,
            int t,
            int sample_idx
        ) override;
        
        /**
         * @brief Allocate optimizer state memory
         * 
         * @param n_samples Number of samples
         * @param output_dim Output dimensionality
         */
        void set_memory(const int n_samples, const int output_dim) override;
        
        /**
         * @brief Save optimizer to binary file
         * 
         * @param file Output file stream
         * @return Number of bytes written
         */
        int saveToFile(std::ofstream& file) override;
        
        /**
         * @brief Load SGD optimizer from binary file
         * 
         * @param file Input file stream
         * @return Pointer to loaded optimizer, caller must delete
         */
        static SGDOptimizer* loadFromFile(std::ifstream& file);
};

/**
 * @brief Adam (Adaptive Moment Estimation) optimizer
 * 
 * Implements the Adam optimization algorithm with bias correction
 * and adaptive learning rates per parameter.
 */
class AdamOptimizer : public Optimizer {
    public:
        /**
         * @brief Construct Adam with default hyperparameters
         * 
         * @param beta_1 First moment decay rate
         * @param beta_2 Second moment decay rate
         * @param eps Numerical stability epsilon
         */
        AdamOptimizer(float beta_1, float beta_2, float eps);
        
        /**
         * @brief Destructor
         */
        ~AdamOptimizer();
        
        /**
         * @brief Construct Adam with constant learning rate
         * 
         * @param schedule_func Scheduler type
         * @param init_lr Initial learning rate
         * @param beta_1 First moment decay rate
         * @param beta_2 Second moment decay rate
         * @param eps Numerical stability epsilon
         */
        AdamOptimizer(
            schedulerFunc schedule_func,
            float init_lr,
            float beta_1,
            float beta_2,
            float eps
        );
        
        /**
         * @brief Construct Adam with learning rate decay
         * 
         * @param schedule_func Scheduler type
         * @param init_lr Initial learning rate
         * @param stop_lr Final learning rate
         * @param T Total iterations
         * @param beta_1 First moment decay rate
         * @param beta_2 Second moment decay rate
         * @param eps Numerical stability epsilon
         */
        AdamOptimizer(
            schedulerFunc schedule_func,
            float init_lr,
            float stop_lr,
            int T,
            float beta_1,
            float beta_2,
            float eps
        );
        
        /**
         * @brief Copy constructor
         * @param other Optimizer to copy from
         */
        AdamOptimizer(const AdamOptimizer& other);
        
        /**
         * @brief Get optimizer configuration
         * @return Pointer to config struct, caller must delete
         */
        optimizerConfig* getConfig() override;

        /**
         * @brief Perform Adam update step
         * 
         * Updates first and second moment estimates and applies
         * bias-corrected adaptive learning rate.
         * 
         * @param theta Parameters to update
         * @param raw_grad_theta Gradient
         * @param t Iteration number
         * @param sample_idx Sample index
         */
        void step(
            float *theta,
            const float *raw_grad_theta,
            int t,
            int sample_idx
        ) override;
        
        /**
         * @brief Allocate optimizer state memory
         * 
         * @param n_samples Number of samples
         * @param output_dim Output dimensionality
         */
        void set_memory(const int n_samples, const int output_dim) override;
        
        /**
         * @brief Save optimizer to binary file
         * 
         * @param file Output file stream
         * @return Number of bytes written
         */
        int saveToFile(std::ofstream& file) override;
        
        /**
         * @brief Load Adam optimizer from binary file
         * 
         * @param file Input file stream
         * @return Pointer to loaded optimizer, caller must delete
         */
        static AdamOptimizer* loadFromFile(std::ifstream& file);

        float beta_1 = 0.9f;        /**< First moment exponential decay rate */
        float beta_2 = 0.99f;       /**< Second moment exponential decay rate */
        float eps = 1e-8f;          /**< Epsilon for numerical stability */
        float* m = nullptr;         /**< First moment estimate */
        float* v = nullptr;         /**< Second moment estimate */
};

#endif // OPTIMIZER_H 