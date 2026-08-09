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
 * @file scheduler.h
 * @brief Learning rate scheduling for optimization
 * 
 * Provides learning rate schedulers that control how the learning rate
 * changes during training. Supports constant and linear decay schedules.
 */

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <iostream>
#include <cstdint>

#include "types.h"

/**
 * @brief Base class for learning rate schedulers
 * 
 * Abstract base class defining the interface for learning rate scheduling.
 * Derived classes implement specific scheduling strategies.
 */
class Scheduler {
    public:
        /**
         * @brief Construct scheduler with initial learning rate
         * @param init_lr Initial learning rate
         */
        Scheduler(float init_lr);
        
        /**
         * @brief Copy constructor
         * @param other Scheduler to copy from
         */
        Scheduler(const Scheduler& other);
        
        /**
         * @brief Virtual destructor
         */
        virtual ~Scheduler() = default;
        
        /**
         * @brief Get learning rate at iteration t
         * 
         * @param t Current iteration number
         * @return Learning rate for this iteration
         */
        virtual float get_lr(int t) = 0;
        
        /**
         * @brief Save scheduler to binary file
         * 
         * @param file Output file stream
         * @return Number of bytes written
         */
        virtual int saveToFile(std::ofstream& file) = 0;
        
        /**
         * @brief Get scheduler type
         * @return Type enum value
         */
        schedulerFunc getType() const;
        
        /**
         * @brief Set scheduler type
         * @param _type Type enum value
         */
        void setType(schedulerFunc _type);
        
        /**
         * @brief Load scheduler from binary file
         * 
         * @param file Input file stream
         * @return Pointer to loaded scheduler, caller must delete
         */
        static Scheduler *loadFromFile(std::ifstream& file);

        float init_lr;      /**< Initial learning rate */

    private:
        schedulerFunc type;  /**< Scheduler type identifier */
};

/**
 * @brief Linear learning rate decay scheduler
 * 
 * Linearly interpolates learning rate from initial to final value
 * over T iterations.
 */
class LinearScheduler : public Scheduler {
    public:
        /**
         * @brief Construct linear scheduler
         * 
         * @param init_lr Initial learning rate
         * @param stop_lr Final learning rate
         * @param T Total number of iterations
         */
        LinearScheduler(float init_lr, float stop_lr, int T);
        
        /**
         * @brief Copy constructor
         * @param other Scheduler to copy from
         */
        LinearScheduler(const LinearScheduler& other);
        
        /**
         * @brief Get learning rate at iteration t
         *
         * Computes: lr(t) = init_lr + clamp(t/T, 0, 1) * (stop_lr - init_lr)
         *
         * so lr(0) == init_lr and lr(t >= T) == stop_lr.  Clamping the progress
         * rather than the rate keeps this correct in both directions: the old
         * `if (lr < stop_lr) return stop_lr` assumed a decaying schedule, which
         * collapsed warmup (stop_lr > init_lr) to a constant stop_lr and then
         * let it grow past stop_lr once t exceeded T.
         *
         * @param t Current iteration number
         * @return Learning rate for this iteration
         */
        inline float get_lr(int t) override {
            if (this->T <= 0)
                return this->stop_lr;
            float progress = static_cast<float>(t) / static_cast<float>(this->T);
            if (progress < 0.0f) progress = 0.0f;
            if (progress > 1.0f) progress = 1.0f;
            return this->init_lr + progress * (this->stop_lr - this->init_lr);
        }
        
        /**
         * @brief Save scheduler to binary file
         * 
         * @param file Output file stream
         * @return Number of bytes written
         */
        int saveToFile(std::ofstream& file) override;
        
        /**
         * @brief Load linear scheduler from binary file
         * 
         * @param file Input file stream
         * @return Pointer to loaded scheduler, caller must delete
         */
        static LinearScheduler* loadFromFile(std::ifstream& file);

        float stop_lr;      /**< Final learning rate */
        int T;              /**< Total number of iterations */
};

/**
 * @brief Constant learning rate scheduler
 * 
 * Maintains a fixed learning rate throughout training.
 */
class ConstScheduler : public Scheduler {
    public:
        /**
         * @brief Construct constant scheduler
         * @param init_lr Learning rate (constant)
         */
        ConstScheduler(float init_lr);
        
        /**
         * @brief Copy constructor
         * @param other Scheduler to copy from
         */
        ConstScheduler(const ConstScheduler& other);
        
        /**
         * @brief Get learning rate (always returns init_lr)
         * 
         * @param t Current iteration (unused)
         * @return Constant learning rate
         */
        inline float get_lr(int t) override {
            (void)t;  // Suppress unused parameter warning
            return this->init_lr;
        }

        /**
         * @brief Save scheduler to binary file
         * 
         * @param file Output file stream
         * @return Number of bytes written
         */
        int saveToFile(std::ofstream& file) override;
        
        /**
         * @brief Load constant scheduler from binary file
         * 
         * @param file Input file stream
         * @return Pointer to loaded scheduler, caller must delete
         */
        static ConstScheduler* loadFromFile(std::ifstream& file);
};

#endif // SCHEDULER_H 