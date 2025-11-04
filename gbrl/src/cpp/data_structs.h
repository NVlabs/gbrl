//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file data_structs.h
 * @brief Fundamental data structures for tree algorithms
 * 
 * Provides basic data structures like stacks used throughout the
 * tree-building and traversal algorithms.
 */

#ifndef DATA_STRUCTS_H
#define DATA_STRUCTS_H

/**
 * @brief Generic stack data structure
 * 
 * @tparam T Type of elements stored in the stack
 * 
 * Provides a simple fixed-size stack implementation for tree traversal
 * and other algorithmic needs. Uses array-based storage for efficiency.
 */
template <typename T>
struct stack {
    T* data;                /**< Array storing stack elements */
    int max_size;           /**< Maximum capacity of the stack */
    int top_index;          /**< Index of the top element (-1 when empty) */

    /**
     * @brief Construct a new stack
     * @param max_elements Maximum number of elements the stack can hold
     */
    stack(int max_elements);
    
    /**
     * @brief Destroy the stack and free memory
     */
    ~stack();

    /**
     * @brief Check if the stack is empty
     * @return true if stack has no elements
     */
    bool is_empty() const;
    
    /**
     * @brief Check if the stack is at full capacity
     * @return true if stack cannot accept more elements
     */
    bool is_full() const;
    
    /**
     * @brief Push an element onto the stack
     * @param element Element to add to the top of the stack
     */
    void push(const T &element);
    
    /**
     * @brief Access the top element without removing it
     * @return Reference to the top element
     * @throws std::out_of_range if stack is empty
     */
    T& top() const;
    
    /**
     * @brief Remove the top element from the stack
     */
    void pop();
};

#endif // DATA_STRUCTS_H 