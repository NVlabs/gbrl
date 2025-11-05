
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
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
 * @file data_structs.cpp
 * @brief Implementation of fundamental data structures
 */

#include <iostream>

#include "data_structs.h"
#include "types.h"

template <typename T>
stack<T>::stack(int max_elements) 
    : max_size(max_elements), top_index(-1) {
    data = new T[max_size];
}

template <typename T>
stack<T>::~stack() {
    delete[] data;
}

template <typename T>
bool stack<T>::is_empty() const {
    return top_index == -1;
}

template <typename T>
bool stack<T>::is_full() const {
    return top_index == max_size - 1;
}

template <typename T>
void stack<T>::push(const T &element) {
    if (is_full()) {
        std::cerr << "Stack overflow: cannot push element\n";
        return;
    }
    data[++top_index] = element;
}

template <typename T>
void stack<T>::pop() {
    if (is_empty()) {
        std::cerr << "Stack underflow: cannot pop from empty stack\n";
        return;
    }
    --top_index;
}

template <typename T>
T& stack<T>::top() const {
    if (is_empty()) {
        throw std::out_of_range("Stack is empty: cannot access top element");
    }
    return data[top_index];
}

// Explicit template instantiations
template struct stack<int>;
template struct stack<nodeInfo>;