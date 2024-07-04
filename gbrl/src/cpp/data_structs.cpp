
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <iostream>

#include "data_structs.h"
#include "types.h"

template <typename T>
stack<T>::stack(int max_elements) : max_size(max_elements), top_index(-1) {
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
        std::cerr << "Stack overflow\n";
        return;
    }
    data[++top_index] = element;
}

template <typename T>
void stack<T>::pop() {
    if (is_empty()) {
        std::cerr << "Stack underflow\n";
        return;
    }
    --top_index;
}


template <typename T>
T& stack<T>::top() const {
    if (is_empty()) {
        throw std::out_of_range("Stack is empty");
    }
    return data[top_index];
}

template struct stack<int>;
template struct stack<nodeInfo>;