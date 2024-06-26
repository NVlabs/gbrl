#ifndef DATA_STRUCTS_H
#define DATA_STRUCTS_H

template <typename T>
struct stack {
    T* data;
    int max_size;
    int top_index;

    stack(int max_elements);
    ~stack();

    bool is_empty() const;
    bool is_full() const;
    void push(const T &element);
    T& top() const;
    void pop();
};

#endif 