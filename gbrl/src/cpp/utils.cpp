//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "utils.h"
#include "config.h"


std::string VectoString(const float* vec, const int vec_size){
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);  // Set precision to 3
    if (vec_size > 1)
        oss << "[";
    for (int i = 0; i < vec_size; ++i) {
        oss << vec[i];
        if (i < vec_size -1)
            oss << ", ";
    }
    if (vec_size > 1)
        oss << "]";
    return oss.str();
}

int binaryToDecimal(const BoolVector& binaryPath) {
    int decimal = 0;
    int i = static_cast<int>(binaryPath.size()) - 1;
    size_t j = 0;
    while (i >= 0){
        decimal += binaryPath[i] * (1 << j);
        j++;
        i--;
    }
    return decimal + (1 << binaryPath.size()) - 1;
}

void write_header(std::ofstream& file, const serializationHeader& header) {
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file.good()) {
        throw std::runtime_error("Failed to write header to file");
    }
}

serializationHeader create_header() {
    serializationHeader header;
    header.major_version = MAJOR_VERSION;
    header.minor_version = MINOR_VERSION;
    header.patch_version = PATCH_VERSION;
    return header;
}

serializationHeader read_header(std::ifstream& file) {
    serializationHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file.good()) {
        throw std::runtime_error("Failed to read header from file");
    }
    return header;
}

void display_header(serializationHeader header){
    std::cout << "Version " << header.major_version << "."
                << header.minor_version << "." << header.patch_version << std::endl;
}

template<typename T>
int count_distinct(T *arr, int n){
    /*
    Count the number of distinct elements in an unsorted array O(n^2)
    */
    int res = 1;

    // Pick all elements one by one
    for (int i = 1; i < n; i++) {
        int j = 0;
        for (j = 0; j < i; j++)
            if (arr[i] == arr[j])
                break;

        if (i == j)
            res++;
    }
    return res;
}
template int count_distinct<int>(int* arr, int n);
template int count_distinct<double>(double* arr, int n);