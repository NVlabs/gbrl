#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "utils.h"
#include "config.h"


std::string VectoString(const float* vec, const int vec_size){
    std::ostringstream oss;
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
    int i = binaryPath.size() - 1;
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
