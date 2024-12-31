#include "matrix.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>

// Initialize static member
std::mt19937 Matrix::gen_{std::random_device{}()};

Matrix::Matrix(int rows, int cols) 
    : rows_(rows), cols_(cols), data_(rows * cols) {}

Matrix::Matrix(int rows, int cols, float init_val) 
    : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

float& Matrix::operator()(int i, int j) {
    return data_[i * cols_ + j];
}

const float& Matrix::operator()(int i, int j) const {
    return data_[i * cols_ + j];
}

void Matrix::xavier_init() {
    float limit = std::sqrt(6.0f / (rows_ + cols_));
    std::uniform_real_distribution<float> dist(-limit, limit);
    
    for (auto& val : data_) {
        val = dist(gen_);
    }
}

void Matrix::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }
    
    file.write(reinterpret_cast<const char*>(data_.data()), 
               data_.size() * sizeof(float));
}

void Matrix::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(data_.data()), 
              data_.size() * sizeof(float));
    
    if (file.gcount() != static_cast<std::streamsize>(data_.size() * sizeof(float))) {
        throw std::runtime_error("File size mismatch when loading weights");
    }
}