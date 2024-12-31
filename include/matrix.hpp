#pragma once

#include <vector>
#include <string>
#include <random>
#include <cstdint>

class Matrix {
public:
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, float init_val);
    
    float& operator()(int i, int j);
    const float& operator()(int i, int j) const;
    
    void xavier_init();
    void save_to_file(const std::string& filename) const;
    void load_from_file(const std::string& filename);
    
    std::vector<float> data_;
    int rows_, cols_;

private:
    static std::mt19937 gen_;
};