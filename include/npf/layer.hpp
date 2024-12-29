// include/npf/layer.hpp
#pragma once
#include <vector>
#include <random>
#include <cmath>

namespace npf {

class Layer {
public:
    Layer(size_t input_size, size_t output_size) 
        : weights(input_size, std::vector<float>(output_size)),
          biases(output_size),
          input_size(input_size),
          output_size(output_size) {
        
        // Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = std::sqrt(2.0f / (input_size + output_size));
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (auto& row : weights) {
            for (float& weight : row) {
                weight = dist(gen);
            }
        }
        
        for (float& bias : biases) {
            bias = dist(gen);
        }
    }
    
    std::vector<float> forward(const std::vector<float>& input) const {
        std::vector<float> output(output_size);
        
        for (size_t j = 0; j < output_size; ++j) {
            float sum = biases[j];
            for (size_t i = 0; i < input_size; ++i) {
                sum += input[i] * weights[i][j];
            }
            output[j] = relu(sum);
        }
        
        return output;
    }
    
    void backward(const std::vector<float>& input,
                 const std::vector<float>& output,
                 const std::vector<float>& output_gradient,
                 float learning_rate) {
        
        for (size_t j = 0; j < output_size; ++j) {
            float grad = output_gradient[j] * relu_derivative(output[j]);
            biases[j] -= learning_rate * grad;
            
            for (size_t i = 0; i < input_size; ++i) {
                weights[i][j] -= learning_rate * input[i] * grad;
            }
        }
    }
    
    size_t get_input_size() const { return input_size; }
    size_t get_output_size() const { return output_size; }
    
private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    size_t input_size;
    size_t output_size;
    
    static float relu(float x) {
        return std::max(0.0f, x);
    }
    
    static float relu_derivative(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }
};

} // namespace npf