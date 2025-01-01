#include "transformer_prefetcher.hpp"
#include <cmath>

LayerNorm::LayerNorm(int size) : gamma_(size, 1.0f), beta_(size, 0.0f) {}

std::vector<float> LayerNorm::forward(const std::vector<float>& input) {
    const float eps = 1e-5f;
    std::vector<float> output(input.size());
    
    // Calculate mean
    float mean = 0.0f;
    for (float val : input) {
        mean += val;
    }
    mean /= input.size();
    
    // Calculate variance
    float var = 0.0f;
    for (float val : input) {
        float diff = val - mean;
        var += diff * diff;
    }
    var /= input.size();
    
    // Normalize
    float std_dev = std::sqrt(var + eps);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = gamma_[i] * ((input[i] - mean) / std_dev) + beta_[i];
    }
    
    return output;
}

void LayerNorm::load_weights(const std::string& gamma_file, const std::string& beta_file) {
    Matrix gamma_mat(1, gamma_.size());
    Matrix beta_mat(1, beta_.size());
    gamma_mat.load_from_file(gamma_file);
    beta_mat.load_from_file(beta_file);
    
    gamma_ = gamma_mat.data_;
    beta_ = beta_mat.data_;
}
