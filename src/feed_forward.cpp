#include "transformer_prefetcher.hpp"
#include <filesystem>

FeedForward::FeedForward()
    : w1_(FF_DIM, EMBEDDING_DIM),
      w2_(EMBEDDING_DIM, FF_DIM) {}

std::vector<float> FeedForward::forward(const std::vector<float>& input) {
    // First layer with ReLU activation
    std::vector<float> hidden(FF_DIM, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < FF_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < EMBEDDING_DIM; ++j) {
            sum += input[j] * w1_(i, j);
        }
        hidden[i] = std::max(0.0f, sum);  // ReLU activation
    }
    
    // Second layer
    std::vector<float> output(EMBEDDING_DIM, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < FF_DIM; ++j) {
            sum += hidden[j] * w2_(i, j);
        }
        output[i] = sum;
    }
    
    return output;
}

void FeedForward::load_weights(const std::string& weight_dir) {
    namespace fs = std::filesystem;
    fs::path dir(weight_dir);
    w1_.load_from_file((dir / "ff_w1.bin").string());
    w2_.load_from_file((dir / "ff_w2.bin").string());
}