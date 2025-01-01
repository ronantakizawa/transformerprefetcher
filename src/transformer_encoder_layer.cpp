#include "transformer_prefetcher.hpp"
#include <filesystem>

TransformerEncoderLayer::TransformerEncoderLayer()
    : norm1_(EMBEDDING_DIM), 
      norm2_(EMBEDDING_DIM),
      norm3_(EMBEDDING_DIM) {}

std::vector<float> TransformerEncoderLayer::add_residual(
    const std::vector<float>& input,
    const std::vector<float>& residual) {
    
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] + residual[i];
    }
    return output;
}

std::vector<float> TransformerEncoderLayer::forward(
    const std::vector<float>& input,
    const std::vector<float>& cross_input) {
    
    // Self attention
    auto self_attention_output = self_attention_.forward(input, input, input, false);
    auto post_attention = norm1_.forward(add_residual(input, self_attention_output));
    
    // Cross attention (if cross_input provided)
    if (!cross_input.empty()) {
        auto cross_attention_output = 
            cross_attention_.forward(post_attention, cross_input, cross_input, false);
        post_attention = norm2_.forward(add_residual(post_attention, cross_attention_output));
    }
    
    // Feed forward
    auto ff_output = ff_network_.forward(post_attention);
    return norm3_.forward(add_residual(post_attention, ff_output));
}

void TransformerEncoderLayer::load_weights(const std::string& weight_dir, int layer_index) {
    namespace fs = std::filesystem;
    fs::path layer_dir = fs::path(weight_dir) / ("layer_" + std::to_string(layer_index));
    
    self_attention_.load_weights(layer_dir.string(), "self_");
    cross_attention_.load_weights(layer_dir.string(), "cross_");
    ff_network_.load_weights(layer_dir.string());
    
    norm1_.load_weights((layer_dir / "ln1_gamma.bin").string(),
                       (layer_dir / "ln1_beta.bin").string());
    norm2_.load_weights((layer_dir / "ln2_gamma.bin").string(),
                       (layer_dir / "ln2_beta.bin").string());
    norm3_.load_weights((layer_dir / "ln3_gamma.bin").string(),
                       (layer_dir / "ln3_beta.bin").string());
}