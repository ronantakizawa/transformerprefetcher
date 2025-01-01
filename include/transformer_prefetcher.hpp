#pragma once

#include "matrix.hpp"
#include <vector>
#include <memory>
#include <array>
#include <string>
#include <cstdint>

// Architecture constants
constexpr int SEQUENCE_LENGTH = 8;
constexpr int EMBEDDING_DIM = 32;
constexpr int NUM_SEGMENTS = 4;
constexpr int NUM_HEADS = 4;
constexpr int HEAD_DIM = EMBEDDING_DIM / NUM_HEADS;
constexpr int FF_DIM = EMBEDDING_DIM * 4;
constexpr int DELTA_BITMAP_SIZE = 64;
constexpr int NUM_ENCODER_LAYERS = 3;  // Number of transformer encoder layers

class PositionalEncoding {
public:
    PositionalEncoding();
    std::vector<float> encode(int position) const;
    void initialize();

private:
    Matrix encodings_;  // Pre-computed positional encodings
};

class LayerNorm {
public:
    LayerNorm(int size);
    std::vector<float> forward(const std::vector<float>& input);
    void load_weights(const std::string& gamma_file, const std::string& beta_file);

private:
    std::vector<float> gamma_;
    std::vector<float> beta_;
};

class MultiHeadAttention {
public:
    MultiHeadAttention();
    std::vector<float> forward(const std::vector<float>& query,
                             const std::vector<float>& key,
                             const std::vector<float>& value,
                             bool use_causal_mask = false);
    void load_weights(const std::string& weight_dir, const std::string& prefix = "");

private:
    Matrix wq_, wk_, wv_, wo_;
    std::vector<float> project(const std::vector<float>& input, const Matrix& weight);
    std::vector<float> compute_attention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       bool use_causal_mask);
};

class FeedForward {
public:
    FeedForward();
    std::vector<float> forward(const std::vector<float>& input);
    void load_weights(const std::string& weight_dir);

private:
    Matrix w1_, w2_;
};

class TransformerEncoderLayer {
public:
    TransformerEncoderLayer();
    std::vector<float> forward(const std::vector<float>& input,
                             const std::vector<float>& cross_input = std::vector<float>());
    void load_weights(const std::string& weight_dir, int layer_index);

private:
    MultiHeadAttention self_attention_;
    MultiHeadAttention cross_attention_;  // For PC-address cross attention
    FeedForward ff_network_;
    LayerNorm norm1_, norm2_, norm3_;
    
    std::vector<float> add_residual(const std::vector<float>& input,
                                   const std::vector<float>& residual);
};

class TransformerPrefetcher {
public:
    TransformerPrefetcher();
    
    std::vector<uint64_t> predict(uint64_t current_addr, uint64_t current_pc);
    void load_weights(const std::string& weight_dir);
    void reset();

private:
    std::vector<std::unique_ptr<TransformerEncoderLayer>> encoder_layers_;
    Matrix addr_embedding_;
    Matrix pc_embedding_;
    Matrix output_layer_;
    PositionalEncoding pos_encoder_;
    
    std::array<uint64_t, SEQUENCE_LENGTH> address_history;
    std::array<uint64_t, SEQUENCE_LENGTH> pc_history;
    int history_ptr;

    static std::vector<float> segment_address(uint64_t addr);
    std::vector<float> project(const std::vector<float>& input, const Matrix& weight);
    std::vector<float> prepare_sequence_input(const std::array<uint64_t, SEQUENCE_LENGTH>& history,
                                            const Matrix& embedding);
};