#include "transformer_prefetcher.hpp"
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <numeric>

// LayerNorm implementation
LayerNorm::LayerNorm(int size) : gamma_(size, 1.0f), beta_(size, 0.0f) {}

std::vector<float> LayerNorm::forward(const std::vector<float>& input) {
    const float eps = 1e-5f;
    std::vector<float> output(input.size());
    
    // Calculate mean
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    
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

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention()
    : wq_(EMBEDDING_DIM, EMBEDDING_DIM),
      wk_(EMBEDDING_DIM, EMBEDDING_DIM),
      wv_(EMBEDDING_DIM, EMBEDDING_DIM),
      wo_(EMBEDDING_DIM, EMBEDDING_DIM) {}

std::vector<float> MultiHeadAttention::project(const std::vector<float>& input, 
                                             const Matrix& weight) {
    std::vector<float> output(weight.rows_, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < weight.rows_; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < weight.cols_; ++j) {
            sum += input[j] * weight(i, j);
        }
        output[i] = sum;
    }
    return output;
}

std::vector<float> MultiHeadAttention::compute_attention(const std::vector<float>& q,
                                                       const std::vector<float>& k,
                                                       const std::vector<float>& v) {
    const int seq_len = q.size() / HEAD_DIM;
    std::vector<float> output(seq_len * HEAD_DIM, 0.0f);
    std::vector<float> attention_scores(seq_len * seq_len, 0.0f);
    
    // Compute attention scores
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += q[i * HEAD_DIM + d] * k[j * HEAD_DIM + d];
            }
            attention_scores[i * seq_len + j] = score / std::sqrt(float(HEAD_DIM));
        }
    }
    
    // Apply softmax row-wise
    #pragma omp parallel for
    for (int i = 0; i < seq_len; ++i) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
            max_score = std::max(max_score, attention_scores[i * seq_len + j]);
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            attention_scores[i * seq_len + j] = std::exp(attention_scores[i * seq_len + j] - max_score);
            sum += attention_scores[i * seq_len + j];
        }
        
        for (int j = 0; j < seq_len; ++j) {
            attention_scores[i * seq_len + j] /= sum;
        }
    }
    
    // Compute weighted values
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                sum += attention_scores[i * seq_len + j] * v[j * HEAD_DIM + d];
            }
            output[i * HEAD_DIM + d] = sum;
        }
    }
    
    return output;
}

std::vector<float> MultiHeadAttention::forward(const std::vector<float>& query,
                                             const std::vector<float>& key,
                                             const std::vector<float>& value) {
    std::vector<float> output(EMBEDDING_DIM, 0.0f);
    std::vector<std::vector<float>> head_outputs(NUM_HEADS);
    
    #pragma omp parallel for
    for (int h = 0; h < NUM_HEADS; ++h) {
        // Project inputs for this head
        auto q = project(query, wq_);
        auto k = project(key, wk_);
        auto v = project(value, wv_);
        
        // Compute attention
        head_outputs[h] = compute_attention(q, k, v);
    }
    
    // Concatenate head outputs
    std::vector<float> concat_output(EMBEDDING_DIM, 0.0f);
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < HEAD_DIM; ++i) {
            concat_output[h * HEAD_DIM + i] = head_outputs[h][i];
        }
    }
    
    // Final projection
    return project(concat_output, wo_);
}

void MultiHeadAttention::load_weights(const std::string& weight_dir) {
    namespace fs = std::filesystem;
    fs::path dir(weight_dir);
    
    wq_.load_from_file((dir / "attention_wq.bin").string());
    wk_.load_from_file((dir / "attention_wk.bin").string());
    wv_.load_from_file((dir / "attention_wv.bin").string());
    wo_.load_from_file((dir / "attention_wo.bin").string());
}

// FeedForward implementation
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

// TransformerEncoderLayer implementation
TransformerEncoderLayer::TransformerEncoderLayer()
    : norm1_(EMBEDDING_DIM), norm2_(EMBEDDING_DIM) {}

std::vector<float> TransformerEncoderLayer::forward(const std::vector<float>& input) {
    // Self attention
    auto attention_output = attention_.forward(input, input, input);
    
    // Add & normalize
    std::vector<float> post_attention(EMBEDDING_DIM);
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        post_attention[i] = input[i] + attention_output[i];
    }
    post_attention = norm1_.forward(post_attention);
    
    // Feed forward
    auto ff_output = ff_network_.forward(post_attention);
    
    // Add & normalize
    std::vector<float> output(EMBEDDING_DIM);
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        output[i] = post_attention[i] + ff_output[i];
    }
    return norm2_.forward(output);
}

void TransformerEncoderLayer::load_weights(const std::string& weight_dir) {
    namespace fs = std::filesystem;
    fs::path dir(weight_dir);
    
    attention_.load_weights(weight_dir);
    ff_network_.load_weights(weight_dir);
    norm1_.load_weights((dir / "ln1_gamma.bin").string(),
                       (dir / "ln1_beta.bin").string());
    norm2_.load_weights((dir / "ln2_gamma.bin").string(),
                       (dir / "ln2_beta.bin").string());
}

// TransformerPrefetcher implementation
TransformerPrefetcher::TransformerPrefetcher()
    : embedding_(EMBEDDING_DIM, NUM_SEGMENTS * 2 * SEQUENCE_LENGTH),
      output_layer_(DELTA_BITMAP_SIZE, EMBEDDING_DIM),
      history_ptr(0) {
    address_history.fill(0);
    pc_history.fill(0);
}

namespace {
    struct PredictionCandidate {
        uint64_t addr;
        float confidence;
        float locality_score;
        bool is_stride_multiple;
        
        float total_score() const {
            return confidence + locality_score + (is_stride_multiple ? 0.5f : 0.0f);
        }
    };
}

std::vector<uint64_t> TransformerPrefetcher::predict(uint64_t current_addr, 
                                                    uint64_t current_pc) {
    std::cout << "Debug: Starting prediction for addr: 0x" << std::hex << current_addr << std::dec << std::endl;
    
    // Update history
    address_history[history_ptr] = current_addr;
    pc_history[history_ptr] = current_pc;
    history_ptr = (history_ptr + 1) % SEQUENCE_LENGTH;
    
    std::cout << "Debug: Updated history" << std::endl;
    
    // Prepare input features
    std::vector<float> input_features;
    input_features.reserve(NUM_SEGMENTS * 2 * SEQUENCE_LENGTH);
    
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        auto addr_segments = segment_address(address_history[i]);
        auto pc_segments = segment_address(pc_history[i]);
        input_features.insert(input_features.end(), addr_segments.begin(), addr_segments.end());
        input_features.insert(input_features.end(), pc_segments.begin(), pc_segments.end());
    }
    
    std::cout << "Debug: Prepared input features" << std::endl;
    
    // Forward through transformer
    auto embedded = project(input_features, embedding_);
    std::cout << "Debug: Embedded input" << std::endl;
    
    auto encoded = encoder_.forward(embedded);
    std::cout << "Debug: Encoded features" << std::endl;
    
    auto logits = project(encoded, output_layer_);
    std::cout << "Debug: Generated logits" << std::endl;
    
    // Initialize predictions
    std::vector<uint64_t> predictions;
    predictions.reserve(8);

    // First add base stride predictions
    for (int i = 1; i <= 3; i++) {
        predictions.push_back(current_addr + (i * 8));
    }

    std::cout << "Debug: Added base stride predictions" << std::endl;

    // Find high-confidence predictions
    struct ScoredPrediction {
        uint64_t addr;
        float score;
    };
    std::vector<ScoredPrediction> candidates;
    candidates.reserve(DELTA_BITMAP_SIZE);

    for (int i = 1; i < DELTA_BITMAP_SIZE; i++) {
        if (logits[i] < 0.3f) continue;
        
        uint64_t delta = i * 8;
        uint64_t addr = current_addr + delta;
        
        // Skip if already predicted
        bool already_predicted = false;
        for (uint64_t pred : predictions) {
            if (pred == addr) {
                already_predicted = true;
                break;
            }
        }
        if (already_predicted) continue;

        float score = logits[i];
        // Add locality bonus
        if (delta <= 64) {  // Near predictions
            score += 0.3f;
        } else if (delta <= 256) {  // Far predictions
            score += 0.1f;
        }
        
        candidates.push_back({addr, score});
    }

    std::cout << "Debug: Processed candidates" << std::endl;

    // Sort candidates by score
    std::sort(candidates.begin(), candidates.end(),
              [](const ScoredPrediction& a, const ScoredPrediction& b) {
                  return a.score > b.score;
              });

    // Add top candidates
    int added_predictions = 0;
    for (const auto& candidate : candidates) {
        if (predictions.size() >= 8) break;
        predictions.push_back(candidate.addr);
        added_predictions++;
        if (added_predictions >= 5) break;  // Limit additional predictions
    }

    std::cout << "Debug: Added candidates" << std::endl;

    // Fill any remaining slots
    int stride_mult = 4;  // Start after our base predictions
    while (predictions.size() < 8) {
        predictions.push_back(current_addr + (stride_mult * 8));
        stride_mult++;
    }

    std::cout << "Debug: Completed predictions" << std::endl;
    
    return predictions;
}

void TransformerPrefetcher::load_weights(const std::string& weight_dir) {
    namespace fs = std::filesystem;
    fs::path dir(weight_dir);
    
    embedding_.load_from_file((dir / "embedding.bin").string());
    encoder_.load_weights(weight_dir);
    output_layer_.load_from_file((dir / "output_weight.bin").string());
}

void TransformerPrefetcher::reset() {
    history_ptr = 0;
    address_history.fill(0);
    pc_history.fill(0);
}

std::vector<float> TransformerPrefetcher::segment_address(uint64_t addr) {
    std::vector<float> segments(NUM_SEGMENTS);
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        segments[i] = static_cast<float>((addr >> (i * 16)) & 0xFFFF) / 65535.0f;
    }
    return segments;
}

std::vector<float> TransformerPrefetcher::project(const std::vector<float>& input, 
                                                 const Matrix& weight) {
    std::vector<float> output(weight.rows_);
    #pragma omp parallel for
    for (int i = 0; i < weight.rows_; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < weight.cols_; ++j) {
            sum += input[j] * weight(i, j);
        }
        output[i] = sum;
    }
    return output;
}