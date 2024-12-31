#include "transformer_prefetcher.hpp"
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstdlib>  // For llabs
#include <unordered_set>

// LayerNorm implementation
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

// MultiHeadAttention implementation
MultiHeadAttention::MultiHeadAttention()
    : wq_(EMBEDDING_DIM, EMBEDDING_DIM),
      wk_(EMBEDDING_DIM, EMBEDDING_DIM),
      wv_(EMBEDDING_DIM, EMBEDDING_DIM),
      wo_(EMBEDDING_DIM, EMBEDDING_DIM) {}

std::vector<float> MultiHeadAttention::project(const std::vector<float>& input, 
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

std::vector<float> MultiHeadAttention::compute_attention(const std::vector<float>& q,
                                                       const std::vector<float>& k,
                                                       const std::vector<float>& v) {
    const int seq_len = q.size() / HEAD_DIM;
    std::vector<float> output(seq_len * HEAD_DIM);
    
    // Compute attention scores
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += q[i * HEAD_DIM + d] * k[j * HEAD_DIM + d];
            }
            score /= std::sqrt(HEAD_DIM);
            
            // Apply softmax
            float max_score = -std::numeric_limits<float>::infinity();
            for (int k = 0; k < seq_len; ++k) {
                max_score = std::max(max_score, score);
            }
            
            float sum = 0.0f;
            std::vector<float> exp_scores(seq_len);
            for (int k = 0; k < seq_len; ++k) {
                exp_scores[k] = std::exp(score - max_score);
                sum += exp_scores[k];
            }
            
            // Compute weighted sum
            for (int d = 0; d < HEAD_DIM; ++d) {
                float weighted_sum = 0.0f;
                for (int k = 0; k < seq_len; ++k) {
                    weighted_sum += (exp_scores[k] / sum) * v[k * HEAD_DIM + d];
                }
                output[i * HEAD_DIM + d] = weighted_sum;
            }
        }
    }
    
    return output;
}

std::vector<float> MultiHeadAttention::forward(const std::vector<float>& query,
                                             const std::vector<float>& key,
                                             const std::vector<float>& value) {
    std::vector<float> output(EMBEDDING_DIM);
    
    // Split into heads and compute attention
    #pragma omp parallel for
    for (int h = 0; h < NUM_HEADS; ++h) {
        // Project q, k, v for this head
        auto q = project(query, wq_);
        auto k = project(key, wk_);
        auto v = project(value, wv_);
        
        // Compute attention
        auto head_output = compute_attention(q, k, v);
        
        // Add to output
        for (int i = 0; i < HEAD_DIM; ++i) {
            output[h * HEAD_DIM + i] = head_output[i];
        }
    }
    
    // Final projection
    return project(output, wo_);
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
    // First layer with ReLU
    std::vector<float> hidden(FF_DIM);
    #pragma omp parallel for
    for (int i = 0; i < FF_DIM; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < EMBEDDING_DIM; ++j) {
            sum += input[j] * w1_(i, j);
        }
        hidden[i] = std::max(0.0f, sum);  // ReLU
    }
    
    // Second layer
    std::vector<float> output(EMBEDDING_DIM);
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
    auto attended = attention_.forward(input, input, input);
    
    // Add & norm
    std::vector<float> post_attention(EMBEDDING_DIM);
    for (int i = 0; i < EMBEDDING_DIM; ++i) {
        post_attention[i] = input[i] + attended[i];
    }
    post_attention = norm1_.forward(post_attention);
    
    // Feed forward
    auto ff_output = ff_network_.forward(post_attention);
    
    // Add & norm
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

std::vector<uint64_t> TransformerPrefetcher::predict(uint64_t current_addr, 
                                                    uint64_t current_pc) {
    // Update history
    address_history[history_ptr] = current_addr;
    pc_history[history_ptr] = current_pc;
    history_ptr = (history_ptr + 1) % SEQUENCE_LENGTH;
    
    // Prepare input sequence
    std::vector<float> input_features;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        auto addr_segments = segment_address(address_history[i]);
        auto pc_segments = segment_address(pc_history[i]);
        input_features.insert(input_features.end(), addr_segments.begin(), addr_segments.end());
        input_features.insert(input_features.end(), pc_segments.begin(), pc_segments.end());
    }
    
    // Forward through network
    auto embedded = project(input_features, embedding_);
    auto encoded = encoder_.forward(embedded);
    auto logits = project(encoded, output_layer_);
    
    // Convert logits to predictions with stride awareness
    std::vector<uint64_t> predictions;
    predictions.reserve(8);

    // Always predict the next few stride addresses first
    for (int i = 1; i <= 4; i++) {
        predictions.push_back(current_addr + (i * 8));
    }

    // Track addresses we've already predicted
    std::unordered_set<uint64_t> predicted_addrs(predictions.begin(), predictions.end());

    struct Prediction {
        uint64_t addr;
        float confidence;
        int64_t distance;  // Distance from current address
    };
    std::vector<Prediction> other_predictions;

    // Consider additional predictions near the current stride pattern
    for (int i = 1; i < DELTA_BITMAP_SIZE; i++) {
        if (logits[i] > 0.3f) {
            uint64_t delta = i * 8;
            uint64_t pred_addr = current_addr + delta;
            
            if (pred_addr == 0 || predicted_addrs.count(pred_addr) > 0) {
                continue;
            }

            // Calculate how far this prediction is from the nearest stride multiple
            int64_t nearest_stride = ((delta + 4) / 8) * 8;
            int64_t distance = std::llabs(delta - nearest_stride);
            
            // Add extra confidence for addresses that are close to stride multiples
            float adjusted_confidence = logits[i];
            if (distance <= 8) {
                adjusted_confidence += 0.3f * (1.0f - float(distance) / 8.0f);
            }
            
            other_predictions.push_back({
                pred_addr,
                adjusted_confidence,
                distance
            });
        }
    }

    // Sort additional predictions by confidence and distance to stride pattern
    std::sort(other_predictions.begin(), other_predictions.end(),
              [](const Prediction& a, const Prediction& b) {
                  if (std::abs(a.confidence - b.confidence) < 0.1f) {
                      return a.distance < b.distance;
                  }
                  return a.confidence > b.confidence;
              });

    // Add remaining predictions, prioritizing those close to stride pattern
    for (const auto& pred : other_predictions) {
        if (predictions.size() >= 8) break;
        predictions.push_back(pred.addr);
    }

    // Fill any remaining slots with more stride predictions
    while (predictions.size() < 8) {
        uint64_t next_addr = current_addr + (predictions.size() + 1) * 8;
        predictions.push_back(next_addr);
    }
    
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