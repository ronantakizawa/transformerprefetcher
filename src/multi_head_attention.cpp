#include "transformer_prefetcher.hpp"
#include <filesystem>
#include <cmath>
#include <limits>

MultiHeadAttention::MultiHeadAttention()
    : wq_(EMBEDDING_DIM, EMBEDDING_DIM),
      wk_(EMBEDDING_DIM, EMBEDDING_DIM),
      wv_(EMBEDDING_DIM, EMBEDDING_DIM),
      wo_(EMBEDDING_DIM, EMBEDDING_DIM) {}

std::vector<float> MultiHeadAttention::forward(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    bool use_causal_mask) {
    
    // Pre-allocate head outputs to avoid race conditions
    std::vector<std::vector<float>> head_outputs(NUM_HEADS, std::vector<float>(HEAD_DIM, 0.0f));
    
    // Process each head sequentially to avoid memory issues
    for (int h = 0; h < NUM_HEADS; ++h) {
        int head_start = h * HEAD_DIM;
        int head_end = head_start + HEAD_DIM;
        
        // Create views for this head's queries, keys, and values
        std::vector<float> q_head(query.begin() + head_start, query.begin() + head_end);
        std::vector<float> k_head(key.begin() + head_start, key.begin() + head_end);
        std::vector<float> v_head(value.begin() + head_start, value.begin() + head_end);
        
        // Project inputs for this head
        q_head = project(q_head, wq_);
        k_head = project(k_head, wk_);
        v_head = project(v_head, wv_);
        
        // Compute attention and store result
        head_outputs[h] = compute_attention(q_head, k_head, v_head, use_causal_mask);
    }
    
    // Concatenate head outputs safely
    std::vector<float> concat_output(EMBEDDING_DIM, 0.0f);
    for (int h = 0; h < NUM_HEADS; ++h) {
        std::copy(head_outputs[h].begin(), 
                 head_outputs[h].end(), 
                 concat_output.begin() + h * HEAD_DIM);
    }
    
    // Final projection
    return project(concat_output, wo_);
}

std::vector<float> MultiHeadAttention::compute_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    bool use_causal_mask) {
    
    const int seq_len = q.size() / HEAD_DIM;
    std::vector<float> attention_scores(seq_len * seq_len, 0.0f);
    
    // Compute attention scores
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (use_causal_mask && j > i) continue;
            
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += q[i * HEAD_DIM + d] * k[j * HEAD_DIM + d];
            }
            attention_scores[i * seq_len + j] = score / std::sqrt(float(HEAD_DIM));
        }
    }
    
    // Apply softmax row by row
    for (int i = 0; i < seq_len; ++i) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
            if (!use_causal_mask || j <= i) {
                max_score = std::max(max_score, attention_scores[i * seq_len + j]);
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            if (!use_causal_mask || j <= i) {
                attention_scores[i * seq_len + j] = 
                    std::exp(attention_scores[i * seq_len + j] - max_score);
                sum += attention_scores[i * seq_len + j];
            }
        }
        
        for (int j = 0; j < seq_len; ++j) {
            if (!use_causal_mask || j <= i) {
                attention_scores[i * seq_len + j] /= sum;
            } else {
                attention_scores[i * seq_len + j] = 0.0f;
            }
        }
    }
    
    // Compute weighted values
    std::vector<float> output(HEAD_DIM, 0.0f);
    for (int d = 0; d < HEAD_DIM; ++d) {
        for (int i = 0; i < seq_len; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                sum += attention_scores[i * seq_len + j] * v[j * HEAD_DIM + d];
            }
            output[d] = sum;
        }
    }
    
    return output;
}

std::vector<float> MultiHeadAttention::project(
    const std::vector<float>& input,
    const Matrix& weight) {
    
    std::vector<float> output(weight.rows_, 0.0f);
    for (int i = 0; i < weight.rows_; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < weight.cols_; ++j) {
            sum += input[j] * weight(i, j);
        }
        output[i] = sum;
    }
    return output;
}

void MultiHeadAttention::load_weights(const std::string& weight_dir, const std::string& prefix) {
    namespace fs = std::filesystem;
    fs::path dir(weight_dir);
    
    wq_.load_from_file((dir / (prefix + "attention_wq.bin")).string());
    wk_.load_from_file((dir / (prefix + "attention_wk.bin")).string());
    wv_.load_from_file((dir / (prefix + "attention_wv.bin")).string());
    wo_.load_from_file((dir / (prefix + "attention_wo.bin")).string());
}