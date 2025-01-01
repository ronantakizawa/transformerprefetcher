#include "transformer_prefetcher.hpp"
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <iostream>
#include <algorithm>

TransformerPrefetcher::TransformerPrefetcher()
    : addr_embedding_(EMBEDDING_DIM, NUM_SEGMENTS * SEQUENCE_LENGTH),
      pc_embedding_(EMBEDDING_DIM, NUM_SEGMENTS * SEQUENCE_LENGTH),
      output_layer_(DELTA_BITMAP_SIZE, EMBEDDING_DIM),
      pos_encoder_(),
      history_ptr(0) {
    
    address_history.fill(0);
    pc_history.fill(0);
    
    // Initialize encoder layers
    encoder_layers_.reserve(NUM_ENCODER_LAYERS);
    for (int i = 0; i < NUM_ENCODER_LAYERS; ++i) {
        encoder_layers_.push_back(std::make_unique<TransformerEncoderLayer>());
    }
}

std::vector<float> TransformerPrefetcher::prepare_sequence_input(
    const std::array<uint64_t, SEQUENCE_LENGTH>& history,
    const Matrix& embedding) {
    
    std::vector<float> input_features;
    input_features.reserve(NUM_SEGMENTS * SEQUENCE_LENGTH);
    
    // Process each sequence element
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        auto segments = segment_address(history[i]);
        input_features.insert(input_features.end(), segments.begin(), segments.end());
    }
    
    // Embed and add positional encoding
    auto embedded = project(input_features, embedding);
    
    // Add positional encoding safely
    for (size_t pos = 0; pos < static_cast<size_t>(SEQUENCE_LENGTH); pos++) {
        auto pos_encoding = pos_encoder_.encode(static_cast<int>(pos));
        size_t start_idx = pos * static_cast<size_t>(EMBEDDING_DIM);
        for (size_t i = 0; i < static_cast<size_t>(EMBEDDING_DIM) && start_idx + i < embedded.size(); i++) {
            embedded[start_idx + i] += pos_encoding[i];
        }
    }
    
    return embedded;
}

std::vector<uint64_t> TransformerPrefetcher::predict(uint64_t current_addr, uint64_t current_pc) {
    // Update history
    address_history[history_ptr] = current_addr;
    pc_history[history_ptr] = current_pc;
    history_ptr = (history_ptr + 1) % SEQUENCE_LENGTH;
    
    // Prepare inputs
    auto addr_features = prepare_sequence_input(address_history, addr_embedding_);
    auto pc_features = prepare_sequence_input(pc_history, pc_embedding_);
    
    // Process through encoder layers
    auto encoded = addr_features;
    for (const auto& layer : encoder_layers_) {
        if (layer) {  // Null check
            encoded = layer->forward(encoded, pc_features);
        }
    }
    
    // Generate predictions
    auto logits = project(encoded, output_layer_);
    std::vector<uint64_t> predictions;
    predictions.reserve(8);  // Pre-allocate space
    
    // Add base stride predictions
    for (int i = 1; i <= 3; i++) {
        predictions.push_back(current_addr + (i * 8));
    }
    
    // Process candidates
    std::vector<std::pair<uint64_t, float>> candidates;
    candidates.reserve(DELTA_BITMAP_SIZE);
    
    for (int i = 1; i < DELTA_BITMAP_SIZE; i++) {
        if (logits[i] < 0.3f) continue;
        
        uint64_t delta = static_cast<uint64_t>(i) * 8;
        uint64_t addr = current_addr + delta;
        
        if (std::find(predictions.begin(), predictions.end(), addr) != predictions.end()) {
            continue;
        }
        
        float score = logits[i];
        score += (delta <= 64) ? 0.4f : (delta <= 128) ? 0.2f : (delta <= 256) ? 0.1f : 0.0f;
        score += (delta % 8 == 0) ? 0.15f : 0.0f;
        score += ((delta & (delta - 1)) == 0) ? 0.25f : 0.0f;
        
        candidates.push_back({addr, score});
    }
    
    // Sort and select top candidates
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (const auto& candidate : candidates) {
        if (predictions.size() >= 8) break;
        predictions.push_back(candidate.first);
    }
    
    // Fill remaining slots with stride predictions
    int stride_mult = 4;
    while (predictions.size() < 8) {
        predictions.push_back(current_addr + (static_cast<uint64_t>(stride_mult) * 8));
        stride_mult++;
    }
    
    return predictions;
}

void TransformerPrefetcher::load_weights(const std::string& weight_dir) {
    namespace fs = std::filesystem;
    fs::path dir(weight_dir);
    
    try {
        addr_embedding_.load_from_file((dir / "addr_embedding.bin").string());
        pc_embedding_.load_from_file((dir / "pc_embedding.bin").string());
        output_layer_.load_from_file((dir / "output_weight.bin").string());
        
        for (size_t i = 0; i < static_cast<size_t>(NUM_ENCODER_LAYERS) && i < encoder_layers_.size(); ++i) {
            if (encoder_layers_[i]) {
                encoder_layers_[i]->load_weights(weight_dir, static_cast<int>(i));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        throw;
    }
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

std::vector<float> TransformerPrefetcher::project(
    const std::vector<float>& input,
    const Matrix& weight) {
    
    std::vector<float> output(weight.rows_, 0.0f);
    for (int i = 0; i < weight.rows_; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < static_cast<size_t>(weight.cols_) && j < input.size(); ++j) {
            sum += input[j] * weight(i, static_cast<int>(j));
        }
        output[i] = sum;
    }
    return output;
}