#pragma once
#include <npf/types.hpp>
#include <npf/layer.hpp>
#include <deque>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace npf {

class NeuralPrefetcher {
public:
    // Constructor with configurable parameters
    NeuralPrefetcher(size_t history_size = 8,
                     size_t hidden_size = 128,
                     size_t batch_size = 32,
                     float learning_rate = 0.001f)
        : history_size_(history_size),
          batch_size_(batch_size),
          learning_rate_(learning_rate),
          input_size_(history_size * (8 + 4)),  // 8 bits per delta + 4 pattern features
          input_layer_(input_size_, hidden_size),
          hidden_layer_(hidden_size, hidden_size/2),
          output_layer_(hidden_size/2, 8),
          stats_({0, 0, 0, 0.0f, 0.0f}),
          adaptive_lr_(learning_rate),
          stop_training_(false) {
        
        pattern_features_.resize(4, 0.0f);
        training_thread_ = std::thread(&NeuralPrefetcher::training_loop, this);
    }
    
    // Destructor ensures training thread is properly stopped
    ~NeuralPrefetcher() {
        stop_training_ = true;
        training_cv_.notify_one();
        if (training_thread_.joinable()) {
            training_thread_.join();
        }
    }

    // Record a new memory access
    void record_access(uint64_t address, uint64_t timestamp) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Update prediction accuracy
        check_prediction(address);
        
        // Calculate and store delta
        if (!access_history_.empty()) {
            int64_t delta = static_cast<int64_t>(address - access_history_.back().address);
            access_deltas_.push_back(delta);
            update_pattern_features(delta);
            
            if (access_deltas_.size() > history_size_) {
                access_deltas_.pop_front();
            }
        }
        
        // Update access history
        access_history_.push_back({address, timestamp, false});
        if (access_history_.size() > history_size_) {
            access_history_.pop_front();
        }
        
        // Generate training sample if we have enough history
        if (access_deltas_.size() >= history_size_) {
            std::vector<float> input = encode_input();
            if (!access_deltas_.empty()) {
                std::vector<float> target = encode_delta(access_deltas_.back());
                training_samples_.push_back({input, target});
                
                if (training_samples_.size() >= batch_size_) {
                    training_cv_.notify_one();
                }
            }
        }
    }

    // Predict next N memory addresses
    std::vector<uint64_t> predict_next_addresses(size_t num_predictions = 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (access_history_.size() < history_size_ || access_deltas_.empty()) {
            return {};
        }
        
        std::vector<uint64_t> predictions;
        predictions.reserve(num_predictions);
        
        auto pattern = detect_current_pattern();
        auto stride = calculate_adaptive_stride();
        
        std::deque<int64_t> prediction_deltas = access_deltas_;
        uint64_t last_address = access_history_.back().address;
        
        for (size_t i = 0; i < num_predictions; ++i) {
            // Neural network prediction
            std::vector<float> input = encode_input();
            std::vector<float> hidden1 = input_layer_.forward(input);
            std::vector<float> hidden2 = hidden_layer_.forward(hidden1);
            std::vector<float> output = output_layer_.forward(hidden2);
            
            int64_t predicted_delta = decode_delta(output);
            
            // Blend prediction with pattern-based stride
            if (pattern.confidence > 0.8f) {
                predicted_delta = pattern.stride;
            } else {
                predicted_delta = (predicted_delta + stride) / 2;
            }
            
            uint64_t predicted_address = last_address + predicted_delta;
            predictions.push_back(predicted_address);
            
            // Update state for next prediction
            prediction_deltas.push_back(predicted_delta);
            prediction_deltas.pop_front();
            last_address = predicted_address;
        }
        
        recent_predictions_ = predictions;
        stats_.total_predictions += predictions.size();
        
        return predictions;
    }

    // Get current statistics
    PrefetchStats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    // Pattern detection structure
    struct Pattern {
        int64_t stride;
        float confidence;
        bool is_regular;
    };

    // Member variables
    size_t history_size_;
    size_t batch_size_;
    float learning_rate_;
    size_t input_size_;
    
    Layer input_layer_;
    Layer hidden_layer_;
    Layer output_layer_;
    
    mutable std::mutex mutex_;
    std::condition_variable training_cv_;
    std::thread training_thread_;
    std::atomic<bool> stop_training_;
    
    std::deque<MemoryAccess> access_history_;
    std::deque<int64_t> access_deltas_;
    std::vector<std::pair<std::vector<float>, std::vector<float>>> training_samples_;
    std::vector<uint64_t> recent_predictions_;
    std::vector<float> pattern_features_;
    
    PrefetchStats stats_;
    float adaptive_lr_;
    
    static constexpr size_t MAX_RECENT_PREDICTIONS = 1000;
    static constexpr uint64_t PREDICTION_THRESHOLD = 64;

    // Detect current access pattern
    Pattern detect_current_pattern() const {
        if (access_deltas_.size() < 3) {
            return {0, 0.0f, false};
        }

        // Calculate differences between consecutive deltas
        std::vector<int64_t> delta_diffs;
        for (size_t i = 1; i < access_deltas_.size(); ++i) {
            delta_diffs.push_back(access_deltas_[i] - access_deltas_[i-1]);
        }

        // Check if stride is regular
        bool is_regular = std::adjacent_find(delta_diffs.begin(), delta_diffs.end(),
            std::not_equal_to<>()) == delta_diffs.end();

        // Calculate average stride
        int64_t stride = std::accumulate(access_deltas_.begin(), access_deltas_.end(), 0LL) / 
                        access_deltas_.size();

        // Calculate confidence based on variance
        float variance = 0.0f;
        for (const auto& delta : access_deltas_) {
            float diff = static_cast<float>(delta - stride);
            variance += diff * diff;
        }
        variance /= access_deltas_.size();
        
        float confidence = 1.0f / (1.0f + std::sqrt(variance));

        return {stride, confidence, is_regular};
    }

    // Calculate adaptive stride for prediction
    int64_t calculate_adaptive_stride() const {
        if (access_deltas_.empty()) return 0;

        auto pattern = detect_current_pattern();
        if (pattern.confidence > 0.9f) {
            return pattern.stride;
        }

        // Use weighted moving average for irregular patterns
        int64_t weighted_stride = 0;
        float weight_sum = 0.0f;
        
        for (size_t i = 0; i < access_deltas_.size(); ++i) {
            float weight = static_cast<float>(i + 1) / access_deltas_.size();
            weighted_stride += static_cast<int64_t>(weight * access_deltas_[i]);
            weight_sum += weight;
        }

        return static_cast<int64_t>(weighted_stride / weight_sum);
    }

    // Update pattern detection features
    void update_pattern_features(int64_t new_delta) {
        auto pattern = detect_current_pattern();
        
        pattern_features_[0] = pattern.confidence;
        pattern_features_[1] = pattern.is_regular ? 1.0f : 0.0f;
        pattern_features_[2] = std::abs(static_cast<float>(new_delta - pattern.stride)) / 
                             std::max(std::abs(static_cast<float>(pattern.stride)), 1.0f);
        pattern_features_[3] = static_cast<float>(access_deltas_.size()) / history_size_;
    }

    // Check if a memory access was correctly predicted
    void check_prediction(uint64_t actual_address) {
        if (recent_predictions_.empty()) {
            return;
        }

        auto it = std::find_if(
            recent_predictions_.begin(),
            recent_predictions_.end(),
            [actual_address](uint64_t predicted) {
                return std::abs(static_cast<int64_t>(predicted - actual_address)) <= PREDICTION_THRESHOLD;
            }
        );

        if (it != recent_predictions_.end()) {
            stats_.correct_predictions++;
            recent_predictions_.erase(recent_predictions_.begin(), std::next(it));
        }

        if (stats_.total_predictions > 0) {
            stats_.accuracy = static_cast<float>(stats_.correct_predictions) / stats_.total_predictions;
            size_t total_accesses = std::max(access_history_.size(), size_t(1));
            stats_.coverage = std::min(
                static_cast<float>(stats_.correct_predictions) / total_accesses,
                1.0f
            );
        }

        if (recent_predictions_.size() > MAX_RECENT_PREDICTIONS) {
            recent_predictions_.erase(
                recent_predictions_.begin(),
                recent_predictions_.begin() + (recent_predictions_.size() - MAX_RECENT_PREDICTIONS)
            );
        }
    }

    // Encode input for neural network
    std::vector<float> encode_input() const {
        std::vector<float> encoded;
        encoded.reserve(input_size_);
        
        // Encode deltas
        for (int64_t delta : access_deltas_) {
            for (int i = 0; i < 8; ++i) {
                encoded.push_back((delta >> i) & 1 ? 1.0f : 0.0f);
            }
        }
        
        // Add pattern features
        encoded.insert(encoded.end(), pattern_features_.begin(), pattern_features_.end());
        
        while (encoded.size() < input_size_) {
            encoded.push_back(0.0f);
        }
        
        return encoded;
    }

    // Encode a single delta value
    std::vector<float> encode_delta(int64_t delta) const {
        std::vector<float> encoded(8);
        for (int i = 0; i < 8; ++i) {
            encoded[i] = (delta >> i) & 1 ? 1.0f : 0.0f;
        }
        return encoded;
    }

    // Decode network output into delta value
    int64_t decode_delta(const std::vector<float>& encoded) const {
        int64_t delta = 0;
        for (int i = 0; i < 8; ++i) {
            if (encoded[i] > 0.5f) {
                delta |= (1LL << i);
            }
        }
        return delta;
    }

    // Background training loop
    void training_loop() {
        while (!stop_training_) {
            std::vector<std::pair<std::vector<float>, std::vector<float>>> batch;
            
            {
                std::unique_lock<std::mutex> lock(mutex_);
                training_cv_.wait_for(lock, 
                    std::chrono::milliseconds(100),
                    [this] { 
                        return stop_training_ || training_samples_.size() >= batch_size_;
                    }
                );
                
                if (training_samples_.size() >= batch_size_) {
                    batch.insert(
                        batch.end(),
                        training_samples_.end() - batch_size_,
                        training_samples_.end()
                    );
                    training_samples_.erase(
                        training_samples_.end() - batch_size_,
                        training_samples_.end()
                    );
                }
            }
            
            if (!batch.empty()) {
                train_batch(batch);
            }
        }
    }

    // Train on a batch of samples
    void train_batch(const std::vector<std::pair<std::vector<float>, 
                                                std::vector<float>>>& batch) {
        for (const auto& sample : batch) {
            const auto& input = sample.first;
            const auto& target = sample.second;
            
            // Forward pass
            std::vector<float> hidden1 = input_layer_.forward(input);
            std::vector<float> hidden2 = hidden_layer_.forward(hidden1);
            std::vector<float> output = output_layer_.forward(hidden2);
            
            // Compute gradients
            std::vector<float> output_gradient(output.size());
            for (size_t i = 0; i < output.size(); ++i) {
                output_gradient[i] = output[i] - target[i];
            }
            
            // Backward pass with current learning rate
            output_layer_.backward(hidden2, output, output_gradient, adaptive_lr_);
            hidden_layer_.backward(hidden1, hidden2, output_gradient, adaptive_lr_);
            input_layer_.backward(input, hidden1, output_gradient, adaptive_lr_);
        }
    }
};

} // namespace npf