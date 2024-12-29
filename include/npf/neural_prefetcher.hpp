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

namespace npf {

class NeuralPrefetcher {
public:
    NeuralPrefetcher(size_t history_size = 8,
                     size_t hidden_size = 128,
                     size_t batch_size = 32,
                     float learning_rate = 0.001f)
        : history_size_(history_size),
          batch_size_(batch_size),
          learning_rate_(learning_rate),
          input_size_(history_size * (8 + 4)),
          input_layer_(input_size_, hidden_size),
          hidden_layer_(hidden_size, hidden_size/2),
          output_layer_(hidden_size/2, 8),
          stats_({0, 0, 0, 0.0f, 0.0f}),
          stop_training_(false) {
        
        pattern_features_.resize(4, 0.0f);
        training_thread_ = std::thread(&NeuralPrefetcher::training_loop, this);
    }
    
    ~NeuralPrefetcher() {
        stop_training_ = true;
        training_cv_.notify_all();
        if (training_thread_.joinable()) {
            training_thread_.join();
        }
    }
    
    void record_access(uint64_t address, uint64_t timestamp) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            return;  // Skip if can't acquire lock immediately
        }
        
        if (!access_history_.empty()) {
            int64_t delta = static_cast<int64_t>(address - access_history_.back().address);
            access_deltas_.push_back(delta);
            
            if (access_deltas_.size() > history_size_) {
                access_deltas_.pop_front();
            }
        }
        
        access_history_.push_back({address, timestamp, false});
        if (access_history_.size() > history_size_) {
            access_history_.pop_front();
        }
        
        if (access_deltas_.size() >= history_size_) {
            auto sample = create_training_sample();
            if (training_samples_.size() < batch_size_ * 2) {  // Limit buffer size
                training_samples_.push_back(std::move(sample));
                if (training_samples_.size() >= batch_size_) {
                    training_cv_.notify_one();
                }
            }
        }
    }
    
    std::vector<uint64_t> predict_next_addresses(size_t num_predictions = 1) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || access_history_.size() < history_size_ || access_deltas_.empty()) {
            return {};
        }
        
        std::vector<uint64_t> predictions;
        predictions.reserve(num_predictions);
        
        int64_t stride = calculate_stride();
        uint64_t last_address = access_history_.back().address;
        
        for (size_t i = 0; i < num_predictions; ++i) {
            uint64_t predicted_address = last_address + stride;
            predictions.push_back(predicted_address);
            last_address = predicted_address;
        }
        
        stats_.total_predictions += predictions.size();
        return predictions;
    }
    
    PrefetchStats get_stats() const {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            return PrefetchStats{};
        }
        return stats_;
    }

private:
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
    std::vector<float> pattern_features_;
    
    PrefetchStats stats_;
    
    static constexpr size_t MAX_SAMPLES = 1000;
    
    std::pair<std::vector<float>, std::vector<float>> create_training_sample() {
        std::vector<float> input;
        input.reserve(input_size_);
        
        for (const auto& delta : access_deltas_) {
            for (int i = 0; i < 8; ++i) {
                input.push_back((delta >> i) & 1 ? 1.0f : 0.0f);
            }
        }
        
        for (int i = 0; i < 4; ++i) {
            input.push_back(0.0f);  // Simplified pattern features
        }
        
        std::vector<float> target(8, 0.0f);
        if (!access_deltas_.empty()) {
            int64_t last_delta = access_deltas_.back();
            for (int i = 0; i < 8; ++i) {
                target[i] = (last_delta >> i) & 1 ? 1.0f : 0.0f;
            }
        }
        
        return {input, target};
    }
    
    int64_t calculate_stride() const {
        if (access_deltas_.empty()) return 0;
        
        int64_t sum = 0;
        size_t count = 0;
        for (auto it = access_deltas_.rbegin(); it != access_deltas_.rend() && count < 3; ++it, ++count) {
            sum += *it;
        }
        return sum / count;
    }
    
    void training_loop() {
        while (!stop_training_) {
            std::vector<std::pair<std::vector<float>, std::vector<float>>> batch;
            
            {
                std::unique_lock<std::mutex> lock(mutex_);
                auto status = training_cv_.wait_for(lock, std::chrono::milliseconds(100),
                    [this] { return stop_training_ || training_samples_.size() >= batch_size_; });
                
                if (!status || training_samples_.size() < batch_size_) {
                    continue;
                }
                
                batch.insert(batch.end(), 
                           training_samples_.begin(),
                           training_samples_.begin() + batch_size_);
                training_samples_.erase(training_samples_.begin(), 
                                      training_samples_.begin() + batch_size_);
            }
            
            if (!batch.empty()) {
                for (const auto& sample : batch) {
                    train_single_sample(sample);
                }
            }
            
            std::this_thread::yield();
        }
    }
    
    void train_single_sample(const std::pair<std::vector<float>, std::vector<float>>& sample) {
        const auto& input = sample.first;
        const auto& target = sample.second;
        
        auto hidden1 = input_layer_.forward(input);
        auto hidden2 = hidden_layer_.forward(hidden1);
        auto output = output_layer_.forward(hidden2);
        
        std::vector<float> output_gradient(output.size());
        for (size_t i = 0; i < output.size(); ++i) {
            output_gradient[i] = output[i] - target[i];
        }
        
        output_layer_.backward(hidden2, output, output_gradient, learning_rate_);
        hidden_layer_.backward(hidden1, hidden2, output_gradient, learning_rate_);
        input_layer_.backward(input, hidden1, output_gradient, learning_rate_);
    }
};

} // namespace npf