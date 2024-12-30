#include "dart_prefetcher.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>   // For std::accumulate

// Simple stride prefetcher for comparison
class StridePrefetcher {
private:
    uint64_t last_addr;
    int64_t last_stride;
    bool initialized;

public:
    StridePrefetcher() : last_addr(0), last_stride(0), initialized(false) {}

    std::vector<uint64_t> predict(uint64_t current_addr, uint64_t) {
        std::vector<uint64_t> predictions;
        
        if (!initialized) {
            initialized = true;
            last_addr = current_addr;
            return predictions;
        }

        int64_t stride = current_addr - last_addr;
        
        // If we detect a consistent stride pattern
        if (stride == last_stride && stride != 0) {
            // Predict next few addresses based on stride
            for (int i = 1; i <= 4; i++) {
                predictions.push_back(current_addr + stride * i);
            }
        }
        
        last_stride = stride;
        last_addr = current_addr;
        return predictions;
    }

    void reset() {
        initialized = false;
        last_addr = 0;
        last_stride = 0;
    }
};

// Test helper functions
double calculate_accuracy(const std::vector<uint64_t>& predictions, 
                        const std::vector<uint64_t>& actual_accesses) {
    if (predictions.empty() || actual_accesses.empty()) return 0.0;
    
    int hits = 0;
    for (auto pred : predictions) {
        for (auto actual : actual_accesses) {
            if (pred == actual) {
                hits++;
                break;
            }
        }
    }
    return static_cast<double>(hits) / predictions.size();
}

double calculate_coverage(const std::vector<uint64_t>& predictions, 
                        const std::vector<uint64_t>& actual_accesses) {
    if (predictions.empty() || actual_accesses.empty()) return 0.0;
    
    int covered = 0;
    for (auto actual : actual_accesses) {
        for (auto pred : predictions) {
            if (pred == actual) {
                covered++;
                break;
            }
        }
    }
    return static_cast<double>(covered) / actual_accesses.size();
}

void test_pattern(const std::string& pattern_name,
                 const std::vector<std::pair<uint64_t, uint64_t>>& access_sequence,
                 const std::vector<uint64_t>& future_accesses,
                 const std::string& weights_dir) {
    std::cout << "\nTesting " << pattern_name << " pattern\n";
    std::cout << "----------------------------------------\n";

    // Initialize prefetchers
    DARTPrefetcher dart_prefetcher;
    StridePrefetcher stride_prefetcher;
    
    try {
        // Load DART weights
        dart_prefetcher.load_weights(weights_dir);
        
        // Test variables
        std::vector<double> dart_accuracies, stride_accuracies;
        std::vector<double> dart_coverages, stride_coverages;
        std::vector<uint64_t> dart_latencies, stride_latencies;
        
        // Process sequence
        for (const auto& [addr, pc] : access_sequence) {
            // Time and get DART predictions
            auto start = std::chrono::high_resolution_clock::now();
            auto dart_preds = dart_prefetcher.predict(addr, pc);
            auto end = std::chrono::high_resolution_clock::now();
            auto dart_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            dart_latencies.push_back(dart_time);
            
            // Time and get stride predictions
            start = std::chrono::high_resolution_clock::now();
            auto stride_preds = stride_prefetcher.predict(addr, pc);
            end = std::chrono::high_resolution_clock::now();
            auto stride_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            stride_latencies.push_back(stride_time);
            
            // Calculate metrics
            dart_accuracies.push_back(calculate_accuracy(dart_preds, future_accesses));
            stride_accuracies.push_back(calculate_accuracy(stride_preds, future_accesses));
            dart_coverages.push_back(calculate_coverage(dart_preds, future_accesses));
            stride_coverages.push_back(calculate_coverage(stride_preds, future_accesses));
        }
        
        // Calculate averages
        double avg_dart_accuracy = std::accumulate(dart_accuracies.begin(), 
                                                 dart_accuracies.end(), 0.0) / dart_accuracies.size();
        double avg_stride_accuracy = std::accumulate(stride_accuracies.begin(), 
                                                   stride_accuracies.end(), 0.0) / stride_accuracies.size();
        double avg_dart_coverage = std::accumulate(dart_coverages.begin(), 
                                                 dart_coverages.end(), 0.0) / dart_coverages.size();
        double avg_stride_coverage = std::accumulate(stride_coverages.begin(), 
                                                   stride_coverages.end(), 0.0) / stride_coverages.size();
        double avg_dart_latency = std::accumulate(dart_latencies.begin(), 
                                                dart_latencies.end(), 0.0) / dart_latencies.size();
        double avg_stride_latency = std::accumulate(stride_latencies.begin(), 
                                                  stride_latencies.end(), 0.0) / stride_latencies.size();
        
        // Print results
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "DART Prefetcher:\n";
        std::cout << "  Accuracy: " << (avg_dart_accuracy * 100) << "%\n";
        std::cout << "  Coverage: " << (avg_dart_coverage * 100) << "%\n";
        std::cout << "  Average Latency: " << avg_dart_latency << " ns\n";
        
        std::cout << "\nStride Prefetcher:\n";
        std::cout << "  Accuracy: " << (avg_stride_accuracy * 100) << "%\n";
        std::cout << "  Coverage: " << (avg_stride_coverage * 100) << "%\n";
        std::cout << "  Average Latency: " << avg_stride_latency << " ns\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Set weights directory
    std::string weights_dir = (argc > 1) ? argv[1] : "../weights";
    
    // 1. Regular stride pattern
    std::vector<std::pair<uint64_t, uint64_t>> stride_sequence;
    std::vector<uint64_t> stride_future;
    for (int i = 0; i < 10; i++) {
        stride_sequence.push_back({0x1000 + i * 8, 0x400500});
        if (i < 5) {  // Future accesses for prediction evaluation
            stride_future.push_back(0x1000 + (i + 1) * 8);
        }
    }
    test_pattern("Regular Stride", stride_sequence, stride_future, weights_dir);

    // 2. Irregular pattern
    std::vector<std::pair<uint64_t, uint64_t>> irregular_sequence = {
        {0x1000, 0x400500},
        {0x1020, 0x400504},
        {0x1008, 0x400508},
        {0x1030, 0x40050C},
        {0x1010, 0x400510}
    };
    std::vector<uint64_t> irregular_future = {0x1028, 0x1038, 0x1018};
    test_pattern("Irregular", irregular_sequence, irregular_future, weights_dir);

    // 3. Mixed pattern (combination of stride and irregular)
    std::vector<std::pair<uint64_t, uint64_t>> mixed_sequence = {
        {0x1000, 0x400500},
        {0x1008, 0x400504},
        {0x1010, 0x400508},
        {0x1030, 0x40050C},
        {0x1038, 0x400510}
    };
    std::vector<uint64_t> mixed_future = {0x1040, 0x1048, 0x1020};
    test_pattern("Mixed", mixed_sequence, mixed_future, weights_dir);

    return 0;
}