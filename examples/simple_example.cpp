// examples/simple_example.cpp
#include <npf/neural_prefetcher.hpp>
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    // Create prefetcher with default parameters
    npf::NeuralPrefetcher prefetcher;
    
    // Simulate memory access pattern (e.g., strided access)
    uint64_t base_address = 0x1000;
    uint64_t stride = 64;  // Common cache line size
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Record some access patterns
    for (int i = 0; i < 1000; ++i) {
        uint64_t address = base_address + (i * stride);
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        
        prefetcher.record_access(address, timestamp);
        
        // Every 100 accesses, try to predict the next few addresses
        if (i % 100 == 0) {
            auto predictions = prefetcher.predict_next_addresses(3);
            
            std::cout << "After " << i << " accesses, predictions:\n";
            for (size_t j = 0; j < predictions.size(); ++j) {
                std::cout << "  Next+" << j << ": 0x" 
                         << std::hex << predictions[j] << std::dec << "\n";
            }
        }
        
        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Get final stats
    auto stats = prefetcher.get_stats();
    std::cout << "\nFinal Statistics:\n"
              << "Total Predictions: " << stats.total_predictions << "\n"
              << "Correct Predictions: " << stats.correct_predictions << "\n"
              << "Accuracy: " << (stats.accuracy * 100.0f) << "%\n"
              << "Coverage: " << (stats.coverage * 100.0f) << "%\n";
    
    return 0;
}