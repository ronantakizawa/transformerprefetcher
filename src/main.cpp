#include "dart_prefetcher.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>

void print_predictions(const std::vector<uint64_t>& predictions) {
    std::cout << "Predicted addresses: ";
    for (const auto& addr : predictions) {
        std::cout << "0x" << std::hex << std::setw(16) << std::setfill('0') << addr << " ";
    }
    std::cout << std::dec << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_weights_directory>" << std::endl;
        return 1;
    }

    try {
        // Check if weights directory exists
        std::filesystem::path weights_dir(argv[1]);
        if (!std::filesystem::exists(weights_dir)) {
            std::cerr << "Error: Directory " << argv[1] << " does not exist" << std::endl;
            return 1;
        }

        // Check if required weight files exist
        std::vector<std::string> required_files = {
            "input_linear.bin",
            "attention_qk.bin",
            "attention_qkv.bin",
            "output_linear.bin"
        };

        for (const auto& file : required_files) {
            if (!std::filesystem::exists(weights_dir / file)) {
                std::cerr << "Error: Required weight file " << file << " not found in " << argv[1] << std::endl;
                return 1;
            }
            
            // Print file size
            auto file_size = std::filesystem::file_size(weights_dir / file);
            std::cout << "Found " << file << " (size: " << file_size << " bytes)" << std::endl;
        }

        // Initialize prefetcher
        std::cout << "\nInitializing prefetcher..." << std::endl;
        DARTPrefetcher prefetcher;
        
        // Load pre-trained weights
        std::cout << "Loading model weights from " << argv[1] << "..." << std::endl;
        prefetcher.load_weights(argv[1]);
        
        // Example memory access sequence
        std::vector<std::pair<uint64_t, uint64_t>> access_sequence = {
            {0x1000, 0x400500}, // addr, pc
            {0x1008, 0x400504},
            {0x1010, 0x400508},
            {0x1018, 0x40050C},
        };

        // Measure prediction time
        auto start = std::chrono::high_resolution_clock::now();
        
        // Process sequence
        std::cout << "\nProcessing memory access sequence..." << std::endl;
        for (const auto& [addr, pc] : access_sequence) {
            std::cout << "\nCurrent address: 0x" << std::hex << std::setw(16) 
                     << std::setfill('0') << addr;
            std::cout << " PC: 0x" << std::hex << std::setw(16) 
                     << std::setfill('0') << pc << std::dec << std::endl;
            
            auto predictions = prefetcher.predict(addr, pc);
            print_predictions(predictions);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "\nAverage prediction time: " 
                  << duration.count() / access_sequence.size() 
                  << " microseconds" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}