#include "dart_prefetcher.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

LinearTable::LinearTable(int in_dim, int out_dim) 
    : input_dim(in_dim), output_dim(out_dim) {
    std::cout << "Initializing LinearTable with dims: " << in_dim << " -> " << out_dim << std::endl;
    std::cout << "Allocating " << NUM_SUBSPACES << " subspaces with " << NUM_PROTOTYPES 
              << " prototypes each of size " << out_dim << std::endl;
              
    table.resize(NUM_SUBSPACES);
    for (auto& subspace : table) {
        subspace.resize(NUM_PROTOTYPES);
        for (auto& entry : subspace) {
            entry.values.resize(output_dim, 0.0f);
        }
    }
    std::cout << "LinearTable initialized successfully" << std::endl;
}

std::vector<float> LinearTable::lookup(const std::vector<float>& input) {
    if (input.size() != input_dim) {
        throw std::runtime_error("Input dimension mismatch. Expected: " + 
                               std::to_string(input_dim) + " Got: " + 
                               std::to_string(input.size()));
    }

    std::cout << "Starting linear lookup with input size: " << input.size() << std::endl;
    std::vector<float> output(output_dim, 0.0f);
    std::vector<int> indices = encode(input);
    
    std::cout << "Encoded indices: ";
    for (int i = 0; i < indices.size(); i++) {
        std::cout << indices[i] << " ";
    }
    std::cout << std::endl;
    
    for (int s = 0; s < NUM_SUBSPACES; s++) {
        std::cout << "Processing subspace " << s << ", index " << indices[s] 
                 << ", table size: " << table[s].size() << std::endl;
                 
        if (indices[s] >= table[s].size()) {
            throw std::runtime_error("Table index out of bounds in subspace " + 
                                   std::to_string(s) + ": index=" + 
                                   std::to_string(indices[s]) + 
                                   ", size=" + 
                                   std::to_string(table[s].size()));
        }
        
        const auto& entry = table[s][indices[s]];
        std::cout << "Entry vector size: " << entry.values.size() 
                 << ", expected: " << output_dim << std::endl;
        
        if (entry.values.size() != output_dim) {
            throw std::runtime_error("Entry dimension mismatch in subspace " + 
                                   std::to_string(s) + ": expected=" + 
                                   std::to_string(output_dim) + 
                                   ", got=" + 
                                   std::to_string(entry.values.size()));
        }
        
        for (int i = 0; i < output_dim; i++) {
            output[i] += entry.values[i];
        }
        std::cout << "Completed subspace " << s << std::endl;
    }
    
    return output;
}

void LinearTable::load_weights(const std::string& filename) {
    std::cout << "Loading weights from: " << filename << std::endl;
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open weights file: " + filename);
    }

    size_t expected_size = NUM_SUBSPACES * NUM_PROTOTYPES * output_dim * sizeof(float);
    std::filesystem::path filepath(filename);
    size_t actual_size = std::filesystem::file_size(filepath);
    
    std::cout << "Expected file size: " << expected_size << " bytes" << std::endl;
    std::cout << "Actual file size: " << actual_size << " bytes" << std::endl;
    std::cout << "Table dimensions: subspaces=" << NUM_SUBSPACES 
              << ", prototypes=" << NUM_PROTOTYPES 
              << ", output_dim=" << output_dim << std::endl;

    if (actual_size != expected_size) {
        throw std::runtime_error("Weight file size mismatch. Expected: " + 
                               std::to_string(expected_size) + " Got: " + 
                               std::to_string(actual_size));
    }

    // Verify table dimensions before loading
    for (int s = 0; s < NUM_SUBSPACES; s++) {
        if (table[s].size() != NUM_PROTOTYPES) {
            throw std::runtime_error("Table dimension mismatch in subspace " + 
                                   std::to_string(s) + ": expected=" + 
                                   std::to_string(NUM_PROTOTYPES) + 
                                   ", got=" + 
                                   std::to_string(table[s].size()));
        }
        for (int p = 0; p < NUM_PROTOTYPES; p++) {
            if (table[s][p].values.size() != output_dim) {
                throw std::runtime_error("Entry dimension mismatch at (" + 
                                       std::to_string(s) + "," + 
                                       std::to_string(p) + "): expected=" + 
                                       std::to_string(output_dim) + 
                                       ", got=" + 
                                       std::to_string(table[s][p].values.size()));
            }
        }
    }

    for (auto& subspace : table) {
        for (auto& entry : subspace) {
            file.read(reinterpret_cast<char*>(entry.values.data()), 
                     entry.values.size() * sizeof(float));
            if (file.fail()) {
                throw std::runtime_error("Failed to read weights from file");
            }
        }
    }
    std::cout << "Successfully loaded weights" << std::endl;
}

std::vector<int> LinearTable::encode(const std::vector<float>& input) {
    std::vector<int> indices(NUM_SUBSPACES);
    int subspace_size = input_dim / NUM_SUBSPACES;
    
    for (int s = 0; s < NUM_SUBSPACES; s++) {
        float sum = 0.0f;
        for (int i = s * subspace_size; i < (s + 1) * subspace_size && i < input.size(); i++) {
            sum += input[i];
        }
        // Ensure positive index and proper modulo
        int index = static_cast<int>(std::abs(sum) * NUM_PROTOTYPES);
        indices[s] = index % NUM_PROTOTYPES;
    }
    return indices;
}

DARTPrefetcher::DARTPrefetcher() : 
    input_linear(NUM_SEGMENTS * 2 * SEQUENCE_LENGTH, HIDDEN_DIM),
    output_linear(HIDDEN_DIM, DELTA_BITMAP_SIZE),
    history_ptr(0) {
    std::cout << "Initializing DARTPrefetcher..." << std::endl;
    address_history.fill(0);
    pc_history.fill(0);
    std::cout << "DARTPrefetcher initialized" << std::endl;
}

std::vector<uint64_t> DARTPrefetcher::predict(uint64_t current_addr, uint64_t current_pc) {
    std::cout << "Starting prediction..." << std::endl;
    
    // Update history
    address_history[history_ptr] = current_addr;
    pc_history[history_ptr] = current_pc;
    history_ptr = (history_ptr + 1) % SEQUENCE_LENGTH;

    // Prepare input sequence
    std::cout << "Preparing input features..." << std::endl;
    std::vector<float> input_features;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        auto addr_segments = segment_address(address_history[i]);
        auto pc_segments = segment_address(pc_history[i]);
        input_features.insert(input_features.end(), addr_segments.begin(), addr_segments.end());
        input_features.insert(input_features.end(), pc_segments.begin(), pc_segments.end());
    }

    std::cout << "Input feature size: " << input_features.size() << std::endl;

    // Forward pass through tables
    std::cout << "Running forward pass..." << std::endl;
    auto hidden = input_linear.lookup(input_features);
    std::cout << "Input linear complete (output size: " << hidden.size() << ")" << std::endl;
    
    auto attention_output = attention.lookup(hidden, hidden, hidden);
    std::cout << "Attention complete (output size: " << attention_output.size() << ")" << std::endl;
    
    std::cout << "Starting output linear..." << std::endl;
    if (attention_output.size() != HIDDEN_DIM) {
        throw std::runtime_error("Attention output dimension mismatch. Expected: " + 
                               std::to_string(HIDDEN_DIM) + " Got: " + 
                               std::to_string(attention_output.size()));
    }
    
    auto logits = output_linear.lookup(attention_output);
    std::cout << "Output linear complete (output size: " << logits.size() << ")" << std::endl;

    // Debug output
    std::cout << "Logits: ";
    for (int i = 0; i < std::min(5, DELTA_BITMAP_SIZE); i++) {
        std::cout << logits[i] << " ";
    }
    std::cout << "..." << std::endl;

    // Convert logits to predicted deltas
    std::vector<uint64_t> predictions;
    for (int i = 0; i < DELTA_BITMAP_SIZE; i++) {
        if (logits[i] > 0.1f) {  // Lower threshold for testing
            predictions.push_back(current_addr + i);
        }
    }
    
    return predictions;
}

void DARTPrefetcher::load_weights(const std::string& model_dir) {
    std::cout << "Loading weights from directory: " << model_dir << std::endl;
    namespace fs = std::filesystem;
    fs::path dir(model_dir);
    
    input_linear.load_weights(dir / "input_linear.bin");
    attention.load_weights(dir / "attention_qk.bin", dir / "attention_qkv.bin");
    output_linear.load_weights(dir / "output_linear.bin");
    std::cout << "All weights loaded successfully" << std::endl;
}

void DARTPrefetcher::reset() {
    history_ptr = 0;
    address_history.fill(0);
    pc_history.fill(0);
}

std::vector<float> DARTPrefetcher::segment_address(uint64_t addr) {
    std::vector<float> segments(NUM_SEGMENTS);
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        segments[i] = static_cast<float>((addr >> (i * 16)) & 0xFFFF) / 65535.0f; // Normalize to [0,1]
    }
    return segments;
}

// AttentionTable implementation
AttentionTable::AttentionTable() {
    std::cout << "Initializing AttentionTable..." << std::endl;
    qk_table.resize(NUM_SUBSPACES);
    qkv_table.resize(NUM_SUBSPACES);
    
    for (auto& subspace : qk_table) {
        subspace.resize(NUM_PROTOTYPES * NUM_PROTOTYPES);
        for (auto& entry : subspace) {
            entry.values.resize(HIDDEN_DIM, 0.0f);
        }
    }
    
    for (auto& subspace : qkv_table) {
        subspace.resize(NUM_PROTOTYPES * NUM_PROTOTYPES);
        for (auto& entry : subspace) {
            entry.values.resize(HIDDEN_DIM, 0.0f);
        }
    }
    std::cout << "AttentionTable initialized" << std::endl;
}

void AttentionTable::load_weights(const std::string& qk_filename, 
                                const std::string& qkv_filename) {
    std::cout << "Loading attention weights..." << std::endl;
    // Load QK table
    std::ifstream qk_file(qk_filename, std::ios::binary);
    if (!qk_file) {
        throw std::runtime_error("Could not open QK weights file: " + qk_filename);
    }

    size_t expected_qk_size = NUM_SUBSPACES * NUM_PROTOTYPES * NUM_PROTOTYPES * HIDDEN_DIM * sizeof(float);
    if (std::filesystem::file_size(qk_filename) != expected_qk_size) {
        throw std::runtime_error("QK weight file size mismatch");
    }

    for (auto& subspace : qk_table) {
        for (auto& entry : subspace) {
            qk_file.read(reinterpret_cast<char*>(entry.values.data()),
                        entry.values.size() * sizeof(float));
        }
    }

    // Load QKV table
    std::ifstream qkv_file(qkv_filename, std::ios::binary);
    if (!qkv_file) {
        throw std::runtime_error("Could not open QKV weights file: " + qkv_filename);
    }

    size_t expected_qkv_size = NUM_SUBSPACES * NUM_PROTOTYPES * NUM_PROTOTYPES * HIDDEN_DIM * sizeof(float);
    if (std::filesystem::file_size(qkv_filename) != expected_qkv_size) {
        throw std::runtime_error("QKV weight file size mismatch");
    }

    for (auto& subspace : qkv_table) {
        for (auto& entry : subspace) {
            qkv_file.read(reinterpret_cast<char*>(entry.values.data()),
                         entry.values.size() * sizeof(float));
        }
    }
    std::cout << "Attention weights loaded successfully" << std::endl;
}

std::vector<float> AttentionTable::lookup(const std::vector<float>& q,
                                        const std::vector<float>& k,
                                        const std::vector<float>& v) {
    if (q.size() != HIDDEN_DIM || k.size() != HIDDEN_DIM || v.size() != HIDDEN_DIM) {
        throw std::runtime_error("Input dimension mismatch in attention lookup");
    }

    std::vector<float> qk_result = qk_lookup(q, k);
    return qkv_lookup(qk_result, v);
}

std::vector<float> AttentionTable::qk_lookup(const std::vector<float>& q,
                                           const std::vector<float>& k) {
    std::vector<float> output(HIDDEN_DIM, 0.0f);
    std::vector<int> q_indices = encode(q);
    std::vector<int> k_indices = encode(k);
    
    for (int s = 0; s < NUM_SUBSPACES; s++) {
        uint64_t table_idx = (static_cast<uint64_t>(q_indices[s]) * NUM_PROTOTYPES + k_indices[s]) % (NUM_PROTOTYPES * NUM_PROTOTYPES);
        if (table_idx >= qk_table[s].size()) {
            throw std::runtime_error("QK table index out of bounds: index=" + 
                                   std::to_string(table_idx) + 
                                   ", size=" + 
                                   std::to_string(qk_table[s].size()) +
                                   ", q_idx=" +
                                   std::to_string(q_indices[s]) +
                                   ", k_idx=" +
                                   std::to_string(k_indices[s]));
        }
        const auto& entry = qk_table[s][table_idx];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            output[i] += entry.values[i];
        }
    }
    return output;
}

std::vector<float> AttentionTable::qkv_lookup(const std::vector<float>& qk,
                                            const std::vector<float>& v) {
    std::vector<float> output(HIDDEN_DIM, 0.0f);
    std::vector<int> qk_indices = encode(qk);
    std::vector<int> v_indices = encode(v);
    
    for (int s = 0; s < NUM_SUBSPACES; s++) {
        uint64_t table_idx = (static_cast<uint64_t>(qk_indices[s]) * NUM_PROTOTYPES + v_indices[s]) % (NUM_PROTOTYPES * NUM_PROTOTYPES);
        if (table_idx >= qkv_table[s].size()) {
            throw std::runtime_error("QKV table index out of bounds: index=" + 
                                   std::to_string(table_idx) + 
                                   ", size=" + 
                                   std::to_string(qkv_table[s].size()) +
                                   ", qk_idx=" +
                                   std::to_string(qk_indices[s]) +
                                   ", v_idx=" +
                                   std::to_string(v_indices[s]));
        }
        const auto& entry = qkv_table[s][table_idx];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            output[i] += entry.values[i];
        }
    }
    return output;
}

std::vector<int> AttentionTable::encode(const std::vector<float>& input) {
    std::vector<int> indices(NUM_SUBSPACES);
    int subspace_size = HIDDEN_DIM / NUM_SUBSPACES;
    
    for (int s = 0; s < NUM_SUBSPACES; s++) {
        float sum = 0.0f;
        for (int i = s * subspace_size; i < (s + 1) * subspace_size && i < input.size(); i++) {
            sum += input[i];
        }
        indices[s] = static_cast<int>(sum * NUM_PROTOTYPES) % NUM_PROTOTYPES;
    }
    return indices;
}