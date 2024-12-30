#pragma once

#include <vector>
#include <unordered_map>
#include <cmath>
#include <array>

// Constants based on the paper configurations
constexpr int SEQUENCE_LENGTH = 8;        // Input sequence length T
constexpr int NUM_SEGMENTS = 4;           // Number of address segments S
constexpr int HIDDEN_DIM = 32;           // Hidden dimension D
constexpr int NUM_PROTOTYPES = 128;      // Number of prototypes K
constexpr int NUM_SUBSPACES = 2;         // Number of subspaces C
constexpr int DELTA_BITMAP_SIZE = 64;    // Size of output delta bitmap

struct TableEntry {
    std::vector<float> values;
    TableEntry() : values(HIDDEN_DIM, 0.0f) {}
};

class LinearTable {
public:
    LinearTable(int in_dim, int out_dim);
    std::vector<float> lookup(const std::vector<float>& input);
    void load_weights(const std::string& filename);

private:
    std::vector<std::vector<TableEntry>> table;
    int input_dim;
    int output_dim;
    std::vector<int> encode(const std::vector<float>& input);
};

class AttentionTable {
public:
    AttentionTable();
    std::vector<float> lookup(const std::vector<float>& q,
                             const std::vector<float>& k,
                             const std::vector<float>& v);
    void load_weights(const std::string& qk_filename, 
                     const std::string& qkv_filename);

private:
    std::vector<std::vector<TableEntry>> qk_table;
    std::vector<std::vector<TableEntry>> qkv_table;
    std::vector<float> qk_lookup(const std::vector<float>& q,
                                const std::vector<float>& k);
    std::vector<float> qkv_lookup(const std::vector<float>& qk,
                                 const std::vector<float>& v);
    std::vector<int> encode(const std::vector<float>& input);
};

class DARTPrefetcher {
public:
    DARTPrefetcher();
    
    // Initialize tables with pre-trained weights
    void load_weights(const std::string& model_dir);
    
    // Main prediction function
    std::vector<uint64_t> predict(uint64_t current_addr, uint64_t current_pc);
    
    // Clear history
    void reset();

private:
    LinearTable input_linear;
    AttentionTable attention;
    LinearTable output_linear;
    
    std::array<uint64_t, SEQUENCE_LENGTH> address_history;
    std::array<uint64_t, SEQUENCE_LENGTH> pc_history;
    int history_ptr;

    static std::vector<float> segment_address(uint64_t addr);
};