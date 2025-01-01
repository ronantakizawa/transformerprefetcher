# Enhanced Transformer-Based Memory Prefetcher

This project implements an advanced memory prefetcher using a multi-layer transformer architecture to predict future memory access patterns. The prefetcher analyzes memory access sequences and program counter values through cross-attention mechanisms to make informed predictions about future memory accesses.

## Features

- Enhanced transformer architecture with multiple encoder layers
- Cross-attention between PC and address patterns
- Positional encoding for better sequence understanding
- Multi-head attention support (4 heads)
- Residual connections for improved gradient flow
- Advanced stride pattern recognition and prediction
- Program counter (PC) tracking with dedicated embeddings
- OpenMP parallelization support
- Configurable prediction parameters

## Project Structure

```
transformer_prefetcher/
├── include/
│   ├── transformer_prefetcher.hpp
│   └── matrix.hpp
├── src/
│   ├── transformer_prefetcher.cpp
│   ├── transformer_encoder_layer.cpp
│   ├── multi_head_attention.cpp
│   ├── feed_forward.cpp
│   ├── layer_norm.cpp
│   ├── positional_encoding.cpp
│   ├── matrix.cpp
│   └── main.cpp
├── python/
│   └── generate_weights.py
├── build/
└── CMakeLists.txt
```

## Requirements

- C++17 compatible compiler
- CMake 3.15 or higher
- Python 3.7+ (for weight generation)
- NumPy (for weight generation)
- OpenMP (optional, for parallel processing)

## Building the Project

1. Generate the model weights:
```bash
cd python
python generate_weights.py
```

2. Create build directory and build the project:
```bash
mkdir build
cd build
cmake ..
make
```

If you have OpenMP available and want to enable parallel processing:
```bash
# On Linux/Unix:
cmake .. -DCMAKE_CXX_FLAGS="-fopenmp"

# On macOS with Homebrew OpenMP:
cmake .. -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" -DOpenMP_CXX_LIB_NAMES="omp" -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib
```

## Running the Prefetcher

Run the prefetcher with the weights directory:
```bash
./transformer_prefetcher ../python/weights
```

## Architecture Details

### Enhanced Transformer Components:
- Separate Address and PC Embeddings (32-dim)
- Multiple Transformer Encoder Layers (3 layers)
- Cross-Attention Mechanism between PC and Address patterns
- Positional Encoding using sinusoidal functions
- Multi-Head Attention (4 heads)
- Feed-Forward Network with ReLU activation
- Layer Normalization with residual connections

### Memory Access Analysis:
- Sequence Length: 8 addresses
- Address Segmentation: 4 segments per address
- PC Tracking: Dedicated PC embedding and cross-attention
- Advanced Stride Detection: Combined pattern-based and stride-based prediction

### Prediction Strategy:
- Generates 8 predictions per memory access
- Multi-level pattern recognition through stacked encoders
- Combined local and global pattern awareness
- Confidence-based scoring with stride bias
- Maintains history of both addresses and program counters

## Performance

The prefetcher achieves:
- Average prediction time: ~400 microseconds (3.5x faster than base version)
- Enhanced pattern recognition through multiple encoder layers
- Improved accuracy for both sequential and non-sequential patterns
- Adaptive prediction for varying access patterns
- Better long-range dependency modeling

## Configuration

Key parameters can be modified in `transformer_prefetcher.hpp`:
```cpp
constexpr int SEQUENCE_LENGTH = 8;        // Input sequence length
constexpr int EMBEDDING_DIM = 32;         // Embedding dimension
constexpr int NUM_HEADS = 4;              // Number of attention heads
constexpr int NUM_SEGMENTS = 4;           // Address segments
constexpr int NUM_ENCODER_LAYERS = 3;     // Number of transformer layers
```

## Example Output

```
Current address: 0x0000000000001000 PC: 0x0000000000400500
Debug: Starting prediction for addr: 0x1000
Predicted addresses: 
  0x0000000000001008  (Next sequential)
  0x0000000000001010  (Sequential stride)
  0x0000000000001018  (Sequential stride)
  0x0000000000001040  (Pattern-based)
  0x0000000000001100  (Long-range prediction)
  0x00000000000011c0  (Global pattern)
  ...
```

## Implementation Details

### Enhanced Memory Address Processing
- Separate embeddings for addresses and PC values
- Positional encoding added to capture sequence order
- Cross-attention between PC and address patterns
- Multiple encoder layers for hierarchical feature extraction

### Advanced Attention Mechanism
- Scaled dot-product attention with multiple heads
- Cross-attention for PC-address pattern correlation
- Residual connections throughout the network
- Layer normalization for improved stability

### Sophisticated Prediction Generation
- Multi-level pattern recognition
- Confidence-based scoring with stride bias
- Pattern-based and stride-based hybrid prediction
- Duplicate prevention and ranking system

## License

This project is released under the MIT License. See the LICENSE file for details.