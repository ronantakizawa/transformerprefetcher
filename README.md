# Transformer-Based Memory Prefetcher

This project implements a memory prefetcher using a transformer architecture to predict future memory access patterns. The prefetcher analyzes memory access sequences and program counter values to make informed predictions about future memory accesses.

## Features

- Transformer-based architecture with self-attention mechanism
- Multi-head attention support (4 heads)
- Stride pattern recognition and prediction
- Program counter (PC) tracking
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

### Transformer Components:
- Input Embedding (32-dim)
- Multi-Head Attention (4 heads)
- Feed-Forward Network
- Layer Normalization

### Memory Access Analysis:
- Sequence Length: 8 addresses
- Address Segmentation: 4 segments per address
- PC Tracking: Monitors program counter values
- Stride Detection: Optimized for common stride patterns

### Prediction Strategy:
- Generates 8 predictions per memory access
- Prioritizes sequential stride patterns
- Maintains history of both addresses and program counters
- Uses confidence scoring for predictions

## Performance

The prefetcher achieves:
- Average prediction time: ~270-300 microseconds
- Consistent prediction of stride patterns
- High accuracy for sequential access patterns
- Adaptive prediction for varying access patterns

## Configuration

Key parameters can be modified in `transformer_prefetcher.hpp`:
```cpp
constexpr int SEQUENCE_LENGTH = 8;        // Input sequence length
constexpr int EMBEDDING_DIM = 32;         // Embedding dimension
constexpr int NUM_HEADS = 4;              // Number of attention heads
constexpr int NUM_SEGMENTS = 4;           // Address segments
```

## Example Output

```
Current address: 0x0000000000001000 PC: 0x0000000000400500
Predicted addresses: 
  0x0000000000001008  (+8)
  0x0000000000001010  (+16)
  0x0000000000001018  (+24)
  0x0000000000001020  (+32)
  ...
```

## Implementation Details

### Memory Address Processing
- Addresses are segmented into 16-bit chunks
- Each segment is normalized to [0,1] range
- Program counter values are processed similarly

### Attention Mechanism
- Uses scaled dot-product attention
- Parallel processing of attention heads
- Layer normalization for training stability

### Prediction Generation
- Primary stride pattern prediction
- Confidence-based additional predictions
- Distance-weighted prediction scoring
- Duplicate prevention

## License

This project is released under the MIT License. See the LICENSE file for details.