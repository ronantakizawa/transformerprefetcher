# Neural Network-based Prefetcher Library

A C++ library implementing a neural network-based memory access predictor for prefetching. The library uses a two-layer neural network to learn and predict memory access patterns, potentially improving application performance through accurate prefetching.

## Features

- Header-only library for easy integration
- Thread-safe design
- Continuous background training
- Configurable history size and model parameters
- Real-time prediction accuracy tracking
- Modern C++17 implementation

## Requirements

- C++17 compatible compiler
- CMake 3.14 or higher
- Google Test (for building tests)
- Threading support

## Building

```bash
mkdir build
cd build
cmake ..
make
```

To build with examples and tests:
```bash
cmake .. -DBUILD_EXAMPLES=ON -DBUILD_TESTING=ON
make
```

## Usage

Basic usage example:

```cpp
#include <npf/neural_prefetcher.hpp>

// Create prefetcher instance
npf::NeuralPrefetcher prefetcher(
    8,     // history size
    64,    // hidden layer size
    32,    // batch size
    0.001f // learning rate
);

// Record memory accesses
prefetcher.record_access(address, timestamp);

// Get predictions
auto predicted_addresses = prefetcher.predict_next_addresses(3);

// Get performance stats
auto stats = prefetcher.get_stats();
```

## Integration

The library is header-only, so you can either:

1. Copy the `include/npf` directory to your project
2. Use CMake to install the library:
```bash
cmake --install .
```

Then in your project's CMakeLists.txt:
```cmake
find_package(neural_prefetcher REQUIRED)
target_link_libraries(your_target PRIVATE neural_prefetcher)
```

## Configuration

The prefetcher can be configured with several parameters:

- `history_size`: Number of past memory accesses to consider (default: 8)
- `hidden_size`: Size of the hidden layer in the neural network (default: 64)
- `batch_size`: Number of samples to accumulate before training (default: 32)
- `learning_rate`: Learning rate for the neural network (default: 0.001)

## Performance Tracking

The prefetcher tracks several metrics:

- Total predictions made
- Correct predictions
- Prediction accuracy
- Memory access coverage
- False positive rate

Access these metrics using the `get_stats()` method.