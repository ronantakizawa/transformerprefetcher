import numpy as np
import os

# Constants (must match C++ implementation)
SEQUENCE_LENGTH = 8
HIDDEN_DIM = 32
NUM_SEGMENTS = 4
NUM_PROTOTYPES = 128
NUM_SUBSPACES = 2
DELTA_BITMAP_SIZE = 64

def save_binary_weights(weights, filename):
    weights = weights.astype(np.float32)
    print(f"Saving {filename}: shape {weights.shape}, size {weights.nbytes} bytes")
    with open(filename, 'wb') as f:
        weights.tofile(f)

def main():
    # Create weights directory
    os.makedirs('weights', exist_ok=True)
    
    # Calculate dimensions
    input_dim = NUM_SEGMENTS * 2 * SEQUENCE_LENGTH  # addr and PC segments for full sequence
    print(f"Input dimension: {input_dim}")
    
    # Generate random weights for each table
    # Input linear table (input_dim -> HIDDEN_DIM)
    input_linear = np.random.randn(NUM_SUBSPACES, NUM_PROTOTYPES, HIDDEN_DIM) * 0.5
    save_binary_weights(input_linear, 'weights/input_linear.bin')
    
    # Attention QK table (HIDDEN_DIM -> HIDDEN_DIM)
    attention_qk = np.random.randn(NUM_SUBSPACES, NUM_PROTOTYPES * NUM_PROTOTYPES, HIDDEN_DIM) * 0.5
    save_binary_weights(attention_qk, 'weights/attention_qk.bin')
    
    # Attention QKV table (HIDDEN_DIM -> HIDDEN_DIM)
    attention_qkv = np.random.randn(NUM_SUBSPACES, NUM_PROTOTYPES * NUM_PROTOTYPES, HIDDEN_DIM) * 0.5
    save_binary_weights(attention_qkv, 'weights/attention_qkv.bin')
    
    # Output linear table (HIDDEN_DIM -> DELTA_BITMAP_SIZE)
    # Make some entries clearly above threshold
    output_linear = np.random.randn(NUM_SUBSPACES, NUM_PROTOTYPES, DELTA_BITMAP_SIZE) * 0.5
    # Ensure some values are likely to be above threshold
    output_linear[output_linear > 0.5] = 0.8
    save_binary_weights(output_linear, 'weights/output_linear.bin')
    
    print("\nGenerated weights with dimensions:")
    print(f"Input Linear: {NUM_SUBSPACES}x{NUM_PROTOTYPES}x{HIDDEN_DIM}")
    print(f"Attention QK: {NUM_SUBSPACES}x{NUM_PROTOTYPES * NUM_PROTOTYPES}x{HIDDEN_DIM}")
    print(f"Attention QKV: {NUM_SUBSPACES}x{NUM_PROTOTYPES * NUM_PROTOTYPES}x{HIDDEN_DIM}")
    print(f"Output Linear: {NUM_SUBSPACES}x{NUM_PROTOTYPES}x{DELTA_BITMAP_SIZE}")

if __name__ == '__main__':
    main()