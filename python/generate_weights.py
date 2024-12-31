import numpy as np
import os
import struct

# Constants (must match C++ implementation)
SEQUENCE_LENGTH = 8
EMBEDDING_DIM = 32
NUM_SEGMENTS = 4
NUM_HEADS = 4
HEAD_DIM = EMBEDDING_DIM // NUM_HEADS
FF_DIM = EMBEDDING_DIM * 4
DELTA_BITMAP_SIZE = 64

def save_matrix(matrix, filename):
    """Save matrix as binary file."""
    matrix = matrix.astype(np.float32)
    print(f"Saving {filename}: shape {matrix.shape}, size {matrix.nbytes} bytes")
    with open(filename, 'wb') as f:
        matrix.tofile(f)

def xavier_init(shape):
    """Initialize matrix using Xavier initialization."""
    limit = np.sqrt(6.0 / sum(shape))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)

def main():
    # Create weights directory
    os.makedirs('weights', exist_ok=True)
    
    # Input dimensions
    input_dim = NUM_SEGMENTS * 2 * SEQUENCE_LENGTH  # addr and PC segments
    
    # Generate embedding weights
    embedding = xavier_init((EMBEDDING_DIM, input_dim))
    save_matrix(embedding, 'weights/embedding.bin')
    
    # Generate attention weights
    wq = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
    wk = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
    wv = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
    wo = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
    
    save_matrix(wq, 'weights/attention_wq.bin')
    save_matrix(wk, 'weights/attention_wk.bin')
    save_matrix(wv, 'weights/attention_wv.bin')
    save_matrix(wo, 'weights/attention_wo.bin')
    
    # Generate feed-forward weights
    ff_w1 = xavier_init((FF_DIM, EMBEDDING_DIM))
    ff_w2 = xavier_init((EMBEDDING_DIM, FF_DIM))
    
    save_matrix(ff_w1, 'weights/ff_w1.bin')
    save_matrix(ff_w2, 'weights/ff_w2.bin')
    
    # Generate layer norm parameters
    ln1_gamma = np.ones(EMBEDDING_DIM, dtype=np.float32)
    ln1_beta = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    ln2_gamma = np.ones(EMBEDDING_DIM, dtype=np.float32)
    ln2_beta = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    
    save_matrix(ln1_gamma, 'weights/ln1_gamma.bin')
    save_matrix(ln1_beta, 'weights/ln1_beta.bin')
    save_matrix(ln2_gamma, 'weights/ln2_gamma.bin')
    save_matrix(ln2_beta, 'weights/ln2_beta.bin')
    
    # Generate output layer weights with strong stride-8 pattern bias
    output_weight = xavier_init((DELTA_BITMAP_SIZE, EMBEDDING_DIM))
    
    # Focus strongly on the stride-8 pattern
    # We'll use indices that correspond to multiples of 8
    stride_8_indices = [i for i in range(DELTA_BITMAP_SIZE) if i % 8 == 0]
    for idx in stride_8_indices:
        if idx < DELTA_BITMAP_SIZE:
            output_weight[idx] *= 5.0  # Much stronger bias for stride-8 patterns
    
    # Add smaller bias for nearby indices to allow for some variation
    for idx in stride_8_indices:
        if idx + 1 < DELTA_BITMAP_SIZE:
            output_weight[idx + 1] *= 1.5
        if idx - 1 >= 0:
            output_weight[idx - 1] *= 1.5
    
    save_matrix(output_weight, 'weights/output_weight.bin')
    
    print("\nGenerated weights with dimensions:")
    print(f"Embedding: {embedding.shape}")
    print(f"Attention Q/K/V: {wq.shape}")
    print(f"Attention Output: {wo.shape}")
    print(f"FF1: {ff_w1.shape}")
    print(f"FF2: {ff_w2.shape}")
    print(f"Output: {output_weight.shape}")
    
    print("\nAll weights have been saved to the 'weights' directory")
    print("Total files generated:", len(os.listdir('weights')))

if __name__ == '__main__':
    main()