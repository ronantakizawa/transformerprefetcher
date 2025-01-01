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
NUM_ENCODER_LAYERS = 3

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

def generate_positional_encoding():
    """Generate positional encoding matrix."""
    position = np.arange(SEQUENCE_LENGTH)[:, np.newaxis]
    div_term = np.exp(np.arange(0, EMBEDDING_DIM, 2) * -(np.log(10000.0) / EMBEDDING_DIM))
    
    pe = np.zeros((SEQUENCE_LENGTH, EMBEDDING_DIM))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe.astype(np.float32)

def main():
    os.makedirs('weights', exist_ok=True)
    
    # Generate separate embeddings for addresses and PCs
    input_dim = NUM_SEGMENTS * SEQUENCE_LENGTH
    addr_embedding = xavier_init((EMBEDDING_DIM, input_dim))
    pc_embedding = xavier_init((EMBEDDING_DIM, input_dim))
    
    save_matrix(addr_embedding, 'weights/addr_embedding.bin')
    save_matrix(pc_embedding, 'weights/pc_embedding.bin')
    
    # Generate positional encodings
    pos_encoding = generate_positional_encoding()
    save_matrix(pos_encoding, 'weights/positional_encoding.bin')
    
    # Generate weights for each encoder layer
    for layer in range(NUM_ENCODER_LAYERS):
        layer_dir = f'layer_{layer}'
        os.makedirs(os.path.join('weights', layer_dir), exist_ok=True)
        
        # Self-attention weights
        wq = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        wk = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        wv = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        wo = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        
        save_matrix(wq, f'weights/{layer_dir}/self_attention_wq.bin')
        save_matrix(wk, f'weights/{layer_dir}/self_attention_wk.bin')
        save_matrix(wv, f'weights/{layer_dir}/self_attention_wv.bin')
        save_matrix(wo, f'weights/{layer_dir}/self_attention_wo.bin')
        
        # Cross-attention weights
        wq = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        wk = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        wv = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        wo = xavier_init((EMBEDDING_DIM, EMBEDDING_DIM))
        
        save_matrix(wq, f'weights/{layer_dir}/cross_attention_wq.bin')
        save_matrix(wk, f'weights/{layer_dir}/cross_attention_wk.bin')
        save_matrix(wv, f'weights/{layer_dir}/cross_attention_wv.bin')
        save_matrix(wo, f'weights/{layer_dir}/cross_attention_wo.bin')
        
        # Feed-forward weights
        ff_w1 = xavier_init((FF_DIM, EMBEDDING_DIM))
        ff_w2 = xavier_init((EMBEDDING_DIM, FF_DIM))
        
        save_matrix(ff_w1, f'weights/{layer_dir}/ff_w1.bin')
        save_matrix(ff_w2, f'weights/{layer_dir}/ff_w2.bin')
        
        # Layer norm parameters
        for norm_idx in range(3):
            ln_gamma = np.ones(EMBEDDING_DIM, dtype=np.float32)
            ln_beta = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            
            save_matrix(ln_gamma, f'weights/{layer_dir}/ln{norm_idx+1}_gamma.bin')
            save_matrix(ln_beta, f'weights/{layer_dir}/ln{norm_idx+1}_beta.bin')
    
    # Generate output layer weights with stride pattern bias
    output_weight = xavier_init((DELTA_BITMAP_SIZE, EMBEDDING_DIM))
    
    # Add stride pattern biases
    stride_8_indices = [i for i in range(DELTA_BITMAP_SIZE) if i % 8 == 0]
    for idx in stride_8_indices:
        if idx < DELTA_BITMAP_SIZE:
            output_weight[idx] *= 5.0
    
    # Add bias for nearby indices
    for idx in stride_8_indices:
        if idx + 1 < DELTA_BITMAP_SIZE:
            output_weight[idx + 1] *= 1.5
        if idx - 1 >= 0:
            output_weight[idx - 1] *= 1.5
    
    save_matrix(output_weight, 'weights/output_weight.bin')
    
    print("\nGenerated weights with dimensions:")
    print(f"Address Embedding: {addr_embedding.shape}")
    print(f"PC Embedding: {pc_embedding.shape}")
    print(f"Positional Encoding: {pos_encoding.shape}")
    print(f"Number of encoder layers: {NUM_ENCODER_LAYERS}")
    print(f"Output: {output_weight.shape}")
    
    print("\nAll weights have been saved to the 'weights' directory")
    
if __name__ == '__main__':
    main()