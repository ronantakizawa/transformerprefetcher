#include "transformer_prefetcher.hpp"
#include <cmath>

PositionalEncoding::PositionalEncoding() : encodings_(SEQUENCE_LENGTH, EMBEDDING_DIM) {
    initialize();
}

void PositionalEncoding::initialize() {
    for (int pos = 0; pos < SEQUENCE_LENGTH; pos++) {
        for (int i = 0; i < EMBEDDING_DIM; i += 2) {
            float angle = pos / std::pow(10000, (2.0f * i) / EMBEDDING_DIM);
            encodings_(pos, i) = std::sin(angle);
            if (i + 1 < EMBEDDING_DIM) {
                encodings_(pos, i + 1) = std::cos(angle);
            }
        }
    }
}

std::vector<float> PositionalEncoding::encode(int position) const {
    std::vector<float> encoding(EMBEDDING_DIM);
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        encoding[i] = encodings_(position, i);
    }
    return encoding;
}