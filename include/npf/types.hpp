// include/npf/types.hpp
#pragma once
#include <cstdint>
#include <cstddef>  // for size_t

namespace npf {

struct MemoryAccess {
    uint64_t address;
    uint64_t timestamp;
    bool was_hit;
};

struct PrefetchStats {
    size_t total_predictions;
    size_t correct_predictions;
    size_t false_positives;
    float accuracy;
    float coverage;
};

} // namespace npf