#include <gtest/gtest.h>
#include <npf/neural_prefetcher.hpp>
#include <thread>
#include <chrono>
#include <random>

class NeuralPrefetcherTest : public ::testing::Test {
protected:
    npf::NeuralPrefetcher prefetcher{8, 64, 32, 0.001f};
};

TEST_F(NeuralPrefetcherTest, PredictionsNotEmptyAfterTraining) {
    uint64_t base_address = 0x1000;
    uint64_t stride = 64;
    
    for (int i = 0; i < 20; ++i) {
        prefetcher.record_access(base_address + (i * stride), i);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    auto predictions = prefetcher.predict_next_addresses(1);
    EXPECT_FALSE(predictions.empty());
}

TEST_F(NeuralPrefetcherTest, PredictionsFollowPattern) {
    uint64_t base_address = 0x1000;
    uint64_t stride = 64;
    
    for (int i = 0; i < 20; ++i) {
        prefetcher.record_access(base_address + (i * stride), i);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    auto predictions = prefetcher.predict_next_addresses(1);
    ASSERT_FALSE(predictions.empty());
    
    uint64_t expected_next = base_address + (20 * stride);
    uint64_t prediction_error = std::abs(static_cast<int64_t>(predictions[0] - expected_next));
    EXPECT_LT(prediction_error, stride * 2);
}

TEST_F(NeuralPrefetcherTest, StatsAreReasonable) {
    uint64_t base_address = 0x1000;
    uint64_t stride = 64;
    
    for (int i = 0; i < 20; ++i) {
        prefetcher.record_access(base_address + (i * stride), i);
        if (i % 5 == 0) {
            prefetcher.predict_next_addresses(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    prefetcher.predict_next_addresses(1);
    auto stats = prefetcher.get_stats();
    
    EXPECT_GT(stats.total_predictions, 0);
    EXPECT_GE(stats.accuracy, 0.0f);
    EXPECT_LE(stats.accuracy, 1.0f);
}

TEST_F(NeuralPrefetcherTest, HandlesRandomAccess) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0x1000, 0x10000);
    
    for (int i = 0; i < 20; ++i) {
        prefetcher.record_access(dis(gen), i);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    auto predictions = prefetcher.predict_next_addresses(1);
    EXPECT_FALSE(predictions.empty());
}

TEST_F(NeuralPrefetcherTest, ThreadSafety) {
    constexpr int num_threads = 4;
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t]() {
            uint64_t base = 0x1000 * (t + 1);
            for (int i = 0; i < 20; ++i) {
                prefetcher.record_access(base + (i * 64), i);
                if (i % 5 == 0) {
                    prefetcher.predict_next_addresses(1);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    SUCCEED();
}