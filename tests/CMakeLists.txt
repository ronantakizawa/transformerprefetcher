find_package(GTest REQUIRED)
add_executable(test_prefetcher test_prefetcher.cpp)
target_link_libraries(test_prefetcher PRIVATE 
    neural_prefetcher
    GTest::GTest 
    GTest::Main
)
add_test(NAME test_prefetcher COMMAND test_prefetcher)