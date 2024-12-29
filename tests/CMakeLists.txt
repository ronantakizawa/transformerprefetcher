# Set the GTest path explicitly to Homebrew's installation
set(GTEST_ROOT "/opt/homebrew/Cellar/googletest/1.15.2")
set(CMAKE_PREFIX_PATH "${GTEST_ROOT}")

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_prefetcher test_prefetcher.cpp)
target_link_libraries(test_prefetcher PRIVATE 
    neural_prefetcher
    GTest::gtest 
    GTest::gtest_main
    pthread
)

add_test(NAME test_prefetcher COMMAND test_prefetcher)