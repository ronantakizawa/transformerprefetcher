cmake_minimum_required(VERSION 3.10)
project(dart_prefetcher)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Create library
add_library(dart_lib
    src/dart_prefetcher.cpp
)

target_include_directories(dart_lib
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# Create main executable
add_executable(dart_prefetcher
    src/main.cpp
)

target_link_libraries(dart_prefetcher
    PRIVATE dart_lib
)

# Create test executable
add_executable(test_prefetcher
    test/test_prefetcher.cpp
)

target_link_libraries(test_prefetcher
    PRIVATE dart_lib
)