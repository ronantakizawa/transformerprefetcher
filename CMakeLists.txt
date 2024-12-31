cmake_minimum_required(VERSION 3.15)
project(transformer_prefetcher CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable optimizations in Release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Make OpenMP optional
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found and enabled")
else()
    message(STATUS "OpenMP not found - parallel processing will be disabled")
endif()

# Create main library
add_library(transformer_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/src/transformer_prefetcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix.cpp
)

target_include_directories(transformer_lib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create main executable
add_executable(transformer_prefetcher
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(transformer_prefetcher
    PRIVATE transformer_lib
)

# Optional: Create test executable if test file exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test_prefetcher.cpp")
    add_executable(test_prefetcher
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test_prefetcher.cpp
    )
    target_link_libraries(test_prefetcher
        PRIVATE transformer_lib
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer_lib PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(transformer_prefetcher PRIVATE OpenMP::OpenMP_CXX)
    if(TARGET test_prefetcher)
        target_link_libraries(test_prefetcher PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()