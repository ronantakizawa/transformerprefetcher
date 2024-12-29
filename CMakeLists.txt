cmake_minimum_required(VERSION 3.14)
project(neural_prefetcher VERSION 0.1.0 LANGUAGES CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create interface library
add_library(neural_prefetcher INTERFACE)
target_include_directories(neural_prefetcher INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add threading support
find_package(Threads REQUIRED)
target_link_libraries(neural_prefetcher INTERFACE Threads::Threads)

# Option to build examples and tests
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTING "Build tests" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation rules
include(GNUInstallDirs)
install(TARGETS neural_prefetcher
    EXPORT neural_prefetcher-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/npf
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT neural_prefetcher-targets
    FILE neural_prefetcher-targets.cmake
    NAMESPACE neural_prefetcher::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/neural_prefetcher
)