cmake_minimum_required(VERSION 3.16)
project(BackTestingFramework VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Set compiler flags for optimization
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -DNDEBUG")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include
)

# Source files for the core library
set(CORE_SOURCES
    src/core/event_loop.cpp
    src/core/market_data_feed.cpp
    src/core/order_book.cpp
    src/core/fill_engine.cpp
    src/core/portfolio.cpp
    src/core/strategy_api.cpp
)

# Source files for Python bindings
set(BINDING_SOURCES
    src/bindings/python_bindings.cpp
)

# Create the core library
add_library(trading_core_lib STATIC ${CORE_SOURCES})

# Set library properties
set_target_properties(trading_core_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Create Python extension module
pybind11_add_module(_core ${BINDING_SOURCES})

# Link the core library to the Python module
target_link_libraries(_core PRIVATE trading_core_lib)

# Set output directory
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trading_core
)

# Installation
install(TARGETS _core
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/trading_core
)

# Add tests if enabled
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # Add test executable
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core trading_core_lib)
    
    # Add tests
    add_test(NAME CoreTests COMMAND test_core)
endif()

# Add benchmarks if enabled
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
    # Add benchmark executable
    add_executable(benchmark_core benchmarks/benchmark_core.cpp)
    target_link_libraries(benchmark_core trading_core_lib)
endif()

# Print configuration summary
message(STATUS "BackTestingFramework Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Python Executable: ${Python3_EXECUTABLE}")
message(STATUS "  Python Version: ${Python3_VERSION}") 