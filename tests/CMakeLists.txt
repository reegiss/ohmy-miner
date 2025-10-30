# Copyright (C) 2025 Regis Araujo Melo
# This program is free software under the GPL-3.0 license. See LICENSE file.

cmake_minimum_required(VERSION 3.22)

# Test executable names
set(TEST_EXECUTABLES
    test_fixed_point
    test_sha256d
    test_difficulty
    test_quantum_simulator
    test_stratum_messages
)

# Include parent directories for headers
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Build each test executable
foreach(test_name ${TEST_EXECUTABLES})
    add_executable(${test_name} ${test_name}.cpp)
    
    # Link against project libraries
    target_link_libraries(${test_name} PRIVATE
        fmt::fmt
        nlohmann_json::nlohmann_json
        OpenSSL::Crypto
    )
    
    # Set C++20 standard
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    
    # Add to test suite
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Link specific dependencies for each test

# Fixed-point test needs fixed_point implementation
target_sources(test_fixed_point PRIVATE
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)

# SHA256d test needs crypto utilities
target_sources(test_sha256d PRIVATE
    ${CMAKE_SOURCE_DIR}/src/crypto/sha256d.cpp
)

# Difficulty test needs difficulty utilities
target_sources(test_difficulty PRIVATE
    ${CMAKE_SOURCE_DIR}/src/crypto/difficulty.cpp
    ${CMAKE_SOURCE_DIR}/src/crypto/sha256d.cpp
)

# Quantum simulator test needs quantum implementation
target_sources(test_quantum_simulator PRIVATE
    ${CMAKE_SOURCE_DIR}/src/quantum/circuit.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cpu_simulator.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/simulator_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)

# Stratum messages test needs pool implementation
target_sources(test_stratum_messages PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pool/messages.cpp
)

# Enable testing
enable_testing()

# Add custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS ${TEST_EXECUTABLES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all unit tests..."
)

# Print test summary
message(STATUS "Configured ${CMAKE_CURRENT_LIST_LINE} unit tests")
message(STATUS "Run 'make run_tests' or 'ctest' to execute tests")
