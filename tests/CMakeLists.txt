# Copyright (C) 2025 Regis Araujo Melo
# This program is free software under the GPL-3.0 license. See LICENSE file.

cmake_minimum_required(VERSION 3.22)

# Experimental tests are opt-in to avoid breaking default test runs
option(OHMY_ENABLE_EXPERIMENTAL_TESTS "Enable experimental/diagnostic tests" OFF)

# Test executable names
set(TEST_EXECUTABLES
    test_fixed_point
    test_sha256d
    test_difficulty
    test_quantum_simulator
    test_stratum_messages
    test_stratum_optional
    test_cuda_backend
    test_performance_baseline
    test_batch_performance
    test_cuquantum_backend
    test_fpm_consensus
    test_sha256_device
    test_fused_qhash_kernel
    test_qhash_debug
)

# Include parent directories for headers
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${asio_SOURCE_DIR}/asio/include)

# Build each test executable
foreach(test_name ${TEST_EXECUTABLES})
    # Special handling for CUDA tests
    if(test_name STREQUAL "test_fpm_consensus" OR 
       test_name STREQUAL "test_sha256_device" OR
    test_name STREQUAL "test_fused_qhash_kernel" OR
    test_name STREQUAL "test_qhash_debug")
        add_executable(${test_name} ${test_name}.cu)
        set_target_properties(${test_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
        )
        target_link_libraries(${test_name} PRIVATE CUDA::cudart)
    else()
        add_executable(${test_name} ${test_name}.cpp)
        
        # Link against project libraries
        target_link_libraries(${test_name} PRIVATE
            fmt::fmt
            nlohmann_json::nlohmann_json
            OpenSSL::Crypto
        )
        
        # Set C++20 standard
        target_compile_features(${test_name} PRIVATE cxx_std_20)
    endif()
    
    # Add to test suite
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Link specific dependencies for each test

# FPM Consensus test (CUDA golden test for fixed-point conversion)
target_sources(test_fpm_consensus PRIVATE
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)
target_include_directories(test_fpm_consensus PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
# Disable device linking for this simple CUDA test
set_target_properties(test_fpm_consensus PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# SHA256 Device test (CUDA test for device-side SHA256)
target_sources(test_sha256_device PRIVATE
    ${CMAKE_SOURCE_DIR}/src/crypto/sha256d.cpp
)
target_include_directories(test_sha256_device PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
set_target_properties(test_sha256_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
target_link_libraries(test_sha256_device PRIVATE OpenSSL::Crypto)

# Fused QHash Kernel test (CUDA test for on-the-fly kernel)
target_sources(test_fused_qhash_kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/src/quantum/fused_qhash_kernel.cu
)
target_include_directories(test_fused_qhash_kernel PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
set_target_properties(test_fused_qhash_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

# QHash Debug test (Golden value validation with intermediate outputs)
target_sources(test_qhash_debug PRIVATE
    ${CMAKE_SOURCE_DIR}/src/quantum/fused_qhash_kernel.cu
)
target_include_directories(test_qhash_debug PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
set_target_properties(test_qhash_debug PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
)

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
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_device.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_simulator.cu
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)
target_link_libraries(test_quantum_simulator PRIVATE CUDA::cudart)

# Stratum messages test needs pool implementation
target_sources(test_stratum_messages PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pool/messages.cpp
)

# Optional Stratum methods & server-initiated requests tests
target_sources(test_stratum_optional PRIVATE
    ${CMAKE_SOURCE_DIR}/src/pool/stratum_client.cpp
)
target_compile_definitions(test_stratum_optional PRIVATE ASIO_STANDALONE)

# CUDA backend validation test
target_sources(test_cuda_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/src/quantum/circuit.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cpu_simulator.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/simulator_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_device.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_simulator.cu
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)

target_link_libraries(test_cuda_backend PRIVATE CUDA::cudart)

# Performance baseline test (CPU vs CUDA comparison)
target_sources(test_performance_baseline PRIVATE
    ${CMAKE_SOURCE_DIR}/src/quantum/circuit.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cpu_simulator.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/simulator_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_device.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_simulator.cu
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)

target_link_libraries(test_performance_baseline PRIVATE CUDA::cudart)

# Batch performance test
target_sources(test_batch_performance PRIVATE
    ${CMAKE_SOURCE_DIR}/src/quantum/circuit.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_device.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/batched_cuda_simulator.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/fused_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)

target_link_libraries(test_batch_performance PRIVATE CUDA::cudart)

# cuQuantum backend validation test needs circuit and factory
target_sources(test_cuquantum_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/src/quantum/circuit.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/simulator_factory.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cpu_simulator.cpp
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_device.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_kernels.cu
    ${CMAKE_SOURCE_DIR}/src/quantum/cuda_simulator.cu
    ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp
)
target_link_libraries(test_cuquantum_backend PRIVATE CUDA::cudart)


# cuQuantum PauliRotation validation (enable macro to exercise that path)
if(OHMY_ENABLE_EXPERIMENTAL_TESTS)
    add_executable(test_cuquantum_pauli_rotation test_cuquantum_pauli_rotation.cpp)
    target_link_libraries(test_cuquantum_pauli_rotation PRIVATE
        fmt::fmt nlohmann_json::nlohmann_json OpenSSL::Crypto CUDA::cudart)
    target_compile_features(test_cuquantum_pauli_rotation PRIVATE cxx_std_20)
    add_test(NAME test_cuquantum_pauli_rotation COMMAND test_cuquantum_pauli_rotation)
    target_sources(test_cuquantum_pauli_rotation PRIVATE
        ${CMAKE_SOURCE_DIR}/src/quantum/circuit.cpp
        ${CMAKE_SOURCE_DIR}/src/quantum/simulator_factory.cpp
        ${CMAKE_SOURCE_DIR}/src/quantum/cpu_simulator.cpp
        ${CMAKE_SOURCE_DIR}/src/quantum/cuda_device.cu
        ${CMAKE_SOURCE_DIR}/src/quantum/cuda_kernels.cu
        ${CMAKE_SOURCE_DIR}/src/quantum/cuda_simulator.cu
        ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp)
endif()

# Link cuQuantum backend into tests when enabled
if(OHMY_WITH_CUQUANTUM AND HAVE_CUQUANTUM)
    foreach(tgt IN ITEMS test_quantum_simulator test_cuda_backend test_performance_baseline test_cuquantum_backend test_batch_performance)
        target_sources(${tgt} PRIVATE
            ${CMAKE_SOURCE_DIR}/src/quantum/custatevec_backend.cpp
            ${CMAKE_SOURCE_DIR}/src/quantum/custatevec_batched.cu)
        target_link_libraries(${tgt} PRIVATE ${CUQUANTUM_custatevec_LIBRARY})
    endforeach()
    # Batched cuQuantum test
    add_executable(test_cuquantum_batched test_cuquantum_batched.cpp)
    target_link_libraries(test_cuquantum_batched PRIVATE fmt::fmt nlohmann_json::nlohmann_json OpenSSL::Crypto CUDA::cudart ${CUQUANTUM_custatevec_LIBRARY})
    target_compile_features(test_cuquantum_batched PRIVATE cxx_std_20)
    target_sources(test_cuquantum_batched PRIVATE
        ${CMAKE_SOURCE_DIR}/src/quantum/circuit.cpp
        ${CMAKE_SOURCE_DIR}/src/quantum/custatevec_backend.cpp
        ${CMAKE_SOURCE_DIR}/src/quantum/custatevec_batched.cu
        ${CMAKE_SOURCE_DIR}/src/fixed_point.cpp)
    add_test(NAME test_cuquantum_batched COMMAND test_cuquantum_batched)
    if(OHMY_ENABLE_EXPERIMENTAL_TESTS)
        # Enable PauliRotation path only for the dedicated test
        target_sources(test_cuquantum_pauli_rotation PRIVATE
            ${CMAKE_SOURCE_DIR}/src/quantum/custatevec_backend.cpp
            ${CMAKE_SOURCE_DIR}/src/quantum/custatevec_batched.cu)
        target_link_libraries(test_cuquantum_pauli_rotation PRIVATE ${CUQUANTUM_custatevec_LIBRARY})
        target_compile_definitions(test_cuquantum_pauli_rotation PRIVATE OHMY_CUQUANTUM_USE_PAULI_ROTATION=1)
    endif()
endif()

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
