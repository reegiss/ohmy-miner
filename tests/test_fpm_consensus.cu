/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

/**
 * @file test_fpm_consensus.cu
 * @brief Golden vector test for fixed-point consensus validation
 * 
 * CRITICAL TEST: This validates that convert_q15_device() is bit-exact with fpm library.
 * 100% pass rate is MANDATORY before deploying to production mining.
 * 
 * Test strategy:
 * 1. Generate 10,000+ test values covering full Q15 range
 * 2. Compute reference outputs using original fpm library (host)
 * 3. Compute actual outputs using convert_q15_device() (device)
 * 4. Compare bit-by-bit (assert exact match)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

// Host-side fixed-point library (reference implementation)
#include "ohmy/fixed_point.hpp"

// Device-side conversion (implementation under test)
#include "../src/quantum/fpm_consensus_device.cuh"

// Type alias for Q15 format used by qhash
using Q15 = ohmy::fixed_point<int32_t, 15>;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Test kernel: Apply convert_q15_device() to array
__global__ void test_conversion_kernel(
    const double* inputs,
    int32_t* outputs,
    int num_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_values) {
        outputs[idx] = convert_q15_device(inputs[idx]);
    }
}

/**
 * @brief Generate comprehensive test vector covering Q15 range
 * 
 * Test cases:
 * - Special values: -1.0, 0.0, +1.0 (common quantum expectations)
 * - Edge cases: ±65536, ±2^-15
 * - Random values in [-1.0, +1.0] (typical quantum range)
 * - Random values in full Q15 range [-65536, 65535.999]
 * - Values near rounding boundaries (x.5)
 */
std::vector<double> generate_test_vector(int num_samples) {
    std::vector<double> test_values;
    test_values.reserve(num_samples);
    
    // Special quantum values
    test_values.push_back(-1.0);
    test_values.push_back(-0.5);
    test_values.push_back(0.0);
    test_values.push_back(0.5);
    test_values.push_back(1.0);
    
    // Edge cases
    test_values.push_back(-65536.0);              // Min Q15
    test_values.push_back(65535.999969482421875); // Max Q15
    test_values.push_back(1.0 / 32768.0);         // Smallest positive step
    test_values.push_back(-1.0 / 32768.0);        // Smallest negative step
    
    // Rounding boundary cases (test round-to-even)
    for (int i = -100; i <= 100; ++i) {
        double base = i / 32768.0;
        test_values.push_back(base);
        test_values.push_back(base + 0.5 / 32768.0); // Exactly .5 case
    }
    
    // Random values in quantum range [-1, +1]
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> quantum_dist(-1.0, 1.0);
    
    for (int i = 0; i < num_samples / 2; ++i) {
        test_values.push_back(quantum_dist(rng));
    }
    
    // Random values in full Q15 range
    std::uniform_real_distribution<double> full_dist(-65536.0, 65535.0);
    
    for (int i = test_values.size(); i < num_samples; ++i) {
        test_values.push_back(full_dist(rng));
    }
    
    return test_values;
}

/**
 * @brief Compute reference outputs using ohmy::fixed_point
 */
std::vector<int32_t> compute_reference_outputs(const std::vector<double>& inputs) {
    std::vector<int32_t> outputs;
    outputs.reserve(inputs.size());
    
    for (double val : inputs) {
        Q15 fixed_val = Q15::from_float(val);
        outputs.push_back(fixed_val.raw());
    }
    
    return outputs;
}

/**
 * @brief Run golden test
 */
bool run_golden_test(int num_samples) {
    printf("=== FPM Consensus Golden Test ===\n");
    printf("Testing %d samples...\n", num_samples);
    
    // Generate test vector
    auto h_inputs = generate_test_vector(num_samples);
    int actual_samples = h_inputs.size();
    
    printf("Generated %d test cases\n", actual_samples);
    
    // Compute reference outputs (host, using ohmy::fixed_point)
    printf("Computing reference outputs (ohmy::fixed_point library)...\n");
    auto h_expected = compute_reference_outputs(h_inputs);
    
    // Allocate device memory
    double* d_inputs = nullptr;
    int32_t* d_outputs = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_inputs, actual_samples * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_outputs, actual_samples * sizeof(int32_t)));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs.data(),
                          actual_samples * sizeof(double),
                          cudaMemcpyHostToDevice));
    
    // Launch kernel
    printf("Computing actual outputs (device kernel)...\n");
    int block_size = 256;
    int grid_size = (actual_samples + block_size - 1) / block_size;
    
    test_conversion_kernel<<<grid_size, block_size>>>(
        d_inputs, d_outputs, actual_samples
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    std::vector<int32_t> h_actual(actual_samples);
    CUDA_CHECK(cudaMemcpy(h_actual.data(), d_outputs,
                          actual_samples * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    
    // Validate results
    printf("Validating bit-exact match...\n");
    int mismatches = 0;
    int first_mismatch_idx = -1;
    
    for (int i = 0; i < actual_samples; ++i) {
        if (h_expected[i] != h_actual[i]) {
            mismatches++;
            if (first_mismatch_idx < 0) {
                first_mismatch_idx = i;
            }
        }
    }
    
    // Report results
    if (mismatches == 0) {
        printf("\n✓ SUCCESS: All %d samples matched bit-exactly!\n", actual_samples);
        printf("convert_q15_device() is consensus-compatible with ohmy::fixed_point.\n");
    } else {
        printf("\n✗ FAILURE: %d/%d samples MISMATCHED!\n", mismatches, actual_samples);
        printf("\nFirst mismatch at index %d:\n", first_mismatch_idx);
        printf("  Input:    %.17g\n", h_inputs[first_mismatch_idx]);
        printf("  Expected: %d (0x%08x)\n", 
               h_expected[first_mismatch_idx], 
               static_cast<uint32_t>(h_expected[first_mismatch_idx]));
        printf("  Actual:   %d (0x%08x)\n", 
               h_actual[first_mismatch_idx], 
               static_cast<uint32_t>(h_actual[first_mismatch_idx]));
        printf("  Diff:     %d\n", h_actual[first_mismatch_idx] - h_expected[first_mismatch_idx]);
        
        // Show a few more mismatches
        int shown = 0;
        for (int i = 0; i < actual_samples && shown < 10; ++i) {
            if (h_expected[i] != h_actual[i]) {
                if (i != first_mismatch_idx) {
                    printf("\nMismatch at index %d:\n", i);
                    printf("  Input:    %.17g\n", h_inputs[i]);
                    printf("  Expected: %d\n", h_expected[i]);
                    printf("  Actual:   %d\n", h_actual[i]);
                    shown++;
                }
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_inputs));
    CUDA_CHECK(cudaFree(d_outputs));
    
    return (mismatches == 0);
}

int main() {
    // Run test with 20,000 samples
    bool success = run_golden_test(20000);
    
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
