/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

/**
 * @file test_qhash_debug.cu
 * @brief Debug test with intermediate value validation (golden vectors)
 * 
 * CRITICAL: This test validates EVERY computational step of qhash against
 * reference values from the Qubitcoin client. This catches logic errors that
 * would result in 100% rejected shares.
 * 
 * Test Strategy:
 * 1. Use known input (block header + nonce)
 * 2. Validate SHA256 output
 * 3. Validate angle extraction
 * 4. Validate quantum simulation (expectation values)
 * 5. Validate Q15 conversion
 * 6. Validate final hash
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ===== GOLDEN VALUES (Reference from Qubitcoin client) =====
// TODO: These need to be populated from actual Qubitcoin reference implementation

// Test input: Simplified block header (first 76 bytes before nonce)
const uint8_t GOLDEN_HEADER_TEMPLATE[76] = {
    0x01, 0x00, 0x00, 0x00, // Version
    // Previous block hash (32 bytes of zeros for genesis)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // Merkle root (32 bytes)
    0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
    0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
    0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
    0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
    // Timestamp (4 bytes)
    0x29, 0xab, 0x5f, 0x49
};

const uint64_t GOLDEN_NONCE = 0x7c2bac1d;  // Bitcoin genesis nonce
const uint32_t GOLDEN_NTIME = 0x495fab29;  // Timestamp

// Expected SHA256d(header || nonce)
// NOTE: These are PLACEHOLDER values - must be computed from actual Qubitcoin client
const uint32_t GOLDEN_H_INITIAL[8] = {
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// Expected rotation angles (72 total: 2 layers × 16 qubits × 2 axes)
// NOTE: Must be computed from qhash reference implementation
[[maybe_unused]] const double GOLDEN_ANGLES[64] = {0}; // Placeholder - skip validation for now

// Expected quantum expectation values <σ_z> before Q15 conversion
// NOTE: Must be computed from reference quantum simulator
const double GOLDEN_EXPECTATIONS[16] = {0}; // Placeholder

// Expected Q15 fixed-point values after conversion
// NOTE: Should match convert_q15_device(GOLDEN_EXPECTATIONS[i])
const int32_t GOLDEN_Q15_RESULTS[16] = {0}; // Placeholder

// Expected final Result_XOR
const uint32_t GOLDEN_RESULT_XOR[8] = {0}; // Placeholder

// ===== DEBUG KERNEL (with intermediate outputs) =====

// Forward declarations from fused_qhash_kernel.cu
extern "C" void launch_fused_qhash_kernel_debug(
    cuDoubleComplex* d_state_vectors,
    const uint8_t* d_header_template,
    uint32_t nTime,
    uint64_t nonce,
    uint32_t* d_result_buffer,
    uint32_t* d_result_count,
    // Debug outputs
    uint32_t* d_debug_h_initial,
    double* d_debug_angles,
    double* d_debug_expectations,
    int32_t* d_debug_q15_results,
    uint32_t* d_debug_result_xor,
    int block_size,
    cudaStream_t stream
);

// Helper: Print hash in hex
void print_hash(const char* label, const uint32_t* hash, int words) {
    printf("%s: ", label);
    for (int i = 0; i < words; i++) {
        printf("%08x", hash[i]);
    }
    printf("\n");
}

// Helper: Compare with tolerance
bool compare_double_array(const char* label, const double* actual, 
                         const double* expected, int count, double tolerance) {
    bool pass = true;
    for (int i = 0; i < count; i++) {
        double diff = fabs(actual[i] - expected[i]);
        if (diff > tolerance) {
            if (pass) {
                printf("\n✗ FAIL: %s\n", label);
                pass = false;
            }
            printf("  [%d] Expected: %.10f, Got: %.10f, Diff: %.2e\n",
                   i, expected[i], actual[i], diff);
        }
    }
    if (pass) {
        printf("✓ PASS: %s (tolerance: %.2e)\n", label, tolerance);
    }
    return pass;
}

// Helper: Compare exact integer arrays
bool compare_int_array(const char* label, const int32_t* actual,
                      const int32_t* expected, int count) {
    bool pass = true;
    for (int i = 0; i < count; i++) {
        if (actual[i] != expected[i]) {
            if (pass) {
                printf("\n✗ FAIL: %s\n", label);
                pass = false;
            }
            printf("  [%d] Expected: %d (0x%08x), Got: %d (0x%08x)\n",
                   i, expected[i], expected[i], actual[i], actual[i]);
        }
    }
    if (pass) {
        printf("✓ PASS: %s (bit-exact)\n", label);
    }
    return pass;
}

int main()
{
    printf("=== QHash Debug Test (Intermediate Value Validation) ===\n\n");
    printf("WARNING: This test uses PLACEHOLDER golden values.\n");
    printf("Before deploying to production, golden values MUST be extracted\n");
    printf("from the actual Qubitcoin reference client.\n\n");
    
    const int block_size = 256;
    const int state_size = 65536;
    
    // Allocate device memory
    cuDoubleComplex* d_state_vectors;
    uint8_t* d_header_template;
    uint32_t* d_result_buffer;
    uint32_t* d_result_count;
    
    // Debug output buffers
    uint32_t* d_debug_h_initial;
    double* d_debug_angles;
    double* d_debug_expectations;
    int32_t* d_debug_q15_results;
    uint32_t* d_debug_result_xor;
    
    CUDA_CHECK(cudaMalloc(&d_state_vectors, state_size * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_header_template, 76));
    CUDA_CHECK(cudaMalloc(&d_result_buffer, 1024 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));
    
    CUDA_CHECK(cudaMalloc(&d_debug_h_initial, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_debug_angles, 64 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_debug_expectations, 16 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_debug_q15_results, 16 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_debug_result_xor, 8 * sizeof(uint32_t)));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_header_template, GOLDEN_HEADER_TEMPLATE, 76,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(uint32_t)));
    
    printf("--- Launching debug kernel (1 nonce) ---\n");
    
    // Launch kernel
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    launch_fused_qhash_kernel_debug(
        d_state_vectors,
        d_header_template,
        GOLDEN_NTIME,
        GOLDEN_NONCE,
        d_result_buffer,
        d_result_count,
        d_debug_h_initial,
        d_debug_angles,
        d_debug_expectations,
        d_debug_q15_results,
        d_debug_result_xor,
        block_size,
        stream
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    
    printf("✓ Kernel completed\n\n");
    
    // ===== VALIDATE ALL INTERMEDIATE VALUES =====
    
    printf("=== Validation Results ===\n\n");
    
    // Allocate host buffers
    uint32_t h_debug_h_initial[8];
    double h_debug_angles[64];
    double h_debug_expectations[16];
    int32_t h_debug_q15_results[16];
    uint32_t h_debug_result_xor[8];
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_debug_h_initial, d_debug_h_initial, 8 * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_debug_angles, d_debug_angles, 64 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_debug_expectations, d_debug_expectations, 16 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_debug_q15_results, d_debug_q15_results, 16 * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_debug_result_xor, d_debug_result_xor, 8 * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    
    bool all_pass = true;
    
    // Test 1: SHA256d validation
    printf("1. SHA256d Hash:\n");
    print_hash("   Expected", GOLDEN_H_INITIAL, 8);
    print_hash("   Got     ", h_debug_h_initial, 8);
    bool sha256_pass = (memcmp(h_debug_h_initial, GOLDEN_H_INITIAL, 32) == 0);
    if (sha256_pass) {
        printf("   ✓ PASS: SHA256d matches\n\n");
    } else {
        printf("   ✗ FAIL: SHA256d mismatch\n\n");
        all_pass = false;
    }
    
    // Test 2: Angle extraction
    printf("2. Rotation Angles:\n");
    printf("   First 8 angles:\n");
    for (int i = 0; i < 8; i++) {
        printf("     [%d] %.6f\n", i, h_debug_angles[i]);
    }
    // NOTE: Can't validate without golden values
    printf("   (⚠ Skipped - no golden reference)\n\n");
    
    // Test 3: Quantum expectation values (CRITICAL TEST)
    printf("3. Quantum Expectation Values <σ_z>:\n");
    for (int i = 0; i < 16; i++) {
        printf("   Qubit %2d: %.10f\n", i, h_debug_expectations[i]);
    }
    bool expectations_pass = compare_double_array(
        "   Quantum expectations",
        h_debug_expectations,
        GOLDEN_EXPECTATIONS,
        16,
        1e-9
    );
    all_pass = all_pass && expectations_pass;
    printf("\n");
    
    // Test 4: Q15 fixed-point conversion
    printf("4. Q15 Fixed-Point Values:\n");
    for (int i = 0; i < 16; i++) {
        printf("   Q15[%2d]: %d (0x%08x)\n", i, 
               h_debug_q15_results[i], h_debug_q15_results[i]);
    }
    bool q15_pass = compare_int_array(
        "   Q15 conversion",
        h_debug_q15_results,
        GOLDEN_Q15_RESULTS,
        16
    );
    all_pass = all_pass && q15_pass;
    printf("\n");
    
    // Test 5: Final XOR result
    printf("5. Final Result_XOR:\n");
    print_hash("   Expected", GOLDEN_RESULT_XOR, 8);
    print_hash("   Got     ", h_debug_result_xor, 8);
    bool xor_pass = (memcmp(h_debug_result_xor, GOLDEN_RESULT_XOR, 32) == 0);
    if (xor_pass) {
        printf("   ✓ PASS: Result_XOR matches\n\n");
    } else {
        printf("   ✗ FAIL: Result_XOR mismatch\n\n");
        all_pass = false;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_state_vectors));
    CUDA_CHECK(cudaFree(d_header_template));
    CUDA_CHECK(cudaFree(d_result_buffer));
    CUDA_CHECK(cudaFree(d_result_count));
    CUDA_CHECK(cudaFree(d_debug_h_initial));
    CUDA_CHECK(cudaFree(d_debug_angles));
    CUDA_CHECK(cudaFree(d_debug_expectations));
    CUDA_CHECK(cudaFree(d_debug_q15_results));
    CUDA_CHECK(cudaFree(d_debug_result_xor));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // ===== FINAL VERDICT =====
    
    printf("=== FINAL VERDICT ===\n");
    if (all_pass) {
        printf("✓ SUCCESS: All intermediate values validated!\n");
        printf("Kernel is ready for integration (Phase 5).\n");
        return 0;
    } else {
        printf("✗ FAILURE: Intermediate value validation failed!\n");
        printf("DO NOT proceed to Phase 5 (Integration).\n");
        printf("Debug the failed step and re-run validation.\n");
        return 1;
    }
}
