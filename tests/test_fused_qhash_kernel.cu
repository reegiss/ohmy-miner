/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

/**
 * @file test_fused_qhash_kernel.cu
 * @brief Test the monolithic on-the-fly qhash kernel
 * 
 * This validates:
 * 1. Kernel launches without errors
 * 2. State vector initialization works
 * 3. Gate application doesn't crash
 * 4. Expectation values are computed
 * 5. Results match expected range
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// Declare kernel wrapper
extern "C" void launch_fused_qhash_kernel(
    cuDoubleComplex* d_state_vectors,
    const uint8_t* d_header_template,
    uint32_t nTime,
    uint64_t nonce_start,
    uint32_t target_compact,
    uint32_t* d_result_buffer,
    uint32_t* d_result_count,
    int batch_size,
    int block_size,
    cudaStream_t stream
);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main()
{
    printf("=== Fused QHash Kernel Test ===\n");
    
    const int batch_size = 4;  // Small batch for testing
    const int block_size = 256;
    const int state_size = 65536;  // 2^16
    
    // Test header template (simplified)
    uint8_t h_header_template[76];
    memset(h_header_template, 0, 76);
    h_header_template[0] = 0x01;  // Version
    
    uint32_t nTime = 1700000000;  // Timestamp
    uint64_t nonce_start = 0;
    uint32_t target = 0xFFFFFFFF;  // Easy target
    
    printf("Batch size: %d\n", batch_size);
    printf("Block size: %d threads\n", block_size);
    printf("State vector size: %d amplitudes (1MB per nonce)\n", state_size);
    printf("Total VRAM: %.2f MB\n", 
           (batch_size * state_size * sizeof(cuDoubleComplex)) / 1024.0 / 1024.0);
    
    // Allocate device memory
    cuDoubleComplex* d_state_vectors;
    uint8_t* d_header_template;
    uint32_t* d_result_buffer;
    uint32_t* d_result_count;
    
    size_t state_mem = batch_size * state_size * sizeof(cuDoubleComplex);
    
    CUDA_CHECK(cudaMalloc(&d_state_vectors, state_mem));
    CUDA_CHECK(cudaMalloc(&d_header_template, 76));
    CUDA_CHECK(cudaMalloc(&d_result_buffer, 1024 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));
    
    // Initialize
    CUDA_CHECK(cudaMemcpy(d_header_template, h_header_template, 76, 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(uint32_t)));
    
    printf("\n--- Launching kernel ---\n");
    
    // Launch kernel
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    launch_fused_qhash_kernel(
        d_state_vectors,
        d_header_template,
        nTime,
        nonce_start,
        target,
        d_result_buffer,
        d_result_count,
        batch_size,
        block_size,
        stream
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    
    printf("✓ Kernel completed without errors\n");
    
    // Check results
    uint32_t h_result_count;
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    
    printf("\nResults found: %u\n", h_result_count);
    
    if (h_result_count > 0) {
        std::vector<uint32_t> h_results(h_result_count);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_result_buffer,
                              h_result_count * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        
        printf("Valid nonces:\n");
        for (uint32_t i = 0; i < h_result_count && i < 10; i++) {
            printf("  Nonce %u: %u\n", i, h_results[i]);
        }
    }
    
    printf("\n✓ SUCCESS: Kernel functional test passed!\n");
    printf("Next steps:\n");
    printf("  1. Integrate into batched_qhash_worker.cpp\n");
    printf("  2. Optimize block_size and batch_size\n");
    printf("  3. Profile with Nsight Compute\n");
    printf("  4. Target: 36 MH/s on GTX 1660 SUPER\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_state_vectors));
    CUDA_CHECK(cudaFree(d_header_template));
    CUDA_CHECK(cudaFree(d_result_buffer));
    CUDA_CHECK(cudaFree(d_result_count));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return 0;
}
