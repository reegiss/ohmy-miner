/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

/**
 * @file test_sha256_device.cu
 * @brief Test device-side SHA256 implementation against known test vectors
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/quantum/sha256_device.cuh"
#include "ohmy/crypto/sha256d.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Test kernel
__global__ void test_sha256d_kernel(
    const uint8_t* input,
    uint32_t* output,
    int num_tests
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    const uint8_t* data = input + idx * 80;
    uint32_t* hash = output + idx * 8;
    
    sha256d_80_bytes(data, hash);
}

int main()
{
    printf("=== SHA256 Device Test ===\n");
    
    // Test case: Bitcoin genesis block header
    uint8_t h_genesis[80] = {
        0x01, 0x00, 0x00, 0x00, // Version
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, // Previous block hash (all zeros)
        0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2, 0x7a, 0xc7, 0x2c, 0x3e,
        0x67, 0x76, 0x8f, 0x61, 0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
        0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a, // Merkle root
        0x29, 0xab, 0x5f, 0x49, // Timestamp
        0xff, 0xff, 0x00, 0x1d, // Bits (difficulty)
        0x1d, 0xac, 0x2b, 0x7c  // Nonce
    };
    
    // Compute reference hash using host implementation
    std::vector<uint8_t> genesis_vec(h_genesis, h_genesis + 80);
    auto reference_hash = ohmy::crypto::sha256d(genesis_vec);
    
    printf("Reference hash (host):\n");
    for (size_t i = 0; i < 32; i++) {
        printf("%02x", reference_hash[i]);
    }
    printf("\n");
    
    // Allocate device memory
    uint8_t* d_input;
    uint32_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, 80));
    CUDA_CHECK(cudaMalloc(&d_output, 8 * sizeof(uint32_t)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_genesis, 80, cudaMemcpyHostToDevice));
    
    // Launch kernel
    test_sha256d_kernel<<<1, 1>>>(d_input, d_output, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    uint32_t h_output[8];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    // Convert to bytes for comparison
    uint8_t device_hash[32];
    for (int i = 0; i < 8; i++) {
        device_hash[i * 4 + 0] = (h_output[i] >> 24) & 0xFF;
        device_hash[i * 4 + 1] = (h_output[i] >> 16) & 0xFF;
        device_hash[i * 4 + 2] = (h_output[i] >> 8) & 0xFF;
        device_hash[i * 4 + 3] = h_output[i] & 0xFF;
    }
    
    printf("Device hash (GPU):    \n");
    for (int i = 0; i < 32; i++) {
        printf("%02x", device_hash[i]);
    }
    printf("\n");
    
    // Compare
    bool match = true;
    for (int i = 0; i < 32; i++) {
        if (device_hash[i] != reference_hash[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        printf("\n✓ SUCCESS: Device SHA256d matches host reference!\n");
    } else {
        printf("\n✗ FAILURE: Hash mismatch!\n");
        return 1;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}
