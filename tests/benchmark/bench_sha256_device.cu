/*
 * Benchmark for SHA256 CUDA device implementation
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#include <ohmy/crypto/sha256_device.cuh>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <chrono>

using namespace ohmy::crypto;

// Kernel: Batch SHA256d processing (one nonce per thread)
__global__ void batch_sha256d_kernel(const uint8_t* headers, uint32_t* nonces, 
                                      uint8_t* outputs, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Simulate block header with nonce
    uint8_t header[84]; // 76 bytes header + 8 bytes nonce
    
    // Copy base header (76 bytes)
    for (int i = 0; i < 76; i++) {
        header[i] = headers[i];
    }
    
    // Append nonce (little-endian, 8 bytes)
    uint64_t nonce = nonces[idx];
    for (int i = 0; i < 8; i++) {
        header[76 + i] = (nonce >> (i * 8)) & 0xFF;
    }
    
    // Compute SHA256d
    sha256d(header, 84, &outputs[idx * 32]);
}

int main() {
    fmt::print("=== SHA256 CUDA Device Benchmark ===\n\n");
    
    // Test parameters
    constexpr int BATCH_SIZE = 4096;
    constexpr int ITERATIONS = 100;
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int BLOCKS = (BATCH_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    fmt::print("Configuration:\n");
    fmt::print("  Batch size: {} hashes\n", BATCH_SIZE);
    fmt::print("  Iterations: {}\n", ITERATIONS);
    fmt::print("  Threads per block: {}\n", THREADS_PER_BLOCK);
    fmt::print("  Blocks: {}\n\n", BLOCKS);
    
    // Allocate host memory
    uint8_t header[76] = {0}; // Dummy header
    uint32_t* h_nonces = new uint32_t[BATCH_SIZE];
    uint8_t* h_outputs = new uint8_t[BATCH_SIZE * 32];
    
    // Initialize nonces
    for (int i = 0; i < BATCH_SIZE; i++) {
        h_nonces[i] = i;
    }
    
    // Allocate device memory
    uint8_t* d_header;
    uint32_t* d_nonces;
    uint8_t* d_outputs;
    
    cudaMalloc(&d_header, 76);
    cudaMalloc(&d_nonces, BATCH_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_outputs, BATCH_SIZE * 32);
    
    // Copy data to device
    cudaMemcpy(d_header, header, 76, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonces, h_nonces, BATCH_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Warmup
    batch_sha256d_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_header, d_nonces, d_outputs, BATCH_SIZE);
    cudaDeviceSynchronize();
    
    fmt::print("Running benchmark...\n");
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < ITERATIONS; i++) {
        batch_sha256d_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_header, d_nonces, d_outputs, BATCH_SIZE);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate metrics
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double seconds = duration / 1e6;
    double total_hashes = BATCH_SIZE * ITERATIONS;
    double hashes_per_sec = total_hashes / seconds;
    double mhashes_per_sec = hashes_per_sec / 1e6;
    
    fmt::print("\nResults:\n");
    fmt::print("  Total time: {:.3f} seconds\n", seconds);
    fmt::print("  Total hashes: {:.0f}\n", total_hashes);
    fmt::print("  Hashrate: {:.2f} MH/s\n", mhashes_per_sec);
    fmt::print("  Time per hash: {:.2f} µs\n", (duration / total_hashes));
    
    // Verify a sample output
    cudaMemcpy(h_outputs, d_outputs, 32, cudaMemcpyDeviceToHost);
    fmt::print("\nSample hash (nonce 0):\n  ");
    for (int i = 0; i < 32; i++) {
        fmt::print("{:02x}", h_outputs[i]);
    }
    fmt::print("\n");
    
    // Cleanup
    cudaFree(d_header);
    cudaFree(d_nonces);
    cudaFree(d_outputs);
    delete[] h_nonces;
    delete[] h_outputs;
    
    return 0;
}
