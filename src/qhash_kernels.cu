/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 *
 * qhash_kernels.cu
 *
 * Device SHA256 implementation + kernel to check nonces in parallel.
 *
 * Build: compiled by nvcc as part of your CMake project.
 */

#include <algorithm>
#include "qhash_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ------------------------------ CUDA error check ----------------------------
#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 0xFFFFFFFFu; \
    } \
} while(0)

// ------------------------------ SHA256 constants ----------------------------
__device__ __constant__ static const uint32_t Kc[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

// rotate right
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// SHA256 small helpers
__device__ __forceinline__ uint32_t S0(uint32_t x) { return rotr(x, 2) ^ rotr(x,13) ^ rotr(x,22); }
__device__ __forceinline__ uint32_t S1(uint32_t x) { return rotr(x, 6) ^ rotr(x,11) ^ rotr(x,25); }
__device__ __forceinline__ uint32_t s0(uint32_t x) { return rotr(x, 7) ^ rotr(x,18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t s1(uint32_t x) { return rotr(x,17) ^ rotr(x,19) ^ (x >>10); }
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

// parse big-endian 32-bit from 4 bytes
__device__ __forceinline__ uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | ((uint32_t)p[3]);
}

// write big-endian 32-bit to bytes
__device__ __forceinline__ void write_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v);
}

// SHA256 transform on a single 64-byte block. Input is big-endian bytes.
__device__ void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = be32(block + i*4);
    }
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        W[i] = s1(W[i-2]) + W[i-7] + s0(W[i-15]) + W[i-16];
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + S1(e) + Ch(e,f,g) + Kc[i] + W[i];
        uint32_t T2 = S0(a) + Maj(a,b,c);
        h = g; g = f; f = e; e = d + T1; d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// Full SHA256 for a message <= 64+ bytes implemented inline (we need 80 bytes header).
__device__ void sha256_full(const uint8_t* data, int len, uint8_t out32[32]) {
    // initialize
    uint32_t h[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    // We will process up to two blocks (80 bytes)
    uint8_t block[64];
    int processed = 0;
    // process first block(s)
    if (len >= 64) {
        // first 64
        for (int i = 0; i < 64; ++i) block[i] = data[i];
        sha256_transform(h, block);
        processed = 64;
    } else {
        // less than 64
        for (int i = 0; i < len; ++i) block[i] = data[i];
        processed = 0;
    }
    int rem = len - processed;
    // copy remaining bytes
    if (rem > 0) {
        for (int i = 0; i < rem; ++i) block[i] = data[processed + i];
    }
    // append 0x80
    block[rem] = 0x80;
    // zero pad
    for (int i = rem + 1; i < 64; ++i) block[i] = 0;
    if (rem >= 56) {
        sha256_transform(h, block);
        // prepare new block of zeros
        for (int i = 0; i < 56; ++i) block[i] = 0;
    }
    // put bit length big-endian (len*8)
    unsigned long long bits = (unsigned long long)len * 8ULL;
    // last 8 bytes big-endian
    for (int i = 0; i < 8; ++i) block[63 - i] = (uint8_t)(bits >> (8 * i));
    sha256_transform(h, block);
    // output big-endian bytes
    for (int i = 0; i < 8; ++i) {
        write_be32(out32 + i*4, h[i]);
    }
}

// double SHA256 (sha256d)
__device__ void sha256d_device(const uint8_t* data, int len, uint8_t out32[32]) {
    uint8_t tmp[32];
    sha256_full(data, len, tmp);    // tmp is big-endian bytes of first sha
    sha256_full(tmp, 32, out32);    // out32 is big-endian bytes of double sha
}

// compare hash (big-endian) <= target (big-endian)
__device__ __forceinline__ bool hash_le_target_be(const uint8_t hash32[32], const uint8_t target32[32]) {
    for (int i = 0; i < 32; ++i) {
        if (hash32[i] < target32[i]) return true;
        if (hash32[i] > target32[i]) return false;
    }
    return true; // equal
}

// ------------------------------ Kernel -------------------------------------
// Each thread tests nonces in stride. header76 points to device memory containing 76 bytes.
__global__ void qhash_kernel_device(
    const uint8_t* header76_d,
    const uint8_t* target_be_d,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce_out) 
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    uint64_t idx0 = (uint64_t)tid;

    // local buffer for header+nonce (80 bytes)
    uint8_t header80[80];
    // copy 76 bytes from global into local (safe; 80B per thread)
    for (int i = 0; i < 76; ++i) header80[i] = header76_d[i];

    // iterate through assigned nonces
    uint32_t limit = start_nonce + num_nonces;
    // start at nonce = start_nonce + tid
    uint64_t nonce = (uint64_t)start_nonce + idx0;
    for (; nonce < (uint64_t)limit; nonce += stride) {
        // write nonce as little-endian into last 4 bytes
        header80[76] = static_cast<uint8_t>(nonce & 0xFFu);
        header80[77] = static_cast<uint8_t>((nonce >> 8) & 0xFFu);
        header80[78] = static_cast<uint8_t>((nonce >> 16) & 0xFFu);
        header80[79] = static_cast<uint8_t>((nonce >> 24) & 0xFFu);

        uint8_t final_hash[32];
        sha256d_device(header80, 80, final_hash); // final_hash is big-endian bytes

        if (hash_le_target_be(final_hash, target_be_d)) {
            // attempt to set minimum nonce
            atomicMin(found_nonce_out, (uint32_t)nonce);
        }
    }
}

extern "C" uint32_t qhash_search_batch(
    const uint8_t* header_h,
    const uint8_t* target_h,
    uint32_t start_nonce,
    uint32_t num_nonces)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // garantir consistência de tipos para evitar conflito com min/max
    uint32_t max_blocks = static_cast<uint32_t>(prop.multiProcessorCount) * 32u;
    uint32_t blocks = std::min<uint32_t>(
        std::max(1u, max_blocks),
        num_nonces
    );

    uint32_t threads = 256; // valor padrão (ajustável)

    // alocação de memória no device
    uint8_t* d_header = nullptr;
    uint8_t* d_target = nullptr;
    uint32_t* d_found_nonce = nullptr;
    uint32_t found_nonce = UINT32_MAX;

    cudaMalloc(&d_header, 80);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_found_nonce, sizeof(uint32_t));

    cudaMemcpy(d_header, header_h, 80, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_h, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &found_nonce, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // chamada do kernel
    qhash_kernel<<<blocks, threads>>>(d_header, d_target, start_nonce, num_nonces, d_found_nonce);

    cudaMemcpy(&found_nonce, d_found_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_found_nonce);

    return found_nonce;
}