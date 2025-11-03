/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

/**
 * @file sha256_device.cuh
 * @brief Device-side SHA256 implementation for on-the-fly qhash kernel
 * 
 * This is a lightweight, unrolled SHA256 implementation optimized for CUDA.
 * Used to compute H_initial = SHA256(block_header || nonce) inside the mining kernel.
 * 
 * Reference: FIPS 180-4 SHA-256 specification
 * https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
 */

// SHA256 constants (first 32 bits of fractional parts of cube roots of first 64 primes)
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA256 helper macros
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

/**
 * @brief SHA256 transform for a single 512-bit block
 * 
 * @param state Current hash state (8 × uint32_t)
 * @param block Input block (16 × uint32_t in big-endian)
 */
__device__ __forceinline__ void sha256_transform(uint32_t state[8], const uint32_t block[16])
{
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;
    
    // Prepare message schedule W[0..63]
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = block[i];
    }
    
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
    }
    
    // Initialize working variables
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    
    // Main loop (64 rounds)
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add compressed chunk to state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

/**
 * @brief Compute SHA256 hash of 80-byte block header
 * 
 * Optimized for Bitcoin-style block headers (exactly 80 bytes).
 * 
 * @param data Input data (80 bytes = block header)
 * @param hash Output hash (32 bytes = 8 × uint32_t)
 */
__device__ void sha256_80_bytes(const uint8_t data[80], uint32_t hash[8])
{
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Block 1: First 64 bytes of header
    // Read bytes in BIG-ENDIAN order (byte 0 is MSB) as required by SHA256
    uint32_t block1[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        block1[i] = ((uint32_t)data[i * 4 + 0] << 24) |
                    ((uint32_t)data[i * 4 + 1] << 16) |
                    ((uint32_t)data[i * 4 + 2] << 8) |
                    ((uint32_t)data[i * 4 + 3]);
    }
    sha256_transform(state, block1);
    
    // Block 2: Last 16 bytes + padding + length
    uint32_t block2[16] = {0};
    
    // Copy remaining 16 bytes (also big-endian)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        block2[i] = ((uint32_t)data[64 + i * 4 + 0] << 24) |
                    ((uint32_t)data[64 + i * 4 + 1] << 16) |
                    ((uint32_t)data[64 + i * 4 + 2] << 8) |
                    ((uint32_t)data[64 + i * 4 + 3]);
    }
    
    // Padding: append '1' bit followed by zeros
    block2[4] = 0x80000000;
    
    // Length in bits (80 bytes = 640 bits) at end of block
    block2[15] = 640;
    
    sha256_transform(state, block2);
    
    // Copy final state to output
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i] = state[i];
    }
}

/**
 * @brief Compute double SHA256 (SHA256d) of 80-byte header
 * 
 * Used by Bitcoin and Qubitcoin: H = SHA256(SHA256(header))
 * 
 * @param data Input data (80 bytes = block header with nonce)
 * @param hash Output hash (32 bytes = 8 × uint32_t)
 */
__device__ void sha256d_80_bytes(const uint8_t data[80], uint32_t hash[8])
{
    uint32_t intermediate[8];
    
    // First SHA256
    sha256_80_bytes(data, intermediate);
    
    // Convert intermediate hash to bytes for second round
    // Output from SHA256 is big-endian internally, keep that for second round
    uint8_t intermediate_bytes[32];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        intermediate_bytes[i * 4 + 0] = (intermediate[i] >> 24) & 0xFF;
        intermediate_bytes[i * 4 + 1] = (intermediate[i] >> 16) & 0xFF;
        intermediate_bytes[i * 4 + 2] = (intermediate[i] >> 8) & 0xFF;
        intermediate_bytes[i * 4 + 3] = intermediate[i] & 0xFF;
    }
    
    // Second SHA256 (on 32 bytes)
    // Initial state
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Single block: 32 bytes + padding + length
    uint32_t block[16] = {0};
    
    // Copy 32 bytes (already in big-endian from SHA256 output, load as-is)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        block[i] = ((uint32_t)intermediate_bytes[i * 4 + 0] << 24) |
                   ((uint32_t)intermediate_bytes[i * 4 + 1] << 16) |
                   ((uint32_t)intermediate_bytes[i * 4 + 2] << 8) |
                   ((uint32_t)intermediate_bytes[i * 4 + 3]);
    }
    
    // Padding
    block[8] = 0x80000000;
    
    // Length in bits (32 bytes = 256 bits)
    block[15] = 256;
    
    sha256_transform(state, block);
    
    // Copy final state to output
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i] = state[i];
    }
}

#undef ROTR
#undef CH
#undef MAJ
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
