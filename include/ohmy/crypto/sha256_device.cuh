#pragma once

#include <cstdint>

namespace ohmy::crypto {

// SHA256 Constants (first 32 bits of fractional parts of cube roots of first 64 primes)
__constant__ const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA256 initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
__constant__ const uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA256 Context
struct SHA256_CTX {
    uint32_t state[8];
    uint32_t count[2];
    uint8_t buffer[64];
};

// Rotate right
__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA256 Functions
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t Sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t Sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Initialize SHA256 context
__device__ void sha256_init(SHA256_CTX* ctx) {
    ctx->count[0] = 0;
    ctx->count[1] = 0;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ctx->state[i] = H0[i];
    }
}

// Transform 512-bit block
__device__ void sha256_transform(SHA256_CTX* ctx, const uint8_t data[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t T1, T2;
    
    // Prepare message schedule (big-endian)
    #pragma unroll 16
    for (int i = 0; i < 16; i++) {
        W[i] = (uint32_t(data[i * 4 + 0]) << 24) |
               (uint32_t(data[i * 4 + 1]) << 16) |
               (uint32_t(data[i * 4 + 2]) << 8)  |
               (uint32_t(data[i * 4 + 3]));
    }
    
    #pragma unroll 48
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }
    
    // Initialize working variables
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];
    
    // 64 rounds
    #pragma unroll 64
    for (int i = 0; i < 64; i++) {
        T1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
        T2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }
    
    // Add compressed chunk to current hash value
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

// Update SHA256 with data
__device__ void sha256_update(SHA256_CTX* ctx, const uint8_t* data, uint32_t len) {
    uint32_t i = 0;
    uint32_t index = (ctx->count[0] >> 3) & 0x3F;
    
    ctx->count[0] += len << 3;
    if (ctx->count[0] < (len << 3)) {
        ctx->count[1]++;
    }
    ctx->count[1] += len >> 29;
    
    uint32_t partLen = 64 - index;
    
    if (len >= partLen) {
        // Fill buffer and transform
        for (uint32_t j = 0; j < partLen; j++) {
            ctx->buffer[index + j] = data[j];
        }
        sha256_transform(ctx, ctx->buffer);
        
        // Transform complete blocks
        for (i = partLen; i + 63 < len; i += 64) {
            sha256_transform(ctx, &data[i]);
        }
        
        index = 0;
    }
    
    // Buffer remaining input
    uint32_t remaining = len - i;
    for (uint32_t j = 0; j < remaining; j++) {
        ctx->buffer[index + j] = data[i + j];
    }
}

// Finalize SHA256 and produce digest
__device__ void sha256_final(SHA256_CTX* ctx, uint8_t hash[32]) {
    uint8_t bits[8];
    uint32_t index = (ctx->count[0] >> 3) & 0x3F;
    uint32_t padLen = (index < 56) ? (56 - index) : (120 - index);
    
    // Encode bit count (big-endian)
    bits[0] = (ctx->count[1] >> 24) & 0xFF;
    bits[1] = (ctx->count[1] >> 16) & 0xFF;
    bits[2] = (ctx->count[1] >> 8) & 0xFF;
    bits[3] = (ctx->count[1]) & 0xFF;
    bits[4] = (ctx->count[0] >> 24) & 0xFF;
    bits[5] = (ctx->count[0] >> 16) & 0xFF;
    bits[6] = (ctx->count[0] >> 8) & 0xFF;
    bits[7] = (ctx->count[0]) & 0xFF;
    
    // Pad with 0x80 followed by zeros
    uint8_t padding[64] = {0x80};
    sha256_update(ctx, padding, padLen);
    
    // Append length
    sha256_update(ctx, bits, 8);
    
    // Output hash (big-endian)
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        hash[i * 4 + 0] = (ctx->state[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (ctx->state[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (ctx->state[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = (ctx->state[i]) & 0xFF;
    }
}

// Convenience function: Single-shot SHA256 (single round)
__device__ void sha256_hash(const uint8_t* input, uint32_t len, uint8_t output[32]) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, input, len);
    sha256_final(&ctx, output);
}

// SHA256d: Double SHA256 (Bitcoin-style)
__device__ void sha256d(const uint8_t* input, uint32_t len, uint8_t output[32]) {
    uint8_t temp[32];
    sha256_hash(input, len, temp);
    sha256_hash(temp, 32, output);
}

} // namespace ohmy::crypto
