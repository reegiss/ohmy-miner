#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <cuComplex.h> 
#include <iostream>

extern "C" {
#include "qhash_miner.h"
}

// =============================================================================
// == SHA256 Implementation for CUDA Devices
// =============================================================================
__device__ const uint32_t K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ uint32_t ROTR(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t S0(uint32_t x) { return ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t S1(uint32_t x) { return ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10); }
__device__ __forceinline__ uint32_t s0(uint32_t x) { return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22); }
__device__ __forceinline__ uint32_t s1(uint32_t x) { return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25); }
__device__ __forceinline__ uint32_t F0(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (z & (x | y)); }
__device__ __forceinline__ uint32_t F1(uint32_t x, uint32_t y, uint32_t z) { return z ^ (x & (y ^ z)); }
__device__ __forceinline__ uint32_t bswap_32(uint32_t x) { return ((x & 0xff000000) >> 24) | ((x & 0x00ff0000) >> 8) | ((x & 0x0000ff00) << 8) | ((x & 0x000000ff) << 24); }

__device__ void sha256_transform_device(uint32_t *state, const uint32_t *block) {
    uint32_t a, b, c, d, e, f, g, h, W[64];
    for (int i = 0; i < 16; i++) W[i] = bswap_32(block[i]);
    for (int i = 16; i < 64; i++) W[i] = S1(W[i - 2]) + W[i - 7] + S0(W[i - 15]) + W[i - 16];
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + s1(e) + F1(e, f, g) + K[i] + W[i];
        uint32_t t2 = s0(a) + F0(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// =============================================================================
// == High Performance qhash Kernel
// =============================================================================

__global__ void qhash_search_kernel(
    const uint8_t* header_template_d, 
    const uint64_t* target_d,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce_out_d
) {
    uint32_t nonce = start_nonce + (blockIdx.x * blockDim.x + threadIdx.x);
    if (nonce >= start_nonce + num_nonces) return;
    if (atomicCAS(found_nonce_out_d, 0xFFFFFFFF, 0xFFFFFFFF) != 0xFFFFFFFF) return;

    uint32_t block_header[20];
    memcpy(block_header, header_template_d, 76);
    ((uint8_t*)block_header)[76] = nonce & 0xff;
    ((uint8_t*)block_header)[77] = (nonce >> 8) & 0xff;
    ((uint8_t*)block_header)[78] = (nonce >> 16) & 0xff;
    ((uint8_t*)block_header)[79] = (nonce >> 24) & 0xff;

    uint32_t sha256_state_1[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    sha256_transform_device(sha256_state_1, block_header);
    
    uint32_t final_block[16];
    memcpy(final_block, (uint8_t*)block_header + 64, 16);
    ((uint8_t*)final_block)[16] = 0x80;
    for(int i = 5; i < 14; ++i) final_block[i] = 0;
    final_block[14] = bswap_32(80 * 8);
    final_block[15] = 0;
    sha256_transform_device(sha256_state_1, final_block);

    // Placeholder for native quantum simulation. For now, we use the first hash.
    const uint64_t* hash64 = (const uint64_t*)sha256_state_1;
    if (hash64[3] < target_d[3] || 
       (hash64[3] == target_d[3] && (hash64[2] < target_d[2] ||
       (hash64[2] == target_d[2] && (hash64[1] < target_d[1] ||
       (hash64[1] == target_d[1] && hash64[0] < target_d[0])))))) {
        atomicMin(found_nonce_out_d, nonce);
    }
}

// =============================================================================
// == C++ Bridge to Launch the Kernel
// =============================================================================

uint32_t qhash_search(const uint8_t* block_header_template, const uint8_t* target, uint32_t start_nonce, uint32_t num_nonces_to_search) {
    uint8_t *d_header;
    uint64_t *d_target;
    uint32_t* d_found_nonce;
    uint32_t h_found_nonce = 0xFFFFFFFF;

    cudaMalloc(&d_header, 76);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_found_nonce, sizeof(uint32_t));

    cudaMemcpy(d_header, block_header_template, 76, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(uint32_t), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (num_nonces_to_search + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 65535) num_blocks = 65535; // Grid size limit

    qhash_search_kernel<<<num_blocks, threads_per_block>>>(
        d_header,
        (const uint64_t*)d_target,
        start_nonce,
        num_nonces_to_search,
        d_found_nonce
    );

    cudaDeviceSynchronize(); // Wait for the kernel to finish
    cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_found_nonce);

    return h_found_nonce;
}