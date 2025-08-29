#include "qhash_kernels.cuh"
#include <cuda_runtime.h>

// =============================================================================
// == SHA256 Implementation for CUDA Devices
//    (This is a self-contained device implementation to fix compilation)
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

__device__ void sha256_transform_device(uint32_t *state, const uint32_t *block, bool swap) {
    uint32_t a, b, c, d, e, f, g, h, W[64];
    for (int i = 0; i < 16; i++) W[i] = swap ? bswap_32(block[i]) : block[i];
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

__device__ void sha256d_device(uint8_t* output, const uint8_t* input, size_t len) {
    uint32_t state[8];
    uint32_t block[16];

    // First hash
    state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
    state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
    
    memcpy(block, input, 64);
    sha256_transform_device(state, (const uint32_t*)block, true);
    memcpy(block, input + 64, 16);
    ((uint8_t*)block)[16] = 0x80;
    for(int i=5; i<14; ++i) block[i] = 0;
    block[14] = bswap_32(len * 8);
    block[15] = 0;
    sha256_transform_device(state, block, false);

    // Second hash
    uint32_t first_hash_as_block[8];
    for(int i=0; i<8; ++i) first_hash_as_block[i] = state[i]; // Use state directly (already big-endian words)
    
    state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
    state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
    // The block for the second round is already in the correct word order, so swap=false
    sha256_transform_device(state, (const uint32_t*)first_hash_as_block, false);
    
    // --- FINAL FIX IS HERE ---
    // Manually write the final state as a big-endian byte array.
    for(int i=0; i < 8; ++i) {
        output[i*4 + 0] = (state[i] >> 24) & 0xff;
        output[i*4 + 1] = (state[i] >> 16) & 0xff;
        output[i*4 + 2] = (state[i] >> 8) & 0xff;
        output[i*4 + 3] = (state[i] >> 0) & 0xff;
    }
}

// =============================================================================
// == High-Performance qhash Search Kernel (Placeholder Logic)
// =============================================================================
__global__ void qhash_search_kernel(
    const uint8_t* header_template_d,
    const uint8_t* target_d,
    uint32_t start_nonce,
    uint32_t num_nonces,
    uint32_t* found_nonce_out_d
) {
    uint32_t nonce = start_nonce + (blockIdx.x * blockDim.x + threadIdx.x);
    if (nonce >= start_nonce + num_nonces) return;
    if (atomicCAS(found_nonce_out_d, 0xFFFFFFFF, 0xFFFFFFFF) != 0xFFFFFFFF) return;

    uint8_t block_header[80];
    memcpy(block_header, header_template_d, 76);
    block_header[76] = nonce & 0xFF;
    block_header[77] = (nonce >> 8) & 0xFF;
    block_header[78] = (nonce >> 16) & 0xFF;
    block_header[79] = (nonce >> 24) & 0xFF;

    uint8_t final_hash[32];
    sha256d_device(final_hash, block_header, 80);

    // --- LÓGICA DE COMPARAÇÃO CORRIGIDA ---
    // Para números big-endian, começamos a comparar do byte mais significativo (índice 0)
    for (int i = 0; i < 32; ++i) {
        if (final_hash[i] < target_d[i]) {
            // O nosso hash é menor, encontramos uma solução.
            atomicMin(found_nonce_out_d, nonce);
            return; // Encerra a thread
        }
        if (final_hash[i] > target_d[i]) {
            // O nosso hash é maior, não é uma solução.
            return; // Encerra a thread
        }
        // Se os bytes forem iguais, continuamos para o próximo.
    }
    
    // Se o loop terminar, os hashes são idênticos, o que também é uma solução válida.
    atomicMin(found_nonce_out_d, nonce);
}

// =============================================================================
// == C++ Bridge to Launch the Kernel
// =============================================================================
uint32_t qhash_search_batch(const uint8_t* header_template_h, const uint8_t* target_h, uint32_t start_nonce, uint32_t num_nonces) {
    uint8_t* d_header;
    uint8_t* d_target;
    uint32_t* d_found_nonce;
    uint32_t h_found_nonce = 0xFFFFFFFF;

    cudaMalloc(&d_header, 76);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_found_nonce, sizeof(uint32_t));

    cudaMemcpy(d_header, header_template_h, 76, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_h, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(uint32_t), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (num_nonces + threads_per_block - 1) / threads_per_block;
    
    qhash_search_kernel<<<num_blocks, threads_per_block>>>(
        d_header,
        d_target,
        start_nonce,
        num_nonces,
        d_found_nonce
    );
    
    cudaDeviceSynchronize();
    cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_found_nonce);

    return h_found_nonce;
}