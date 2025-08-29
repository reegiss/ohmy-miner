#include "qhash_kernels.cuh"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>

// Constantes do algoritmo portadas de qhash_miner.h
#define NUM_QUBITS 16
#define NUM_LAYERS 2
#define SHA256_BLOCK_SIZE 32

// =============================================================================
// SHA256 DEVICE IMPLEMENTATION
// =============================================================================
__device__ const uint32_t K[64] = { 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
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
    a = state[0]; b = state[1]; c = state[2]; d = state[3]; e = state[4]; f = state[5]; g = state[6]; h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + s1(e) + F1(e, f, g) + K[i] + W[i];
        uint32_t t2 = s0(a) + F0(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void sha256_full_device(uint32_t *state, const uint32_t *data, int len_bytes) {
    state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a; state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
    uint32_t block[16];
    int n_blocks = len_bytes / 64;
    for (int i = 0; i < n_blocks; ++i) {
        sha256_transform_device(state, data + i * 16);
    }
    int rem = len_bytes % 64;
    memcpy(block, data + n_blocks * 16, rem);
    ((uint8_t*)block)[rem] = 0x80;
    for (int i = rem + 1; i < 56; ++i) ((uint8_t*)block)[i] = 0;
    block[14] = bswap_32((uint32_t)len_bytes * 8);
    block[15] = 0;
    sha256_transform_device(state, block);
}

// =============================================================================
// QUANTUM CIRCUIT DEVICE IMPLEMENTATION
// =============================================================================
constexpr int STATE_VECTOR_SIZE = 1 << NUM_QUBITS;

__device__ void apply_rotation(cuFloatComplex* sv, int target_q, float angle, bool is_y) {
    const float cos_a = cosf(0.5f * angle);
    const float sin_a = sinf(0.5f * angle);
    const uint32_t k = 1 << target_q;

    #pragma unroll
    for (uint32_t i = 0; i < STATE_VECTOR_SIZE / 2; ++i) {
        uint32_t i0 = (i & (k - 1)) | ((i & ~(k - 1)) << 1);
        uint32_t i1 = i0 | k;

        cuFloatComplex v0 = sv[i0];
        cuFloatComplex v1 = sv[i1];

        if (is_y) { // RY Gate
            sv[i0] = make_cuFloatComplex(v0.x * cos_a + v1.y * sin_a, v0.y * cos_a - v1.x * sin_a);
            sv[i1] = make_cuFloatComplex(v1.x * cos_a - v0.y * sin_a, v1.y * cos_a + v0.x * sin_a);
        } else { // RZ Gate
            sv[i0] = make_cuFloatComplex(v0.x * cos_a + v0.y * sin_a, v0.y * cos_a - v0.x * sin_a);
            sv[i1] = make_cuFloatComplex(v1.x * cos_a - v1.y * sin_a, v1.y * cos_a + v1.x * sin_a);
        }
    }
}

__device__ void apply_cnot(cuFloatComplex* sv, int control_q, int target_q) {
    const uint32_t c = 1 << control_q;
    const uint32_t t = 1 << target_q;
    const uint32_t mask = c | t;
    
    #pragma unroll
    for (uint32_t i = 0; i < STATE_VECTOR_SIZE; ++i) {
        if ((i & mask) == c) {
            cuFloatComplex temp = sv[i];
            sv[i] = sv[i | t];
            sv[i | t] = temp;
        }
    }
}

__device__ float get_expectation_z(cuFloatComplex* sv, int target_q) {
    const uint32_t k = 1 << target_q;
    float expectation = 0.0f;
    
    #pragma unroll
    for (uint32_t i = 0; i < STATE_VECTOR_SIZE; ++i) {
        float sign = ((i & k) == 0) ? 1.0f : -1.0f;
        expectation += sign * (sv[i].x * sv[i].x + sv[i].y * sv[i].y);
    }
    return expectation;
}

__device__ int16_t toFixed_device(float x) {
    const int32_t fractionMult = 1 << 15;
    return (x >= 0.0f) ? (x * fractionMult + 0.5f) : (x * fractionMult - 0.5f);
}

// =============================================================================
// QPOW KERNEL
// =============================================================================
__global__ void qpow_search_kernel(
    const uint8_t* header_template_d, const uint8_t* target_d,
    uint32_t start_nonce, uint32_t num_nonces, uint32_t* found_nonce_out_d) {
    
    uint32_t nonce = start_nonce + (blockIdx.x * blockDim.x + threadIdx.x);
    if (nonce >= start_nonce + num_nonces || *found_nonce_out_d != 0xFFFFFFFF) return;

    // --- Stage 1: Initial SHA256 ---
    uint8_t block_header[80];
    memcpy(block_header, header_template_d, 76);
    block_header[76] = nonce & 0xFF; block_header[77] = (nonce >> 8) & 0xFF; block_header[78] = (nonce >> 16) & 0xFF; block_header[79] = (nonce >> 24) & 0xFF;
    
    uint32_t hash1_state[8];
    sha256_full_device(hash1_state, (const uint32_t*)block_header, 80);

    // --- Split Nibbles ---
    uint8_t nibbles[2 * SHA256_BLOCK_SIZE];
    for (int i = 0; i < SHA256_BLOCK_SIZE; ++i) {
        uint8_t byte = ((uint8_t*)hash1_state)[i];
        nibbles[2 * i] = (byte >> 4) & 0xF;
        nibbles[2 * i + 1] = byte & 0xF;
    }

    // --- Stage 2: Quantum Simulation ---
    cuFloatComplex sv[STATE_VECTOR_SIZE];
    sv[0] = make_cuFloatComplex(1.0f, 0.0f);
    for (int i = 1; i < STATE_VECTOR_SIZE; ++i) sv[i] = make_cuFloatComplex(0.0f, 0.0f);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        for (int i = 0; i < NUM_QUBITS; ++i) {
            apply_rotation(sv, i, -nibbles[(2 * l * NUM_QUBITS + i) % 64] * CUDART_PI_F / 16.0f, true); // RY
            apply_rotation(sv, i, -nibbles[((2 * l + 1) * NUM_QUBITS + i) % 64] * CUDART_PI_F / 16.0f, false); // RZ
        }
        for (int i = 0; i < NUM_QUBITS - 1; ++i) {
            apply_cnot(sv, i, i + 1);
        }
    }
    
    // --- Stage 3: Final Hash ---
    uint8_t final_input[SHA256_BLOCK_SIZE + NUM_QUBITS * sizeof(int16_t)];
    memcpy(final_input, hash1_state, SHA256_BLOCK_SIZE);

    for (int i = 0; i < NUM_QUBITS; ++i) {
        float exp_val = get_expectation_z(sv, i);
        int16_t fixed_val = toFixed_device(exp_val);
        memcpy(final_input + SHA256_BLOCK_SIZE + i * sizeof(int16_t), &fixed_val, sizeof(int16_t));
    }

    uint32_t final_hash_state[8];
    sha256_full_device(final_hash_state, (const uint32_t*)final_input, sizeof(final_input));
    
    // --- Check against Target ---
    for (int i = 0; i < 8; ++i) {
        uint32_t hash_word = bswap_32(final_hash_state[i]);
        uint32_t target_word = bswap_32(((const uint32_t*)target_d)[i]);
        if (hash_word < target_word) { atomicMin(found_nonce_out_d, nonce); return; }
        if (hash_word > target_word) { return; }
    }
    atomicMin(found_nonce_out_d, nonce);
}

// =============================================================================
// C++ Bridge to Launch Kernel
// =============================================================================
uint32_t qhash_search_batch(const uint8_t* header_h, const uint8_t* target_h, uint32_t start_nonce, uint32_t num_nonces) {
    uint8_t *d_header, *d_target;
    uint32_t* d_found_nonce;
    uint32_t h_found_nonce = 0xFFFFFFFF;

    cudaMalloc(&d_header, 76);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_found_nonce, sizeof(uint32_t));

    cudaMemcpy(d_header, header_h, 76, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_h, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_nonce, &h_found_nonce, sizeof(uint32_t), cudaMemcpyHostToDevice);

    int threads_per_block = 64; // Reduzido devido ao alto uso de registradores/memória
    int num_blocks = (num_nonces + threads_per_block - 1) / threads_per_block;
    
    qpow_search_kernel<<<num_blocks, threads_per_block>>>(d_header, d_target, start_nonce, num_nonces, d_found_nonce);
    
    cudaDeviceSynchronize();
    cudaMemcpy(&h_found_nonce, d_found_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_found_nonce);

    return h_found_nonce;
}