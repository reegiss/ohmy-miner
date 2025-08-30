/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 *
 * qhash_kernels.cu
 *
 * GPU kernel for QTC core (QHash miner). Device-safe SHA256 (ctx/update/final),
 * double-SHA256 finalization, cooperative quantum state simulation using shared memory,
 * and robust target comparison.
 *
 * NOTES:
 * - Default NUM_QUBITS = 12 (4096 amplitudes). This fits typical shared memory budgets.
 * - For NUM_QUBITS > 12 you must redesign to use global-memory tiling / multi-kernel pipeline.
 * - Host must launch kernel with shared mem size >= sizeof(cuFloatComplex) * STATE_VECTOR_SIZE
 *   + sizeof(float) * NUM_QUBITS * threads_per_block (as used in host bridge below).
 */

#include "qhash_kernels.cuh"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>

#ifndef NUM_QUBITS
#define NUM_QUBITS 12
#endif

#define NUM_LAYERS 2
#define SHA256_DIGEST_SIZE 32
constexpr int STATE_VECTOR_SIZE = 1 << NUM_QUBITS;
static_assert(NUM_QUBITS >= 1 && NUM_QUBITS <= 16, "NUM_QUBITS must be in [1..16] for this implementation");

// ----------------------- CUDA Helpers --------------------------------------
#define CUDA_CHECK_RET(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return 0xFFFFFFFFu; \
    } \
} while(0)

__constant__ static const uint32_t Kc[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t s0_dev(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
__device__ __forceinline__ uint32_t s1_dev(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
__device__ __forceinline__ uint32_t S0_dev(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t S1_dev(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }
__device__ __forceinline__ uint32_t Ch_dev(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj_dev(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }

// ----------------------- SHA256 device (ctx/update/final) ------------------
struct Sha256Ctx {
    uint32_t h[8];
    uint8_t buf[64];
    int idx;
    uint64_t bits;
};

__device__ void sha256_init(Sha256Ctx* c) {
    c->h[0] = 0x6a09e667u; c->h[1] = 0xbb67ae85u; c->h[2] = 0x3c6ef372u; c->h[3] = 0xa54ff53au;
    c->h[4] = 0x510e527fu; c->h[5] = 0x9b05688cu; c->h[6] = 0x1f83d9abu; c->h[7] = 0x5be0cd19u;
    c->idx = 0; c->bits = 0;
}

__device__ void sha256_transform(uint32_t* h, const uint8_t* blk) {
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)blk[4*i] << 24) | ((uint32_t)blk[4*i+1] << 16) | ((uint32_t)blk[4*i+2] << 8) | ((uint32_t)blk[4*i+3]);
    }
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        W[i] = S1_dev(W[i-2]) + W[i-7] + S0_dev(W[i-15]) + W[i-16];
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = hh + s1_dev(e) + Ch_dev(e,f,g) + Kc[i] + W[i];
        uint32_t t2 = s0_dev(a) + Maj_dev(a,b,c);
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

__device__ void sha256_update(Sha256Ctx* c, const uint8_t* data, int len) {
    c->bits += (uint64_t)len * 8ULL;
    int i = 0;
    if (c->idx) {
        int n = min(64 - c->idx, len);
        for (int k = 0; k < n; ++k) c->buf[c->idx + k] = data[k];
        c->idx += n; i += n;
        if (c->idx == 64) { sha256_transform(c->h, c->buf); c->idx = 0; }
    }
    for (; i + 64 <= len; i += 64) sha256_transform(c->h, data + i);
    int rem = len - i;
    for (int k = 0; k < rem; ++k) c->buf[c->idx + k] = data[i + k];
    c->idx += rem;
}

__device__ void sha256_final(Sha256Ctx* c, uint8_t out[32]) {
    c->buf[c->idx++] = 0x80;
    if (c->idx > 56) {
        for (int k = c->idx; k < 64; ++k) c->buf[k] = 0;
        sha256_transform(c->h, c->buf);
        c->idx = 0;
    }
    for (int k = c->idx; k < 56; ++k) c->buf[k] = 0;
    uint64_t bits = c->bits;
    for (int i = 0; i < 8; ++i) c->buf[63 - i] = (uint8_t)(bits >> (8 * i));
    sha256_transform(c->h, c->buf);
    for (int i = 0; i < 8; ++i) {
        uint32_t w = c->h[i];
        out[4*i + 0] = (uint8_t)(w >> 24);
        out[4*i + 1] = (uint8_t)(w >> 16);
        out[4*i + 2] = (uint8_t)(w >> 8);
        out[4*i + 3] = (uint8_t)(w);
    }
}

// device-side lexicographic compare (both arrays big-endian bytes)
__device__ __forceinline__ bool hash_le_target_be(const uint8_t hash[32], const uint8_t target[32]) {
    for (int i = 0; i < 32; ++i) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true; // equal
}

// ------------------- Complex helpers for state vector ops -------------------
__device__ __forceinline__ cuFloatComplex c_add(const cuFloatComplex &a, const cuFloatComplex &b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ cuFloatComplex c_sub(const cuFloatComplex &a, const cuFloatComplex &b) {
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ cuFloatComplex c_mul_real(const cuFloatComplex &a, float r) {
    return make_cuFloatComplex(a.x * r, a.y * r);
}

// Ry on a pair (v0, v1)
__device__ __forceinline__ void apply_Ry_pair(cuFloatComplex &v0, cuFloatComplex &v1, float theta) {
    float ca = cosf(0.5f * theta);
    float sa = sinf(0.5f * theta);
    cuFloatComplex nv0 = c_sub(c_mul_real(v0, ca), c_mul_real(v1, sa));
    cuFloatComplex nv1 = c_add(c_mul_real(v0, sa), c_mul_real(v1, ca));
    v0 = nv0; v1 = nv1;
}

// Rx on a pair (v0, v1)
__device__ __forceinline__ void apply_Rx_pair(cuFloatComplex &v0, cuFloatComplex &v1, float theta) {
    float ca = cosf(0.5f * theta);
    float sa = sinf(0.5f * theta);
    // -i * z  => (z.y, -z.x)
    cuFloatComplex minus_i_v1 = make_cuFloatComplex(v1.y, -v1.x);
    cuFloatComplex minus_i_v0 = make_cuFloatComplex(v0.y, -v0.x);
    cuFloatComplex nv0 = c_sub(c_mul_real(v0, ca), c_mul_real(minus_i_v1, sa));
    cuFloatComplex nv1 = c_add(c_mul_real(minus_i_v0, sa), c_mul_real(v1, ca));
    v0 = nv0; v1 = nv1;
}

// ------------------------- QTC Kernel --------------------------------------
// Cooperative kernel: one block per nonce, threads in block cooperate on state vector
extern "C" __global__ void qtc_kernel(
    const uint8_t* header_template_d, const uint8_t* target_d,
    uint32_t start_nonce, uint32_t total_nonces, uint32_t* found_nonce_out_d)
{
    uint32_t nonce_index = blockIdx.x;
    if (nonce_index >= total_nonces) return;
    uint32_t nonce = start_nonce + nonce_index;

    // quick atomic read (safe hint)
    if (atomicAdd(found_nonce_out_d, 0) != 0xFFFFFFFFu) return;

    const int tid = threadIdx.x;
    const int tcount = blockDim.x;

    // Shared memory layout (host must pass enough shared mem):
    // [0 .. STATE_VECTOR_SIZE-1] : cuFloatComplex state vector
    // [sv_end .. sv_end + NUM_QUBITS * tcount -1] : partial reductions (float)
    extern __shared__ uint8_t shared_mem_raw[];
    cuFloatComplex* s_sv = reinterpret_cast<cuFloatComplex*>(shared_mem_raw);
    float* s_partials = reinterpret_cast<float*>(shared_mem_raw + sizeof(cuFloatComplex) * STATE_VECTOR_SIZE);

    // --- Stage 1: Build header (avoid memcpy in device) ---
    uint8_t header_block[80];
    for (int i = 0; i < 76; ++i) header_block[i] = header_template_d[i];
    header_block[76] = (uint8_t)(nonce & 0xFFu);
    header_block[77] = (uint8_t)((nonce >> 8) & 0xFFu);
    header_block[78] = (uint8_t)((nonce >> 16) & 0xFFu);
    header_block[79] = (uint8_t)((nonce >> 24) & 0xFFu);

    Sha256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, header_block, 80);
    uint8_t hash1[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, hash1);

    // --- split nibbles from hash1 ---
    uint8_t nibbles[SHA256_DIGEST_SIZE * 2];
    for (int i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        uint8_t b = hash1[i];
        nibbles[2*i] = (b >> 4) & 0x0F;
        nibbles[2*i + 1] = b & 0x0F;
    }

    // --- Stage 2: Initialize state vector cooperatively (shared memory) ---
    for (uint32_t idx = tid; idx < (uint32_t)STATE_VECTOR_SIZE; idx += tcount) {
        s_sv[idx] = make_cuFloatComplex((idx == 0u) ? 1.0f : 0.0f, 0.0f);
    }
    __syncthreads();

    // Apply layers: per qubit rotations and CNOT chain
    for (int l = 0; l < NUM_LAYERS; ++l) {
        for (int q = 0; q < NUM_QUBITS; ++q) {
            float angle0 = - (float)nibbles[(2*l*NUM_QUBITS + q) % (SHA256_DIGEST_SIZE*2)] * CUDART_PI_F / 16.0f; // Ry
            float angle1 = - (float)nibbles[((2*l + 1)*NUM_QUBITS + q) % (SHA256_DIGEST_SIZE*2)] * CUDART_PI_F / 16.0f; // Rx
            uint32_t mask = 1u << q;

            // Ry on pairs (idx where bit q == 0)
            for (uint32_t idx = tid; idx < (uint32_t)STATE_VECTOR_SIZE; idx += tcount) {
                if ((idx & mask) == 0u) {
                    uint32_t idx0 = idx;
                    uint32_t idx1 = idx | mask;
                    cuFloatComplex v0 = s_sv[idx0];
                    cuFloatComplex v1 = s_sv[idx1];
                    apply_Ry_pair(v0, v1, angle0);
                    s_sv[idx0] = v0;
                    s_sv[idx1] = v1;
                }
            }
            __syncthreads();

            // Rx on same pairs
            for (uint32_t idx = tid; idx < (uint32_t)STATE_VECTOR_SIZE; idx += tcount) {
                if ((idx & mask) == 0u) {
                    uint32_t idx0 = idx;
                    uint32_t idx1 = idx | mask;
                    cuFloatComplex v0 = s_sv[idx0];
                    cuFloatComplex v1 = s_sv[idx1];
                    apply_Rx_pair(v0, v1, angle1);
                    s_sv[idx0] = v0;
                    s_sv[idx1] = v1;
                }
            }
            __syncthreads();
        }

        // CNOT chain: control i -> target i+1
        for (int i = 0; i < NUM_QUBITS - 1; ++i) {
            uint32_t c = 1u << i;
            uint32_t t = 1u << (i + 1);
            uint32_t mask = c | t;
            for (uint32_t idx = tid; idx < (uint32_t)STATE_VECTOR_SIZE; idx += tcount) {
                if ((idx & mask) == c) {
                    uint32_t idx0 = idx;
                    uint32_t idx1 = idx | t;
                    cuFloatComplex tmp = s_sv[idx0];
                    s_sv[idx0] = s_sv[idx1];
                    s_sv[idx1] = tmp;
                }
            }
            __syncthreads();
        }
    }

    // --- Stage 3: Measure Z expectations per qubit (block reduction) ---
    // Each thread computes partial sums for all qubits and stores in s_partials
    for (int q = 0; q < NUM_QUBITS; ++q) {
        s_partials[q * tcount + tid] = 0.0f;
    }
    __syncthreads();

    for (uint32_t idx = tid; idx < (uint32_t)STATE_VECTOR_SIZE; idx += tcount) {
        float prob = s_sv[idx].x * s_sv[idx].x + s_sv[idx].y * s_sv[idx].y;
        for (int q = 0; q < NUM_QUBITS; ++q) {
            float sign = ((idx & (1u << q)) == 0u) ? 1.0f : -1.0f;
            s_partials[q * tcount + tid] += sign * prob;
        }
    }
    __syncthreads();

    // Thread 0 reduces partials into expectations[]
    float expectations[NUM_QUBITS];
    if (tid == 0) {
        for (int q = 0; q < NUM_QUBITS; ++q) {
            float sum = 0.0f;
            for (int t = 0; t < tcount; ++t) sum += s_partials[q * tcount + t];
            expectations[q] = sum;
        }

        // Build final input: hash1 (32 bytes) + fixed-point expectations (int16_t little-endian)
        uint8_t final_input[SHA256_DIGEST_SIZE + NUM_QUBITS * 2];
        for (int i = 0; i < SHA256_DIGEST_SIZE; ++i) final_input[i] = hash1[i];
        for (int q = 0; q < NUM_QUBITS; ++q) {
            int16_t fixedv = (int16_t) (expectations[q] * (1 << 15));
            final_input[SHA256_DIGEST_SIZE + 2*q + 0] = (uint8_t)(fixedv & 0xFF);
            final_input[SHA256_DIGEST_SIZE + 2*q + 1] = (uint8_t)((fixedv >> 8) & 0xFF);
        }

        // Double SHA256 (sha256d)
        Sha256Ctx ctx2;
        sha256_init(&ctx2);
        sha256_update(&ctx2, final_input, sizeof(final_input));
        uint8_t tmp_hash[SHA256_DIGEST_SIZE];
        sha256_final(&ctx2, tmp_hash);

        Sha256Ctx ctx3;
        sha256_init(&ctx3);
        sha256_update(&ctx3, tmp_hash, SHA256_DIGEST_SIZE);
        uint8_t final_hash[SHA256_DIGEST_SIZE];
        sha256_final(&ctx3, final_hash);

        // Compare final_hash (big-endian bytes) to target_d (assumed big-endian bytes from host)
        if (hash_le_target_be(final_hash, target_d)) {
            atomicMin(found_nonce_out_d, nonce);
        }
    }
    // end kernel
}

// ------------------------- Host bridge -------------------------------------
extern "C" uint32_t qhash_search_batch(const uint8_t* header_h, const uint8_t* target_h, uint32_t start_nonce, uint32_t num_nonces) {
    if (!header_h || !target_h || num_nonces == 0) return 0xFFFFFFFFu;

    uint8_t* d_header = nullptr;
    uint8_t* d_target = nullptr;
    uint32_t* d_found = nullptr;
    uint32_t h_found = 0xFFFFFFFFu;

    CUDA_CHECK_RET(cudaMalloc(&d_header, 76));
    CUDA_CHECK_RET(cudaMalloc(&d_target, 32));
    CUDA_CHECK_RET(cudaMalloc(&d_found, sizeof(uint32_t)));

    CUDA_CHECK_RET(cudaMemcpy(d_header, header_h, 76, cudaMemcpyHostToDevice));
    CUDA_CHECK_RET(cudaMemcpy(d_target, target_h, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK_RET(cudaMemcpy(d_found, &h_found, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch: one block per nonce (caller must cap workload if > max grid)
    uint32_t blocks = num_nonces;
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK_RET(cudaGetDevice(&device));
    CUDA_CHECK_RET(cudaGetDeviceProperties(&prop, device));

    // conservative max blocks for 1D grid on older devices
    uint32_t maxBlocks = (uint32_t)prop.maxGridSize[0];
    if (blocks > maxBlocks) blocks = maxBlocks;

    int threads_per_block = 256;
    size_t sv_bytes = sizeof(cuFloatComplex) * STATE_VECTOR_SIZE;
    size_t partials_bytes = sizeof(float) * NUM_QUBITS * threads_per_block;
    size_t shared_bytes = sv_bytes + partials_bytes;

    // try smaller threads if shared needed is too large
    if (shared_bytes > prop.sharedMemPerBlock) {
        threads_per_block = 128;
        partials_bytes = sizeof(float) * NUM_QUBITS * threads_per_block;
        shared_bytes = sv_bytes + partials_bytes;
        if (shared_bytes > prop.sharedMemPerBlock) {
            // cannot allocate; bail out
            printf("[HOST] Insufficient shared memory per block: required=%zu available=%zu\n", shared_bytes, prop.sharedMemPerBlock);
            cudaFree(d_header); cudaFree(d_target); cudaFree(d_found);
            return 0xFFFFFFFFu;
        }
    }

    qtc_kernel<<<blocks, threads_per_block, shared_bytes>>>(d_header, d_target, start_nonce, num_nonces, d_found);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[HOST] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_header); cudaFree(d_target); cudaFree(d_found);
        return 0xFFFFFFFFu;
    }

    CUDA_CHECK_RET(cudaDeviceSynchronize());
    CUDA_CHECK_RET(cudaMemcpy(&h_found, d_found, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_found);
    return h_found;
}
