/*
 * Test standalone: Compara SHA256d device vs OpenSSL com header exato
 */
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

// Inline do sha256_device.cuh para teste
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

__device__ void sha256_transform_dev(uint32_t state[8], const uint32_t block[16]) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    uint32_t w[64];
    
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = block[i];
    
    #pragma unroll
    for (int i = 16; i < 64; i++)
        w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];
    
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e, f, g) + k[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void sha256d_80_bytes_dev(const uint8_t data[80], uint32_t hash[8]) {
    // First SHA256
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Block 1: First 64 bytes
    uint32_t block1[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // Read as BIG-ENDIAN: byte 0 is MSB
        block1[i] = ((uint32_t)data[i * 4 + 0] << 24) |
                    ((uint32_t)data[i * 4 + 1] << 16) |
                    ((uint32_t)data[i * 4 + 2] << 8) |
                    ((uint32_t)data[i * 4 + 3]);
    }
    
    printf("First block[0] = %08x (should be big-endian: 01000000)\n", block1[0]);
    printf("First block[9] = %08x (should be: 3ba3edfd)\n", block1[9]);
    
    sha256_transform_dev(state, block1);
    printf("After block1, state[0] = %08x\n", state[0]);
    
    // Block 2: Last 16 bytes + padding + length
    uint32_t block2[16] = {0};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Read as BIG-ENDIAN: byte 0 is MSB
        block2[i] = ((uint32_t)data[64 + i * 4 + 0] << 24) |
                    ((uint32_t)data[64 + i * 4 + 1] << 16) |
                    ((uint32_t)data[64 + i * 4 + 2] << 8) |
                    ((uint32_t)data[64 + i * 4 + 3]);
    }
    block2[4] = 0x80000000;
    block2[15] = 640;
    
    printf("Second block[0] = %08x (ntime)\n", block2[0]);
    printf("Second block[1] = %08x (bits)\n", block2[1]);
    printf("Second block[2] = %08x (nonce)\n", block2[2]);
    
    sha256_transform_dev(state, block2);
    
    printf("After block2 (first SHA256 done), state[0] = %08x\n", state[0]);
    
    // Second SHA256
    uint8_t intermediate_bytes[32];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        intermediate_bytes[i * 4 + 0] = (state[i] >> 24) & 0xFF;
        intermediate_bytes[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        intermediate_bytes[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        intermediate_bytes[i * 4 + 3] = state[i] & 0xFF;
    }
    
    uint32_t state2[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    uint32_t block[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        block[i] = ((uint32_t)intermediate_bytes[i * 4 + 0] << 24) |
                   ((uint32_t)intermediate_bytes[i * 4 + 1] << 16) |
                   ((uint32_t)intermediate_bytes[i * 4 + 2] << 8) |
                   ((uint32_t)intermediate_bytes[i * 4 + 3]);
    }
    block[8] = 0x80000000;
    block[15] = 256;
    
    sha256_transform_dev(state2, block);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) hash[i] = state2[i];
}

__global__ void test_kernel(const uint8_t* d_header, uint32_t* d_hash) {
    sha256d_80_bytes_dev(d_header, d_hash);
}

int main() {
    // Header exato do teste
    const uint8_t header[80] = {
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2, 0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
        0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32, 0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
        0x29, 0xab, 0x5f, 0x49,
        0x00, 0x00, 0x00, 0x00,
        0x1d, 0xac, 0x2b, 0x7c
    };
    
    // OpenSSL
    uint8_t openssl_hash[32];
    uint8_t temp[32];
    SHA256(header, 80, temp);
    SHA256(temp, 32, openssl_hash);
    
    printf("OpenSSL SHA256d: ");
    for (int i = 0; i < 8; i++) {
        uint32_t word = (openssl_hash[i*4] << 24) | (openssl_hash[i*4+1] << 16) |
                        (openssl_hash[i*4+2] << 8) | openssl_hash[i*4+3];
        printf("%08x ", word);
    }
    printf("\n");
    
    // CUDA
    uint8_t* d_header;
    uint32_t* d_hash;
    uint32_t h_hash[8];
    
    cudaMalloc(&d_header, 80);
    cudaMalloc(&d_hash, 32);
    cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice);
    
    test_kernel<<<1, 1>>>(d_header, d_hash);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_hash, d_hash, 32, cudaMemcpyDeviceToHost);
    
    printf("Device SHA256d:  ");
    for (int i = 0; i < 8; i++) printf("%08x ", h_hash[i]);
    printf("\n");
    
    // Compare
    bool match = true;
    for (int i = 0; i < 8; i++) {
        uint32_t openssl_word = (openssl_hash[i*4] << 24) | (openssl_hash[i*4+1] << 16) |
                                (openssl_hash[i*4+2] << 8) | openssl_hash[i*4+3];
        if (h_hash[i] != openssl_word) {
            match = false;
            break;
        }
    }
    
    printf("\nResult: %s\n", match ? "✓ MATCH" : "✗ MISMATCH");
    
    cudaFree(d_header);
    cudaFree(d_hash);
    
    return match ? 0 : 1;
}
