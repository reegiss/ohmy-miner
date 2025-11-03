/*
 * Test para validar ordem de bytes no SHA256 device
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Inline minimal SHA256 transform para teste
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

__device__ void sha256_transform_test(uint32_t state[8], const uint32_t block[16]) {
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

__global__ void test_sha256_kernel(uint32_t* result) {
    // Mesmo input do teste OpenSSL: {0xAA, 0xBB, 0xCC, 0xDD} (4 bytes total)
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    printf("Kernel started, initial state[0] = %08x\n", state[0]);
    
    // Um único bloco: 4 bytes de dados + padding + length
    uint32_t block[16] = {0};
    
    // TESTE: Big-endian load (0xAA no MSB)
    block[0] = (0xAA << 24) | (0xBB << 16) | (0xCC << 8) | 0xDD;
    block[1] = 0x80000000;  // Padding bit imediatamente após dados
    block[15] = 32;         // Length: 4 bytes = 32 bits
    
    printf("block[0] = %08x, block[1] = %08x, block[15] = %08x\n", 
           block[0], block[1], block[15]);
    
    sha256_transform_test(state, block);
    printf("After transform, state[0] = %08x\n", state[0]);
    
    result[0] = state[0];
}

int main() {
    uint32_t* d_result;
    uint32_t h_result;
    
    cudaMalloc(&d_result, sizeof(uint32_t));
    cudaMemset(d_result, 0xFF, sizeof(uint32_t));  // Init com 0xFF para detectar se não executou
    
    test_sha256_kernel<<<1, 1>>>(d_result);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    printf("Device first word: %08x\n", h_result);
    printf("Expected (OpenSSL): 8d70d691\n");
    
    if (h_result == 0x8d70d691) {
        printf("✓ MATCH! Byte order is correct.\n");
    } else if (h_result == 0x91d6708d) {
        printf("✗ REVERSED! Need to swap byte order.\n");
    } else {
        printf("✗ DIFFERENT! Check SHA256 implementation.\n");
    }
    
    return 0;
}
