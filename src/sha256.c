#include <string.h>
#include <stdint.h>
#include "sha256-hash.h"

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define S0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define S1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))
#define s0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define s1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define F0(x, y, z) ((x) & (y)) | ((z) & ((x) | (y)))
#define F1(x, y, z) (z) ^ ((x) & ((y) ^ (z)))

#ifdef __GNUC__
#define bswap_32(x) __builtin_bswap32(x)
#else
static inline uint32_t bswap_32(uint32_t x) {
    return (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >> 8) |
           (((x) & 0x0000ff00) << 8) | (((x) & 0x000000ff) << 24);
}
#endif

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static void sha256_transform(uint32_t *state, const uint32_t *block) {
    uint32_t a, b, c, d, e, f, g, h, W[64];
    int i;
    for (i = 0; i < 16; i++) W[i] = bswap_32(block[i]);
    for (i = 16; i < 64; i++) W[i] = S1(W[i - 2]) + W[i - 7] + S0(W[i - 15]) + W[i - 16];
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];
    for (i = 0; i < 64; i++) {
        uint32_t t1 = h + s1(e) + F1(e, f, g) + K[i] + W[i];
        uint32_t t2 = s0(a) + F0(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void sha256_full(uint32_t *state, const uint32_t *p, int len) {
    int i;
    uint32_t block[16];
    state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
    state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
    for (i = 0; len >= 64; i++, len -= 64) {
        sha256_transform(state, p + i * 16);
    }
    memcpy(block, p + i * 16, len);
    ((unsigned char*)block)[len] = 0x80;
    if (len >= 56) {
        memset((unsigned char*)block + len + 1, 0, 64 - len - 1);
        sha256_transform(state, block);
        memset(block, 0, 56);
    } else {
        memset((unsigned char*)block + len + 1, 0, 56 - len - 1);
    }
    block[14] = bswap_32((uint32_t)(p - block) * 512 + len * 8);
    sha256_transform(state, block);
}