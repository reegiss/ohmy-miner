/*
 * Bruteforce para encontrar o nonce do Block 1 do Qubitcoin
 * Target hash: 000000fd8ed7cba05121b1f66cf955328300949d5cc8a7a1b36ec7a7d1934a63
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <openssl/sha.h>

// Target hash (reversed for little-endian comparison)
const uint8_t TARGET_HASH[32] = {
    0x63,0x4a,0x93,0xd1,0xa7,0xc7,0x6e,0xb3,0xa1,0xa7,0xc8,0x5c,0x9d,0x94,0x00,0x83,
    0x32,0x55,0xf9,0x6c,0xf6,0xb1,0x21,0x51,0xa0,0xcb,0xd7,0x8e,0xfd,0x00,0x00,0x00
};

const uint8_t HEADER_TEMPLATE[76] = {
    0x01,0x00,0x00,0x00,
    0x2b,0x36,0x9b,0xbf,0x52,0x7c,0xf0,0x18,0x28,0x25,0xa1,0x87,0x3f,0x78,0x96,0x26,
    0xec,0xc6,0x38,0x5f,0x54,0xc5,0xff,0x53,0x45,0xcd,0xe0,0xcc,0x21,0x00,0x00,0x00,
    0x8b,0x1a,0xb8,0x71,0xf8,0xf4,0x49,0x8b,0x91,0x92,0xe8,0x6f,0xe4,0x48,0x14,0x86,
    0x7c,0x2f,0x56,0xfc,0x20,0x9d,0xde,0x43,0x39,0xc9,0xa4,0x96,0x7e,0x82,0xf3,0x8c,
    0x29,0xe0,0x17,0x67,
    0xff,0xff,0x00,0x1d
};

void sha256d(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint8_t hash1[32];
    SHA256(data, len, hash1);
    SHA256(hash1, 32, out);
}

int main() {
    uint8_t header[80];
    uint8_t hash[32];
    
    memcpy(header, HEADER_TEMPLATE, 76);
    
    printf("Searching for nonce...\n");
    printf("Target hash: 000000fd8ed7cba05121b1f66cf955328300949d5cc8a7a1b36ec7a7d1934a63\n\n");
    
    for (uint64_t nonce = 0; nonce < 0x100000000ULL; nonce++) {
        // Set nonce (little-endian)
        header[76] = (nonce >> 0) & 0xFF;
        header[77] = (nonce >> 8) & 0xFF;
        header[78] = (nonce >> 16) & 0xFF;
        header[79] = (nonce >> 24) & 0xFF;
        
        // Compute SHA256d
        sha256d(header, 80, hash);
        
        // Check if matches target
        if (memcmp(hash, TARGET_HASH, 32) == 0) {
            printf("✓ FOUND NONCE: 0x%08llx (%llu)\n\n", 
                   (unsigned long long)nonce, (unsigned long long)nonce);
            
            printf("Header (80 bytes):\n");
            for (int i = 0; i < 80; i++) {
                printf("%02x ", header[i]);
                if ((i+1) % 16 == 0) printf("\n");
            }
            printf("\n");
            
            printf("SHA256d hash:\n");
            for (int i = 0; i < 32; i++) {
                printf("%02x", hash[i]);
            }
            printf("\n\n");
            
            printf("Copy this to golden_extractor.cpp:\n");
            printf("const uint64_t NONCE = 0x%08llx;\n", (unsigned long long)nonce);
            
            return 0;
        }
        
        // Progress update every 10M nonces
        if (nonce % 10000000 == 0 && nonce > 0) {
            printf("Tried %llu million nonces...\n", (unsigned long long)(nonce / 1000000));
        }
    }
    
    printf("✗ Nonce not found in range 0-0xFFFFFFFF\n");
    printf("   Check if header format is correct!\n");
    
    return 1;
}
