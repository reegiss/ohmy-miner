#include <stdio.h>
#include <string.h> // <-- ADD THIS LINE
#include "sha256-hash.h"
#include "qhash_miner.h"

#define NIBBLE_MASK (unsigned char)0xF
#define NIBBLE_SIZE 4

// Declaração da função que agora será definida em C++
int16_t toFixed(double x);

static void split_nibbles(const unsigned char input[SHA256_BLOCK_SIZE],
                          unsigned char output[2 * SHA256_BLOCK_SIZE]) {
    for (size_t i = 0; i < SHA256_BLOCK_SIZE; ++i) {
        output[2 * i] = (input[i] >> NIBBLE_SIZE) & NIBBLE_MASK;
        output[2 * i + 1] = input[i] & NIBBLE_MASK;
    }
}

int qhash_hash(void *output, const void *input, int thr_id) {
    unsigned char buf[SHA256_BLOCK_SIZE + NUM_QUBITS * sizeof(int16_t)];

    sha256_full((uint32_t*)buf, (const uint32_t*)input, INPUT_SIZE);

    unsigned char nibbles[2 * SHA256_BLOCK_SIZE];
    split_nibbles(buf, nibbles);

    double expectations[NUM_QUBITS];
    run_simulation(nibbles, expectations);

    for (size_t i = 0; i < NUM_QUBITS; ++i) {
        const size_t j = SHA256_BLOCK_SIZE + i * sizeof(int16_t);
        int16_t fixed_val = toFixed(expectations[i]);
        // Copia a representação little-endian
        memcpy((void*)(buf + j), (void*)&fixed_val, sizeof(int16_t));
    }

    sha256_full((uint32_t*)output, (const uint32_t*)buf, sizeof(buf));

    return 1;
}