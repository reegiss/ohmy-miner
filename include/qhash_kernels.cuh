#ifndef QHASH_KERNELS_CUH
#define QHASH_KERNELS_CUH

#include <cstdint>

// Launches the qhash search kernel on the GPU.
// Searches a range of nonces [start_nonce, start_nonce + num_nonces).
// Returns the first valid nonce found, or 0xFFFFFFFF if none is found.
uint32_t qhash_search_batch(
    const uint8_t* header_template, // 76-byte header without the nonce
    const uint8_t* target,          // 32-byte target hash
    uint32_t start_nonce,
    uint32_t num_nonces
);

#endif // QHASH_KERNELS_CUH