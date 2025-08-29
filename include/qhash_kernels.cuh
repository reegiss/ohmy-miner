#ifndef QHASH_KERNELS_CUH
#define QHASH_KERNELS_CUH

#include <cstdint>

// Launches the high-performance qPoW search kernel on the GPU.
uint32_t qhash_search_batch(
    const uint8_t* header_template, // 76-byte header without the nonce
    const uint8_t* target,          // 32-byte target hash
    uint32_t start_nonce,
    uint32_t num_nonces
);

#endif // QHASH_KERNELS_CUH