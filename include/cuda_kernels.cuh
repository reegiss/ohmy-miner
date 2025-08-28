#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// The high-performance search function that will be called from C++.
// It orchestrates the launch of the CUDA kernel.
uint32_t qhash_search(const uint8_t* block_header_template,
                      const uint8_t* target,
                      uint32_t start_nonce,
                      uint32_t num_nonces_to_search);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H