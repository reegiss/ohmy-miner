#pragma once

#include <cstdint>

namespace ohmy::crypto {

// Device-side stubs to be implemented in CUDA source later.
__device__ void sha256_init(void* ctx);
__device__ void sha256_update(void* ctx, const std::uint8_t* data, int len);
__device__ void sha256_final(void* ctx, std::uint8_t out[32]);

} // namespace ohmy::crypto
