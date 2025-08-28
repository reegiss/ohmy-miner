#ifndef CRYPTO_UTILS_H__
#define CRYPTO_UTILS_H__

#include <vector>
#include <cstdint>
#include <cstddef>

// Calculates SHA256D (SHA256(SHA256(data)))
void sha256d(uint8_t* output, const uint8_t* input, size_t len);

#endif // CRYPTO_UTILS_H__