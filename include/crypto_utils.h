#ifndef CRYPTO_UTILS_H__
#define CRYPTO_UTILS_H__

#include <vector>
#include <cstdint>
#include <cstddef>

/**
 * @brief Calculates the double SHA-256 hash (SHA256(SHA256(data))) of the input data.
 *
 * This function computes the SHA-256 hash of the input data, and then computes
 * the SHA-256 hash of the resulting hash. The final hash is written to the output buffer.
 *
 * @param output Pointer to a buffer where the resulting 32-byte hash will be stored.
 * @param input Pointer to the input data to be hashed.
 * @param len Length of the input data in bytes.
 */
void sha256d(uint8_t* output, const uint8_t* input, size_t len);

#endif // CRYPTO_UTILS_H__