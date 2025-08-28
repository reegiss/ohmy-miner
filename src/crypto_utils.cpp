#include "crypto_utils.h"

extern "C" {
#include "sha256-hash.h"
}

void sha256d(uint8_t* output, const uint8_t* input, size_t len) {
    uint8_t first_hash[32];
    sha256_full(reinterpret_cast<uint32_t*>(first_hash),
                reinterpret_cast<const uint32_t*>(input),
                len);
    sha256_full(reinterpret_cast<uint32_t*>(output),
                reinterpret_cast<const uint32_t*>(first_hash),
                32);
}