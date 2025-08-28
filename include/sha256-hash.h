#ifndef SHA256_HASH_H__
#define SHA256_HASH_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// This function is what qhash.c needs.
void sha256_full(uint32_t *state, const uint32_t *p, int len);

#ifdef __cplusplus
}
#endif

#endif // SHA256_HASH_H__