/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 *
 * qhash_kernels.cuh
 *
 * Declaration for GPU SHA256d batch search.
 */

#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * qhash_search_batch
 *
 * header_h: pointer to 76-byte header template (version..nbits), host memory.
 * target_h: pointer to 32-byte target (big-endian byte order).
 * start_nonce: first nonce to test.
 * num_nonces: number of nonces to test.
 *
 * Returns: found nonce (minimum) or 0xFFFFFFFF if none found.
 *
 * Notes:
 * - This function copies inputs to device and launches a kernel that computes
 *   SHA256d(header||nonce) for nonces in [start_nonce, start_nonce+num_nonces).
 * - The kernel compares the final hash (big-endian bytes) lexicographically
 *   against target_h and uses atomicMin to record the smallest matching nonce.
 */
uint32_t qhash_search_batch(const uint8_t* header_h, const uint8_t* target_h, uint32_t start_nonce, uint32_t num_nonces);

#ifdef __cplusplus
}
#endif
