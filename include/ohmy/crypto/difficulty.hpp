/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <vector>
#include <cstdint>

namespace ohmy {
namespace crypto {

/**
 * Decode compact target representation (Bitcoin nBits format)
 * @param bits Compact representation (4 bytes)
 * @return 32-byte target value (big-endian)
 */
std::vector<uint8_t> decode_compact_target(uint32_t bits);

/**
 * Check if a hash meets the difficulty target
 * @param hash 32-byte hash to check (big-endian)
 * @param bits Compact target representation
 * @return true if hash <= target
 */
bool hash_meets_target(const uint8_t* hash, uint32_t bits);

} // namespace crypto
} // namespace ohmy
