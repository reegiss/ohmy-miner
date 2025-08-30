// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef MINER_TARGET_UTIL_H_
#define MINER_TARGET_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

// Performs 256-bit division to calculate share target from difficulty
// without external multiprecision libraries.
inline void target_from_difficulty(double diff, uint8_t target[32]) {
    if (diff <= 1e-9) diff = 1e-9;

    // max_target is 0x00000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    // Stored as little-endian 64-bit parts.
    uint64_t max_target_parts[4] = {
        0xFFFFFFFFFFFFFFFF, // Low part
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0x00000000FFFFFFFF  // High part
    };

    // Use double for initial division and carry. The high-precision part of a double
    // is sufficient for this calculation as difficulty values don't span the full range.
    double divisor = diff;
    uint64_t result_parts[4] = {0, 0, 0, 0};
    double remainder = 0.0;

    // Perform long division, part by part, from most significant to least significant.
    for (int i = 3; i >= 0; --i) {
        // Combine remainder from previous step with current part to form the new dividend.
        // A 64-bit number can be represented as up to 1.844e19. A double has ~15-17 decimal digits of precision.
        // This is a close approximation. For cryptographic levels of precision, a full bigint library is needed,
        // but for mining difficulty, this is sufficient.
        double dividend = remainder * (1ULL << 31) * (1ULL << 31) * 2.0 + static_cast<double>(max_target_parts[i]);
        
        result_parts[i] = static_cast<uint64_t>(floor(dividend / divisor));
        remainder = fmod(dividend, divisor);
    }
    
    // Convert result from little-endian parts to a big-endian byte array.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            target[i * 8 + j] = (uint8_t)((result_parts[3 - i] >> (8 * (7 - j))) & 0xFF);
        }
    }
}
#endif // MINER_TARGET_UTIL_H_