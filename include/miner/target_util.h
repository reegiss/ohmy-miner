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

// Performs a high-precision 256-bit by 64-bit integer division to calculate 
// the share target from a numeric difficulty, without external libraries.
// This implementation uses a standard long division algorithm to ensure mathematical correctness
// and avoid floating-point precision issues.
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

    uint64_t result_parts[4] = {0, 0, 0, 0};
    
    // Scale the difficulty by 2^32 to maintain precision while using integer math.
    // This is a common technique in difficulty calculation.
    uint64_t divisor = static_cast<uint64_t>(diff * (1ULL << 32));
    if (divisor == 0) divisor = 1;

    // Perform 256-bit / 64-bit long division.
    // The remainder from the division of the higher part becomes the high part
    // of the next dividend.
    unsigned __int128 remainder = 0;
    for (int i = 3; i >= 0; --i) {
        // Use 128-bit integer to hold intermediate dividend to prevent overflow
        unsigned __int128 dividend = (remainder << 64) | max_target_parts[i];
        result_parts[i] = dividend / divisor;
        remainder = dividend % divisor;
    }
    
    // The final target needs to be scaled back down by 2^32.
    // We achieve this with a 256-bit right shift, carrying over bits.
    result_parts[0] = (result_parts[1] << 32) | (result_parts[0] >> 32);
    result_parts[1] = (result_parts[2] << 32) | (result_parts[1] >> 32);
    result_parts[2] = (result_parts[3] << 32) | (result_parts[2] >> 32);
    result_parts[3] = (result_parts[3] >> 32);

    // Convert result from little-endian parts to a big-endian byte array.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            target[i * 8 + j] = (uint8_t)((result_parts[3 - i] >> (8 * (7 - j))) & 0xFF);
        }
    }
}
#endif // MINER_TARGET_UTIL_H_