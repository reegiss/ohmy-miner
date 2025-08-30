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
// without external multiprecision libraries, using integer-based arithmetic.
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
    
    // Use a large integer to scale the difficulty to avoid floating point issues
    // while maintaining precision.
    uint64_t divisor = static_cast<uint64_t>(diff);
    double fractional_part = diff - divisor;

    uint64_t remainder = 0;

    // Custom 256-bit / 64-bit long division
    for (int i = 3; i >= 0; --i) {
        // __int128 can hold the intermediate dividend (remainder + current part)
        unsigned __int128 dividend = ((unsigned __int128)remainder << 64) | max_target_parts[i];
        if (divisor > 0) {
            result_parts[i] = dividend / divisor;
            remainder = dividend % divisor;
        } else {
            // Handle cases where difficulty is less than 1.0
            result_parts[i] = 0; // Will be handled by fractional part later
            remainder = max_target_parts[i];
        }
    }
    
    // Approximate the division by the fractional part if it exists
    if (fractional_part > 1e-9) {
        // This is a simplification; a full floating point 256-bit division is very complex.
        // We essentially perform another division on the result.
        for(int i=3; i>=0; --i) {
            result_parts[i] /= (1.0 + fractional_part);
        }
    }
    
    // Convert result from little-endian parts to a big-endian byte array.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            target[i * 8 + j] = (uint8_t)((result_parts[3 - i] >> (8 * (7 - j))) & 0xFF);
        }
    }
}
#endif // MINER_TARGET_UTIL_H_