/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/crypto/difficulty.hpp"
#include <cstring>
#include <algorithm>

namespace ohmy {
namespace crypto {

std::vector<uint8_t> decode_compact_target(uint32_t bits) {
    // Extract exponent and coefficient from compact format
    // bits = 0xMMNNNNNN where MM is exponent and NNNNNN is coefficient
    uint32_t exponent = (bits >> 24) & 0xFF;
    uint32_t coefficient = bits & 0x00FFFFFF;
    
    // Initialize 32-byte target (256 bits) with zeros
    std::vector<uint8_t> target(32, 0);
    
    // Calculate the position where the coefficient starts
    // exponent indicates the number of bytes from the end
    if (exponent <= 3) {
        // Small targets (exponent <= 3)
        int offset = 32 - exponent;
        if (offset >= 0 && offset < 32) {
            // Store coefficient in little-endian at the position
            for (int i = 0; i < 3 && (offset + i) < 32; i++) {
                target[offset + i] = (coefficient >> (i * 8)) & 0xFF;
            }
        }
    } else {
        // Normal case: exponent > 3
        int offset = 32 - exponent;
        if (offset >= 0 && offset < 32) {
            // Store coefficient bytes in big-endian
            target[offset] = (coefficient >> 16) & 0xFF;
            if (offset + 1 < 32) target[offset + 1] = (coefficient >> 8) & 0xFF;
            if (offset + 2 < 32) target[offset + 2] = coefficient & 0xFF;
        }
    }
    
    return target;
}

bool hash_meets_target(const uint8_t* hash, uint32_t bits) {
    std::vector<uint8_t> target = decode_compact_target(bits);
    
    // Compare hash with target (both are big-endian, 32 bytes)
    // Hash must be <= target to meet difficulty
    for (int i = 0; i < 32; i++) {
        if (hash[i] < target[i]) {
            return true;  // Hash is less than target
        } else if (hash[i] > target[i]) {
            return false; // Hash is greater than target
        }
        // If equal, continue to next byte
    }
    
    // Hash equals target exactly - this meets the target
    return true;
}

} // namespace crypto
} // namespace ohmy
