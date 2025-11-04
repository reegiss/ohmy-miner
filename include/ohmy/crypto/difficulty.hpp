#pragma once

#include <array>
#include <cstdint>

namespace ohmy::crypto {

// Compact target (nBits) to target array (little-endian 256-bit)
std::array<std::uint8_t, 32> compact_to_target(std::uint32_t nBits);

// Check if hash (32-byte LE) meets target
bool meets_target(const std::uint8_t hash[32], const std::uint8_t target[32]);

} // namespace ohmy::crypto
