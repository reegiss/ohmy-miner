/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ohmy {
struct MiningJob; // forward declaration (from pool_connection.hpp)

namespace crypto {

// Hex helpers
std::string bytes_to_hex(const std::array<uint8_t, 32>& bytes);
std::vector<uint8_t> hex_to_bytes(const std::string& hex);

// Block header construction from a pool job
std::vector<uint8_t> build_block_header(const MiningJob& job,
                                        uint32_t nonce,
                                        const std::string& extra_nonce1,
                                        const std::string& extra_nonce2);

// Difficulty check (hash is big-endian, as returned by SHA256)
bool check_difficulty(const std::array<uint8_t, 32>& hash_be, double difficulty);

} // namespace crypto
} // namespace ohmy
