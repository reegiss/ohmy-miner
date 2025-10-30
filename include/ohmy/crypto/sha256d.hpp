/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace ohmy {
namespace crypto {

/**
 * Compute double SHA256 hash (Bitcoin standard)
 * @param data Input data to hash
 * @return 32-byte hash output
 */
std::vector<uint8_t> sha256d(const std::vector<uint8_t>& data);

/**
 * Compute double SHA256 hash from string
 * @param data Input string to hash
 * @return 32-byte hash output
 */
std::vector<uint8_t> sha256d(const std::string& data);

} // namespace crypto
} // namespace ohmy
