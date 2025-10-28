/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "crypto_utils.hpp"
#include "pool_connection.hpp"

#include <openssl/sha.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace ohmy {
namespace crypto {

std::string bytes_to_hex(const std::array<uint8_t, 32>& bytes) {
    std::stringstream ss;
    for (size_t i = 0; i < 32; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
    }
    return ss.str();
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::strtol(byte_str.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

static std::array<uint8_t, 32> sha256d_bytes(const std::vector<uint8_t>& data) {
    std::array<uint8_t, 32> h1;
    SHA256(data.data(), data.size(), h1.data());
    std::array<uint8_t, 32> h2;
    SHA256(h1.data(), h1.size(), h2.data());
    return h2;
}

static std::array<uint8_t, 32> be32_to_le32(const std::array<uint8_t, 32>& be) {
    std::array<uint8_t, 32> le{};
    std::reverse_copy(be.begin(), be.end(), le.begin());
    return le;
}

static std::array<uint8_t, 32> compute_merkle_root_le(const MiningJob& job,
                                                      const std::string& extra_nonce1,
                                                      const std::string& extra_nonce2) {
    std::string coinbase_hex = job.coinbase1 + extra_nonce1 + extra_nonce2 + job.coinbase2;
    auto coinbase_bytes = hex_to_bytes(coinbase_hex);
    std::array<uint8_t, 32> cur = sha256d_bytes(coinbase_bytes); // big-endian bytes

    if (!job.merkle_branch.empty()) {
        for (const auto& branch_hex : job.merkle_branch) {
            auto branch_bytes = hex_to_bytes(branch_hex);
            std::vector<uint8_t> concat;
            concat.reserve(64);
            // Use little-endian for internal merkle concatenation
            concat.insert(concat.end(), cur.rbegin(), cur.rend());
            concat.insert(concat.end(), branch_bytes.rbegin(), branch_bytes.rend());
            cur = sha256d_bytes(concat); // still big-endian
        }
    }

    std::array<uint8_t, 32> merkle_le{};
    std::reverse_copy(cur.begin(), cur.end(), merkle_le.begin());
    return merkle_le;
}

std::vector<uint8_t> build_block_header(const MiningJob& job,
                                        uint32_t nonce,
                                        const std::string& extra_nonce1,
                                        const std::string& extra_nonce2) {
    std::vector<uint8_t> header;

    // Version (little-endian in header)
    auto version_bytes = hex_to_bytes(job.version);
    std::reverse(version_bytes.begin(), version_bytes.end());
    header.insert(header.end(), version_bytes.begin(), version_bytes.end());

    // Previous block hash (reversed for little-endian)
    auto prev_hash_bytes = hex_to_bytes(job.prev_hash);
    std::reverse(prev_hash_bytes.begin(), prev_hash_bytes.end());
    header.insert(header.end(), prev_hash_bytes.begin(), prev_hash_bytes.end());

    // Merkle root (little-endian) from coinbase + merkle branches
    auto merkle_root_le = compute_merkle_root_le(job, extra_nonce1, extra_nonce2);
    header.insert(header.end(), merkle_root_le.begin(), merkle_root_le.end());

    // ntime (little-endian)
    auto ntime_bytes = hex_to_bytes(job.ntime);
    std::reverse(ntime_bytes.begin(), ntime_bytes.end());
    header.insert(header.end(), ntime_bytes.begin(), ntime_bytes.end());

    // nbits (little-endian)
    auto nbits_bytes = hex_to_bytes(job.nbits);
    std::reverse(nbits_bytes.begin(), nbits_bytes.end());
    header.insert(header.end(), nbits_bytes.begin(), nbits_bytes.end());

    // nonce (4 bytes, little-endian)
    header.push_back(nonce & 0xFF);
    header.push_back((nonce >> 8) & 0xFF);
    header.push_back((nonce >> 16) & 0xFF);
    header.push_back((nonce >> 24) & 0xFF);

    return header;
}

static std::array<uint8_t, 32> compute_target_from_difficulty(double difficulty) {
    std::array<uint32_t, 8> limbs{};
    limbs.fill(0);
    limbs[6] = 0xFFFF0000u;
    limbs[7] = 0x00000000u;

    const uint32_t SCALE = 1000000000u; // 1e9
    std::array<uint32_t, 9> num{}; // up to 288-bit after scaling
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        unsigned __int128 prod = static_cast<unsigned __int128>(limbs[i]) * SCALE + carry;
        num[i] = static_cast<uint32_t>(prod & 0xFFFFFFFFu);
        carry = static_cast<uint64_t>(prod >> 32);
    }
    num[8] = static_cast<uint32_t>(carry); // may be 0

    long double dl = static_cast<long double>(difficulty);
    uint64_t divisor = static_cast<uint64_t>(ceill(dl * static_cast<long double>(SCALE)));
    if (divisor == 0) divisor = 1;

    std::array<uint32_t, 9> quo{};
    unsigned __int128 rem = 0;
    for (int i = 8; i >= 0; --i) {
        unsigned __int128 cur = (rem << 32) + num[i];
        uint64_t q = static_cast<uint64_t>(cur / divisor);
        rem = cur % divisor;
        quo[i] = static_cast<uint32_t>(q & 0xFFFFFFFFu);
    }

    std::array<uint8_t, 32> target{};
    for (int i = 0; i < 8; ++i) {
        uint32_t w = quo[i];
        target[i * 4 + 0] = static_cast<uint8_t>(w & 0xFF);
        target[i * 4 + 1] = static_cast<uint8_t>((w >> 8) & 0xFF);
        target[i * 4 + 2] = static_cast<uint8_t>((w >> 16) & 0xFF);
        target[i * 4 + 3] = static_cast<uint8_t>((w >> 24) & 0xFF);
    }
    return target;
}

static bool le256_less_equal(const std::array<uint8_t, 32>& a, const std::array<uint8_t, 32>& b) {
    for (int i = 31; i >= 0; --i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return true;
}

bool check_difficulty(const std::array<uint8_t, 32>& hash_be, double difficulty) {
    auto target_le = compute_target_from_difficulty(difficulty);
    auto hash_le = be32_to_le32(hash_be);
    return le256_less_equal(hash_le, target_le);
}

} // namespace crypto
} // namespace ohmy
