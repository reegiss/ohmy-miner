// Copyright (c) 2025 The GPU-Miner Authors.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#include "qhash_algorithm.h"
#include "crypto_utils.h"
#include "logger.h"

#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>

// Convert hex string -> bytes
std::vector<uint8_t> QHashAlgorithm::hex_to_bytes(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("Invalid hex string length");
    }

    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);

    for (size_t i = 0; i < hex.size(); i += 2) {
        uint8_t b = static_cast<uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16));
        bytes.push_back(b);
    }

    return bytes;
}

// Build Merkle root from coinbase hash + branches
std::vector<uint8_t> QHashAlgorithm::build_merkle_root(
    const std::vector<uint8_t>& coinbase_hash_le,
    const std::vector<std::string>& branches_hex) 
{
    if (branches_hex.empty()) {
        return coinbase_hash_le;
    }

    std::vector<uint8_t> hash = coinbase_hash_le;

    for (const auto& branch_hex : branches_hex) {
        auto branch = hex_to_bytes(branch_hex);
        if (branch.size() != 32) {
            throw std::runtime_error("Invalid merkle branch size");
        }

        std::vector<uint8_t> concat;
        concat.reserve(64);
        concat.insert(concat.end(), hash.begin(), hash.end());
        concat.insert(concat.end(), branch.begin(), branch.end());

        uint8_t tmp[32];
        sha256d(tmp, concat.data(), concat.size());
        hash.assign(tmp, tmp + 32);
    }

    return hash;
}

// Compact nBits -> 32-byte target
void QHashAlgorithm::set_target_from_nbits(const std::string& nbits_hex, uint8_t* target) {
    if (nbits_hex.size() != 8) {
        throw std::runtime_error("Invalid nBits size");
    }

    auto nbits = hex_to_bytes(nbits_hex);
    if (nbits.size() != 4) {
        throw std::runtime_error("nBits must be 4 bytes");
    }

    uint32_t compact = (nbits[0] << 24) | (nbits[1] << 16) | (nbits[2] << 8) | nbits[3];
    uint32_t exp = compact >> 24;
    uint32_t mant = compact & 0x007fffff;

    std::memset(target, 0, 32);
    if (exp <= 3) {
        mant >>= 8 * (3 - exp);
        std::memcpy(target + 28, &mant, 4);
    } else {
        std::memcpy(target + 32 - exp, &mant, 3);
    }
}

// Check hash <= target
bool QHashAlgorithm::check_hash(const uint8_t* hash, const uint8_t* target) {
    return std::memcmp(hash, target, 32) <= 0;
}

// Main batch execution
uint32_t QHashAlgorithm::search_batch(
    int device_id,
    const MiningJob& job,
    const uint8_t* target,
    uint32_t nonce_start,
    uint32_t num_nonces,
    ThreadSafeQueue<FoundShare>& result_queue)
{
    // Build coinbase
    auto coinb1 = hex_to_bytes(job.coinb1);
    auto coinb2 = hex_to_bytes(job.coinb2);
    auto extranonce1 = hex_to_bytes(job.extranonce1);
    auto extranonce2 = hex_to_bytes(job.extranonce2);

    std::vector<uint8_t> coinbase;
    coinbase.reserve(coinb1.size() + coinb2.size() + extranonce1.size() + extranonce2.size());
    coinbase.insert(coinbase.end(), coinb1.begin(), coinb1.end());
    coinbase.insert(coinbase.end(), extranonce1.begin(), extranonce1.end());
    coinbase.insert(coinbase.end(), extranonce2.begin(), extranonce2.end());
    coinbase.insert(coinbase.end(), coinb2.begin(), coinb2.end());

    // Coinbase hash (double SHA256, little endian)
    uint8_t coinbase_hash[32];
    sha256d(coinbase_hash, coinbase.data(), coinbase.size());
    std::vector<uint8_t> coinbase_hash_le(coinbase_hash, coinbase_hash + 32);

    // Merkle root
    const auto merkle_root_le = build_merkle_root(coinbase_hash_le, job.merkle_branches);

    // Block header (little endian fields)
    std::vector<uint8_t> header(80, 0);

    // Version
    auto ver = hex_to_bytes(job.version);
    if (ver.size() != 4) throw std::runtime_error("Invalid version size");
    std::memcpy(header.data(), ver.data(), 4);

    // Prevhash
    auto prev = hex_to_bytes(job.prev_hash);
    if (prev.size() != 32) throw std::runtime_error("Invalid prevhash size");
    std::memcpy(header.data() + 4, prev.data(), 32);

    // Merkle root
    std::memcpy(header.data() + 36, merkle_root_le.data(), 32);

    // nTime
    auto ntime = hex_to_bytes(job.ntime);
    if (ntime.size() != 4) throw std::runtime_error("Invalid ntime size");
    std::memcpy(header.data() + 68, ntime.data(), 4);

    // nBits
    auto nbits = hex_to_bytes(job.nbits);
    if (nbits.size() != 4) throw std::runtime_error("Invalid nbits size");
    std::memcpy(header.data() + 72, nbits.data(), 4);

    uint32_t valid_shares = 0;

    for (uint32_t i = 0; i < num_nonces; i++) {
        uint32_t nonce = nonce_start + i;
        std::memcpy(header.data() + 76, &nonce, 4);

        uint8_t final_hash[32];
        sha256d(final_hash, header.data(), header.size());

        if (check_hash(final_hash, target)) {
            FoundShare share;
            share.job_id = job.job_id;
            share.extranonce2 = job.extranonce2;
            share.ntime = job.ntime;

            std::ostringstream oss;
            oss << std::hex << std::setw(8) << std::setfill('0') << nonce;
            share.nonce_hex = oss.str();

            result_queue.push(share);
            valid_shares++;
        }
    }

    return valid_shares;
}
