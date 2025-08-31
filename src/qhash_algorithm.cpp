// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#include "qhash_algorithm.h"
#include "crypto_utils.h"
#include "miner/endian_util.h"
#include "qhash_kernels.cuh"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cstring>

uint32_t QHashAlgorithm::search_batch(int, const MiningJob& job, const uint8_t* target, uint32_t nonce_start, uint32_t num_nonces, ThreadSafeQueue<FoundShare>& result_queue) {
    if (job.job_id.empty()) {
        return 0xFFFFFFFF;
    }

    std::vector<uint8_t> block_header_template(76);
    auto version_bytes = hex_to_bytes(job.version);
    auto prev_hash_bytes = hex_to_bytes(job.prev_hash);
    auto merkle_root_bytes = build_merkle_root(job);
    auto ntime_bytes = hex_to_bytes(job.ntime);
    auto nbits_bytes = hex_to_bytes(job.nbits);

    memcpy(&block_header_template[0], version_bytes.data(), 4);
    memcpy(&block_header_template[4], prev_hash_bytes.data(), 32);
    memcpy(&block_header_template[36], merkle_root_bytes.data(), 32);
    memcpy(&block_header_template[68], ntime_bytes.data(), 4);
    memcpy(&block_header_template[72], nbits_bytes.data(), 4);

    swap_endian_words(&block_header_template[4], 32);
    swap_endian_words(&block_header_template[36], 32);

    uint32_t found_nonce = qhash_search_batch(
        block_header_template.data(),
        target,
        nonce_start,
        num_nonces
    );

    if (found_nonce != 0xFFFFFFFF) {
        FoundShare share;
        share.job_id = job.job_id;
        share.extranonce2 = job.extranonce2;
        share.ntime = job.ntime;
        std::stringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(8) << found_nonce;
        share.nonce_hex = ss.str();
        result_queue.push(share);
    }
    
    return found_nonce;
}

std::vector<uint8_t> QHashAlgorithm::hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.length() / 2);
    for (unsigned int i = 0; i < hex.length(); i += 2) {
        bytes.push_back((uint8_t)strtol(hex.substr(i, 2).c_str(), NULL, 16));
    }
    return bytes;
}

std::vector<uint8_t> QHashAlgorithm::build_merkle_root(const MiningJob& job) {
    auto coinbase_p1 = hex_to_bytes(job.coinb1);
    auto coinbase_en1 = hex_to_bytes(job.extranonce1);
    auto coinbase_en2 = hex_to_bytes(job.extranonce2);
    auto coinbase_p2 = hex_to_bytes(job.coinb2);

    std::vector<uint8_t> coinbase_tx_data;
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_p1.begin(), coinbase_p1.end());
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_en1.begin(), coinbase_en1.end());
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_en2.begin(), coinbase_en2.end());
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_p2.begin(), coinbase_p2.end());

    std::vector<uint8_t> current_hash(32);
    sha256d(current_hash.data(), coinbase_tx_data.data(), coinbase_tx_data.size());

    for (const auto& branch_hex : job.merkle_branches) {
        auto branch_bytes = hex_to_bytes(branch_hex);
        std::vector<uint8_t> combined(64);
        memcpy(combined.data(), current_hash.data(), 32);
        memcpy(combined.data() + 32, branch_bytes.data(), 32);
        sha256d(current_hash.data(), combined.data(), 64);
    }
    return current_hash;
}