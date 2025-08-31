// Copyright (c) 2025 The GPU-Miner Authors.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef QHASH_ALGORITHM_H_
#define QHASH_ALGORITHM_H_

#include "ialgorithm.h"
#include <vector>
#include <cstdint>
#include <string>

class QHashAlgorithm : public IAlgorithm {
public:
    // Run a batch of nonces on a specific GPU
    uint32_t search_batch(
        int device_id,
        const MiningJob& job,
        const uint8_t* target,
        uint32_t nonce_start,
        uint32_t num_nonces,
        ThreadSafeQueue<FoundShare>& result_queue) override;

    // Validate a hash against a target
    bool check_hash(const uint8_t* hash, const uint8_t* target);

    // Decode compact nBits to 32-byte target
    void set_target_from_nbits(const std::string& nbits_hex, uint8_t* target);

private:
    // Helpers
    std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    std::vector<uint8_t> build_merkle_root(
        const std::vector<uint8_t>& coinbase_hash_le,
        const std::vector<std::string>& branches_hex);
};

#endif // QHASH_ALGORITHM_H_