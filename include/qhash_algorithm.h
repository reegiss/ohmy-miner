// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
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
    // --- FIX: Ensure this signature matches IAlgorithm ---
    uint32_t search_batch(
        int device_id, 
        const MiningJob& job, 
        const uint8_t* target,
        uint32_t nonce_start, 
        uint32_t num_nonces, 
        ThreadSafeQueue<FoundShare>& result_queue
    ) override;

private:
    std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    std::vector<uint8_t> build_merkle_root(const MiningJob& job);
};
#endif // QHASH_ALGORITHM_H_