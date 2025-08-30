// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef IALGORITHM_H_
#define IALGORITHM_H_

#include "mining_job.h"
#include "found_share.h"
#include "thread_safe_queue.h"
#include <cstdint>

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;

    virtual uint32_t search_batch(
        int device_id, 
        const MiningJob& job, 
        const uint8_t* target,
        uint32_t nonce_start, 
        uint32_t num_nonces, 
        ThreadSafeQueue<FoundShare>& result_queue
    ) = 0;
};
#endif // IALGORITHM_H_