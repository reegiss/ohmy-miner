// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef MINER_MINING_CONTEXT_H_
#define MINER_MINING_CONTEXT_H_

#include "mining_job.h"
#include <mutex>
#include <cstdint>
#include <cstring>

struct MiningContext {
    std::mutex mtx;
    MiningJob current_job;
    uint8_t share_target[32];
    double difficulty;

    MiningContext() : difficulty(1.0) {
        // Initialize target to max_target (difficulty 1) to prevent
        // starting with an invalid target before the pool sends one.
        memset(share_target, 0xFF, sizeof(share_target));
        share_target[0] = 0x00;
        share_target[1] = 0x00;
        share_target[2] = 0x00;
        share_target[3] = 0x00;
    }
};

#endif // MINER_MINING_CONTEXT_H_