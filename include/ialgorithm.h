/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 */
#ifndef IALGORITHM_H
#define IALGORITHM_H

#include "mining_job.h"
#include "found_share.h"
#include "thread_safe_queue.h"
#include <cstdint>

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;

    // Correct signature for the high-performance batch processing architecture.
    virtual uint32_t search_batch(int device_id, const MiningJob& job, uint32_t nonce_start, uint32_t num_nonces, ThreadSafeQueue<FoundShare>& result_queue) = 0;
};

#endif // IALGORITHM_H