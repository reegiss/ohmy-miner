// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef IALGORITHM_H_
#define IALGORITHM_H_

#include "mining_job.h"
#include "found_share.h"
#include "thread_safe_queue.h"
#include <cstdint>

/**
 * @class IAlgorithm
 * @brief Defines the interface for all mining algorithm implementations.
 * 
 * This abstract class provides a standardized method for searching a batch 
 * of nonces to find valid shares, ensuring that all derived classes 
 * implement this fundamental functionality.
 */
class IAlgorithm {
public:
    /**
     * @brief Virtual destructor for the interface.
     * 
     * Ensures derived classes are destructed correctly.
     */
    virtual ~IAlgorithm() = default;

    /**
     * @brief Searches a batch of nonces for valid shares.
     * 
     * This method must be implemented by any mining algorithm class. It 
     * processes a given range of nonces using the specified device and mining 
     * job parameters, comparing results against a target to identify valid shares.
     * 
     * @param device_id Identifier of the GPU or device to be used for mining.
     * @param job The mining job containing necessary data such as the block header.
     * @param target The target hash value to be used for validating shares.
     * @param nonce_start The nonce value from which to start the search.
     * @param num_nonces The total number of nonces to search within this batch.
     * @param result_queue A thread-safe queue for storing found valid shares.
     * @return The number of valid shares found within the nonce batch.
     */
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