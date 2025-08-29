#ifndef IALGORITHM_H
#define IALGORITHM_H

#include "mining_job.h"
#include "found_share.h"
#include "thread_safe_queue.h"
#include <cstdint>

class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;
    virtual bool thread_init(int device_id) = 0;
    virtual void thread_destroy() = 0;

    // Searches a single batch of nonces for a valid solution.
    // Returns the found nonce or 0xFFFFFFFF if none is found in the batch.
    virtual uint32_t search_batch(int device_id, const MiningJob& job, uint32_t nonce_start, uint32_t num_nonces, ThreadSafeQueue<FoundShare>& result_queue) = 0;
};

#endif // IALGORITHM_H