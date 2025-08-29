#ifndef IALGORITHM_H
#define IALGORITHM_H

#include "mining_job.h"
#include "found_share.h"
#include "thread_safe_queue.h"

// Abstract base class for all mining algorithms.
class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;

    // Initializes thread-specific resources for a given GPU device.
    virtual bool thread_init(int device_id) = 0;

    // Releases thread-specific resources.
    virtual void thread_destroy() = 0;

    // The main processing loop for a given mining job.
    // This function will block until the job is done or the miner is shut down.
    virtual void process_job(int device_id, const MiningJob& job, ThreadSafeQueue<FoundShare>& result_queue) = 0;
};

#endif // IALGORITHM_H