#ifndef MINER_BRIDGE_H__
#define MINER_BRIDGE_H__

#include "mining_job.h"
#include "found_share.h"
#include "thread_safe_queue.h"
#include <vector>
#include <cstdint>

namespace MinerBridge {

std::vector<uint8_t> hex_to_bytes(const std::string& hex);

// FIX: Add int device_id as the first parameter
void process_job(int device_id, const MiningJob& job, ThreadSafeQueue<FoundShare>& result_queue);

} // namespace MinerBridge

#endif // MINER_BRIDGE_H__