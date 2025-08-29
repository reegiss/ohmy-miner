#ifndef QHASH_ALGORITHM_H
#define QHASH_ALGORITHM_H

#include "ialgorithm.h"
#include <vector>
#include <cstdint>
#include <string>

class QHashAlgorithm : public IAlgorithm {
public:
    bool thread_init(int device_id) override;
    void thread_destroy(void) override;
    uint32_t search_batch(int device_id, const MiningJob& job, uint32_t nonce_start, uint32_t num_nonces, ThreadSafeQueue<FoundShare>& result_queue) override;
private:
    std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    std::vector<uint8_t> build_merkle_root(const MiningJob& job);
    bool check_hash(const uint8_t* hash, const uint8_t* target);
    void set_target_from_nbits(const std::string& nbits_hex, uint8_t* target);
};

#endif // QHASH_ALGORITHM_H