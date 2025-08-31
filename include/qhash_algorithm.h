// qhash_algorithm.h
#ifndef QHASH_ALGORITHM_H
#define QHASH_ALGORITHM_H

#include "ialgorithm.h"
#include <vector>
#include <cstdint>
#include <string>

class QHashAlgorithm : public IAlgorithm {
public:
    uint32_t search_batch(int device_id, const MiningJob& job, const uint8_t* target, uint32_t nonce_start, uint32_t num_nonces, ThreadSafeQueue<FoundShare>& result_queue) override;

private:
    std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    // Certifique-se de que a assinatura da função aceite dois parâmetros
    std::vector<uint8_t> build_merkle_root(const std::vector<uint8_t>& coinbase_hash, const std::vector<std::string>& merkle_branches);
    bool check_hash(const uint8_t* hash, const uint8_t* target);
    void set_target_from_nbits(const std::string& nbits_hex, uint8_t* target);
};

#endif // QHASH_ALGORITHM_H