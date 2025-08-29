#include "qhash_algorithm.h"
#include "crypto_utils.h"
#include "qhash_kernels.cuh" // Include our new batch search header
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <chrono>

// Include the C header for the qPoW hash function
extern "C" {
#include "qhash_miner.h"
}

extern std::atomic<bool> g_shutdown;

// --- C Linkage for toFixed ---
// This C function is defined in qhash_algo.c but its toFixed conversion
// is implemented here for easy access to C++ features.
extern "C" {
    #define FRACTION_BITS 15
    int16_t toFixed(double x) {
        static_assert(FRACTION_BITS <= (sizeof(int16_t) * 8 - 1), "Fraction bits exceeds type size");
        const int32_t fractionMult = 1 << FRACTION_BITS;
        return (x >= 0.0) ? (x * fractionMult + 0.5) : (x * fractionMult - 0.5);
    }
}
// --- IAlgorithm Implementation ---


// Renamed from process_job and removed the outer nonce loop
uint32_t QHashAlgorithm::search_batch(int device_id, const MiningJob& job, uint32_t nonce_start, uint32_t num_nonces, ThreadSafeQueue<FoundShare>& result_queue) {
    std::vector<uint8_t> block_header_template(76);
    auto version_bytes = hex_to_bytes(job.version);
    auto prev_hash_bytes = hex_to_bytes(job.prev_hash);
    auto merkle_root_bytes = build_merkle_root(job);
    auto ntime_bytes = hex_to_bytes(job.ntime);
    auto nbits_bytes = hex_to_bytes(job.nbits);

    std::reverse(prev_hash_bytes.begin(), prev_hash_bytes.end());
    std::reverse(merkle_root_bytes.begin(), merkle_root_bytes.end());

    memcpy(&block_header_template[0], version_bytes.data(), 4);
    memcpy(&block_header_template[4], prev_hash_bytes.data(), 32);
    memcpy(&block_header_template[36], merkle_root_bytes.data(), 32);
    memcpy(&block_header_template[68], ntime_bytes.data(), 4);
    memcpy(&block_header_template[72], nbits_bytes.data(), 4);

    uint8_t target[32];
    set_target_from_nbits(job.nbits, target);

    uint32_t found_nonce = qhash_search_batch(
        block_header_template.data(),
        target,
        nonce_start,
        num_nonces
    );

    if (found_nonce != 0xFFFFFFFF) {
        FoundShare share;
        share.job_id = job.job_id;
        share.extranonce2 = job.extranonce2;
        share.ntime = job.ntime;
        
        std::stringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(8) << found_nonce;
        share.nonce_hex = ss.str();
        
        result_queue.push(share);
    }
    
    return found_nonce;
}

// --- Helper Implementations ---

std::vector<uint8_t> QHashAlgorithm::hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.length() / 2);
    for (unsigned int i = 0; i < hex.length(); i += 2) {
        bytes.push_back((uint8_t)strtol(hex.substr(i, 2).c_str(), NULL, 16));
    }
    return bytes;
}

std::vector<uint8_t> QHashAlgorithm::build_merkle_root(const MiningJob& job) {
    auto coinbase_p1 = hex_to_bytes(job.coinb1);
    auto coinbase_en1 = hex_to_bytes(job.extranonce1);
    auto coinbase_en2 = hex_to_bytes(job.extranonce2);
    auto coinbase_p2 = hex_to_bytes(job.coinb2);
    std::vector<uint8_t> coinbase_tx_data;
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_p1.begin(), coinbase_p1.end());
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_en1.begin(), coinbase_en1.end());
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_en2.begin(), coinbase_en2.end());
    coinbase_tx_data.insert(coinbase_tx_data.end(), coinbase_p2.begin(), coinbase_p2.end());
    std::vector<uint8_t> current_hash(32);
    sha256d(current_hash.data(), coinbase_tx_data.data(), coinbase_tx_data.size());
    for (const auto& branch_hex : job.merkle_branches) {
        auto branch_bytes = hex_to_bytes(branch_hex);
        std::vector<uint8_t> combined(64);
        memcpy(combined.data(), current_hash.data(), 32);
        memcpy(combined.data() + 32, branch_bytes.data(), 32);
        sha256d(current_hash.data(), combined.data(), 64);
    }
    return current_hash;
}

bool QHashAlgorithm::check_hash(const uint8_t* hash, const uint8_t* target) {
    for (int i = 31; i >= 0; --i) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;
}

void QHashAlgorithm::set_target_from_nbits(const std::string& nbits_hex, uint8_t* target) {
    uint32_t nbits = stoul(nbits_hex, nullptr, 16);
    uint32_t mantissa = nbits & 0x007fffff;
    uint8_t exponent = nbits >> 24;
    
    memset(target, 0, 32);
    int byte_pos = 32 - exponent;
    if (byte_pos >= 0 && byte_pos <= 29) {
        target[byte_pos]     = (mantissa >> 16) & 0xff;
        target[byte_pos + 1] = (mantissa >> 8) & 0xff;
        target[byte_pos + 2] = mantissa & 0xff;
    }
}