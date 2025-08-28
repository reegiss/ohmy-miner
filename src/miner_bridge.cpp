#include "miner_bridge.h"
#include "crypto_utils.h"
#include "found_share.h"
#include "thread_safe_queue.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstring>

extern "C" {
#include "qhash_miner.h"
}

extern std::atomic<bool> g_shutdown;

namespace MinerBridge {

    // Helper to convert hex strings to byte vectors
    std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    // Helper to build the merkle root correctly
    std::vector<uint8_t> build_merkle_root(const MiningJob& job);
    // Optimized hash comparison
    bool check_hash(const uint8_t* hash, const uint8_t* target);
    // nbits to target conversion
    void set_target_from_nbits(const std::string& nbits_hex, uint8_t* target);

    #define FIXED_FRACTION int16_t
    #define FIXED_INTERMEDIATE int32_t
    #define FRACTION_BITS 15

    // C-compatible function now defined in this C++ file
    extern "C" int16_t toFixed(double x) {
        static_assert(FRACTION_BITS <= (sizeof(int16_t) * 8 - 1), "Fraction bits exceeds type size");
        const int32_t fractionMult = 1 << FRACTION_BITS;
        return (x >= 0.0) ? (x * fractionMult + 0.5) : (x * fractionMult - 0.5);
    }

    void process_job(int device_id, const MiningJob& job, ThreadSafeQueue<FoundShare>& result_queue) {
        // 1. Build the block header template (everything except the nonce)
        std::vector<uint8_t> block_header(80);
        auto version_bytes = hex_to_bytes(job.version);
        auto prev_hash_bytes = hex_to_bytes(job.prev_hash);
        auto nbits_bytes = hex_to_bytes(job.nbits);
        auto ntime_bytes = hex_to_bytes(job.ntime);

        // This is a critical step: calculate the real merkle root.
        auto merkle_root_bytes = build_merkle_root(job);

        // Block headers require prev_hash and merkle_root to be in little-endian byte order.
        std::reverse(prev_hash_bytes.begin(), prev_hash_bytes.end());
        std::reverse(merkle_root_bytes.begin(), merkle_root_bytes.end());

        memcpy(&block_header[0], version_bytes.data(), 4);
        memcpy(&block_header[4], prev_hash_bytes.data(), 32);
        memcpy(&block_header[36], merkle_root_bytes.data(), 32);
        memcpy(&block_header[68], ntime_bytes.data(), 4);
        memcpy(&block_header[72], nbits_bytes.data(), 4);

        // 2. Calculate the real difficulty target
        uint8_t target[32];
        set_target_from_nbits(job.nbits, target);
        
        std::cout << "[MINER " << device_id << "] Starting hash loop for job " << job.job_id << "..." << std::endl;
        
        uint32_t nonce = 0;
        uint8_t final_hash[32];
        
        // This entire loop is the performance bottleneck.
        while (nonce < 0xFFFFFFFF && !g_shutdown) {
            memcpy(&block_header[76], &nonce, 4);
            qhash_hash(final_hash, block_header.data(), device_id);

            if (check_hash(final_hash, target)) {
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                std::cout << "!!! [MINER " << device_id << "] Found a valid share! Nonce: " << nonce << std::endl;
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                
                FoundShare share;
                share.job_id = job.job_id;
                share.extranonce2 = job.extranonce2;
                share.ntime = job.ntime;
                
                std::stringstream ss;
                ss << std::hex << std::setw(8) << std::setfill('0') << nonce;
                share.nonce_hex = ss.str();
                
                result_queue.push(share);
                break; // Stop searching on this job
            }
            nonce++;
        }
        std::cout << "[MINER " << device_id << "] Finished hash loop for job " << job.job_id << ". Hashes done: " << nonce << std::endl;
    }

    // --- Implementation of helper functions ---

    std::vector<uint8_t> build_merkle_root(const MiningJob& job) {
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

    bool check_hash(const uint8_t* hash, const uint8_t* target) {
        const uint64_t* hash64 = reinterpret_cast<const uint64_t*>(hash);
        const uint64_t* target64 = reinterpret_cast<const uint64_t*>(target);
        if (hash64[3] < target64[3]) return true;
        if (hash64[3] > target64[3]) return false;
        if (hash64[2] < target64[2]) return true;
        if (hash64[2] > target64[2]) return false;
        if (hash64[1] < target64[1]) return true;
        if (hash64[1] > target64[1]) return false;
        if (hash64[0] < target64[0]) return true;
        return false;
    }

    void set_target_from_nbits(const std::string& nbits_hex, uint8_t* target) {
        uint32_t nbits = stoul(nbits_hex, nullptr, 16);
        int exponent = nbits >> 24;
        uint32_t mantissa = nbits & 0x007fffff;
        int byte_pos = 32 - exponent;
        memset(target, 0, 32);
        if (byte_pos >= 0 && byte_pos <= 29) {
            target[byte_pos]     = (mantissa >> 16) & 0xff;
            target[byte_pos + 1] = (mantissa >> 8) & 0xff;
            target[byte_pos + 2] = mantissa & 0xff;
        }
    }
    
    std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
        std::vector<uint8_t> bytes;
        bytes.reserve(hex.length() / 2);
        for (unsigned int i = 0; i < hex.length(); i += 2) {
            std::string byteString = hex.substr(i, 2);
            bytes.push_back((uint8_t)strtol(byteString.c_str(), NULL, 16));
        }
        return bytes;
    }

} // namespace MinerBridge