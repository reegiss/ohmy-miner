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
#include <cstring> // For memset/memcpy

// Forward declare the C functions we will be calling
extern "C" {
#include "qhash_miner.h"
}

// Global shutdown flag
extern std::atomic<bool> g_shutdown;

namespace MinerBridge {

// Implementation of the hex-to-bytes helper
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.length() / 2);
    for (unsigned int i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        bytes.push_back((uint8_t)strtol(byteString.c_str(), NULL, 16));
    }
    return bytes;
}

// Calculates the Merkle Root
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

// A simple function to check the hash against the target difficulty
bool check_hash(const uint8_t* hash, const uint8_t* target) {
    for (int i = 31; i >= 0; --i) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;
}

// Converts the compact nbits format to a 256-bit target
void set_target_from_nbits(const std::string& nbits_hex, uint8_t* target) {
    uint32_t nbits = stoul(nbits_hex, nullptr, 16);
    int exponent = nbits >> 24;
    uint32_t mantissa = nbits & 0x007fffff;
    int byte_pos = 32 - exponent;
    memset(target, 0, 32);
    if (byte_pos >= 0 && byte_pos <= 29) {
        target[byte_pos] = (mantissa >> 16) & 0xff;
        target[byte_pos + 1] = (mantissa >> 8) & 0xff;
        target[byte_pos + 2] = mantissa & 0xff;
    }
}

// --- This is the main function that connects everything ---
void process_job(const MiningJob& job, ThreadSafeQueue<FoundShare>& result_queue) {
    auto merkle_root_bytes = build_merkle_root(job);
    std::reverse(merkle_root_bytes.begin(), merkle_root_bytes.end());
    
    std::vector<uint8_t> block_header(80);
    auto version_bytes = hex_to_bytes(job.version);
    auto prev_hash_bytes = hex_to_bytes(job.prev_hash);
    auto nbits_bytes = hex_to_bytes(job.nbits);
    auto ntime_bytes = hex_to_bytes(job.ntime);

    std::reverse(prev_hash_bytes.begin(), prev_hash_bytes.end());

    memcpy(&block_header[0], version_bytes.data(), 4);
    memcpy(&block_header[4], prev_hash_bytes.data(), 32);
    memcpy(&block_header[36], merkle_root_bytes.data(), 32);
    memcpy(&block_header[68], ntime_bytes.data(), 4);
    memcpy(&block_header[72], nbits_bytes.data(), 4);

    uint8_t target[32];
    set_target_from_nbits(job.nbits, target);
    
    std::cout << "[MINER] Starting hash loop for job " << job.job_id << "..." << std::endl;
    
    uint32_t nonce = 0;
    uint8_t final_hash[32];
    
    while (nonce < 10000000 && !g_shutdown) { // Increased hash count
        memcpy(&block_header[76], &nonce, 4);
        qhash_hash(final_hash, block_header.data(), 0);

        if (check_hash(final_hash, target)) {
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            std::cout << "!!! YAY! Found a valid share! Nonce: " << nonce << std::endl;
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            
            // FIX: Declare the share variable here, before using it.
            FoundShare share;
            share.job_id = job.job_id;
            share.extranonce2 = job.extranonce2;
            share.ntime = job.ntime;
            
            std::stringstream ss;
            ss << std::hex << std::setw(8) << std::setfill('0') << nonce;
            share.nonce_hex = ss.str();
            
            result_queue.push(share);
            break;
        }
        nonce++;
    }
    std::cout << "[MINER] Finished hash loop for job " << job.job_id << ". Hashes done: " << nonce << std::endl;
}

} // namespace MinerBridge