#ifndef MINING_JOB_H
#define MINING_JOB_H

#include <string>
#include <vector>
#include <array>

struct MiningJob {
    std::string job_id;  // Unique identifier for the job
    std::array<uint8_t, 32> prev_hash;  // Previous block hash
    std::vector<uint8_t> coinb1;  // First part of the coinbase transaction
    std::vector<uint8_t> coinb2;  // Second part of the coinbase transaction
    std::vector<std::string> merkle_branches;  // Merkle branches for constructing the Merkle root
    std::array<uint8_t, 4> version;  // Block version
    std::array<uint8_t, 4> nbits;  // Encoded network difficulty
    std::array<uint8_t, 4> ntime;  // Current timestamp
    bool clean_jobs;  // Whether the job requires a clean state
    std::vector<uint8_t> extranonce1;  // Part 1 of the extra nonce
    std::vector<uint8_t> extranonce2;  // Part 2 of the extra nonce

    MiningJob() : clean_jobs(false) {}  // Default constructor with initialization
};

#endif // MINING_JOB_H