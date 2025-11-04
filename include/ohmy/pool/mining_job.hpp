#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace ohmy::pool {

struct MiningJob {
    std::string job_id;
    std::string prev_hash;         // prevhash (hex)
    std::string coinbase1;         // coinb1 (hex)
    std::string coinbase2;         // coinb2 (hex)
    std::vector<std::string> merkle_branch;  // array of merkle hashes
    std::string version;           // block version (hex)
    std::string nbits;             // difficulty bits (hex)
    std::string ntime;             // timestamp (hex)
    bool clean_jobs{false};        // if true, discard old jobs
};

// Parse mining.notify params array into MiningJob
// params format: [job_id, prevhash, coinb1, coinb2, merkle_branch[], version, nbits, ntime, clean]
std::optional<MiningJob> parse_mining_notify(const std::vector<nlohmann::json>& params);

} // namespace ohmy::pool
