#include <ohmy/pool/mining_job.hpp>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ohmy::pool {

std::optional<MiningJob> parse_mining_notify(const std::vector<json>& params) {
    // Stratum mining.notify params:
    // [0] job_id (string)
    // [1] prevhash (hex string)
    // [2] coinb1 (hex string)
    // [3] coinb2 (hex string)
    // [4] merkle_branch (array of hex strings)
    // [5] version (hex string)
    // [6] nbits (hex string)
    // [7] ntime (hex string)
    // [8] clean_jobs (bool)
    
    if (params.size() < 9) {
        return std::nullopt; // Invalid params
    }
    
    try {
        MiningJob job;
        job.job_id = params[0].get<std::string>();
        job.prev_hash = params[1].get<std::string>();
        job.coinbase1 = params[2].get<std::string>();
        job.coinbase2 = params[3].get<std::string>();
        
        // Merkle branch is an array
        if (params[4].is_array()) {
            for (const auto& branch : params[4]) {
                job.merkle_branch.push_back(branch.get<std::string>());
            }
        }
        
        job.version = params[5].get<std::string>();
        job.nbits = params[6].get<std::string>();
        job.ntime = params[7].get<std::string>();
        job.clean_jobs = params[8].get<bool>();
        
        return job;
    } catch (const json::exception&) {
        return std::nullopt;
    }
}

} // namespace ohmy::pool
