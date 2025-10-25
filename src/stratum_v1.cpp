/*
 * Implementation of Stratum v1 helper
 */

#include "miner/stratum_v1.hpp"

#include <stdexcept>

namespace miner {
namespace stratum_v1 {

using json = nlohmann::json;

json StratumV1::make_subscribe(const std::string& agent) {
    return json{
        {"id", 1},
        {"method", "mining.subscribe"},
        {"params", {agent}}
    };
}

json StratumV1::make_authorize(const std::string& user, const std::string& pass) {
    return json{
        {"id", 2},
        {"method", "mining.authorize"},
        {"params", {user, pass}}
    };
}

json StratumV1::make_submit(const net::Share& share, int id) {
    return json{
        {"id", id},
        {"method", "mining.submit"},
        {"params", {share.job_id, share.extranonce2, share.ntime, share.nonce}}
    };
}

bool StratumV1::is_notify(const json& j) {
    return j.is_object() && j.contains("method") && j["method"] == "mining.notify";
}

net::Job StratumV1::parse_notify(const json& j) {
    if (!is_notify(j)) {
        throw std::invalid_argument("Not a mining.notify message");
    }

    const auto& params = j.at("params");
    if (!params.is_array() || params.size() < 9) {
        throw std::invalid_argument("Invalid 'mining.notify' parameters");
    }

    net::Job job;
    job.job_id = params[0].get<std::string>();
    job.prev_hash = params[1].get<std::string>();
    job.coinb1 = params[2].get<std::string>();
    job.coinb2 = params[3].get<std::string>();
    job.merkle_branch = params[4].get<std::vector<std::string>>();
    job.version = params[5].get<std::string>();
    job.nbits = params[6].get<std::string>();
    job.ntime = params[7].get<std::string>();
    job.clean_jobs = params[8].get<bool>();

    return job;
}

} // namespace stratum_v1
} // namespace miner
