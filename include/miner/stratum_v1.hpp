/*
 * Stratum v1 protocol helper
 *
 * Small utility for creating/parsing Stratum v1 JSON messages.
 * This is a thin, header-only contract used by the networking layer
 * to build subscribe/authorize/submit messages and to parse
 * `mining.notify` notifications into `miner::net::Job`.
 */
#pragma once

#include "miner/net.hpp"
#include <nlohmann/json.hpp>
#include <string>

namespace miner {
namespace stratum_v1 {

using json = nlohmann::json;

class StratumV1 {
public:
    // Create a mining.subscribe message
    static json make_subscribe(const std::string& agent = "ohmy-miner/0.1.0");

    // Create a mining.authorize message
    static json make_authorize(const std::string& user, const std::string& pass);

    // Create a mining.submit message from a Share
    static json make_submit(const net::Share& share, int id = 1);

    // Detects if the JSON is a mining.notify message
    static bool is_notify(const json& j);

    // Parses a mining.notify JSON object into a net::Job
    // Throws std::invalid_argument on malformed input
    static net::Job parse_notify(const json& j);
};

} // namespace stratum_v1
} // namespace miner
