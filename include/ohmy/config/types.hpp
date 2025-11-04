#pragma once

#include <string>
#include <optional>

namespace ohmy::config {

struct MinerConfig {
    std::string algo{"qhash"};
    std::string url;
    std::string user;
    std::string pass{"x"};
};

struct ParseResult {
    std::optional<MinerConfig> cfg; // present when valid and ready to run
    std::string config_path{"miner.conf"};
    bool show_only{false}; // true if --help/--version was printed
    bool debug{false};     // true if --debug was passed on CLI
    bool stratum_connect{false}; // true if --stratum-connect probe requested
    bool stratum_listen{false};  // true if --stratum-listen (keep connection open)
};

} // namespace ohmy::config
