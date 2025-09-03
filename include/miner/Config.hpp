// include/miner/Config.hpp

/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <string>
#include <optional>

namespace miner {

struct Config {
    std::string algo;
    std::string url;
    std::string user;
    std::string pass;
};

// Parses command line arguments.
// Returns a populated Config struct on success.
// Returns std::nullopt if parsing fails or if --help was requested,
// in which case the program should exit.
std::optional<Config> parse_arguments(int argc, char** argv);

} // namespace miner