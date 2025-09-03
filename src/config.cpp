// src/config.cpp

/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "miner/Config.hpp"
#include <cxxopts.hpp>
#include <fmt/core.h>
#include <optional>

namespace miner {

std::optional<Config> parse_arguments(int argc, char** argv) {
    cxxopts::Options options("ohmy-miner", "A high-performance, open-source GPU miner.");

    options.add_options()
        ("a,algo", "Specify the hash algorithm to use (e.g., qhash)", cxxopts::value<std::string>())
        ("o,url", "URL of the mining pool (e.g., stratum+tcp://host:port)", cxxopts::value<std::string>())
        ("u,user", "Username or wallet address for the mining pool", cxxopts::value<std::string>())
        ("p,pass", "Password for the mining pool", cxxopts::value<std::string>()->default_value("x"))
        ("h,help", "Print help message");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return std::nullopt; // Indicates successful exit for --help
        }

        // Validate that all required arguments are present
        if (result.count("algo") == 0 || result.count("url") == 0 || result.count("user") == 0) {
            fmt::print(stderr, "Error: Missing required arguments: --algo, --url, and --user are mandatory.\n");
            fmt::print(stderr, "Use --help for more information.\n");
            return std::nullopt; // Indicates failure due to missing args
        }

        Config config;
        config.algo = result["algo"].as<std::string>();
        config.url = result["url"].as<std::string>();
        config.user = result["user"].as<std::string>();
        config.pass = result["pass"].as<std::string>();
        
        return config;

    } catch (const cxxopts::exceptions::exception& e) {
        fmt::print(stderr, "Error parsing options: {}\n", e.what());
        fmt::print(stderr, "Use --help for more information.\n");
        return std::nullopt; // Indicates failure due to parsing error
    }
}

} // namespace miner