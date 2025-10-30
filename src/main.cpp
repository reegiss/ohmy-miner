/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include <iostream>
#include <string>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <fmt/color.h>

// Banner art - use raw string literal for multi-line string
const char* BANNER = R"(
╔═══════════════════════════════════════════════════════════════════╗
║                      OhMyMiner v1.0.0-GPU                         ║
║          High-Performance Quantum Circuit Mining on GPU           ║
╚═══════════════════════════════════════════════════════════════════╝
)";

void print_banner() {
    fmt::print(fg(fmt::color::cyan), "{}\n", BANNER);
}

void print_usage(const std::string& message = "") {
    if (!message.empty()) {
        fmt::print(fg(fmt::color::red), "Error: {}\n\n", message);
    }

    fmt::print(
        "Usage: ohmy-miner [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --algo ALGORITHM   Mining algorithm (required, supported: qhash)\n"
        "  --url URL         Pool URL (required, format: hostname:port)\n"
        "  --user WALLET     Wallet address for mining rewards (required)\n"
        "  --pass PASSWORD   Pool password (default: x)\n"
        "  --help           Show this help message\n"
        "\n"
        "Example:\n"
        "  ohmy-miner --algo qhash \\\n"
        "            --url qubitcoin.luckypool.io:8610 \\\n"
        "            --user bc1q...wallet... \\\n"
        "            --pass x\n"
        "\n"
    );
}

// Structure to hold command line parameters
struct MinerParams {
    std::string algo;
    std::string url;
    std::string user;
    std::string pass;

    bool validate() const {
        if (algo.empty() || algo != "qhash") {
            print_usage("Missing or invalid algorithm. Only 'qhash' is supported.");
            return false;
        }
        if (url.empty()) {
            print_usage("Pool URL is required.");
            return false;
        }
        if (user.empty()) {
            print_usage("Wallet address is required.");
            return false;
        }
        return true;
    }
};

MinerParams parse_command_line(int argc, char* argv[]) {
    MinerParams params;
    
    try {
        cxxopts::Options options("ohmy-miner", "High-Performance Quantum Circuit Mining on GPU");
        
        options.add_options()
            ("algo", "Mining algorithm (required, supported: qhash)", 
             cxxopts::value<std::string>())
            ("url", "Pool URL (required, format: hostname:port)", 
             cxxopts::value<std::string>())
            ("user", "Wallet address for mining rewards (required)", 
             cxxopts::value<std::string>())
            ("pass", "Pool password", 
             cxxopts::value<std::string>()->default_value("x"))
            ("help", "Show help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            print_banner();
            print_usage();
            exit(0);
        }

        // Store parameters
        if (result.count("algo")) params.algo = result["algo"].as<std::string>();
        if (result.count("url")) params.url = result["url"].as<std::string>();
        if (result.count("user")) params.user = result["user"].as<std::string>();
        if (result.count("pass")) params.pass = result["pass"].as<std::string>();

    } catch (const std::exception& e) {
        print_usage(fmt::format("Error parsing options: {}", e.what()));
        exit(1);
    }

    return params;
}

int main(int argc, char* argv[]) {
    print_banner();

    // Parse and validate command line
    auto params = parse_command_line(argc, argv);
    if (!params.validate()) {
        return 1;
    }

    // Log startup configuration
    fmt::print("\nStarting miner with configuration:\n");
    fmt::print("  Algorithm: {}\n", params.algo);
    fmt::print("  Pool URL:  {}\n", params.url);
    fmt::print("  Wallet:    {}\n", params.user);
    fmt::print("  Password:  {}\n", params.pass);
    fmt::print("\nInitializing GPU mining...\n\n");

    // TODO: Initialize mining components
    // TODO: Start mining loop

    return 0;
}
