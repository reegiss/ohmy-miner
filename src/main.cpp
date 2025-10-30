/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include <iostream>
#include <string>
#include <memory>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <fmt/color.h>
#include <asio.hpp>

#include "ohmy/pool/stratum.hpp"
#include "ohmy/pool/work.hpp"
#include "ohmy/pool/monitor.hpp"
#include "ohmy/mining/qhash_worker.hpp"
#include "ohmy/quantum/simulator.hpp"

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

    try {
        // Initialize ASIO
        asio::io_context io_context;

        // Create work manager
        auto work_manager = std::make_shared<ohmy::pool::WorkManager>();

        // Create job dispatcher
        auto dispatcher = std::make_shared<ohmy::pool::JobDispatcher>(work_manager);

        // Create job monitor
        auto monitor = std::make_shared<ohmy::pool::JobMonitor>(work_manager, dispatcher);

        // Create and configure stratum client
        auto stratum = std::make_shared<ohmy::pool::StratumClient>(
            io_context, params.url, params.user, params.pass);

        // Set work callback to add jobs to queue
        stratum->set_work_callback([work_manager](const ohmy::pool::WorkPackage& work) {
            work_manager->add_job(work);
        });

        // Set share callback for result tracking
        stratum->set_share_callback([work_manager](const ohmy::pool::ShareResult& share) {
            work_manager->track_share_result(share);
            if (share.accepted) {
                fmt::print(fg(fmt::color::green), "Share accepted by pool!\n");
            } else {
                fmt::print(fg(fmt::color::red), "Share rejected by pool: {}\n", share.reason);
            }
        });

        // Connect to pool
        stratum->connect();

        // Start job monitoring
        monitor->start_monitoring();

        // Create and configure mining workers
        const int num_workers = 2;  // Start with 2 workers for testing
        std::vector<std::shared_ptr<ohmy::mining::QHashWorker>> workers;
        
        for (int i = 0; i < num_workers; ++i) {
            // Create quantum simulator for each worker
            auto simulator = ohmy::quantum::SimulatorFactory::create(
                ohmy::quantum::SimulatorFactory::Backend::CPU_BASIC, 4);
            
            // Create worker
            auto worker = std::make_shared<ohmy::mining::QHashWorker>(
                std::move(simulator), i);
            
            // Set share callback to submit to pool
            worker->set_share_callback([stratum](const ohmy::pool::ShareResult& share) {
                stratum->submit_share(share);
            });
            
            // Add to dispatcher
            dispatcher->add_worker(worker);
            workers.push_back(worker);
        }

        // Start job dispatcher
        dispatcher->start_dispatching();

        fmt::print("Mining system initialized:\n");
        fmt::print("  ✓ Pool connection established\n");
        fmt::print("  ✓ Job dispatcher started\n");
        fmt::print("  ✓ Job monitor started\n");
        fmt::print("  ✓ {} quantum mining workers created\n", num_workers);
        fmt::print("  ✓ Using {} quantum simulator backend\n\n", 
                  workers[0]->get_stats().current_job_id.empty() ? "CPU_BASIC" : "CPU_BASIC");

        // Run ASIO event loop
        fmt::print("Starting event loop...\n");
        io_context.run();

        // Cleanup
        monitor->stop_monitoring();
        dispatcher->stop_dispatching();

    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Fatal error: {}\n", e.what());
        return 1;
    }

    return 0;
}
