/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <cxxopts.hpp>
#include <fmt/format.h>
#include <asio.hpp>
#include <cuda_runtime.h>
#include "ohmy/log.hpp"
#include <cmath>

#include "ohmy/pool/stratum.hpp"
#include "ohmy/pool/work.hpp"
#include "ohmy/pool/monitor.hpp"
#include "ohmy/mining/batched_qhash_worker.hpp"
#include "ohmy/quantum/simulator.hpp"
#include "ohmy/quantum/cuda_types.hpp"
#include "ohmy/quantum/batched_cuda_simulator.hpp"


void print_usage(const std::string& message = "") {
    if (!message.empty()) {
        ohmy::log::line("Error: {}", message);
    }

    fmt::print(
        "Usage: ohmy-miner [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --algo ALGORITHM          Mining algorithm (required, supported: qhash)\n"
        "  --url URL                 Pool URL (required, format: hostname:port)\n"
        "  --user WALLET             Wallet address for mining rewards (required)\n"
        "  --pass PASSWORD           Pool password (default: x)\n"
        "  --diff DIFFICULTY         Static difficulty (optional, e.g., 60K, 1M)\n"
        "  --extranonce-subscribe    Enable dynamic extranonce support (for specific pools)\n"
    "  --no-mining               Connect and exchange protocol only (no GPU mining)\n"
    "  --send-capabilities       Send mining.capabilities after authorization\n"
    "  --suggest-target HEX     Advisory 256-bit target (hex) sent after authorization\n"
    "  --suggest-diff N         Advisory difficulty sent after authorization\n"
        "  --help                    Show this help message\n"
        "\n"
        "Example:\n"
        "  ohmy-miner --algo qhash \\\n"
        "            --url qubitcoin.luckypool.io:8610 \\\n"
        "            --user bc1q...wallet... \\\n"
        "            --diff 60K \\\n"
        "            --pass x\n"
        "\n"
        "Note: --extranonce-subscribe is only needed for pools that use mining.set_extranonce\n"
        "      (most standard pools don't require this)\n"
        "\n"
    );
}

// Structure to hold command line parameters
struct MinerParams {
    std::string algo;
    std::string url;
    std::string user;
    std::string pass;
    std::string diff;
    bool extranonce_subscribe = false;  // Enable mining.extranonce.subscribe
    bool no_mining = false;
    bool send_capabilities = false;
    std::string suggest_target_hex;
    double suggest_diff = 0.0;
    bool have_suggest_diff = false;

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
            ("diff", "Static difficulty (optional, e.g., 60K, 1M)", 
             cxxopts::value<std::string>()->default_value(""))
            ("extranonce-subscribe", "Enable mining.extranonce.subscribe for pools with dynamic extranonce")
              ("no-mining", "Connect only; do not start GPU workers")
              ("send-capabilities", "Send mining.capabilities after authorization")
              ("suggest-target", "Send mining.suggest_target with full 256-bit hex target", cxxopts::value<std::string>())
              ("suggest-diff", "Send mining.suggest_difficulty with numeric value", cxxopts::value<double>())
            ("help", "Show help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            print_usage();
            exit(0);
        }

        // Store parameters
        if (result.count("algo")) params.algo = result["algo"].as<std::string>();
        if (result.count("url")) params.url = result["url"].as<std::string>();
        if (result.count("user")) params.user = result["user"].as<std::string>();
        if (result.count("pass")) params.pass = result["pass"].as<std::string>();
        if (result.count("diff")) params.diff = result["diff"].as<std::string>();
        if (result.count("extranonce-subscribe")) params.extranonce_subscribe = true;
    if (result.count("no-mining")) params.no_mining = true;
    if (result.count("send-capabilities")) params.send_capabilities = true;
    if (result.count("suggest-target")) params.suggest_target_hex = result["suggest-target"].as<std::string>();
    if (result.count("suggest-diff")) { params.suggest_diff = result["suggest-diff"].as<double>(); params.have_suggest_diff = true; }

    } catch (const std::exception& e) {
        print_usage(fmt::format("Error parsing options: {}", e.what()));
        exit(1);
    }

    return params;
}

int main(int argc, char* argv[]) {
    // Parse and validate command line
    auto params = parse_command_line(argc, argv);
    if (!params.validate()) {
        return 1;
    }

    if (!params.no_mining) {
        ohmy::log::line("Initializing GPUs...");
    }

    try {
        // Initialize ASIO
        asio::io_context io_context;

        // Create work manager
        auto work_manager = std::make_shared<ohmy::pool::WorkManager>();

        // Create job dispatcher
        auto dispatcher = std::make_shared<ohmy::pool::JobDispatcher>(work_manager);

    // Create job monitor
        auto monitor = std::make_shared<ohmy::pool::JobMonitor>(work_manager, dispatcher);

        // Format username with static difficulty if provided
        // Pool format: ADDRESS=DIFF.WORKER or ADDRESS=DIFF
        std::string formatted_user = params.user;
        if (!params.diff.empty()) {
            // Check if user already has a worker suffix (e.g., wallet.worker)
            size_t dot_pos = formatted_user.find('.');
            if (dot_pos != std::string::npos) {
                // wallet.worker -> wallet=DIFF.worker
                std::string wallet = formatted_user.substr(0, dot_pos);
                std::string worker = formatted_user.substr(dot_pos + 1);
                formatted_user = fmt::format("{}={}.{}", wallet, params.diff, worker);
            } else {
                // wallet -> wallet=DIFF
                formatted_user = fmt::format("{}={}", params.user, params.diff);
            }
            ohmy::log::line("Using static difficulty: {} (formatted as: {})", params.diff, formatted_user);
        }

        // Create and configure stratum client
        auto stratum = std::make_shared<ohmy::pool::StratumClient>(
            io_context, params.url, formatted_user, params.pass, params.extranonce_subscribe);

        // Configure optional advisories to send after authorization
        stratum->set_send_capabilities(params.send_capabilities, params.suggest_target_hex);
        if (!params.suggest_target_hex.empty()) {
            stratum->set_suggest_target(params.suggest_target_hex);
        }
        if (params.have_suggest_diff) {
            stratum->set_suggest_difficulty(params.suggest_diff);
        }

        // Set work callback to add jobs to queue
        stratum->set_work_callback([work_manager](const ohmy::pool::WorkPackage& work) {
            work_manager->add_job(work);
        });

        // Set share callback for result tracking
        stratum->set_share_callback([work_manager](const ohmy::pool::ShareResult& share) {
            work_manager->track_share_result(share);
            if (share.accepted) {
                ohmy::log::line("Share accepted");
            } else {
                ohmy::log::line("Share rejected: {}", share.reason);
            }
        });

        // Track pool difficulty updates for ETA calculations/monitor
        stratum->set_difficulty_callback([work_manager](double diff){
            work_manager->set_difficulty(diff);
        });

        // Connect to pool
        stratum->connect();

        // Start job monitoring
        monitor->start_monitoring();

        // Setup signal handling (Ctrl+C / SIGTERM)
        asio::signal_set signals(io_context, SIGINT, SIGTERM);
        signals.async_wait([&](const std::error_code&, int){
            ohmy::log::line("Ctrl+C received");
            ohmy::log::line("exiting...");
            monitor->stop_monitoring();
            dispatcher->stop_dispatching();
            dispatcher->stop_all_workers();
            stratum->disconnect();
            io_context.stop();
        });

        // Auto-detect GPU and create appropriate workers (unless no-mining)
        ohmy::quantum::cuda::DeviceInfo gpu_info;

        if (!params.no_mining) {
            try {
                gpu_info = ohmy::quantum::cuda::DeviceInfo::query(0);

                if (!gpu_info.is_compatible()) {
                    ohmy::log::line("GPU compute capability {}.{} < 7.5 (minimum required)",
                              gpu_info.compute_capability_major, gpu_info.compute_capability_minor);
                    return 1;
                }

                int drv=0, rt=0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
                int arch_major = gpu_info.compute_capability_major;
                int arch_minor = gpu_info.compute_capability_minor;
                // Attempt to get bus id from CUDA props
                cudaDeviceProp prop{};
                int busId = -1;
                if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
                    busId = prop.pciBusID;
                }
                ohmy::log::line("GPU #0: {} [busID: {}] [arch: sm{}{}] [driver: {}]",
                                gpu_info.name, busId, arch_major, arch_minor, drv/1000);
            } catch (const std::exception& e) {
                ohmy::log::line("No CUDA GPU detected: {}", e.what());
                return 1;
            }
        }

        // GPU MODE: Create single batched worker
        constexpr int BATCH_SIZE = 1000;
        constexpr int NUM_QUBITS = 16;

        if (!params.no_mining) {
            try {
                // Create batched CUDA simulator
                auto simulator = std::make_unique<ohmy::quantum::cuda::BatchedCudaSimulator>(
                    NUM_QUBITS, BATCH_SIZE, 0);

                // Create GPU worker
                auto worker = std::make_shared<ohmy::mining::BatchedQHashWorker>(
                    std::move(simulator), 0, BATCH_SIZE);

                // Set share callback
                worker->set_share_callback([stratum](const ohmy::pool::ShareResult& share) {
                    stratum->submit_share(share);
                });

                // Add to dispatcher
                dispatcher->add_worker(worker);

                // Print threads/intensity/compute units/memory line
                const int threads = 1; // single GPU worker thread
                const int cu = gpu_info.multiprocessor_count;
                double free_mb = static_cast<double>(gpu_info.free_memory) / (1024.0 * 1024.0);
                // Heuristic intensity from batch size (tuned to resemble typical values)
                int intensity = static_cast<int>(std::round(std::log2(BATCH_SIZE))) + 13;
                ohmy::log::line("threads: {}, intensity: {}, cu: {}, mem: {}Mb",
                                threads, intensity, cu, static_cast<int>(free_mb));
            } catch (const std::exception& e) {
                ohmy::log::line("Failed to initialize GPU worker: {}", e.what());
                return 1;
            }
        }

        // Start job dispatcher unless in protocol-only mode
        if (!params.no_mining) {
            dispatcher->start_dispatching();
        } else {
            ohmy::log::line("Protocol-only mode: not starting GPU workers");
        }

        // Run ASIO event loop
        io_context.run();

        // Cleanup
        monitor->stop_monitoring();
        dispatcher->stop_dispatching();
    } catch (const std::exception& e) {
        ohmy::log::line("Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
