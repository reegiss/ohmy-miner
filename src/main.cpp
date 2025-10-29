/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "gpu_info.hpp"
#include "pool_connection.hpp"
#include "miner.hpp"
#include <fmt/core.h>
#include <fmt/color.h>
#include <cxxopts.hpp>
#include <cuda_runtime.h>
#include <asio.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>

using namespace ohmy;

// Global flag for graceful shutdown
std::atomic<bool> should_exit{false};


void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
            "\n\nReceived shutdown signal. Exiting gracefully...\n");
        should_exit = true;
    }
}

void print_banner() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        R"(
  ___  _     __  __       __  __ _                 
 / _ \| |__ |  \/  |_   _|  \/  (_)_ __   ___ _ __ 
| | | | '_ \| |\/| | | | | |\/| | | '_ \ / _ \ '__|
| |_| | | | | |  | | |_| | |  | | | | | |  __/ |   
 \___/|_| |_|_|  |_|\__, |_|  |_|_|_| |_|\___|_|   
                    |___/                          
)");
    
    fmt::print(fg(fmt::color::yellow), 
        "Quantum Circuit Simulation Miner for Qubitcoin (QTC)\n");
    fmt::print(fg(fmt::color::white), 
        "Version: 0.1.0 | License: GPL-3.0\n");
    fmt::print(fg(fmt::color::green), 
        "High-Performance GPU-Accelerated Mining Framework\n\n");
}


int main(int argc, char* argv[]) {
    print_banner();

    try {
        cxxopts::Options options("ohmy-miner", "Quantum Proof-of-Work Miner");
        
        options.add_options()
            ("a,algo", "Mining algorithm", 
                cxxopts::value<std::string>()->default_value("qhash"))
            ("o,url", "Mining pool URL (host:port)", 
                cxxopts::value<std::string>())
            ("u,user", "Pool username (wallet.worker)", 
                cxxopts::value<std::string>())
            ("p,pass", "Pool password", 
                cxxopts::value<std::string>()->default_value("x"))
            ("h,help", "Print usage information");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Validate required parameters
        if (!result.count("url")) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
                "Error: Mining pool URL (--url) is required\n\n");
            std::cout << options.help() << std::endl;
            return 1;
        }

        if (!result.count("user")) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
                "Error: Pool username (--user) is required\n\n");
            std::cout << options.help() << std::endl;
            return 1;
        }

        // Extract parameters
        auto algo = result["algo"].as<std::string>();
        auto url = result["url"].as<std::string>();
        auto user = result["user"].as<std::string>();
        auto pass = result["pass"].as<std::string>();
        constexpr int batch_size = 256;  // Fixed batch size for optimal performance

        // Use default device 0
        int device = 0;

        // Detect available GPUs
        std::vector<GPUInfo> gpus;
        if (!GPUDetector::detect_all(gpus)) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: GPU detection failed. Cannot continue.\n");
            return 1;
        }

        // Validate selected GPU device
        if (!GPUDetector::validate_device(device, gpus)) {
            return 1;
        }

        // Display configuration
        fmt::print(fg(fmt::color::magenta) | fmt::emphasis::bold, 
            "=== Mining Configuration ===\n");
        fmt::print("Algorithm:    {}\n", algo);
        fmt::print("Pool URL:     {}\n", url);
        fmt::print("Username:     {}\n", user);
        fmt::print("Password:     {}\n", pass);
        fmt::print("CUDA Device:  {} (auto-selected)\n", device);
        fmt::print("Batch size:   {} (fixed)\n", batch_size);
        fmt::print(fg(fmt::color::magenta) | fmt::emphasis::bold, 
            "============================\n\n");

        // Initialize CUDA device
        cudaError_t cuda_error = cudaSetDevice(device);
        if (cuda_error != cudaSuccess) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "CUDA Error: {}\n", cudaGetErrorString(cuda_error));
            return 1;
        }
        fmt::print(fg(fmt::color::green), "✓ CUDA device {} initialized\n", device);

        // Setup ASIO io_context and pool
        asio::io_context io_context;
        fmt::print("\n");
        PoolConnection pool(io_context, url, user, pass);

        pool.set_job_callback([](const MiningJob& job) {
            fmt::print(fg(fmt::color::magenta),
                "New job received: {} (Clean: {})\n",
                job.job_id, job.clean_jobs ? "Yes" : "No");
        });
        pool.set_difficulty_callback([](double difficulty) {
            fmt::print(fg(fmt::color::yellow), "Difficulty updated: {}\n", difficulty);
        });
        pool.set_error_callback([](const std::string& error) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "Pool error: {}\n", error);
        });

        if (!pool.connect()) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: Failed to connect to pool\n");
            return 1;
        }
        if (!pool.subscribe()) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: Failed to subscribe to pool\n");
            return 1;
        }
        if (!pool.authorize()) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: Failed to authorize with pool\n");
            return 1;
        }

        pool.start_receive_loop();

        // Setup signal handlers
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        fmt::print("\n");
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ Mining initialization complete!\n");
        fmt::print(fg(fmt::color::yellow),
            "Press Ctrl+C to stop mining\n\n");

        // Run io_context in separate thread
        std::thread io_thread([&io_context]() {
            io_context.run();
        });

        // Start mining (delegated)
        fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
            "=== Mining Started ===\n\n");
        const int num_qubits = 16;  // QTC standard
        Miner miner(io_context, pool, num_qubits, batch_size, [&](){ return should_exit.load(); });
        miner.run();

        // Cleanup
        fmt::print("\n");
        fmt::print(fg(fmt::color::yellow), "Shutting down...\n");
        pool.disconnect();
        io_context.stop();
        if (io_thread.joinable()) {
            io_thread.join();
        }

        fmt::print(fg(fmt::color::green), "Shutdown complete. Goodbye!\n");

        return 0;
    } catch (const cxxopts::exceptions::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
            "Error parsing options: {}\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, 
            "Unexpected error: {}\n", e.what());
        return 1;
    }
}
