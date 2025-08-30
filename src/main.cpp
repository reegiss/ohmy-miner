/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 */
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <boost/asio.hpp>

#include "cxxopts.hpp"
#include "pool_connection.h"
#include "thread_safe_queue.h"
#include "mining_job.h"
#include "gpu_manager.h"
#include "ialgorithm.h"
#include "qhash_algorithm.h"
#include "stats.h"

// =============================================================================
// Global Variables & Forward Declarations
// =============================================================================
std::atomic<bool> g_shutdown(false);
std::vector<GpuStats> g_gpu_stats;
std::mutex g_stats_mutex;

struct Config {
    std::string algo;
    std::string url;
    std::string host;
    uint16_t port;
    std::string user;
    std::string pass;
};

// Forward declarations for helper and thread functions
void signal_handler(int signal);
bool parse_url(const std::string& url, std::string& host, uint16_t& port);
std::string format_hashrate(double hashrate);
void miner_thread_func(int device_id, IAlgorithm* algorithm, ThreadSafeQueue<MiningJob>& job_queue, ThreadSafeQueue<FoundShare>& result_queue, GpuStats& stats);
void telemetry_thread_func();

// =============================================================================
// Main Application Entry Point
// =============================================================================
int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto available_gpus = detect_gpus();
    if (available_gpus.empty()) {
        std::cerr << "Fatal: No CUDA-compatible GPUs found. Exiting." << std::endl;
        return 1;
    }
    
    for(const auto& gpu : available_gpus) {
        g_gpu_stats.push_back({gpu.device_id, gpu.name, 0.0});
    }

    Config config;
    try {
        cxxopts::Options options("QtcMiner", "Qubitcoin (QTC) CUDA Miner");
        options.add_options()
            ("a,algo", "Hashing algorithm (required)", cxxopts::value<std::string>())
            ("o,url", "Pool URL with port (required)", cxxopts::value<std::string>())
            ("u,user", "Username or wallet address (required)", cxxopts::value<std::string>())
            ("p,pass", "Password for the pool", cxxopts::value<std::string>()->default_value("x"))
            ("h,help", "Print usage");
        
        auto result = options.parse(argc, argv);
        if (result.count("help") || !result.count("algo") || !result.count("url") || !result.count("user")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        config.algo = result["algo"].as<std::string>();
        config.url = result["url"].as<std::string>();
        config.user = result["user"].as<std::string>();
        config.pass = result["pass"].as<std::string>();
        if (!parse_url(config.url, config.host, config.port)) {
            return 1;
        }
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        return 1;
    }

    std::unique_ptr<IAlgorithm> algorithm;
    if (config.algo == "qhash") {
        algorithm = std::make_unique<QHashAlgorithm>();
    } else {
        std::cerr << "Error: Unknown algorithm '" << config.algo << "'" << std::endl;
        return 1;
    }

    std::cout << "Configuration loaded. Initializing miner for " << available_gpus.size() << " GPU(s)..." << std::endl;
    std::cout << "Press Ctrl+C to exit." << std::endl;

    boost::asio::io_context io_context;
    
    ThreadSafeQueue<MiningJob> job_queue;
    ThreadSafeQueue<FoundShare> result_queue;

    auto pool = std::make_shared<PoolConnection>(io_context, config.host, config.port, job_queue, result_queue);
    
    pool->start(config.user, config.pass);

    std::thread network_thread([&io_context](){ 
        try {
            io_context.run(); 
        } catch (const std::exception& e) {
            std::cerr << "Fatal Network Exception: " << e.what() << std::endl;
            g_shutdown = true;
        }
    });

    std::thread telemetry_thread(telemetry_thread_func);

    std::vector<std::thread> miner_threads;
    for (auto& stats : g_gpu_stats) {
        miner_threads.emplace_back(miner_thread_func, stats.device_id, algorithm.get(), std::ref(job_queue), std::ref(result_queue), std::ref(stats));
    }

    while(!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\n[MAIN] Shutdown signal received. Cleaning up..." << std::endl;
    
    pool->close();
    job_queue.shutdown();
    
    network_thread.join();
    telemetry_thread.join();
    for (auto& t : miner_threads) {
        t.join();
    }

    std::cout << "[MAIN] All threads have been joined. Exiting cleanly." << std::endl;
    return 0;
}

// =============================================================================
// Helper and Thread Function Implementations
// =============================================================================

// --- DEFINIÇÃO DA FUNÇÃO QUE FALTAVA ---
void signal_handler(int signal) {
    std::cout << "\nCaught signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_shutdown = true;
}

void miner_thread_func(int device_id, IAlgorithm* algorithm, ThreadSafeQueue<MiningJob>& job_queue, ThreadSafeQueue<FoundShare>& result_queue, GpuStats& stats) {
    cudaSetDevice(device_id);

    MiningJob current_job;
    if (!job_queue.wait_and_pop(current_job)) {
        return; 
    }

    while (!g_shutdown) {
        const uint32_t NONCES_PER_BATCH = 1024 * 512;
        uint64_t total_hashes_done_for_job = 0;
        auto job_start_time = std::chrono::high_resolution_clock::now();

        for (uint32_t nonce_base = 0; nonce_base < 0xFFFFFFFF && !g_shutdown; nonce_base += NONCES_PER_BATCH) {
            MiningJob new_job;
            if (job_queue.try_pop(new_job)) {
                if(new_job.clean_jobs) {
                    current_job = new_job;
                    nonce_base = 0; 
                    total_hashes_done_for_job = 0;
                    job_start_time = std::chrono::high_resolution_clock::now();
                }
            }

            uint32_t num_nonces_in_batch = std::min((uint32_t)NONCES_PER_BATCH, 0xFFFFFFFF - nonce_base);
            uint32_t found_nonce = algorithm->search_batch(device_id, current_job, nonce_base, num_nonces_in_batch, result_queue);

            if (found_nonce != 0xFFFFFFFF) {
                total_hashes_done_for_job += (found_nonce - nonce_base + 1);
                break; 
            } else {
                total_hashes_done_for_job += num_nonces_in_batch;
            }
            
            auto now = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - job_start_time).count();
            if (duration_ms > 500) {
                double seconds = static_cast<double>(duration_ms) / 1000.0;
                double hashrate = static_cast<double>(total_hashes_done_for_job) / seconds;
                std::lock_guard<std::mutex> lock(g_stats_mutex);
                stats.hashrate = hashrate;
            }
        }
        
        if (!job_queue.wait_and_pop(current_job)) {
            break;
        }
    }
}

void telemetry_thread_func() {
    auto last_display = std::chrono::steady_clock::now();
    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - last_display).count() >= 10) {
            std::cout << "\n--- MINER STATS ---\n";
            double total_hashrate = 0.0;
            std::lock_guard<std::mutex> lock(g_stats_mutex);
            for (const auto& stats : g_gpu_stats) {
                std::cout << "GPU " << stats.device_id << " (" << std::left << std::setw(25) << stats.name + "): " 
                          << format_hashrate(stats.hashrate) << std::endl;
                total_hashrate += stats.hashrate;
            }
            std::cout << "-------------------\n" << "TOTAL: " << format_hashrate(total_hashrate) << std::endl;
            last_display = std::chrono::steady_clock::now();
        }
    }
}

std::string format_hashrate(double hashrate) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    if (hashrate >= 1e6) {
        ss << (hashrate / 1e6) << " MH/s";
    } else if (hashrate >= 1e3) {
        ss << (hashrate / 1e3) << " kH/s";
    } else {
        ss << hashrate << " H/s";
    }
    return ss.str();
}

bool parse_url(const std::string& url, std::string& host, uint16_t& port) {
    size_t colon_pos = url.find_last_of(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Error: Invalid URL format. Port is missing (e.g., host:port)." << std::endl; 
        return false;
    }
    host = url.substr(0, colon_pos);
    try {
        unsigned long p = std::stoul(url.substr(colon_pos + 1));
        if (p == 0 || p > 65535) {
            std::cerr << "Error: Port number " << p << " is out of valid range (1-65535)." << std::endl;
            return false;
        }
        port = static_cast<uint16_t>(p);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid port number in URL." << std::endl; 
        return false;
    }
    return true;
}