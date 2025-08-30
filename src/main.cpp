// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

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
#include "miner/pool_connection.h"
#include "thread_safe_queue.h"
#include "mining_job.h"
#include "found_share.h"
#include "gpu_manager.h"
#include "ialgorithm.h"
#include "qhash_algorithm.h"
#include "stats.h"

// Globals
std::atomic<bool> g_shutdown(false);
ThreadSafeQueue<MiningJob> g_job_queue;
ThreadSafeQueue<FoundShare> g_result_queue;
std::vector<GpuStats> g_gpu_stats;
std::mutex g_stats_mutex;

void signal_handler(int) {
    if (g_shutdown.load()) {
        std::cout << "\nForcing exit..." << std::endl;
        exit(1);
    }
    std::cout << "\nShutdown signal received. Stopping threads..." << std::endl;
    g_shutdown = true;
    g_job_queue.shutdown();
    g_result_queue.shutdown();
}

struct Config { std::string url, host, user, pass, algo; uint16_t port; };
bool parse_url(const std::string& url, std::string& host, uint16_t& port);
std::string format_hashrate(double hashrate);

void miner_thread_func(int device_id, IAlgorithm* algorithm, GpuStats& stats);
void telemetry_thread_func();
void submission_thread_func(std::shared_ptr<PoolConnection> pool);

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);
    Config config;

    try {
        cxxopts::Options options("QtcMiner", "A high-performance CUDA miner");
        options.add_options()
            ("a,algo", "Hashing algorithm", cxxopts::value<std::string>())
            ("o,url", "Pool URL", cxxopts::value<std::string>())
            ("u,user", "Username/wallet", cxxopts::value<std::string>())
            ("p,pass", "Password", cxxopts::value<std::string>()->default_value("x"))
            ("h,help", "Print usage");
        auto result = options.parse(argc, argv);
        if (result.count("help") || !result.count("url") || !result.count("user") || !result.count("algo")) { std::cout << options.help() << std::endl; return 1; }
        config.url = result["url"].as<std::string>(); 
        config.user = result["user"].as<std::string>(); 
        config.pass = result["pass"].as<std::string>();
        config.algo = result["algo"].as<std::string>();
        if (!parse_url(config.url, config.host, config.port)) { return 1; }
    } catch (const cxxopts::exceptions::exception& e) { std::cerr << "Error parsing options: " << e.what() << std::endl; return 1; }

    auto available_gpus = detect_gpus();
    if (available_gpus.empty()) { std::cerr << "No CUDA-compatible GPUs found." << std::endl; return 1; }
    for(const auto& gpu : g_gpu_stats) { g_gpu_stats.push_back({gpu.device_id, gpu.name, 0.0}); }

    std::unique_ptr<IAlgorithm> algorithm;
    if (config.algo == "qhash") {
        algorithm = std::make_unique<QHashAlgorithm>();
    } else {
        std::cerr << "Error: Unknown algorithm '" << config.algo << "'" << std::endl; return 1;
    }
    
    boost::asio::io_context io_context;
    auto pool = std::make_shared<PoolConnection>(io_context);

    pool->on_connected = []() { std::cout << "[MAIN] Pool connection established." << std::endl; };
    pool->on_disconnected = [](const std::string& reason) { std::cout << "[MAIN] Pool connection lost: " << reason << std::endl; };
    pool->on_new_job = [](const MiningJob& job) { g_job_queue.push(job); };
    pool->on_submit_result = [](uint64_t id, bool accepted, const std::string& err) {
        if (accepted) { std::cout << ">> Share #" << id << " ACCEPTED" << std::endl; } 
        else { std::cerr << ">> Share #" << id << " REJECTED: " << err << std::endl; }
    };

    pool->connect(config.host, std::to_string(config.port), config.user, config.pass);
    
    auto work_guard = boost::asio::make_work_guard(io_context);
    std::thread network_thread([&io_context](){ try { io_context.run(); } catch (const std::exception& e) { std::cerr << "[NETWORK] Exception: " << e.what() << std::endl; } });

    std::vector<std::thread> miner_threads;
    for (auto& stats : g_gpu_stats) { miner_threads.emplace_back(miner_thread_func, stats.device_id, algorithm.get(), std::ref(stats)); }
    std::thread submit_thread(submission_thread_func, pool);
    std::thread telemetry_thread(telemetry_thread_func);

    // Main thread now simply waits for shutdown signal
    while(!g_shutdown.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }

    std::cout << "[MAIN] Shutting down..." << std::endl;
    pool->disconnect();
    work_guard.reset();
    io_context.stop();
    
    if(network_thread.joinable()) network_thread.join();
    if(submit_thread.joinable()) submit_thread.join();
    for (auto& t : miner_threads) { if(t.joinable()) t.join(); }
    if(telemetry_thread.joinable()) telemetry_thread.join();

    std::cout << "[MAIN] All threads joined. Exiting." << std::endl;
    return 0;
}

// Implementations of thread and helper functions
bool parse_url(const std::string& url, std::string& host, uint16_t& port) {
    size_t colon_pos = url.find_last_of(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: Invalid URL format. Expected host:port" << std::endl; return false; }
    host = url.substr(0, colon_pos);
    try {
        unsigned long p = std::stoul(url.substr(colon_pos + 1));
        if (p == 0 || p > 65535) { std::cerr << "Error: Invalid port number." << std::endl; return false; }
        port = static_cast<uint16_t>(p);
    } catch (const std::exception&) { std::cerr << "Error: Invalid port number." << std::endl; return false; }
    return true;
}

void submission_thread_func(std::shared_ptr<PoolConnection> pool) {
    while(true) {
        FoundShare share;
        if (g_result_queue.wait_and_pop(share)) {
            pool->submit(share.job_id, share.extranonce2, share.ntime, share.nonce_hex);
        } else { break; }
    }
}

void miner_thread_func(int device_id, IAlgorithm* algorithm, GpuStats& stats) {
    cudaSetDevice(device_id);
    MiningJob current_job;
    if (!g_job_queue.wait_and_pop(current_job)) { return; }

    while (!g_shutdown.load()) {
        const uint32_t NONCES_PER_BATCH = 1024 * 1024;
        uint64_t total_hashes_done_for_job = 0;
        auto job_start_time = std::chrono::high_resolution_clock::now();

        for (uint32_t nonce_base = 0; nonce_base < 0xFFFFFFFF && !g_shutdown.load(); nonce_base += NONCES_PER_BATCH) {
            MiningJob new_job;
            if (g_job_queue.try_pop(new_job)) {
                current_job = new_job;
                nonce_base = 0;
                total_hashes_done_for_job = 0;
                job_start_time = std::chrono::high_resolution_clock::now();
            }

            uint32_t num_nonces_in_batch = std::min((uint32_t)NONCES_PER_BATCH, 0xFFFFFFFF - nonce_base);
            algorithm->search_batch(device_id, current_job, nonce_base, num_nonces_in_batch, g_result_queue);
            total_hashes_done_for_job += num_nonces_in_batch;
            
            auto now = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - job_start_time).count();
            if (duration_ms > 500) {
                double seconds = static_cast<double>(duration_ms) / 1000.0;
                std::lock_guard<std::mutex> lock(g_stats_mutex);
                stats.hashrate = static_cast<double>(total_hashes_done_for_job) / seconds;
            }
        }
        
        if (!g_job_queue.wait_and_pop(current_job)) { break; }
    }
}

void telemetry_thread_func() {
    while (!g_shutdown.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        if (g_shutdown.load()) break;
        
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        std::cout << "\n--- MINER STATS ---\n";
        double total_hashrate = 0.0;
        for (const auto& stats : g_gpu_stats) {
            std::cout << "GPU " << stats.device_id << " (" << stats.name << "): " 
                      << format_hashrate(stats.hashrate) << std::endl;
            total_hashrate += stats.hashrate;
        }
        std::cout << "-------------------\n";
        std::cout << "TOTAL: " << format_hashrate(total_hashrate) << std::endl;
    }
}

std::string format_hashrate(double hashrate) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    if (hashrate >= 1e6) ss << (hashrate / 1e6) << " MH/s";
    else if (hashrate >= 1e3) ss << (hashrate / 1e3) << " kH/s";
    else ss << hashrate << " H/s";
    return ss.str();
}