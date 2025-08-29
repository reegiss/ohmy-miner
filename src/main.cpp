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

#include "cxxopts.hpp"
#include "pool_connection.h"
#include "thread_safe_queue.h"
#include "mining_job.h"
#include "gpu_manager.h"
#include "ialgorithm.h"
#include "qhash_algorithm.h"
#include "stats.h" // Include our new stats header

// Global variables
std::atomic<bool> g_shutdown(false);
std::vector<GpuStats> g_gpu_stats;
std::mutex g_stats_mutex;

void signal_handler(int signal) {
    std::cout << "\nCaught signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_shutdown = true;
}

// --- Helper Functions ---
struct Config {
    std::string algo;
    std::string url;
    std::string host;
    uint16_t port;
    std::string user;
    std::string pass;
};
bool parse_url(const std::string& url, std::string& host, uint16_t& port);
std::string format_hashrate(double hashrate);

// --- Thread Functions ---
void miner_thread_func(int device_id, IAlgorithm* algorithm, ThreadSafeQueue<MiningJob>& job_queue, ThreadSafeQueue<FoundShare>& result_queue, GpuStats& stats);
void telemetry_thread_func();

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    auto available_gpus = detect_gpus();
    if (available_gpus.empty()) {
        std::cerr << "No CUDA-compatible GPUs found. Miner cannot continue." << std::endl;
        return 1;
    }
    
    // Initialize stats for each GPU
    for(const auto& gpu : available_gpus) {
        g_gpu_stats.push_back({gpu.device_id, gpu.name, 0.0});
    }

    Config config;
    try {
        cxxopts::Options options("QtcMiner", "A high-performance CUDA miner");
        options.add_options()
            ("a,algo", "Hashing algorithm (e.g., qhash)", cxxopts::value<std::string>())
            ("o,url", "Pool URL (e.g., host:port)", cxxopts::value<std::string>())
            ("u,user", "Username/wallet", cxxopts::value<std::string>())
            ("p,pass", "Password", cxxopts::value<std::string>()->default_value("x"))
            ("h,help", "Print usage");
        
        auto result = options.parse(argc, argv);
        if (result.count("help") || !result.count("algo") || !result.count("url") || !result.count("user")) {
            std::cout << options.help() << std::endl; return 1;
        }
        config.algo = result["algo"].as<std::string>();
        config.url = result["url"].as<std::string>();
        config.user = result["user"].as<std::string>();
        config.pass = result["pass"].as<std::string>();
        if (!parse_url(config.url, config.host, config.port)) { return 1; }
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl; return 1;
    }

    std::unique_ptr<IAlgorithm> algorithm;
    if (config.algo == "qhash") {
        algorithm = std::make_unique<QHashAlgorithm>();
    } else {
        std::cerr << "Error: Unknown algorithm '" << config.algo << "'" << std::endl; return 1;
    }

    std::cout << "Configuration loaded. Initializing miner..." << std::endl;
    std::cout << "Press Ctrl+C to exit." << std::endl;

    ThreadSafeQueue<MiningJob> job_queue;
    ThreadSafeQueue<FoundShare> result_queue;
    PoolConnection pool(config.host, config.port, job_queue, result_queue);
    
    std::thread network_thread(&PoolConnection::run, &pool, config.user, config.pass);
    std::thread telemetry_thread(telemetry_thread_func);

    std::vector<std::thread> miner_threads;
    for (auto& stats : g_gpu_stats) {
        miner_threads.emplace_back(miner_thread_func, stats.device_id, algorithm.get(), std::ref(job_queue), std::ref(result_queue), std::ref(stats));
    }

    network_thread.join();
    
    std::cout << "[MAIN] Network thread finished. Signaling all threads to stop." << std::endl;
    g_shutdown = true;
    job_queue.shutdown();
    result_queue.shutdown();

    for (auto& t : miner_threads) {
        t.join();
    }
    telemetry_thread.join();

    std::cout << "[MAIN] All threads have been joined. Exiting." << std::endl;
    return 0;
}

// --- Implementation of Thread and Helper Functions ---

void miner_thread_func(int device_id, IAlgorithm* algorithm, ThreadSafeQueue<MiningJob>& job_queue, ThreadSafeQueue<FoundShare>& result_queue, GpuStats& stats) {
    cudaSetDevice(device_id);
    if (!algorithm->thread_init(device_id)) {
        std::cerr << "[MINER " << device_id << "] CRITICAL: Failed to initialize algorithm." << std::endl;
        g_shutdown = true;
        return;
    }

    MiningJob current_job;
    // Start by waiting for the very first job
    if (!job_queue.wait_and_pop(current_job)) {
        return; // Shutdown while waiting for the first job
    }

    while (!g_shutdown) {
        const uint32_t NONCES_PER_BATCH = 1024 * 1024;
        uint64_t total_hashes_done_for_job = 0;
        auto job_start_time = std::chrono::high_resolution_clock::now();

        // Main nonce search loop for the current job
        for (uint32_t nonce_base = 0; nonce_base < 0xFFFFFFFF && !g_shutdown; nonce_base += NONCES_PER_BATCH) {
            // --- Check for a new job BEFORE starting the next batch ---
            MiningJob new_job;
            if (job_queue.try_pop(new_job)) {
                std::cout << "[MINER " << device_id << "] New job received, switching work." << std::endl;
                current_job = new_job;
                nonce_base = 0; // Reset nonce search
                total_hashes_done_for_job = 0;
                job_start_time = std::chrono::high_resolution_clock::now();
            }

            uint32_t num_nonces_in_batch = std::min((uint32_t)NONCES_PER_BATCH, 0xFFFFFFFF - nonce_base);
            uint32_t found_nonce = algorithm->search_batch(device_id, current_job, nonce_base, num_nonces_in_batch, result_queue);

            if (found_nonce != 0xFFFFFFFF) {
                total_hashes_done_for_job += (found_nonce - nonce_base + 1);
                break; // Found share, break from nonce loop to wait for next job
            } else {
                total_hashes_done_for_job += num_nonces_in_batch;
            }
            
            // Update hashrate
            auto now = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - job_start_time).count();
            if (duration_ms > 500) { // Update hashrate if job has been running for a bit
                double seconds = duration_ms / 1000.0;
                double hashrate = static_cast<double>(total_hashes_done_for_job) / seconds;
                std::lock_guard<std::mutex> lock(g_stats_mutex);
                stats.hashrate = hashrate;
            }
        }
        
        // After finding a share or being interrupted, wait for the next job
        if (!job_queue.wait_and_pop(current_job)) {
            break; // Shutdown signaled
        }
    }

    algorithm->thread_destroy();
    std::cout << "[MINER " << device_id << "] Thread shutting down." << std::endl;
}

void telemetry_thread_func() {
    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        if (g_shutdown) break;

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
    if (hashrate > 1e6) {
        ss << (hashrate / 1e6) << " MH/s";
    } else if (hashrate > 1e3) {
        ss << (hashrate / 1e3) << " kH/s";
    } else {
        ss << hashrate << " H/s";
    }
    return ss.str();
}

bool parse_url(const std::string& url, std::string& host, uint16_t& port) {
    size_t colon_pos = url.find_last_of(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Error: Invalid URL format." << std::endl; return false;
    }
    host = url.substr(0, colon_pos);
    try {
        port = static_cast<uint16_t>(std::stoul(url.substr(colon_pos + 1)));
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid port number." << std::endl; return false;
    }
    return true;
}