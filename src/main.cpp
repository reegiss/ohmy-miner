#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#include <vector>
#include <cuda_runtime.h>

extern "C" {
#include "qhash_miner.h"
}

#include "cxxopts.hpp"
#include "pool_connection.h"
#include "thread_safe_queue.h"
#include "mining_job.h"
#include "miner_bridge.h"
#include "gpu_manager.h"

// Global atomic flag to signal that the program should shut down.
std::atomic<bool> g_shutdown(false);

// Signal handler for Ctrl+C (SIGINT).
void signal_handler(int signal) {
    std::cout << "\nCaught signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_shutdown = true;
}

// Struct to hold all our configuration settings
struct Config {
    std::string algo;
    std::string url;
    std::string host;
    uint16_t port;
    std::string user;
    std::string pass;
};

// Function to parse the URL into host and port
bool parse_url(const std::string& url, std::string& host, uint16_t& port);

// The function for the miner thread
void miner_thread_func(int device_id, ThreadSafeQueue<MiningJob>& job_queue, ThreadSafeQueue<FoundShare>& result_queue);

int main(int argc, char* argv[]) {
    // Register our signal handler for the interrupt signal (Ctrl+C)
    std::signal(SIGINT, signal_handler);

    auto available_gpus = detect_gpus();
    if (available_gpus.empty()) {
        std::cerr << "No CUDA-compatible GPUs found. Miner cannot continue." << std::endl;
        return 1;
    }
    std::cout << "Found " << available_gpus.size() << " GPUs:" << std::endl;
    for(const auto& gpu : available_gpus) {
        std::cout << "  - GPU " << gpu.device_id << ": " << gpu.name << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
    
    Config config;
    try {
        cxxopts::Options options("QtcMiner", "A high-performance CUDA miner");
        options.add_options()
            ("a,algo", "Hashing algorithm", cxxopts::value<std::string>())
            ("o,url", "Pool URL with port (e.g., host:port)", cxxopts::value<std::string>())
            ("u,user", "Username or wallet address for the pool", cxxopts::value<std::string>())
            ("p,pass", "Password for the pool", cxxopts::value<std::string>()->default_value("x"))
            ("h,help", "Print usage");
        
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        if (!result.count("algo") || !result.count("url") || !result.count("user")) {
            std::cerr << "Error: Missing required arguments: --algo, --url, --user" << std::endl;
            return 1;
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

    std::cout << "Configuration loaded. Initializing miner for " << available_gpus.size() << " GPU(s)..." << std::endl;
    std::cout << "Press Ctrl+C to exit gracefully." << std::endl;

    ThreadSafeQueue<MiningJob> job_queue;
    ThreadSafeQueue<FoundShare> result_queue;
    PoolConnection pool(config.host, config.port, job_queue, result_queue);
    
    std::thread network_thread(&PoolConnection::run, &pool, config.user, config.pass);

    std::vector<std::thread> miner_threads;
    for (const auto& gpu : available_gpus) {
        miner_threads.emplace_back(miner_thread_func, gpu.device_id, std::ref(job_queue), std::ref(result_queue));
    }

    network_thread.join();
    
    std::cout << "[MAIN] Network thread finished. Signaling miner threads to stop." << std::endl;
    g_shutdown = true;
    job_queue.shutdown();
    result_queue.shutdown();

    for (auto& t : miner_threads) {
        t.join();
    }

    std::cout << "[MAIN] All threads have been joined. Exiting." << std::endl;

    return 0;
}

// --- Implementation of helper functions ---

void miner_thread_func(int device_id, ThreadSafeQueue<MiningJob>& job_queue, ThreadSafeQueue<FoundShare>& result_queue) {
    cudaSetDevice(device_id);
    std::cout << "[MINER " << device_id << "] Thread started." << std::endl;

    if (!qhash_thread_init(device_id)) {
        std::cerr << "[MINER " << device_id << "] CRITICAL: Failed to initialize qhash. Shutting down thread." << std::endl;
        g_shutdown = true;
        return;
    }

    while (!g_shutdown) {
        MiningJob job;
        if (job_queue.wait_and_pop(job)) {
            MinerBridge::process_job(device_id, job, result_queue);
        } else {
            break;
        }
    }

    qhash_thread_destroy();
    std::cout << "[MINER " << device_id << "] Miner thread shutting down." << std::endl;
}

bool parse_url(const std::string& url, std::string& host, uint16_t& port) {
    size_t colon_pos = url.find_last_of(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Error: Invalid URL format. Port is missing." << std::endl;
        return false;
    }
    host = url.substr(0, colon_pos);
    try {
        unsigned long p = std::stoul(url.substr(colon_pos + 1));
        if (p > 65535) {
            std::cerr << "Error: Port number " << p << " is out of range." << std::endl;
            return false;
        }
        port = static_cast<uint16_t>(p);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid port number in URL." << std::endl;
        return false;
    }
    return true;
}