#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>

#include "cxxopts.hpp"
#include "pool_connection.h"
#include "thread_safe_queue.h"
#include "mining_job.h"

// Global atomic flag to signal that the program should shut down.
std::atomic<bool> g_shutdown(false);

// Signal handler for Ctrl+C (SIGINT).
void signal_handler(int signal) {
    std::cout << "\nCaught signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_shutdown = true;
}

// A struct to hold all our configuration settings
struct Config {
    std::string algo;
    std::string url;
    std::string host;
    uint16_t port;
    std::string user;
    std::string pass;
};

// Function to parse the URL into host and port
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

// The function for the miner thread, now with shutdown logic
void miner_thread_func(ThreadSafeQueue<MiningJob>& job_queue) {
    std::cout << "[MINER] Miner thread started. Waiting for jobs..." << std::endl;
    
    while (!g_shutdown) {
        MiningJob job;
        // wait_and_pop now returns false if the queue was shut down
        if (job_queue.wait_and_pop(job)) {
            std::cout << "[MINER] Received new job with ID: " << job.job_id << std::endl;
            if (job.clean_jobs) {
                std::cout << "[MINER] Clearing previous jobs." << std::endl;
            }
            // Logic to call the CUDA kernel to mine this job will go here
        } else {
            // The queue was shut down, so we exit the loop
            break;
        }
    }
    std::cout << "[MINER] Miner thread shutting down." << std::endl;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    // Bloco de parsing do cxxopts (permanece o mesmo)
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

    std::cout << "Configuration loaded. Initializing miner..." << std::endl;
    std::cout << "Press Ctrl+C to exit gracefully." << std::endl;

    ThreadSafeQueue<MiningJob> job_queue;
    PoolConnection pool(config.host, config.port, job_queue);
    
    std::thread network_thread(&PoolConnection::run, &pool, config.user, config.pass);
    std::thread miner_thread(miner_thread_func, std::ref(job_queue));

    // A thread principal espera ativamente pelo sinal de desligamento
    while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // --- SEQUÊNCIA DE DESLIGAMENTO CORRETA ---
    std::cout << "[MAIN] Shutdown signal received. Closing connections..." << std::endl;

    // 1. SINALIZAR: Desbloqueia todas as threads que estiverem esperando.
    pool.close();
    job_queue.shutdown();

    // 2. ESPERAR: Agora que as threads foram desbloqueadas, esperamos por elas.
    network_thread.join();
    miner_thread.join();
    // ----------------------------------------

    std::cout << "[MAIN] All threads have been joined. Exiting." << std::endl;

    return 0;
}