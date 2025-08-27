#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "cxxopts.hpp" // <-- Nossa nova biblioteca de parsing
#include "cuda_kernels.cuh"

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
        // std::stoul is C++11, we are using C++17
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

int main(int argc, char* argv[]) {
    cxxopts::Options options("QtcMiner", "A high-performance CUDA miner");

    // Define the command-line arguments we accept
    options.add_options()
        ("a,algo", "Hashing algorithm", cxxopts::value<std::string>())
        ("o,url", "Pool URL with port (e.g., host:port)", cxxopts::value<std::string>())
        ("u,user", "Username or wallet address for the pool", cxxopts::value<std::string>())
        ("p,pass", "Password for the pool", cxxopts::value<std::string>()->default_value("x"))
        ("h,help", "Print usage");

    Config config;
    
    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // Check for required arguments
        if (!result.count("algo") || !result.count("url") || !result.count("user")) {
            std::cerr << "Error: Missing required arguments: --algo, --url, --user" << std::endl;
            std::cerr << options.help() << std::endl;
            return 1;
        }

        // Store parsed values
        config.algo = result["algo"].as<std::string>();
        config.url = result["url"].as<std::string>();
        config.user = result["user"].as<std::string>();
        config.pass = result["pass"].as<std::string>();

        // Parse the URL into host and port
        if (!parse_url(config.url, config.host, config.port)) {
            return 1;
        }

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        return 1;
    }

    // --- Verification Step ---
    // Print the configuration to confirm everything was parsed correctly.
    std::cout << "--------------------------\n";
    std::cout << " QtcMiner Configuration \n";
    std::cout << "--------------------------\n";
    std::cout << "Algorithm: " << config.algo << std::endl;
    std::cout << "Pool URL:  " << config.url << std::endl;
    std::cout << "  -> Host: " << config.host << std::endl;
    std::cout << "  -> Port: " << config.port << std::endl;
    std::cout << "User:      " << config.user << std::endl;
    std::cout << "Password:  " << config.pass << std::endl;
    std::cout << "--------------------------\n\n";

    // The rest of the miner logic will go here...
    std::cout << "Configuration loaded. Starting miner...\n";


    // (Placeholder for next steps)
    // PoolConnection pool(config.host, config.port, ...);
    // pool.connect();
    // ...

    return 0;
}