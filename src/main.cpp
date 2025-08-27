#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "cxxopts.hpp"
#include "cuda_kernels.cuh"
#include "pool_connection.h"

// A struct to hold all our configuration settings
struct Config {
    std::string algo;
    std::string url;
    std::string host;
    uint16_t port;
    std::string user;
    std::string pass;
};
// ... (cole a função parse_url aqui)
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


int main(int argc, char* argv[]) {
    // (A lógica de parsing do cxxopts permanece a mesma)
    cxxopts::Options options("QtcMiner", "A high-performance CUDA miner");
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

    std::cout << "Configuration loaded. Initializing miner...\n";

    PoolConnection pool(config.host, config.port);

    // Tenta conectar e depois fazer o handshake
    if (pool.connect()) {
        if (pool.handshake(config.user, config.pass)) {
            std::cout << "Handshake complete. Ready to receive mining jobs..." << std::endl;
            // O loop principal da mineração começará aqui no futuro.
        } else {
            std::cerr << "Failed to complete Stratum handshake. Exiting." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Failed to connect to the pool. Exiting." << std::endl;
        return 1;
    }

    return 0;
}