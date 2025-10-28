/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "gpu_info.hpp"
#include "pool_connection.hpp"
#include "quantum_kernel.cuh"
#include "quantum/simulator.hpp"
#include "batched_quantum.cuh"
#include "circuit_generator.hpp"
#include "fixed_point.hpp"
#include <fmt/core.h>
#include <fmt/color.h>
#include <cxxopts.hpp>
#include <cuda_runtime.h>
#include <asio.hpp>
#include <openssl/sha.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace ohmy;
using namespace ohmy::quantum;

// Global flag for graceful shutdown
std::atomic<bool> should_exit{false};

// Global statistics
std::atomic<uint64_t> total_hashes{0};
std::atomic<uint64_t> shares_found{0};
std::atomic<uint64_t> shares_accepted{0};
std::atomic<uint64_t> shares_rejected{0};

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

std::string bytes_to_hex(const std::array<uint8_t, 32>& bytes) {
    std::stringstream ss;
    for (size_t i = 0; i < 32; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
    }
    return ss.str();
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::strtol(byte_str.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

static std::array<uint8_t, 32> sha256d_bytes(const std::vector<uint8_t>& data) {
    std::array<uint8_t, 32> h1;
    SHA256(data.data(), data.size(), h1.data());
    std::array<uint8_t, 32> h2;
    SHA256(h1.data(), h1.size(), h2.data());
    return h2;
}

static std::array<uint8_t, 32> compute_merkle_root_le(const MiningJob& job,
                                                      const std::string& extra_nonce1,
                                                      const std::string& extra_nonce2) {
    // Build coinbase transaction from parts and compute its double SHA256
    std::string coinbase_hex = job.coinbase1 + extra_nonce1 + extra_nonce2 + job.coinbase2;
    auto coinbase_bytes = hex_to_bytes(coinbase_hex);
    std::array<uint8_t, 32> cur = sha256d_bytes(coinbase_bytes); // big-endian bytes

    // Apply merkle branches: concatenate in little-endian order and double SHA256
    if (!job.merkle_branch.empty()) {
        for (const auto& branch_hex : job.merkle_branch) {
            auto branch_bytes = hex_to_bytes(branch_hex);
            std::vector<uint8_t> concat;
            concat.reserve(64);
            // Use little-endian for internal merkle concatenation
            concat.insert(concat.end(), cur.rbegin(), cur.rend());
            concat.insert(concat.end(), branch_bytes.rbegin(), branch_bytes.rend());
            cur = sha256d_bytes(concat); // still big-endian
        }
    }

    // Header expects merkle root in little-endian
    std::array<uint8_t, 32> merkle_le{};
    std::reverse_copy(cur.begin(), cur.end(), merkle_le.begin());
    return merkle_le;
}

std::vector<uint8_t> build_block_header(const MiningJob& job, 
                                        uint32_t nonce,
                                        const std::string& extra_nonce1,
                                        const std::string& extra_nonce2) {
    // QTC block header format (similar to Bitcoin):
    // version (4 bytes) + prev_hash (32 bytes) + merkle_root (32 bytes) + 
    // ntime (4 bytes) + nbits (4 bytes) + nonce (4 bytes) = 80 bytes
    
    std::vector<uint8_t> header;
    
    // Version (little-endian in header)
    auto version_bytes = hex_to_bytes(job.version);
    std::reverse(version_bytes.begin(), version_bytes.end());
    header.insert(header.end(), version_bytes.begin(), version_bytes.end());
    
    // Previous block hash (reversed for little-endian)
    auto prev_hash_bytes = hex_to_bytes(job.prev_hash);
    std::reverse(prev_hash_bytes.begin(), prev_hash_bytes.end());
    header.insert(header.end(), prev_hash_bytes.begin(), prev_hash_bytes.end());
    
    // Merkle root (little-endian) from coinbase + merkle branches
    auto merkle_root_le = compute_merkle_root_le(job, extra_nonce1, extra_nonce2);
    header.insert(header.end(), merkle_root_le.begin(), merkle_root_le.end());
    
    // ntime (little-endian)
    auto ntime_bytes = hex_to_bytes(job.ntime);
    std::reverse(ntime_bytes.begin(), ntime_bytes.end());
    header.insert(header.end(), ntime_bytes.begin(), ntime_bytes.end());
    
    // nbits (little-endian)
    auto nbits_bytes = hex_to_bytes(job.nbits);
    std::reverse(nbits_bytes.begin(), nbits_bytes.end());
    header.insert(header.end(), nbits_bytes.begin(), nbits_bytes.end());
    
    // nonce (4 bytes, little-endian)
    header.push_back(nonce & 0xFF);
    header.push_back((nonce >> 8) & 0xFF);
    header.push_back((nonce >> 16) & 0xFF);
    header.push_back((nonce >> 24) & 0xFF);
    
    return header;
}

// Convert 32-byte big-endian buffer to little-endian
static std::array<uint8_t, 32> be32_to_le32(const std::array<uint8_t, 32>& be) {
    std::array<uint8_t, 32> le{};
    std::reverse_copy(be.begin(), be.end(), le.begin());
    return le;
}

// Compute target = target1 / difficulty, where target1 = 0x00000000FFFF0000... (Bitcoin-style base)
static std::array<uint8_t, 32> compute_target_from_difficulty(double difficulty) {
    // Represent big integers as base-2^32 little-endian limbs
    // target1 limbs: limb[6] = 0xFFFF0000, limb[7] = 0x00000000, others 0
    std::array<uint32_t, 8> limbs{};
    limbs.fill(0);
    limbs[6] = 0xFFFF0000u;
    limbs[7] = 0x00000000u;

    // Multiply by scale to preserve fractional difficulty precision
    const uint32_t SCALE = 1000000000u; // 1e9
    std::array<uint32_t, 9> num{}; // up to 288-bit after scaling
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        unsigned __int128 prod = static_cast<unsigned __int128>(limbs[i]) * SCALE + carry;
        num[i] = static_cast<uint32_t>(prod & 0xFFFFFFFFu);
        carry = static_cast<uint64_t>(prod >> 32);
    }
    num[8] = static_cast<uint32_t>(carry); // may be 0

    // Divisor = round(difficulty * SCALE), clamp to at least 1
    long double dl = static_cast<long double>(difficulty);
    uint64_t divisor = static_cast<uint64_t>(ceill(dl * static_cast<long double>(SCALE)));
    if (divisor == 0) divisor = 1;

    // Long division by 64-bit divisor across base-2^32 digits
    std::array<uint32_t, 9> quo{};
    unsigned __int128 rem = 0;
    for (int i = 8; i >= 0; --i) {
        unsigned __int128 cur = (rem << 32) + num[i];
        uint64_t q = static_cast<uint64_t>(cur / divisor);
        rem = cur % divisor;
        quo[i] = static_cast<uint32_t>(q & 0xFFFFFFFFu);
    }

    // Convert quotient (lower 8 limbs) to 32 bytes little-endian
    std::array<uint8_t, 32> target{};
    for (int i = 0; i < 8; ++i) {
        uint32_t w = quo[i];
        target[i * 4 + 0] = static_cast<uint8_t>(w & 0xFF);
        target[i * 4 + 1] = static_cast<uint8_t>((w >> 8) & 0xFF);
        target[i * 4 + 2] = static_cast<uint8_t>((w >> 16) & 0xFF);
        target[i * 4 + 3] = static_cast<uint8_t>((w >> 24) & 0xFF);
    }
    return target;
}

static bool le256_less_equal(const std::array<uint8_t, 32>& a, const std::array<uint8_t, 32>& b) {
    // Compare as 256-bit little-endian integers
    for (int i = 31; i >= 0; --i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return true; // equal
}

bool check_difficulty(const std::array<uint8_t, 32>& hash_be, double difficulty) {
    auto target_le = compute_target_from_difficulty(difficulty);
    auto hash_le = be32_to_le32(hash_be);
    return le256_less_equal(hash_le, target_le);
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
            ("t,threads", "Number of mining threads", 
                cxxopts::value<int>()->default_value("1"))
            ("d,device", "CUDA device ID", 
                cxxopts::value<int>()->default_value("0"))
            ("b,batch", "Batch size (>=1). Uses batched simulator when >1 (custom backend only)",
                cxxopts::value<int>()->default_value("1"))
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
    auto threads = result["threads"].as<int>();
    auto device = result["device"].as<int>();
    int batch_size = result["batch"].as<int>();
    if (batch_size < 1) batch_size = 1;

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

        // Auto-tune batch size if user did not specify --batch
        if (result.count("batch") == 0) {
            // For 16 qubits, each state has 2^16 complex amps; Complex is cuDoubleComplex (16B)
            const size_t state_bytes = (1ULL << 16) * sizeof(Complex);
            // Use only a fraction of free memory to stay safe (25%)
            const double mem_budget = gpus[device].free_memory * 0.25;
            int recommended = static_cast<int>(mem_budget / static_cast<double>(state_bytes));
            if (recommended < 1) recommended = 1;
            if (recommended > 512) recommended = 512; // cap to avoid huge batches by default
            batch_size = recommended;
        }

        // Display configuration
        fmt::print(fg(fmt::color::magenta) | fmt::emphasis::bold, 
            "=== Mining Configuration ===\n");
        fmt::print("Algorithm:    {}\n", algo);
        fmt::print("Pool URL:     {}\n", url);
        fmt::print("Username:     {}\n", user);
        fmt::print("Password:     {}\n", pass);
        fmt::print("Threads:      {}\n", threads);
        fmt::print("CUDA Device:  {}\n", device);
        fmt::print("Batch size:   {}\n", batch_size);
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

        // Create ASIO io_context for async operations
        asio::io_context io_context;

        // Create pool connection
        fmt::print("\n");
        PoolConnection pool(io_context, url, user, pass);

        // Set up callbacks
        pool.set_job_callback([](const MiningJob& job) {
            fmt::print(fg(fmt::color::magenta),
                "New job received: {} (Clean: {})\n",
                job.job_id, job.clean_jobs ? "Yes" : "No");
        });

        pool.set_difficulty_callback([](double difficulty) {
            fmt::print(fg(fmt::color::yellow),
                "Difficulty updated: {}\n", difficulty);
        });

        pool.set_error_callback([](const std::string& error) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Pool error: {}\n", error);
        });

        // Connect to pool
        if (!pool.connect()) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: Failed to connect to pool\n");
            return 1;
        }

        // Subscribe to mining
        if (!pool.subscribe()) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: Failed to subscribe to pool\n");
            return 1;
        }

        // Authorize with pool
        if (!pool.authorize()) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Fatal: Failed to authorize with pool\n");
            return 1;
        }

        // Start async receive loop
        pool.start_receive_loop();

        // Setup signal handlers for graceful shutdown
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

        // Main mining loop (placeholder)
        fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
            "=== Mining Started ===\n\n");

        // Initialize quantum simulator (factory chooses best available backend)
        const int num_qubits = 16;  // QTC uses 16 qubits (not 8!)
        auto simulator = ohmy::quantum::create_simulator(num_qubits);
        
        fmt::print(fg(fmt::color::green), 
            "✓ Quantum simulator initialized: {} qubits (QTC standard)\n", num_qubits);
        fmt::print("  Backend: {}\n", simulator->backend_name());
        fmt::print(fg(fmt::color::yellow),
            "  State vector size: {} complex numbers ({:.2f} KB)\n\n",
            simulator->get_state_size(),
            (simulator->get_state_size() * sizeof(Complex)) / 1024.0);

        // Prepare batched simulator if requested and backend is custom
        std::unique_ptr<BatchedQuantumSimulator> batched_sim;
        const bool can_batch = (batch_size > 1) && (std::string(simulator->backend_name()) == "custom");
        if (can_batch) {
            batched_sim = std::make_unique<BatchedQuantumSimulator>(num_qubits, batch_size);
            // Optional: use a dedicated stream for batched compute
            cudaStream_t stream{};
            if (cudaStreamCreate(&stream) == cudaSuccess) {
                batched_sim->set_stream(stream);
            }
            fmt::print(fg(fmt::color::green),
                "✓ Batched simulator ready: batch={} (GPU memory ~{:.2f} MB)\n\n",
                batch_size, batched_sim->get_memory_usage() / (1024.0 * 1024.0));
        } else if (batch_size > 1) {
            fmt::print(fg(fmt::color::yellow),
                "Note: Batch>1 requested but backend '{}' does not support batching yet. Proceeding without batching.\n\n",
                simulator->backend_name());
        }

        // Mining statistics tracking
        auto start_time = std::chrono::steady_clock::now();
        auto last_report_time = start_time;
        uint64_t hashes_since_report = 0;
        
        while (!should_exit && pool.is_connected()) {
            auto job = pool.get_current_job();
            
            if (!job.is_valid()) {
                // No job yet, wait a bit
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // Choose a work quantum (how many nonces to process this outer loop)
            uint32_t nonce_start = total_hashes.load();
            uint32_t nonce_batch = static_cast<uint32_t>(std::max(1, batch_size));
            
            // Get extra_nonce1 from pool
            std::string extra_nonce1 = pool.get_extra_nonce1();
            
            if (can_batch) {
                // Batched path with duplo-buffer simples (CPU prepara próximo lote enquanto GPU calcula o atual)
                struct PreparedBatch {
                    uint32_t nonce_start{0};
                    std::vector<QuantumCircuit> circuits;
                    std::vector<std::array<uint8_t, 32>> initial_hashes;
                    std::vector<std::string> extra_nonce2_vec;
                } cur, next;

                auto prepare_batch = [&](PreparedBatch& out, uint32_t start_nonce, const MiningJob& jb, const std::string& en1){
                    out.nonce_start = start_nonce;
                    out.circuits.assign(batch_size, QuantumCircuit(num_qubits));
                    out.initial_hashes.assign(batch_size, {});
                    out.extra_nonce2_vec.assign(batch_size, {});
                    int en2_size_local = pool.get_extra_nonce2_size();
                    if (en2_size_local <= 0) en2_size_local = 4;
                    uint64_t en2_mask_local = (en2_size_local >= 8) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (en2_size_local * 8)) - 1ULL);
                    for (int i = 0; i < batch_size; ++i) {
                        uint32_t n = start_nonce + static_cast<uint32_t>(i);
                        uint64_t en2_val = static_cast<uint64_t>(n) & en2_mask_local;
                        std::stringstream ss;
                        ss << std::hex << std::setw(en2_size_local * 2) << std::setfill('0') << en2_val;
                        out.extra_nonce2_vec[i] = ss.str();
                        auto header = build_block_header(jb, n, en1, out.extra_nonce2_vec[i]);
                        SHA256(header.data(), header.size(), out.initial_hashes[i].data());
                        out.circuits[i] = CircuitGenerator::build_from_hash(out.initial_hashes[i], num_qubits);
                    }
                };

                static std::string last_job_id;
                static bool has_cur = false;
                static PreparedBatch cur_cached, next_cached;

                if (!has_cur || last_job_id != job.job_id) {
                    // Reset pipeline for new job
                    prepare_batch(cur_cached, nonce_start, job, extra_nonce1);
                    has_cur = true;
                    last_job_id = job.job_id;
                }

                // Launch compute for current batch
                if (!batched_sim->initialize_states()) {
                    fmt::print(fg(fmt::color::red), "Error: Failed to initialize batched states\n");
                    should_exit = true;
                    break;
                }
                if (!batched_sim->apply_circuits_optimized(cur_cached.circuits)) {
                    fmt::print(fg(fmt::color::red), "Error: Failed to apply batched circuits\n");
                    should_exit = true;
                    break;
                }

                // Enquanto GPU calcula, prepare o próximo lote no CPU
                prepare_batch(next_cached, cur_cached.nonce_start + static_cast<uint32_t>(batch_size), job, extra_nonce1);

                // Medir (sincroniza no final) e copiar resultados
                std::vector<std::vector<double>> expectations_batch;
                if (!batched_sim->measure_all(expectations_batch)) {
                    fmt::print(fg(fmt::color::red), "Error: Failed to measure batched states\n");
                    should_exit = true;
                    break;
                }

                // Process results
                for (int i = 0; i < batch_size && !should_exit; ++i) {
                    auto qhash = QHashProcessor::compute_qhash(cur_cached.initial_hashes[i], expectations_batch[i]);

                    // (removed verbose first-hash debug dump)

                    total_hashes++;
                    hashes_since_report++;

                    double difficulty = pool.get_difficulty();
                    if (check_difficulty(qhash, difficulty)) {
                        shares_found++;
                        uint32_t n = cur_cached.nonce_start + static_cast<uint32_t>(i);
                        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
                            "\n✓ SHARE FOUND! Nonce: {}, Hash: {}...\n",
                            n, bytes_to_hex(qhash).substr(0, 16));

                        std::stringstream nonce_hex;
                        nonce_hex << std::hex << std::setw(8) << std::setfill('0') << n;
                        std::string ntime_hex = job.ntime;

                        if (pool.submit_share(job.job_id, nonce_hex.str(), ntime_hex, cur_cached.extra_nonce2_vec[i])) {
                            shares_accepted++;
                            fmt::print(fg(fmt::color::green), "  Share submitted successfully\n\n");
                        } else {
                            shares_rejected++;
                            fmt::print(fg(fmt::color::red), "  Share submission failed\n\n");
                        }
                    }
                }

                // Avança o pipeline
                cur_cached = std::move(next_cached);
            } else {
                // Single-nonce path
                for (uint32_t nonce = nonce_start; nonce < nonce_start + nonce_batch && !should_exit; nonce++) {
                    // Generate extra_nonce2 with correct width from pool
                    int en2_size = pool.get_extra_nonce2_size();
                    if (en2_size <= 0) en2_size = 4; // default to 4 bytes if unknown
                    uint64_t en2_mask = (en2_size >= 8) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (en2_size * 8)) - 1ULL);
                    uint64_t en2_val = static_cast<uint64_t>(nonce) & en2_mask;
                    std::stringstream extra_nonce2_ss;
                    extra_nonce2_ss << std::hex << std::setw(en2_size * 2) << std::setfill('0') 
                                   << en2_val;
                    std::string extra_nonce2 = extra_nonce2_ss.str();
                    
                    // Step 1: Build real block header from job data
                    auto block_header_bytes = build_block_header(job, nonce, extra_nonce1, extra_nonce2);
                    
                    // Step 2: SHA256 hash of block header
                    std::array<uint8_t, 32> initial_hash;
                    SHA256(block_header_bytes.data(), block_header_bytes.size(), initial_hash.data());
                    
                    // Step 2: Generate quantum circuit from hash
                    auto circuit = CircuitGenerator::build_from_hash(initial_hash, num_qubits);
                    
                    // Step 3: Simulate quantum circuit
                    if (!simulator->initialize_state()) {
                        fmt::print(fg(fmt::color::red), "Error: Failed to initialize quantum state\n");
                        should_exit = true;
                        break;
                    }
                    
                    if (!simulator->apply_circuit(circuit)) {
                        fmt::print(fg(fmt::color::red), "Error: Failed to apply quantum circuit\n");
                        should_exit = true;
                        break;
                    }
                    
                    std::vector<double> expectations;
                    if (!simulator->measure(expectations)) {
                        fmt::print(fg(fmt::color::red), "Error: Failed to measure quantum state\n");
                        should_exit = true;
                        break;
                    }
                    
                    // Step 4: Compute final qhash (fixed-point → SHA256)
                    auto qhash = QHashProcessor::compute_qhash(initial_hash, expectations);
                    
                    // (removed verbose first-hash debug dump)
                    
                    // Update hash counter
                    total_hashes++;
                    hashes_since_report++;
                    
                    // Periodic status/debug as before handled below
                    
                    // Step 5: Check against difficulty
                    double difficulty = pool.get_difficulty();
                    if (check_difficulty(qhash, difficulty)) {
                        shares_found++;
                        
                        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
                            "\n✓ SHARE FOUND! Nonce: {}, Hash: {}...\n",
                            nonce, bytes_to_hex(qhash).substr(0, 16));
                        
                        // Convert nonce to hex string for submission (8 hex chars = 4 bytes)
                        std::stringstream nonce_hex;
                        nonce_hex << std::hex << std::setw(8) << std::setfill('0') << nonce;
                        
                        // Use job-provided ntime to avoid server rejections
                        std::string ntime_hex = job.ntime;
                        
                        // Submit share to pool (extra_nonce2 already generated above)
                        if (pool.submit_share(job.job_id, nonce_hex.str(), ntime_hex, extra_nonce2)) {
                            shares_accepted++;
                            fmt::print(fg(fmt::color::green), "  Share submitted successfully\n\n");
                        } else {
                            shares_rejected++;
                            fmt::print(fg(fmt::color::red), "  Share submission failed\n\n");
                        }
                    }
                }
            }

            // Periodic status report (every 5 seconds)
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time);

            if (elapsed.count() >= 5) {
                double hashrate = hashes_since_report / static_cast<double>(elapsed.count());
                auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                double cur_diff = pool.get_difficulty();
                double expected_sec_per_share = (4294967296.0 * cur_diff) / std::max(hashrate, 1e-9);
                int eta_h = static_cast<int>(expected_sec_per_share / 3600.0);
                int eta_m = static_cast<int>((expected_sec_per_share - eta_h * 3600) / 60);
                int eta_s = static_cast<int>(expected_sec_per_share) % 60;

                fmt::print(fg(fmt::color::cyan),
                    "[{:02d}:{:02d}:{:02d}] Hashrate: {:.2f} H/s | Total: {} | Shares: {} accepted, {} rejected | ETA@diff {:.2f}: ~{:02d}h{:02d}m{:02d}s\n",
                    total_elapsed.count() / 3600,
                    (total_elapsed.count() / 60) % 60,
                    total_elapsed.count() % 60,
                    hashrate,
                    total_hashes.load(),
                    shares_accepted.load(),
                    shares_rejected.load(),
                    cur_diff,
                    eta_h, eta_m, eta_s);

                last_report_time = now;
                hashes_since_report = 0;
            }
    }

        fmt::print("\n");
        fmt::print(fg(fmt::color::yellow), "Shutting down...\n");
        
        // Cleanup
        pool.disconnect();
        io_context.stop();
        if (io_thread.joinable()) {
            io_thread.join();
        }

        // Destroy stream if created
        if (batched_sim) {
            if (batched_sim->stream() != nullptr) {
                cudaStreamDestroy(batched_sim->stream());
            }
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
