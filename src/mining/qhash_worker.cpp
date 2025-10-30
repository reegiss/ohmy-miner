/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/mining/qhash_worker.hpp"
#include <fmt/format.h>
#include <fmt/color.h>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace ohmy {
namespace mining {

QHashWorker::QHashWorker(std::unique_ptr<quantum::IQuantumSimulator> simulator, int worker_id)
    : simulator_(std::move(simulator))
    , worker_id_(worker_id)
    , last_hashrate_update_(std::chrono::steady_clock::now()) {
    
    fmt::print("QHash worker {} initialized with {} backend\n", 
               worker_id_, simulator_->backend_name());
}

QHashWorker::~QHashWorker() {
    stop_work();
}

void QHashWorker::process_work(const ohmy::pool::WorkPackage& work) {
    if (is_working_.load()) {
        fmt::print("Worker {} already processing work, ignoring new job\n", worker_id_);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.current_job_id = work.job_id;
        stats_.is_working = true;
    }

    fmt::print("Worker {} starting work on job: {}\n", worker_id_, work.job_id);
    
    // Start mining in a separate thread to avoid blocking the dispatcher
    std::thread mining_thread(&QHashWorker::mine_job, this, work);
    mining_thread.detach();
}

void QHashWorker::stop_work() {
    should_stop_.store(true);
    
    // Wait for current work to stop
    while (is_working_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.is_working = false;
        stats_.current_job_id.clear();
    }
    
    fmt::print("Worker {} stopped\n", worker_id_);
}

void QHashWorker::set_share_callback(std::function<void(const ohmy::pool::ShareResult&)> callback) {
    share_callback_ = std::move(callback);
}

QHashWorker::WorkerStats QHashWorker::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void QHashWorker::mine_job(const ohmy::pool::WorkPackage& work) {
    is_working_.store(true);
    should_stop_.store(false);
    
    uint32_t nonce = 0;
    [[maybe_unused]] auto start_time = std::chrono::steady_clock::now();
    
    try {
        while (!should_stop_.load()) {
            // Try current nonce
            if (try_nonce(work, nonce)) {
                // Found a share!
                ohmy::pool::ShareResult share;
                share.job_id = work.job_id;
                share.nonce = nonce;
                share.ntime = work.time;  // Use job's timestamp
                share.extranonce2 = "00000000";  // TODO: Implement proper extranonce2
                share.difficulty = 1.0; // TODO: Calculate actual difficulty
                share.accepted = true;
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.shares_found++;
                }
                
                if (share_callback_) {
                    share_callback_(share);
                }
                
                fmt::print(fg(fmt::color::green), 
                          "Worker {} found share! Nonce: 0x{:08x}\n", 
                          worker_id_, nonce);
            }
            
            // Update statistics periodically
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.hashes_computed++;
                
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_hashrate_update_);
                
                if (elapsed.count() >= 5) { // Update every 5 seconds
                    uint64_t hashes_done = stats_.hashes_computed - last_hash_count_;
                    stats_.hashrate = static_cast<double>(hashes_done) / elapsed.count();
                    
                    last_hashrate_update_ = now;
                    last_hash_count_ = stats_.hashes_computed;
                }
            }
            
            nonce++;
            
            // Prevent overflow and give other workers a chance
            if (nonce == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } catch (const std::exception& e) {
        fmt::print("Worker {} error: {}\n", worker_id_, e.what());
    }
    
    is_working_.store(false);
    fmt::print("Worker {} finished job: {}\n", worker_id_, work.job_id);
}

bool QHashWorker::try_nonce(const ohmy::pool::WorkPackage& work, uint32_t nonce) {
    try {
        // 1. Format block header with nonce
        std::string block_header = format_block_header(work, nonce);
        
        // 2. Compute qhash
        std::string result_hash = compute_qhash(block_header, nonce);
        
        // 3. Check if meets target
        return meets_target(result_hash, work.bits);
        
    } catch (const std::exception& e) {
        fmt::print("Error trying nonce {}: {}\n", nonce, e.what());
        return false;
    }
}

std::string QHashWorker::compute_qhash(const std::string& block_header, [[maybe_unused]] uint32_t nonce) {
    // 1. Hash → Circuit Parameters: SHA256d(block_header) seeds quantum gate rotation angles
    std::string seed_hash = sha256d(block_header);
    
    // 2. Generate quantum circuit from hash
    auto circuit = generate_circuit_from_hash(seed_hash);
    
    // 3. Simulate quantum circuit
    auto expectations = simulate_circuit(circuit);
    
    // 4. Convert expectations to deterministic fixed-point representation
    std::stringstream ss;
    for (const auto& exp : expectations) {
        ss << std::hex << std::setfill('0') << std::setw(8) << exp.raw();
    }
    
    // 5. Final Hash: XOR quantum output with initial hash → SHA256d (Bitcoin standard)
    std::string quantum_output = ss.str();
    
    // XOR the quantum output with seed hash (simplified)
    std::string combined = seed_hash + quantum_output;
    
    // Final SHA256d as per Bitcoin/Qubitcoin standard
    return sha256d(combined);
}

quantum::QuantumCircuit QHashWorker::generate_circuit_from_hash(const std::string& hash_hex) {
    // Create a simple quantum circuit based on hash
    // For now, create a 4-qubit circuit
    quantum::QuantumCircuit circuit(4);
    
    // Use hash bytes to generate rotation angles
    for (size_t i = 0; i < std::min(hash_hex.length() / 2, size_t(4)); ++i) {
        std::string byte_str = hash_hex.substr(i * 2, 2);
        uint8_t byte_val = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
        
        // Convert byte to angle [0, 2π]
        double angle = (byte_val / 255.0) * 2.0 * M_PI;
        circuit.add_rotation(i, angle);
    }
    
    // Add some CNOT gates for entanglement
    circuit.add_cnot(0, 1);
    circuit.add_cnot(1, 2);
    circuit.add_cnot(2, 3);
    
    return circuit;
}

std::vector<ohmy::Q15> QHashWorker::simulate_circuit(const quantum::QuantumCircuit& circuit) {
    // Simulate the circuit
    simulator_->simulate(circuit);
    
    // Measure expectations for all qubits
    std::vector<int> all_qubits;
    for (int i = 0; i < circuit.num_qubits(); ++i) {
        all_qubits.push_back(i);
    }
    
    return simulator_->measure_expectations(all_qubits);
}

std::string QHashWorker::sha256d(const std::string& input) {
    // Convert string to bytes
    std::vector<uint8_t> input_bytes(input.begin(), input.end());
    
    // Compute double SHA256
    auto result = sha256d_raw(input_bytes);
    
    // Convert to hex string
    std::stringstream ss;
    for (uint8_t byte : result) {
        ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
    }
    
    return ss.str();
}

std::vector<uint8_t> QHashWorker::sha256d_raw(const std::vector<uint8_t>& input) {
    // First SHA256
    std::vector<uint8_t> first_hash(SHA256_DIGEST_LENGTH);
    
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) {
        throw std::runtime_error("Failed to create EVP_MD_CTX for first hash");
    }
    
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, input.data(), input.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to compute first SHA256");
    }
    
    unsigned int hash_len;
    if (EVP_DigestFinal_ex(ctx, first_hash.data(), &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize first SHA256");
    }
    
    // Second SHA256 (SHA256 of the first hash)
    std::vector<uint8_t> second_hash(SHA256_DIGEST_LENGTH);
    
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, first_hash.data(), hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to compute second SHA256");
    }
    
    if (EVP_DigestFinal_ex(ctx, second_hash.data(), &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize second SHA256");
    }
    
    EVP_MD_CTX_free(ctx);
    
    return second_hash;
}

std::string QHashWorker::format_block_header(const ohmy::pool::WorkPackage& work, uint32_t nonce) {
    // Bitcoin-style block header format (80 bytes total)
    // Structure: version(4) + prevhash(32) + merkleroot(32) + time(4) + bits(4) + nonce(4)
    
    std::stringstream ss;
    
    // Version (4 bytes, little-endian)
    ss << work.version;
    
    // Previous block hash (32 bytes, little-endian)
    ss << work.previous_hash;
    
    // Merkle root (32 bytes) - constructed from coinbase + merkle branch
    // For now, use a simplified approach
    std::string merkle_root = work.coinbase1 + work.coinbase2;
    if (merkle_root.length() > 64) merkle_root = merkle_root.substr(0, 64);
    while (merkle_root.length() < 64) merkle_root += "0";
    ss << merkle_root;
    
    // Timestamp (4 bytes, little-endian)
    ss << work.time;
    
    // Target/difficulty bits (4 bytes, little-endian)
    ss << work.bits;
    
    // Nonce (4 bytes, little-endian)
    ss << std::hex << std::setfill('0') << std::setw(8) << nonce;
    
    return ss.str();
}

bool QHashWorker::meets_target(const std::string& hash, const std::string& target_bits) {
    // Convert compact bits format to full target
    // Format: 0x1d00ffff -> exponent=0x1d, coefficient=0x00ffff
    uint32_t bits_value = std::stoul(target_bits, nullptr, 16);
    
    // Extract exponent (most significant byte) and coefficient
    uint32_t exponent = bits_value >> 24;
    uint32_t coefficient = bits_value & 0x00FFFFFF;
    
    // Calculate target as coefficient * 256^(exponent-3)
    // Target is a 256-bit number
    std::vector<uint8_t> target(32, 0);
    
    if (exponent <= 3) {
        // Coefficient fits in first 3 bytes
        target[29] = (coefficient >> 16) & 0xFF;
        target[30] = (coefficient >> 8) & 0xFF;
        target[31] = coefficient & 0xFF;
    } else if (exponent < 32) {
        // Place coefficient at position determined by exponent
        int pos = 32 - exponent;
        if (pos >= 0 && pos < 29) {
            target[pos] = (coefficient >> 16) & 0xFF;
            target[pos + 1] = (coefficient >> 8) & 0xFF;
            target[pos + 2] = coefficient & 0xFF;
        }
    }
    
    // Convert hash hex string to bytes (big-endian)
    std::vector<uint8_t> hash_bytes(32, 0);
    for (size_t i = 0; i < 64 && i < hash.length(); i += 2) {
        std::string byte_str = hash.substr(i, 2);
        hash_bytes[i/2] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    }
    
    // Compare hash < target (byte by byte, big-endian)
    bool meets = false;
    for (size_t i = 0; i < 32; i++) {
        if (hash_bytes[i] < target[i]) {
            meets = true;
            break;
        } else if (hash_bytes[i] > target[i]) {
            meets = false;
            break;
        }
    }
    
    if (meets) {
        // Count leading zeros for display
        int leading_zeros = 0;
        for (char c : hash) {
            if (c == '0') leading_zeros++;
            else break;
        }
        
        fmt::print(fg(fmt::color::green), 
                  "✓ Valid share! Leading zeros: {} | Bits: {}\n",
                  leading_zeros, target_bits);
        fmt::print("  Hash: {}...\n", hash.substr(0, 32));
    }
    
    return meets;
}

} // namespace mining
} // namespace ohmy