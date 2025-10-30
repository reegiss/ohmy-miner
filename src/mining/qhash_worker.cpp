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
        
        // 2. Convert time string to uint32_t for temporal fork calculations
        uint32_t nTime = static_cast<uint32_t>(std::stoul(work.time, nullptr, 16));
        
        // 3. Compute qhash with nTime for temporal fork support
        std::string result_hash = compute_qhash(block_header, nonce, nTime);
        
        // 3. Check if meets target
        return meets_target(result_hash, work.bits);
        
    } catch (const std::exception& e) {
        fmt::print("Error trying nonce {}: {}\n", nonce, e.what());
        return false;
    }
}

std::string QHashWorker::compute_qhash(const std::string& block_header, [[maybe_unused]] uint32_t nonce, uint32_t nTime) {
    // 1. Hash → Circuit Parameters: SHA256d(block_header) seeds quantum gate rotation angles
    std::string seed_hash = sha256d(block_header);
    
    // 2. Generate quantum circuit from hash (with temporal fork awareness)
    auto circuit = generate_circuit_from_hash(seed_hash, nTime);
    
    // 3. Simulate quantum circuit
    auto expectations = simulate_circuit(circuit);
    
    // 4. Convert expectations to deterministic fixed-point representation
    // Each Q15 expectation is 2 bytes (int16_t), 32 qubits = 64 bytes total
    std::vector<uint8_t> fixed_point_bytes;
    fixed_point_bytes.reserve(64);
    
    for (const auto& exp : expectations) {
        int16_t raw = static_cast<int16_t>(exp.raw());
        // Little-endian encoding
        fixed_point_bytes.push_back(static_cast<uint8_t>(raw & 0xFF));
        fixed_point_bytes.push_back(static_cast<uint8_t>((raw >> 8) & 0xFF));
    }
    
    // 5. Zero Validation (Fork #1, #2, #3)
    // Count zero bytes in fixed-point representation
    int zero_count = 0;
    for (uint8_t byte : fixed_point_bytes) {
        if (byte == 0) {
            zero_count++;
        }
    }
    
    // Apply temporal validation rules
    double zero_percentage = (zero_count * 100.0) / fixed_point_bytes.size();
    
    // Note: Forks apply cumulatively - later timestamps override earlier ones
    if (nTime >= 1754220531) {  // Fork #3: Jul 11, 2025 - 25% validation
        if (zero_percentage < 25.0) {
            return std::string(64, 'f');
        }
    } else if (nTime >= 1753305380) {  // Fork #2: Jun 30, 2025 - 75% validation
        if (zero_percentage < 75.0) {
            return std::string(64, 'f');
        }
    } else if (nTime >= 1753105444) {  // Fork #1: Jun 28, 2025 - 100% validation
        if (zero_percentage < 100.0) {
            return std::string(64, 'f');  // Return invalid hash (all f's)
        }
    }
    
    // 6. Convert to hex string for XOR operation
    std::stringstream ss;
    for (uint8_t byte : fixed_point_bytes) {
        ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
    }
    
    std::string quantum_output = ss.str();
    
    // 7. Final Hash: XOR quantum output with initial hash → SHA256d (Bitcoin standard)
    // XOR the quantum output with seed hash (simplified)
    std::string combined = seed_hash + quantum_output;
    
    // Final SHA256d as per Bitcoin/Qubitcoin standard
    return sha256d(combined);
}

quantum::QuantumCircuit QHashWorker::generate_circuit_from_hash(const std::string& hash_hex, uint32_t nTime) {
    // Official qhash specification: 32 qubits, 94 operations (32 R_Y + 31 CNOT + 31 R_Z)
    // Reference: super-quantum/qubitcoin qhash.cpp
    constexpr int NUM_QUBITS = 32;
    quantum::QuantumCircuit circuit(NUM_QUBITS);
    
    // Temporal flag for Fork #4 (Sep 17, 2025 16:00 UTC)
    // Changes angle parametrization after this timestamp
    const int temporal_flag = (nTime >= 1758762000) ? 1 : 0;
    
    // Extract 64 nibbles from 32-byte hash (256 bits = 64 nibbles of 4 bits each)
    std::vector<uint8_t> nibbles;
    nibbles.reserve(64);
    
    for (size_t i = 0; i < hash_hex.length() && nibbles.size() < 64; ++i) {
        char c = hash_hex[i];
        uint8_t nibble;
        if (c >= '0' && c <= '9') {
            nibble = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            nibble = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            nibble = c - 'A' + 10;
        } else {
            continue; // Skip non-hex characters
        }
        nibbles.push_back(nibble);
    }
    
    // Ensure we have exactly 64 nibbles (pad with zeros if needed)
    while (nibbles.size() < 64) {
        nibbles.push_back(0);
    }
    
    // Phase 1: Apply R_Y gates to all 32 qubits (operations 0-31)
    // Formula: angle = -(2*nibble + temporal_flag) * π/32
    for (int i = 0; i < NUM_QUBITS; ++i) {
        uint8_t nibble = nibbles[i * 2]; // Use even-indexed nibbles for R_Y
        double angle = -(2.0 * nibble + temporal_flag) * M_PI / 32.0;
        circuit.add_rotation(i, angle); // R_Y rotation
    }
    
    // Phase 2: Apply CNOT chain (operations 32-62)
    // Creates entanglement: CNOT(i, i+1) for i=0..30
    for (int i = 0; i < NUM_QUBITS - 1; ++i) {
        circuit.add_cnot(i, i + 1);
    }
    
    // Phase 3: Apply R_Z gates to qubits 1-31 (operations 63-93)
    // Uses odd-indexed nibbles, same angle formula
    for (int i = 1; i < NUM_QUBITS; ++i) {
        uint8_t nibble = nibbles[i * 2 - 1]; // Use odd-indexed nibbles for R_Z
        double angle = -(2.0 * nibble + temporal_flag) * M_PI / 32.0;
        circuit.add_rotation(i, angle); // R_Z rotation (need to add support for R_Z)
    }
    
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