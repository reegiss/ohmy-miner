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
    // 1. Initial Hash: SHA256 of block header (NOT double SHA256!)
    // Bug #6 fix: Official uses single SHA256 for input, we need to match that
    std::vector<uint8_t> header_bytes(block_header.begin(), block_header.end());
    std::vector<uint8_t> initial_hash = sha256_raw(header_bytes);
    
    // Convert to hex for circuit generation (temporary, will refactor)
    std::stringstream ss;
    for (uint8_t byte : initial_hash) {
        ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
    }
    std::string hash_hex = ss.str();
    
    // 2. Generate quantum circuit from hash (with temporal fork awareness)
    auto circuit = generate_circuit_from_hash(hash_hex, nTime);
    
    // 3. Simulate quantum circuit
    auto expectations = simulate_circuit(circuit);
    
    // 4. Convert expectations to deterministic fixed-point representation
    // Each Q15 expectation is 2 bytes (int16_t), 16 qubits = 32 bytes total
    std::vector<uint8_t> fixed_point_bytes;
    fixed_point_bytes.reserve(32);
    
    for (const auto& exp : expectations) {
        int16_t raw = static_cast<int16_t>(exp.raw());
        // Little-endian encoding
        fixed_point_bytes.push_back(static_cast<uint8_t>(raw & 0xFF));
        fixed_point_bytes.push_back(static_cast<uint8_t>((raw >> 8) & 0xFF));
    }
    
    // 5. Zero Validation (Fork #1, #2, #3)
    // Reference: qhash.cpp lines 158-167
    // Reject hashes with TOO MANY zeros (pathological quantum states)
    int zero_count = 0;
    for (uint8_t byte : fixed_point_bytes) {
        if (byte == 0) {
            zero_count++;
        }
    }
    
    // Apply temporal fork rules (progressive restrictions)
    bool is_invalid = false;
    
    // Fork #1 (Jun 28, 2025): Reject if ALL 32 bytes are zero
    if (nTime >= 1753105444 && zero_count == 32) {
        is_invalid = true;
    }
    
    // Fork #2 (Jun 30, 2025): Reject if >= 75% (24/32) bytes are zero
    if (nTime >= 1753305380 && zero_count >= 24) {
        is_invalid = true;
    }
    
    // Fork #3 (Jul 11, 2025): Reject if >= 25% (8/32) bytes are zero
    if (nTime >= 1754220531 && zero_count >= 8) {
        is_invalid = true;
    }
    
    if (is_invalid) {
        return std::string(64, 'f');  // Return invalid hash (all f's)
    }
    
    // 6. Final Hash: SHA256(initial_hash + quantum_fixed_point) - Bug #6 fix
    // Official: Write initial_hash bytes, then fixed_point bytes, then single SHA256
    std::vector<uint8_t> combined_data;
    combined_data.reserve(32 + 32);  // 32 bytes hash + 32 bytes quantum
    
    // Append initial hash bytes
    combined_data.insert(combined_data.end(), initial_hash.begin(), initial_hash.end());
    
    // Append quantum fixed-point bytes
    combined_data.insert(combined_data.end(), fixed_point_bytes.begin(), fixed_point_bytes.end());
    
    // Single SHA256 (NOT double!)
    std::vector<uint8_t> final_hash = sha256_raw(combined_data);
    
    // Convert final hash to hex string
    std::stringstream final_ss;
    for (uint8_t byte : final_hash) {
        final_ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
    }
    
    return final_ss.str();
}

quantum::QuantumCircuit QHashWorker::generate_circuit_from_hash(const std::string& hash_hex, uint32_t nTime) {
    // Official qhash specification: 16 qubits, 2 layers
    // Per layer: R_Y[i] → R_Z[i] for each qubit, then CNOT chain
    // Reference: super-quantum/qubitcoin qhash.cpp lines 61-85
    constexpr int NUM_QUBITS = 16;
    constexpr int NUM_LAYERS = 2;
    quantum::QuantumCircuit circuit(NUM_QUBITS);
    
    // Temporal flag for Fork #4 (Sep 17, 2025 16:00 UTC)
    const int temporal_flag = (nTime >= 1758762000) ? 1 : 0;
    
    // Convert hex string to raw bytes
    std::vector<uint8_t> hash_bytes;
    hash_bytes.reserve(32);
    
    for (size_t i = 0; i + 1 < hash_hex.length(); i += 2) {
        std::string byte_str = hash_hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
        hash_bytes.push_back(byte);
    }
    
    // Ensure we have 32 bytes
    while (hash_bytes.size() < 32) {
        hash_bytes.push_back(0);
    }
    
    // Extract nibbles from bytes using bit operations (Bug #3 fix)
    // Official: splitNibbles() in qhash.h:39-49
    std::vector<uint8_t> nibbles;
    nibbles.reserve(64);
    
    for (uint8_t byte : hash_bytes) {
        nibbles.push_back((byte >> 4) & 0xF);  // High nibble
        nibbles.push_back(byte & 0xF);          // Low nibble
    }
    
    // Apply gates layer by layer
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        // Bug #1 & #2 fix: Interleave R_Y and R_Z for each qubit
        // Official order: R_Y[i] → R_Z[i] for i in [0..15], then CNOT chain
        for (int qubit = 0; qubit < NUM_QUBITS; ++qubit) {
            // Bug #4 fix: Use correct nibble index formulas
            // R_Y uses: (2 * layer * NUM_QUBITS + qubit) % 64
            size_t ry_index = (2 * layer * NUM_QUBITS + qubit) % 64;
            uint8_t ry_nibble = nibbles[ry_index];
            double ry_angle = -(2.0 * ry_nibble + temporal_flag) * M_PI / 32.0;
            circuit.add_rotation(qubit, ry_angle, quantum::RotationAxis::Y);
            
            // R_Z uses: ((2 * layer + 1) * NUM_QUBITS + qubit) % 64
            size_t rz_index = ((2 * layer + 1) * NUM_QUBITS + qubit) % 64;
            uint8_t rz_nibble = nibbles[rz_index];
            double rz_angle = -(2.0 * rz_nibble + temporal_flag) * M_PI / 32.0;
            circuit.add_rotation(qubit, rz_angle, quantum::RotationAxis::Z);
        }
        
        // Apply CNOT chain: CNOT(i, i+1) for i in [0..14]
        for (int i = 0; i < NUM_QUBITS - 1; ++i) {
            circuit.add_cnot(i, i + 1);
        }
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

std::vector<uint8_t> QHashWorker::sha256_raw(const std::vector<uint8_t>& input) {
    // Single SHA256
    std::vector<uint8_t> hash(SHA256_DIGEST_LENGTH);
    
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx == nullptr) {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }
    
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, input.data(), input.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to compute SHA256");
    }
    
    unsigned int hash_len;
    if (EVP_DigestFinal_ex(ctx, hash.data(), &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize SHA256");
    }
    
    EVP_MD_CTX_free(ctx);
    
    return hash;
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
    // Block header must be BINARY (80 bytes), not hex string!
    // Reference: Bitcoin/Qubitcoin block header format (little-endian)
    std::vector<uint8_t> header;
    header.reserve(80);
    
    // Helper: Convert hex string to bytes
    auto hex_to_bytes = [](const std::string& hex) -> std::vector<uint8_t> {
        std::vector<uint8_t> bytes;
        for (size_t i = 0; i < hex.length(); i += 2) {
            std::string byte_str = hex.substr(i, 2);
            uint8_t byte = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
            bytes.push_back(byte);
        }
        return bytes;
    };
    
    // 1. Version (4 bytes, little-endian)
    auto version_bytes = hex_to_bytes(work.version);
    std::reverse(version_bytes.begin(), version_bytes.end());
    header.insert(header.end(), version_bytes.begin(), version_bytes.end());
    
    // 2. Previous block hash (32 bytes, reversed for little-endian)
    auto prev_hash_bytes = hex_to_bytes(work.previous_hash);
    std::reverse(prev_hash_bytes.begin(), prev_hash_bytes.end());
    header.insert(header.end(), prev_hash_bytes.begin(), prev_hash_bytes.end());
    
    // 3. Merkle root (32 bytes, reversed for little-endian)
    std::string merkle_root = work.coinbase1 + work.coinbase2;
    if (merkle_root.length() > 64) merkle_root = merkle_root.substr(0, 64);
    while (merkle_root.length() < 64) merkle_root += "0";
    auto merkle_bytes = hex_to_bytes(merkle_root);
    std::reverse(merkle_bytes.begin(), merkle_bytes.end());
    header.insert(header.end(), merkle_bytes.begin(), merkle_bytes.end());
    
    // 4. Timestamp (4 bytes, little-endian)
    auto time_bytes = hex_to_bytes(work.time);
    std::reverse(time_bytes.begin(), time_bytes.end());
    header.insert(header.end(), time_bytes.begin(), time_bytes.end());
    
    // 5. Bits (4 bytes, little-endian)
    auto bits_bytes = hex_to_bytes(work.bits);
    std::reverse(bits_bytes.begin(), bits_bytes.end());
    header.insert(header.end(), bits_bytes.begin(), bits_bytes.end());
    
    // 6. Nonce (4 bytes, little-endian)
    header.push_back(nonce & 0xFF);
    header.push_back((nonce >> 8) & 0xFF);
    header.push_back((nonce >> 16) & 0xFF);
    header.push_back((nonce >> 24) & 0xFF);
    
    // Convert back to string for compatibility with existing code
    return std::string(header.begin(), header.end());
}

bool QHashWorker::meets_target(const std::string& hash, const std::string& target_param) {
    // Build 32-byte big-endian target from either full 64-hex or compact 8-hex bits
    std::vector<uint8_t> target(32, 0);
    if (target_param.size() >= 64) {
        for (size_t i = 0; i < 64; i += 2) {
            target[i/2] = static_cast<uint8_t>(std::stoul(target_param.substr(i, 2), nullptr, 16));
        }
    } else {
        uint32_t bits_value = std::stoul(target_param, nullptr, 16);
        uint32_t exponent = bits_value >> 24;
        uint32_t coefficient = bits_value & 0x00FFFFFF;
        if (exponent <= 3) {
            target[29] = static_cast<uint8_t>((coefficient >> 16) & 0xFF);
            target[30] = static_cast<uint8_t>((coefficient >> 8) & 0xFF);
            target[31] = static_cast<uint8_t>(coefficient & 0xFF);
        } else if (exponent <= 32) {
            int pos = 32 - static_cast<int>(exponent);
            if (pos >= 0 && pos + 2 < 32) {
                target[static_cast<size_t>(pos)]     = static_cast<uint8_t>((coefficient >> 16) & 0xFF);
                target[static_cast<size_t>(pos + 1)] = static_cast<uint8_t>((coefficient >> 8) & 0xFF);
                target[static_cast<size_t>(pos + 2)] = static_cast<uint8_t>(coefficient & 0xFF);
            }
        }
    }

    // Convert hash hex string to bytes (big-endian)
    std::vector<uint8_t> hash_bytes(32, 0);
    for (size_t i = 0; i < 64 && i < hash.length(); i += 2) {
        hash_bytes[i/2] = static_cast<uint8_t>(std::stoul(hash.substr(i, 2), nullptr, 16));
    }

    // Compare hash < target (byte by byte, big-endian)
    for (size_t i = 0; i < 32; i++) {
        if (hash_bytes[i] < target[i]) {
            int leading_zeros = 0;
            for (char c : hash) {
                if (c == '0') leading_zeros++; else break;
            }
            fmt::print(fg(fmt::color::green), "✓ Valid share! Leading zeros: {}\n", leading_zeros);
            fmt::print("  Hash: {}...\n", hash.substr(0, 32));
            return true;
        } else if (hash_bytes[i] > target[i]) {
            return false;
        }
    }
    return true; // equal
}

} // namespace mining
} // namespace ohmy