/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/mining/batched_qhash_worker.hpp"
#include "ohmy/fixed_point.hpp"
#include <fmt/format.h>
#include "ohmy/log.hpp"
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>

namespace ohmy {
namespace mining {

BatchedQHashWorker::BatchedQHashWorker(
    std::unique_ptr<quantum::cuda::BatchedCudaSimulator> simulator,
    int worker_id,
    int batch_size)
    : simulator_(std::move(simulator))
    , worker_id_(worker_id)
    , batch_size_(batch_size)
    , last_hashrate_update_(std::chrono::steady_clock::now()) {
    
    stats_.batch_size = batch_size_;
    
    // quiet init
}

BatchedQHashWorker::~BatchedQHashWorker() {
    stop_work();
}

void BatchedQHashWorker::process_work(const ohmy::pool::WorkPackage& work) {
    if (is_working_.load()) {
        ohmy::log::line("Worker {} already processing work, ignoring new job", worker_id_);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.current_job_id = work.job_id;
        stats_.is_working = true;
    }

    // quiet
    
    // Start mining in a separate thread
    std::thread mining_thread(&BatchedQHashWorker::mine_job, this, work);
    mining_thread.detach();
}

void BatchedQHashWorker::stop_work() {
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
    
    ohmy::log::line("GPU #{}: shutting down...", worker_id_);
}

void BatchedQHashWorker::set_share_callback(
    std::function<void(const ohmy::pool::ShareResult&)> callback) {
    share_callback_ = std::move(callback);
}

BatchedQHashWorker::WorkerStats BatchedQHashWorker::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void BatchedQHashWorker::mine_job(const ohmy::pool::WorkPackage& work) {
    is_working_.store(true);
    should_stop_.store(false);
    
    uint32_t start_nonce = 0;
    
    // quiet
    
    // Generate unique extranonce2 for this job (NOT per batch)
    // In Stratum protocol:
    // - extranonce1 is fixed per connection (from pool)
    // - extranonce2 is miner's nonce space extension (when 32-bit nonce exhausted)
    // - nonce is the primary iteration variable (0 to 2^32-1)
    // For GPU mining, we use a FIXED extranonce2 per job and iterate nonce.
    // Only increment extranonce2 if we exhaust all 4 billion nonces (very rare).
    
    // Use worker_id to avoid collisions if running multiple GPU workers
    extranonce2_counter_ = static_cast<uint64_t>(worker_id_);
    
    std::string job_extranonce2 = format_extranonce2(
        extranonce2_counter_, 
        work.extranonce2.length()  // Match pool's extranonce2 size
    );
    
    // Create modified work package with our unique extranonce2
    ohmy::pool::WorkPackage modified_work = work;
    modified_work.extranonce2 = job_extranonce2;
    
    // quiet
    
    try {
        while (!should_stop_.load()) {
            // Process batch of nonces with SAME extranonce2
            
            auto valid_nonces = try_nonce_batch(modified_work, start_nonce, batch_size_);
            
            // Submit all valid shares with same extranonce2
            for (uint32_t nonce : valid_nonces) {
                ohmy::pool::ShareResult share;
                share.job_id = modified_work.job_id;
                share.nonce = nonce;
                share.ntime = modified_work.time;
                share.extranonce2 = job_extranonce2;  // Fixed extranonce2 for this job
                share.difficulty = (modified_work.share_difficulty > 0.0) ? modified_work.share_difficulty : 1.0;
                share.accepted = true;
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.shares_found++;
                }
                
                if (share_callback_) {
                    share_callback_(share);
                }
                
                // Compact share found line handled by Stratum submit result
            }
            
            // Update statistics
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.hashes_computed += batch_size_;
                
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_hashrate_update_);
                
                if (elapsed.count() >= 5) { // Update every 5 seconds
                    uint64_t hashes_done = stats_.hashes_computed - last_hash_count_;
                    stats_.hashrate = static_cast<double>(hashes_done) / elapsed.count();
                    
                    // Periodic hashrate reported by JobMonitor table
                    
                    last_hashrate_update_ = now;
                    last_hash_count_ = stats_.hashes_computed;
                }
            }
            
            start_nonce += batch_size_;
            
            // Prevent overflow
            if (start_nonce == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } catch (const std::exception& e) {
    ohmy::log::line("Worker {} error: {}", worker_id_, e.what());
    }
    
    is_working_.store(false);
    // quiet
}

std::vector<uint32_t> BatchedQHashWorker::try_nonce_batch(
    const ohmy::pool::WorkPackage& work,
    uint32_t start_nonce,
    int count) {
    
    std::vector<uint32_t> valid_nonces;
    
    
    try {
        // Convert time string to uint32_t for temporal fork calculations
        uint32_t nTime = static_cast<uint32_t>(std::stoul(work.time, nullptr, 16));
        
        // Generate nonces for this batch
        std::vector<uint32_t> nonces;
        nonces.reserve(count);
        for (int i = 0; i < count; ++i) {
            nonces.push_back(start_nonce + i);
        }
        
        // Generate quantum circuits for each nonce
        auto circuits = generate_circuits_batch(work, nonces, nTime);
        
        
        
        // Prepare qubits to measure (all 16 qubits for qhash)
        std::vector<int> qubits_to_measure;
        for (int i = 0; i < 16; ++i) {
            qubits_to_measure.push_back(i);
        }
        
        // Simulate all circuits in parallel on GPU
        auto all_expectations = simulator_->simulate_and_measure_batch(
            circuits, qubits_to_measure);
        
        
        
    // Debug removed for clean logs
        
        // Check each result against target
        for (size_t i = 0; i < nonces.size(); ++i) {
            const auto& expectations = all_expectations[i];
            
            // Convert expectations to fixed-point bytes (same as QHashWorker)
            std::vector<uint8_t> fixed_point_bytes;
            fixed_point_bytes.reserve(32);
            
            for (const auto& exp : expectations) {
                int16_t raw = static_cast<int16_t>(exp.raw());
                fixed_point_bytes.push_back(static_cast<uint8_t>(raw & 0xFF));
                fixed_point_bytes.push_back(static_cast<uint8_t>((raw >> 8) & 0xFF));
            }
            
            // Zero Validation (Fork #1, #2, #3)
            // Zero Validation: Count zero bytes in fixed-point representation
            // Reference: qhash.cpp lines 158-167
            // Reject hashes with TOO MANY zeros (pathological quantum states)
            int zero_count = 0;
            for (uint8_t byte : fixed_point_bytes) {
                if (byte == 0) zero_count++;
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
                continue;  // Skip this nonce (pathological quantum state)
            }
            
            // Compute final hash: SHA256(initial_hash + quantum_fixed_point)
            std::string block_header = format_block_header(work, nonces[i]);
            std::vector<uint8_t> header_bytes(block_header.begin(), block_header.end());
            std::vector<uint8_t> initial_hash = sha256_raw(header_bytes);
            
            std::vector<uint8_t> combined_data;
            combined_data.reserve(64);
            combined_data.insert(combined_data.end(), initial_hash.begin(), initial_hash.end());
            combined_data.insert(combined_data.end(), fixed_point_bytes.begin(), fixed_point_bytes.end());
            
            std::vector<uint8_t> final_hash = sha256_raw(combined_data);
            
            // Convert to hex string
            std::stringstream final_ss;
            for (uint8_t byte : final_hash) {
                final_ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
            }
            std::string hash_hex = final_ss.str();
            
            // Debug removed for clean logs
            
            // Check if meets share target (preferred), else network bits
            const std::string& target_param = !work.share_target_hex.empty() ? work.share_target_hex : work.bits;
            if (meets_target(hash_hex, target_param)) {
                valid_nonces.push_back(nonces[i]);
            }
        }
        
    // quiet summary
        
    } catch (const std::exception& e) {
        ohmy::log::line("Error in batch processing: {}", e.what());
    }
    
    return valid_nonces;
}

std::vector<quantum::QuantumCircuit> BatchedQHashWorker::generate_circuits_batch(
    const ohmy::pool::WorkPackage& work,
    const std::vector<uint32_t>& nonces,
    uint32_t nTime) {
    
    std::vector<quantum::QuantumCircuit> circuits;
    circuits.reserve(nonces.size());
    
    for (uint32_t nonce : nonces) {
        // 1. Format block header with this nonce
        std::string block_header = format_block_header(work, nonce);
        
        // 2. Compute SHA256 of block header
        std::vector<uint8_t> header_bytes(block_header.begin(), block_header.end());
        std::vector<uint8_t> hash_bytes = sha256_raw(header_bytes);
        
        // 3. Convert to hex string
        std::stringstream ss;
        for (uint8_t byte : hash_bytes) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(byte);
        }
        std::string hash_hex = ss.str();
        
        // 4. Generate circuit from hash
        circuits.push_back(generate_circuit_from_hash(hash_hex, nTime));
    }
    
    return circuits;
}

quantum::QuantumCircuit BatchedQHashWorker::generate_circuit_from_hash(
    const std::string& hash_hex,
    uint32_t nTime) {
    
    // Official qhash specification: 16 qubits, 2 layers
    constexpr int NUM_QUBITS = 16;
    constexpr int NUM_LAYERS = 2;
    quantum::QuantumCircuit circuit(NUM_QUBITS);
    
    // Temporal flag for Fork #4
    const int temporal_flag = (nTime >= 1758762000) ? 1 : 0;
    
    // Convert hex string to raw bytes
    std::vector<uint8_t> hash_bytes;
    hash_bytes.reserve(32);
    
    for (size_t i = 0; i + 1 < hash_hex.length(); i += 2) {
        std::string byte_str = hash_hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
        hash_bytes.push_back(byte);
    }
    
    while (hash_bytes.size() < 32) {
        hash_bytes.push_back(0);
    }
    
    // Extract nibbles from bytes
    std::vector<uint8_t> nibbles;
    nibbles.reserve(64);
    
    for (uint8_t byte : hash_bytes) {
        nibbles.push_back((byte >> 4) & 0xF);  // High nibble
        nibbles.push_back(byte & 0xF);          // Low nibble
    }
    
    // Apply gates layer by layer
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        for (int qubit = 0; qubit < NUM_QUBITS; ++qubit) {
            // R_Y rotation
            size_t ry_index = (2 * layer * NUM_QUBITS + qubit) % 64;
            uint8_t ry_nibble = nibbles[ry_index];
            double ry_angle = -(2.0 * ry_nibble + temporal_flag) * M_PI / 32.0;
            circuit.add_rotation(qubit, ry_angle, quantum::RotationAxis::Y);
            
            // R_Z rotation
            size_t rz_index = ((2 * layer + 1) * NUM_QUBITS + qubit) % 64;
            uint8_t rz_nibble = nibbles[rz_index];
            double rz_angle = -(2.0 * rz_nibble + temporal_flag) * M_PI / 32.0;
            circuit.add_rotation(qubit, rz_angle, quantum::RotationAxis::Z);
        }
        
        // CNOT chain
        for (int i = 0; i < NUM_QUBITS - 1; ++i) {
            circuit.add_cnot(i, i + 1);
        }
    }
    
    return circuit;
}

std::vector<uint8_t> BatchedQHashWorker::sha256_raw(const std::vector<uint8_t>& input) {
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

std::vector<uint8_t> BatchedQHashWorker::sha256d_raw(const std::vector<uint8_t>& input) {
    // Double SHA256 (Bitcoin standard)
    auto first_hash = sha256_raw(input);
    return sha256_raw(first_hash);
}

std::string BatchedQHashWorker::format_block_header(
    const ohmy::pool::WorkPackage& work,
    uint32_t nonce) {
    
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
    
    // 3. Merkle root (32 bytes) - CORRECTLY calculated from coinbase + merkle branches
    // Step 3a: Construct coinbase transaction
    std::string coinbase_tx_hex = work.coinbase1 + work.extranonce1 + work.extranonce2 + work.coinbase2;
    auto coinbase_tx_bytes = hex_to_bytes(coinbase_tx_hex);
    
    // Step 3b: Double SHA256 of coinbase transaction
    auto coinbase_hash = sha256d_raw(coinbase_tx_bytes);
    
    // Step 3c: Apply merkle branches (each branch is SHA256d(hash + branch))
    std::vector<uint8_t> merkle_root = coinbase_hash;
    for (const auto& branch_hex : work.merkle_branch) {
        auto branch_bytes = hex_to_bytes(branch_hex);
        
        // Concatenate current hash + branch
        std::vector<uint8_t> combined;
        combined.insert(combined.end(), merkle_root.begin(), merkle_root.end());
        combined.insert(combined.end(), branch_bytes.begin(), branch_bytes.end());
        
        // SHA256d of the combination
        merkle_root = sha256d_raw(combined);
    }
    
    // Step 3d: Merkle root is in internal byte order (little-endian for block header)
    std::reverse(merkle_root.begin(), merkle_root.end());
    header.insert(header.end(), merkle_root.begin(), merkle_root.end());
    
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
    // (sha256_raw expects vector<uint8_t> anyway, so this conversion is temporary)
    return std::string(header.begin(), header.end());
}

bool BatchedQHashWorker::meets_target(
    const std::string& hash,
    const std::string& target_param) {
    
    // Build 32-byte big-endian target from either:
    // - Full target hex (64 chars), or
    // - Compact bits (8 chars)
    std::vector<uint8_t> target(32, 0);
    if (target_param.size() >= 64) {
        // Parse 64-hex full target
        for (size_t i = 0; i < 64; i += 2) {
            const std::string byte_str = target_param.substr(i, 2);
            target[i/2] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
        }
    } else {
        // Interpret as compact bits (network target)
        uint32_t bits_value = std::stoul(target_param, nullptr, 16);
        uint32_t exponent = bits_value >> 24;
        uint32_t coefficient = bits_value & 0x00FFFFFF;
        if (exponent <= 3) {
            target[29] = (coefficient >> 16) & 0xFF;
            target[30] = (coefficient >> 8) & 0xFF;
            target[31] = coefficient & 0xFF;
        } else if (exponent < 32) {
            int pos = 32 - static_cast<int>(exponent);
            if (pos >= 0 && pos < 29) {
                target[pos] = (coefficient >> 16) & 0xFF;
                target[pos + 1] = (coefficient >> 8) & 0xFF;
                target[pos + 2] = coefficient & 0xFF;
            }
        }
    }
    
    // Convert hash hex string to bytes
    std::vector<uint8_t> hash_bytes(32, 0);
    for (size_t i = 0; i < 64 && i < hash.length(); i += 2) {
        std::string byte_str = hash.substr(i, 2);
        hash_bytes[i/2] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    }
    
    // Compare hash < target (byte by byte, big-endian)
    for (size_t i = 0; i < 32; i++) {
        if (hash_bytes[i] < target[i]) {
            // Found a valid share
            return true;
        } else if (hash_bytes[i] > target[i]) {
            return false;  // Hash is too high
        }
        // If equal, continue to next byte
    }
    
    // All bytes equal - hash == target, which is valid
    return true;
}

std::string BatchedQHashWorker::format_extranonce2(uint64_t counter, size_t hex_length) {
    // Convert counter to hex string with proper padding
    // hex_length is in characters (2 chars per byte)
    std::string result;
    result.reserve(hex_length);
    
    // Convert to hex with leading zeros
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(static_cast<int>(hex_length)) << counter;
    result = ss.str();
    
    // Ensure we don't exceed the requested length
    if (result.length() > hex_length) {
        result = result.substr(result.length() - hex_length);
    }
    
    return result;
}

} // namespace mining
} // namespace ohmy
