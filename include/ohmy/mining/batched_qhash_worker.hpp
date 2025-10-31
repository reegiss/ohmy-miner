/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "ohmy/pool/work.hpp"
#include "ohmy/quantum/batched_cuda_simulator.hpp"
#include <memory>
#include <atomic>
#include <vector>
#include <functional>

namespace ohmy {
namespace mining {

/**
 * GPU-accelerated qhash mining worker using batched processing
 * 
 * Processes multiple nonces in parallel on GPU for maximum performance.
 * Each nonce generates a unique circuit (different hash → different angles).
 */
class BatchedQHashWorker : public ohmy::pool::IWorker {
public:
    BatchedQHashWorker(
        std::unique_ptr<quantum::cuda::BatchedCudaSimulator> simulator,
        int worker_id = 0,
        int batch_size = 1000
    );
    ~BatchedQHashWorker() override;

    // IWorker interface
    void process_work(const ohmy::pool::WorkPackage& work) override;
    void stop_work() override;
    void set_share_callback(std::function<void(const ohmy::pool::ShareResult&)> callback) override;

    // Worker statistics
    struct WorkerStats {
        uint64_t hashes_computed = 0;
        uint64_t shares_found = 0;
        double hashrate = 0.0;
        std::string current_job_id;
        bool is_working = false;
        int batch_size = 0;
    };

    WorkerStats get_stats() const;

private:
    // Core mining functions
    void mine_job(const ohmy::pool::WorkPackage& work);
    
    // Batch processing - process multiple nonces in parallel
    std::vector<uint32_t> try_nonce_batch(
        const ohmy::pool::WorkPackage& work,
        uint32_t start_nonce,
        int count
    );
    
    // Circuit generation for batches
    std::vector<quantum::QuantumCircuit> generate_circuits_batch(
        const ohmy::pool::WorkPackage& work,
        const std::vector<uint32_t>& nonces,
        uint32_t nTime
    );
    
    // Hash utilities
    std::vector<uint8_t> sha256_raw(const std::vector<uint8_t>& input);
    std::vector<uint8_t> sha256d_raw(const std::vector<uint8_t>& input);
    std::string format_block_header(const ohmy::pool::WorkPackage& work, uint32_t nonce);
    bool meets_target(const std::string& hash, const std::string& target_bits);
    
    // Generate qhash for single nonce (for validation)
    std::string compute_qhash_single(
        const std::string& block_header,
        uint32_t nonce,
        uint32_t nTime
    );
    
    // Generate circuit from hash (matches QHashWorker implementation)
    quantum::QuantumCircuit generate_circuit_from_hash(
        const std::string& hash_hex,
        uint32_t nTime
    );

    // Worker state
    std::unique_ptr<quantum::cuda::BatchedCudaSimulator> simulator_;
    int worker_id_;
    int batch_size_;
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_working_{false};
    std::function<void(const ohmy::pool::ShareResult&)> share_callback_;
    
    // Extranonce2 management for unique share identification
    uint64_t extranonce2_counter_ = 0;
    std::string format_extranonce2(uint64_t counter, size_t hex_length);
    
    // Statistics
    mutable std::mutex stats_mutex_;
    WorkerStats stats_;
    std::chrono::steady_clock::time_point last_hashrate_update_;
    uint64_t last_hash_count_ = 0;
};

} // namespace mining
} // namespace ohmy
