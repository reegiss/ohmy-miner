/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "ohmy/pool/work.hpp"
#include "ohmy/quantum/simulator.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include <functional>

namespace ohmy {
namespace mining {

/**
 * qhash mining worker that implements the quantum proof-of-work algorithm
 */
class QHashWorker : public ohmy::pool::IWorker {
public:
    QHashWorker(std::unique_ptr<quantum::IQuantumSimulator> simulator, int worker_id = 0);
    ~QHashWorker() override;

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
    };

    WorkerStats get_stats() const;

private:
    // Core mining functions
    void mine_job(const ohmy::pool::WorkPackage& work);
    bool try_nonce(const ohmy::pool::WorkPackage& work, uint32_t nonce);
    std::string compute_qhash(const std::string& block_header, uint32_t nonce);
    
    // Quantum circuit generation
    quantum::QuantumCircuit generate_circuit_from_hash(const std::string& hash_hex);
    std::vector<ohmy::Q15> simulate_circuit(const quantum::QuantumCircuit& circuit);
    
    // Hash utilities
    std::string sha256d(const std::string& input);  // Double SHA256 like Bitcoin
    std::vector<uint8_t> sha256d_raw(const std::vector<uint8_t>& input);
    std::string format_block_header(const ohmy::pool::WorkPackage& work, uint32_t nonce);
    bool meets_target(const std::string& hash, const std::string& target_bits);

    // Worker state
    std::unique_ptr<quantum::IQuantumSimulator> simulator_;
    int worker_id_;
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_working_{false};
    std::function<void(const ohmy::pool::ShareResult&)> share_callback_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    WorkerStats stats_;
    std::chrono::steady_clock::time_point last_hashrate_update_;
    uint64_t last_hash_count_ = 0;
};

} // namespace mining
} // namespace ohmy