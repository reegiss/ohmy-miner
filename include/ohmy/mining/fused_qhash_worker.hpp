/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "ohmy/pool/work.hpp"
#include "ohmy/quantum/cuda_types.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ohmy {
namespace mining {

/**
 * High-performance qhash worker using the monolithic fused CUDA kernel.
 * 1 Block = 1 Nonce. Processes batches of nonces per launch.
 */
class FusedQHashWorker : public ohmy::pool::IWorker {
public:
    FusedQHashWorker(int worker_id = 0, int batch_size = 4096, int block_size = 256);
    ~FusedQHashWorker() override;

    // IWorker interface
    void process_work(const ohmy::pool::WorkPackage& work) override;
    void stop_work() override;
    void set_share_callback(std::function<void(const ohmy::pool::ShareResult&)> callback) override;

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
    // Core mining loop
    void mine_job(const ohmy::pool::WorkPackage& work);

    // Build 76-byte block header template (without nonce)
    std::vector<uint8_t> build_header_template76(const ohmy::pool::WorkPackage& work) const;

    // Utilities
    std::string format_extranonce2(uint64_t counter, size_t hex_length) const;

    // State
    int worker_id_;
    int batch_size_;
    int block_size_;
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> is_working_{false};
    std::function<void(const ohmy::pool::ShareResult&)> share_callback_;

    // Stats
    mutable std::mutex stats_mutex_;
    WorkerStats stats_;
    std::chrono::steady_clock::time_point last_hashrate_update_;
    uint64_t last_hash_count_ = 0;

    // Extranonce2
    uint64_t extranonce2_counter_ = 0;
};

} // namespace mining
} // namespace ohmy
