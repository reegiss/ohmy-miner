/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "ohmy/pool/work.hpp"
#include "ohmy/quantum/batched_cuda_simulator.hpp"
#include "ohmy/quantum/custatevec_backend.hpp"
#include <memory>
#include <atomic>
#include <vector>
#include <functional>
#include <cuda_runtime.h>

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
    
    // Process batch results and return valid nonces
    std::vector<uint32_t> process_batch_results(
        const ohmy::pool::WorkPackage& work,
        const std::vector<uint32_t>& nonces,
        const std::vector<quantum::QuantumCircuit>& circuits,
        uint32_t nTime,
        const std::vector<int>& qubits_to_measure,
        [[maybe_unused]] int buffer_idx,
        const double* results_ptr
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
    
    // Triple-buffering pipeline (3x buffers for H2D | Compute | D2H overlap)
    static constexpr int kNumBuffers = 3;
    
    // Host pinned memory for async transfers (3x)
    std::array<std::unique_ptr<quantum::HostPinnedBuffers>, kNumBuffers> h_pinned_buffers_;
    
    // Device buffers (3x complete buffers: states + angles + matrices + workspace)
    std::array<std::unique_ptr<quantum::GpuBatchBuffers>, kNumBuffers> d_io_buffers_;
    
    // CUDA streams and events for pipeline orchestration
    std::unique_ptr<quantum::GpuPipelineStreams> streams_;
    std::array<cudaEvent_t, kNumBuffers> h2d_events_;
    std::array<cudaEvent_t, kNumBuffers> compute_events_;
    std::array<cudaEvent_t, kNumBuffers> d2h_events_;
    
    // CPU work buffers (prepared while GPU processes previous batch)
    std::array<std::vector<quantum::QuantumCircuit>, kNumBuffers> cpu_circuits_buf_;
    std::array<std::vector<uint32_t>, kNumBuffers> cpu_nonces_buf_;
    
    // Pipeline state
    int num_qubits_{16};  // qhash uses 16 qubits
    
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
