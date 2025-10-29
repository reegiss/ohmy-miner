/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "miner.hpp"

#include "pool_connection.hpp"
#include "crypto_utils.hpp"
#include "circuit_generator.hpp"
#include "fixed_point.hpp"
#include "batched_quantum.cuh"
#include "quantum/simulator.hpp"
#if defined(OHMY_WITH_CUQUANTUM)
#include "quantum/custatevec_batched.hpp"
#endif

#include <asio.hpp>
#include <fmt/color.h>
#include <fmt/core.h>
#include <openssl/sha.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace ohmy;

Miner::Miner(asio::io_context& io,
             PoolConnection& pool,
             int num_qubits,
             int batch_size,
             std::function<bool()> stop_requested)
    : io_(io)
    , pool_(pool)
    , num_qubits_(num_qubits)
    , batch_size_(std::max(1, batch_size))
    , stop_requested_(std::move(stop_requested)) {}

void Miner::run() {
    using namespace std::chrono;

    // Initialize batched simulator directly (always used in production)
    std::unique_ptr<ohmy::quantum::BatchedQuantumSimulator> batched_sim;
    const bool use_custom_batched = (batch_size_ > 1);
    
    if (use_custom_batched) {
        // Custom batched implementation
        fmt::print(fmt::fg(fmt::color::yellow),
                   "⚠ Switching to custom backend for batching (cuQuantum can't batch effectively)\n");
        
        batched_sim = std::make_unique<ohmy::quantum::BatchedQuantumSimulator>(num_qubits_, batch_size_);
        cudaStream_t stream{};
        if (cudaStreamCreate(&stream) == cudaSuccess) {
            batched_sim->set_stream(stream);
        }
        fmt::print(fmt::fg(fmt::color::green),
                   "✓ Batched custom simulator ready: batch={} nonces/iteration\n",
                   batch_size_);
        fmt::print("  GPU memory: ~{:.2f} MB ({} states × {} amplitudes × 16 bytes)\n",
                   batched_sim->get_memory_usage() / (1024.0 * 1024.0),
                   batch_size_,
                   (1ULL << num_qubits_));
        fmt::print(fmt::fg(fmt::color::cyan),
                   "  Expected throughput: {}× vs single-nonce mode\n\n",
                   batch_size_);
    }

    // Tracking
    auto start_time = steady_clock::now();
    auto last_report_time = start_time;
    uint64_t hashes_since_report = 0;

    while (!stop_requested_() && pool_.is_connected()) {
        auto job = pool_.get_current_job();
        if (!job.is_valid()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Work quantum (outer loop chunk)
        uint32_t nonce_start = static_cast<uint32_t>(total_hashes_.load());
        std::string extra_nonce1 = pool_.get_extra_nonce1();

        // Always use batched path (use_custom_batched always true)
        struct PreparedBatch {
                uint32_t nonce_start{0};
                std::vector<ohmy::quantum::QuantumCircuit> circuits;
                std::vector<std::array<uint8_t, 32>> initial_hashes;
                std::vector<std::string> extra_nonce2_vec;
            } cur, next;

            auto prepare_batch = [&](PreparedBatch& out, uint32_t start_nonce, const MiningJob& jb, const std::string& en1){
                out.nonce_start = start_nonce;
                out.initial_hashes.assign(batch_size_, {});
                out.extra_nonce2_vec.assign(batch_size_, {});
                int en2_size_local = pool_.get_extra_nonce2_size();
                if (en2_size_local <= 0) en2_size_local = 4;
                uint64_t en2_mask_local = (en2_size_local >= 8) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (en2_size_local * 8)) - 1ULL);

                    // CRITICAL: Extract nTime from job for temporal fork handling
                    // nTime is hex string (8 chars = 4 bytes) in network byte order
                    uint32_t nTime_value = 0;
                    if (jb.ntime.size() == 8) {
                        nTime_value = std::stoul(jb.ntime, nullptr, 16);
                    }
                
                        // DEBUG: Log nTime and temporal fork status (first batch only)
                        static bool logged_ntime = false;
                        if (!logged_ntime) {
                            logged_ntime = true;
                            const uint32_t FORK_TIMESTAMP = 1758762000;
                            fmt::print(fmt::fg(fmt::color::cyan),
                                       "\n[INFO] Block nTime: {} (0x{:08x})\n",
                                       nTime_value, nTime_value);
                            fmt::print("[INFO] Fork timestamp: {} (0x{:08x})\n",
                                       FORK_TIMESTAMP, FORK_TIMESTAMP);
                            if (nTime_value >= FORK_TIMESTAMP) {
                                fmt::print(fmt::fg(fmt::color::yellow) | fmt::emphasis::bold,
                                           "[INFO] ⚠️  POST-FORK MODE: Using -(2*nibble+1)*π/32 angle formula\n\n");
                            } else {
                                fmt::print(fmt::fg(fmt::color::green),
                                           "[INFO] ✓ PRE-FORK MODE: Using -nibble*π/16 angle formula\n\n");
                            }
                        }

                // Build headers and SHA256 hashes for entire batch (OpenMP outside file scope)
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < batch_size_; ++i) {
                    uint32_t n = start_nonce + static_cast<uint32_t>(i);
                    uint64_t en2_val = static_cast<uint64_t>(n) & en2_mask_local;
                    std::stringstream ss;
                    ss << std::hex << std::setw(en2_size_local * 2) << std::setfill('0') << en2_val;
                    out.extra_nonce2_vec[i] = ss.str();
                    auto header = ohmy::crypto::build_block_header(jb, n, en1, out.extra_nonce2_vec[i]);
                    SHA256(header.data(), header.size(), out.initial_hashes[i].data());
                }

                    // Build circuits with temporal fork handling
                    out.circuits = ohmy::quantum::CircuitGenerator::build_from_hash_batch(out.initial_hashes, num_qubits_, nTime_value);
            };

            static std::string last_job_id;
            static bool has_cur = false;
            static PreparedBatch cur_cached, next_cached;
            if (!has_cur || last_job_id != job.job_id) {
                prepare_batch(cur_cached, nonce_start, job, extra_nonce1);
                has_cur = true;
                last_job_id = job.job_id;
            }

            // Compute using custom batched backend (true parallel processing)
            if (!batched_sim->initialize_states()) {
                fmt::print(fmt::fg(fmt::color::red), "Error: Failed to initialize batched states\n");
                break;
            }
            
            // DEBUG: Check if states initialized correctly
            if (total_hashes_.load() == 0) {
                fmt::print(fmt::fg(fmt::color::cyan),
                           "[DEBUG] Initialized {} states\n", batch_size_);
            }
            
            // Use monolithic kernel for maximum performance (100-500× faster)
            if (!batched_sim->apply_circuits_monolithic(cur_cached.circuits)) {
                fmt::print(fmt::fg(fmt::color::red), "Error: Failed to apply batched circuits\n");
                break;
            }

            // Prepare next batch on CPU while GPU runs
            prepare_batch(next_cached, cur_cached.nonce_start + static_cast<uint32_t>(batch_size_), job, extra_nonce1);

            // Measure using custom batched backend
            std::vector<std::vector<double>> expectations_batch;
            if (!batched_sim->measure_all(expectations_batch)) {
                fmt::print(fmt::fg(fmt::color::red), "Error: Failed to measure batched states\n");
                break;
            }

            // Process results
            for (int i = 0; i < batch_size_ && !stop_requested_(); ++i) {
                    // CRITICAL: Extract nTime for qhash validation with temporal fork
                    uint32_t nTime_value = 0;
                    if (job.ntime.size() == 8) {
                        nTime_value = std::stoul(job.ntime, nullptr, 16);
                    }
                
                    auto qhash = ohmy::quantum::QHashProcessor::compute_qhash(cur_cached.initial_hashes[i], expectations_batch[i], nTime_value);
                total_hashes_++;
                hashes_since_report++;

                double difficulty = pool_.get_difficulty();
                
                // DEBUG: Log first hash to verify calculation
                if (total_hashes_.load() == 1) {
                    fmt::print(fmt::fg(fmt::color::yellow),
                               "\n[DEBUG] First qhash: {}\n",
                               ohmy::crypto::bytes_to_hex(qhash));
                    fmt::print("[DEBUG] Difficulty: {}\n",
                               difficulty);
                    fmt::print("[DEBUG] Expectations[0-3]: {:.6f} {:.6f} {:.6f} {:.6f}\n\n",
                               expectations_batch[i][0], expectations_batch[i][1],
                               expectations_batch[i][2], expectations_batch[i][3]);
                }
                
                if (ohmy::crypto::check_difficulty(qhash, difficulty)) {
                    shares_found_++;
                    uint32_t n = cur_cached.nonce_start + static_cast<uint32_t>(i);
                    fmt::print(fmt::fg(fmt::color::green) | fmt::emphasis::bold,
                               "\n✓ SHARE FOUND! Nonce: {}, Hash: {}...\n",
                               n, ohmy::crypto::bytes_to_hex(qhash).substr(0, 16));

                    std::stringstream nonce_hex;
                    nonce_hex << std::hex << std::setw(8) << std::setfill('0') << n;
                    std::string ntime_hex = job.ntime;

                    if (pool_.submit_share(job.job_id, nonce_hex.str(), ntime_hex, cur_cached.extra_nonce2_vec[i])) {
                        shares_accepted_++;
                        fmt::print(fmt::fg(fmt::color::green), "  Share submitted successfully\n\n");
                    } else {
                        shares_rejected_++;
                        fmt::print(fmt::fg(fmt::color::red), "  Share submission failed\n\n");
                    }
                }
            }

            cur_cached = std::move(next_cached);

        // Periodic status report (every 5 seconds)
        auto now = steady_clock::now();
        auto elapsed = duration_cast<std::chrono::seconds>(now - last_report_time);
        if (elapsed.count() >= 5) {
            double hashrate = hashes_since_report / static_cast<double>(elapsed.count());
            auto total_elapsed = duration_cast<std::chrono::seconds>(now - start_time);
            double cur_diff = pool_.get_difficulty();
            double expected_sec_per_share = (4294967296.0 * cur_diff) / std::max(hashrate, 1e-9);
            int eta_h = static_cast<int>(expected_sec_per_share / 3600.0);
            int eta_m = static_cast<int>((expected_sec_per_share - eta_h * 3600) / 60);
            int eta_s = static_cast<int>(expected_sec_per_share) % 60;

            fmt::print(fmt::fg(fmt::color::cyan),
                       "[{:02d}:{:02d}:{:02d}] Hashrate: {:.2f} H/s | Total: {} | Shares: {} accepted, {} rejected | ETA@diff {:.2f}: ~{:02d}h{:02d}m{:02d}s\n",
                       total_elapsed.count() / 3600,
                       (total_elapsed.count() / 60) % 60,
                       total_elapsed.count() % 60,
                       hashrate,
                       total_hashes_.load(),
                       shares_accepted_.load(),
                       shares_rejected_.load(),
                       cur_diff,
                       eta_h, eta_m, eta_s);

            last_report_time = now;
            hashes_since_report = 0;
        }
    }
}
