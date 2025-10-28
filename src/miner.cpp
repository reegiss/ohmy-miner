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

    // Initialize simulator via factory
    auto simulator = ohmy::quantum::create_simulator(num_qubits_);
    fmt::print(fmt::fg(fmt::color::green),
               "\n✓ Quantum simulator initialized: {} qubits\n",
               num_qubits_);
    fmt::print("  Backend: {}\n", simulator->backend_name());
    fmt::print(fmt::fg(fmt::color::yellow),
               "  State vector size: {} complex numbers ({:.2f} KB)\n\n",
               simulator->get_state_size(),
               (simulator->get_state_size() * sizeof(ohmy::quantum::Complex)) / 1024.0);

    // Prepare batching capability
    std::unique_ptr<ohmy::quantum::BatchedQuantumSimulator> batched_sim;
    const std::string backend_name = simulator->backend_name();
#if defined(OHMY_WITH_CUQUANTUM)
    const bool backend_supports_batch = (backend_name == "custom") || (backend_name == "cuquantum");
#else
    const bool backend_supports_batch = (backend_name == "custom");
#endif
    const bool can_batch = (batch_size_ > 1) && backend_supports_batch;
    if (can_batch && backend_name == "custom") {
    batched_sim = std::make_unique<ohmy::quantum::BatchedQuantumSimulator>(num_qubits_, batch_size_);
        cudaStream_t stream{};
        if (cudaStreamCreate(&stream) == cudaSuccess) {
            batched_sim->set_stream(stream);
        }
        fmt::print(fmt::fg(fmt::color::green),
                   "✓ Batched simulator ready (custom): batch={} (GPU memory ~{:.2f} MB)\n\n",
                   batch_size_, batched_sim->get_memory_usage() / (1024.0 * 1024.0));
    } else if (batch_size_ > 1 && !backend_supports_batch) {
        fmt::print(fmt::fg(fmt::color::yellow),
                   "Note: Batch>1 requested but backend '{}' does not support batching yet. Proceeding without batching.\n\n",
                   backend_name);
    } else if (can_batch) {
        fmt::print(fmt::fg(fmt::color::green),
                   "✓ Batched simulator ready (cuQuantum): batch={}\n\n",
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
        uint32_t nonce_batch = static_cast<uint32_t>(std::max(1, batch_size_));
        std::string extra_nonce1 = pool_.get_extra_nonce1();

        if (can_batch) {
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

                out.circuits = ohmy::quantum::CircuitGenerator::build_from_hash_batch(out.initial_hashes, num_qubits_);
            };

            static std::string last_job_id;
            static bool has_cur = false;
            static PreparedBatch cur_cached, next_cached;
            if (!has_cur || last_job_id != job.job_id) {
                prepare_batch(cur_cached, nonce_start, job, extra_nonce1);
                has_cur = true;
                last_job_id = job.job_id;
            }

            // Compute using appropriate backend
#if defined(OHMY_WITH_CUQUANTUM)
            static std::unique_ptr<ohmy::quantum::BatchedCuQuantumSimulator> batched_cq;
            if (backend_name == "cuquantum") {
                if (!batched_cq) {
                    batched_cq = std::make_unique<ohmy::quantum::BatchedCuQuantumSimulator>(num_qubits_, batch_size_);
                    fmt::print("Using cuQuantum batched backend (float32) for batch={}\n", batch_size_);
                }
                if (!batched_cq->initialize_states()) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to initialize cuQuantum batched states\n");
                    break;
                }
                if (!batched_cq->apply_circuits_optimized(cur_cached.circuits)) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to apply cuQuantum batched circuits\n");
                    break;
                }
            } else
#endif
            {
                if (!batched_sim->initialize_states()) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to initialize batched states\n");
                    break;
                }
                if (!batched_sim->apply_circuits_optimized(cur_cached.circuits)) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to apply batched circuits\n");
                    break;
                }
            }

            // Prepare next batch on CPU while GPU runs
            prepare_batch(next_cached, cur_cached.nonce_start + static_cast<uint32_t>(batch_size_), job, extra_nonce1);

            // Measure
            std::vector<std::vector<double>> expectations_batch;
#if defined(OHMY_WITH_CUQUANTUM)
            if (backend_name == "cuquantum") {
                if (!batched_cq->measure_all(expectations_batch)) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to measure cuQuantum batched states\n");
                    break;
                }
            } else
#endif
            {
                if (!batched_sim->measure_all(expectations_batch)) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to measure batched states\n");
                    break;
                }
            }

            // Process results
            for (int i = 0; i < batch_size_ && !stop_requested_(); ++i) {
                auto qhash = ohmy::quantum::QHashProcessor::compute_qhash(cur_cached.initial_hashes[i], expectations_batch[i]);
                total_hashes_++;
                hashes_since_report++;

                double difficulty = pool_.get_difficulty();
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
        } else {
            // Single-nonce path
            for (uint32_t nonce = nonce_start; nonce < nonce_start + nonce_batch && !stop_requested_(); nonce++) {
                int en2_size = pool_.get_extra_nonce2_size();
                if (en2_size <= 0) en2_size = 4;
                uint64_t en2_mask = (en2_size >= 8) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (en2_size * 8)) - 1ULL);
                uint64_t en2_val = static_cast<uint64_t>(nonce) & en2_mask;
                std::stringstream extra_nonce2_ss;
                extra_nonce2_ss << std::hex << std::setw(en2_size * 2) << std::setfill('0') << en2_val;
                std::string extra_nonce2 = extra_nonce2_ss.str();

                auto block_header_bytes = ohmy::crypto::build_block_header(job, nonce, extra_nonce1, extra_nonce2);
                std::array<uint8_t, 32> initial_hash{};
                SHA256(block_header_bytes.data(), block_header_bytes.size(), initial_hash.data());

                auto circuit = ohmy::quantum::CircuitGenerator::build_from_hash(initial_hash, num_qubits_);
                if (!simulator->initialize_state()) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to initialize quantum state\n");
                    return;
                }
                if (!simulator->apply_circuit(circuit)) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to apply quantum circuit\n");
                    return;
                }
                std::vector<double> expectations;
                if (!simulator->measure(expectations)) {
                    fmt::print(fmt::fg(fmt::color::red), "Error: Failed to measure quantum state\n");
                    return;
                }

                auto qhash = ohmy::quantum::QHashProcessor::compute_qhash(initial_hash, expectations);
                total_hashes_++;
                hashes_since_report++;

                double difficulty = pool_.get_difficulty();
                if (ohmy::crypto::check_difficulty(qhash, difficulty)) {
                    shares_found_++;
                    fmt::print(fmt::fg(fmt::color::green) | fmt::emphasis::bold,
                               "\n✓ SHARE FOUND! Nonce: {}, Hash: {}...\n",
                               nonce, ohmy::crypto::bytes_to_hex(qhash).substr(0, 16));
                    std::stringstream nonce_hex;
                    nonce_hex << std::hex << std::setw(8) << std::setfill('0') << nonce;
                    std::string ntime_hex = job.ntime;
                    if (pool_.submit_share(job.job_id, nonce_hex.str(), ntime_hex, extra_nonce2)) {
                        shares_accepted_++;
                        fmt::print(fmt::fg(fmt::color::green), "  Share submitted successfully\n\n");
                    } else {
                        shares_rejected_++;
                        fmt::print(fmt::fg(fmt::color::red), "  Share submission failed\n\n");
                    }
                }
            }
        }

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
