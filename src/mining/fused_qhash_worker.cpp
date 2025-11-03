/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/mining/fused_qhash_worker.hpp"
#include "ohmy/log.hpp"
#include "ohmy/crypto/difficulty.hpp"
#include <fmt/format.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <thread>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cuComplex.h>

// Extern kernel launcher (implemented in fused_qhash_kernel.cu)
extern "C" void launch_fused_qhash_kernel(
    cuDoubleComplex* d_state_vectors,
    const uint8_t* d_header_template,
    uint32_t nTime,
    uint64_t nonce_start,
    const uint8_t* d_full_target,
    uint32_t* d_result_buffer,
    uint32_t* d_result_count,
    int batch_size,
    int block_size,
    cudaStream_t stream
);

namespace ohmy {
namespace mining {

FusedQHashWorker::FusedQHashWorker(int worker_id, int batch_size, int block_size)
    : worker_id_(worker_id)
    , batch_size_(batch_size)
    , block_size_(block_size)
    , last_hashrate_update_(std::chrono::steady_clock::now())
{
    stats_.batch_size = batch_size_;
    ohmy::log::line("GPU #{}: FusedQHashWorker initialized (batch={}, block={})",
                    worker_id_, batch_size_, block_size_);
}

FusedQHashWorker::~FusedQHashWorker() {
    stop_work();
}

void FusedQHashWorker::process_work(const ohmy::pool::WorkPackage& work) {
    if (is_working_.load()) {
        ohmy::log::line("Worker {} already processing work, ignoring new job", worker_id_);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.current_job_id = work.job_id;
        stats_.is_working = true;
    }

    std::thread mining_thread(&FusedQHashWorker::mine_job, this, work);
    mining_thread.detach();
}

void FusedQHashWorker::stop_work() {
    should_stop_.store(true);
    while (is_working_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.is_working = false;
        stats_.current_job_id.clear();
    }
    ohmy::log::line("GPU #{}: Fused worker stopped", worker_id_);
}

void FusedQHashWorker::set_share_callback(
    std::function<void(const ohmy::pool::ShareResult&)> callback) {
    share_callback_ = std::move(callback);
}

FusedQHashWorker::WorkerStats FusedQHashWorker::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

static inline std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

std::vector<uint8_t> FusedQHashWorker::build_header_template76(const ohmy::pool::WorkPackage& work) const {
    std::vector<uint8_t> header;
    header.reserve(76);

    // 1. Version (4 bytes, little-endian)
    auto version_bytes = hex_to_bytes(work.version);
    std::reverse(version_bytes.begin(), version_bytes.end());
    header.insert(header.end(), version_bytes.begin(), version_bytes.end());

    // 2. Previous block hash (32 bytes, reversed for little-endian)
    auto prev_hash_bytes = hex_to_bytes(work.previous_hash);
    std::reverse(prev_hash_bytes.begin(), prev_hash_bytes.end());
    header.insert(header.end(), prev_hash_bytes.begin(), prev_hash_bytes.end());

    // 3. Merkle root (computed from coinbase + branches)
    std::string coinbase_tx_hex = work.coinbase1 + work.extranonce1 + work.extranonce2 + work.coinbase2;
    auto coinbase_tx_bytes = hex_to_bytes(coinbase_tx_hex);

    // Double SHA256 of coinbase
    auto sha256 = [](const std::vector<uint8_t>& in) {
        std::vector<uint8_t> out(SHA256_DIGEST_LENGTH);
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) throw std::runtime_error("EVP_MD_CTX_new failed");
        if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
            EVP_DigestUpdate(ctx, in.data(), in.size()) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("SHA256 init/update failed");
        }
        unsigned int outlen = 0;
        if (EVP_DigestFinal_ex(ctx, out.data(), &outlen) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("SHA256 failed");
        }
        EVP_MD_CTX_free(ctx);
        return out;
    };
    auto sha256d = [&](const std::vector<uint8_t>& in) {
        return sha256(sha256(in));
    };

    auto merkle_root = sha256d(coinbase_tx_bytes);
    for (const auto& branch_hex : work.merkle_branch) {
        auto branch_bytes = hex_to_bytes(branch_hex);
        std::vector<uint8_t> combined;
        combined.reserve(merkle_root.size() + branch_bytes.size());
        combined.insert(combined.end(), merkle_root.begin(), merkle_root.end());
        combined.insert(combined.end(), branch_bytes.begin(), branch_bytes.end());
        merkle_root = sha256d(combined);
    }
    // Merkle root into header as little-endian
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

    // (nonce will be appended by the kernel)
    if (header.size() != 76) {
        throw std::runtime_error("Header template size mismatch");
    }
    return header;
}

std::string FusedQHashWorker::format_extranonce2(uint64_t counter, size_t hex_length) const {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(static_cast<int>(hex_length)) << counter;
    auto s = ss.str();
    if (s.size() > hex_length) {
        s = s.substr(s.size() - hex_length);
    }
    return s;
}

void FusedQHashWorker::mine_job(const ohmy::pool::WorkPackage& work) {
    is_working_.store(true);
    should_stop_.store(false);

    try {
        // Prepare job context
        uint32_t nTime = static_cast<uint32_t>(std::stoul(work.time, nullptr, 16));
        // Generate worker-unique extranonce2
        extranonce2_counter_ = static_cast<uint64_t>(worker_id_);
        std::string job_extranonce2 = format_extranonce2(extranonce2_counter_, work.extranonce2.length());

        // Local copy of work to include our extranonce2
        ohmy::pool::WorkPackage job = work;
        job.extranonce2 = job_extranonce2;

        // Build 76-byte header template
        auto header76 = build_header_template76(job);

        // Allocate device buffers
        const size_t state_elems_per_nonce = (1ULL << 16); // 2^16 amplitudes
        const size_t total_state_elems = state_elems_per_nonce * static_cast<size_t>(batch_size_);
        ohmy::quantum::cuda::DeviceMemory<cuDoubleComplex> d_states(total_state_elems);
        ohmy::quantum::cuda::DeviceMemory<uint8_t> d_header(76);
        ohmy::quantum::cuda::DeviceMemory<uint32_t> d_results(4096);
        ohmy::quantum::cuda::DeviceMemory<uint32_t> d_count(1);

        // Copy header template
        CUDA_CHECK(cudaMemcpy(d_header.get(), header76.data(), 76, cudaMemcpyHostToDevice));

        // Loop over nonce space in batches
        uint64_t nonce_start = 0;
        // Prepare 32-byte big-endian target for share difficulty if available, else use network bits
        std::vector<uint8_t> target_bytes(32, 0);
        if (!job.share_target_hex.empty() && job.share_target_hex.size() >= 64) {
            // Parse 64-hex into 32 bytes big-endian
            for (size_t i = 0; i < 64; i += 2) {
                target_bytes[i/2] = static_cast<uint8_t>(std::stoul(job.share_target_hex.substr(i, 2), nullptr, 16));
            }
        } else {
            // Fallback to network bits
            uint32_t bits_compact = static_cast<uint32_t>(std::stoul(job.bits, nullptr, 16));
            auto v = ohmy::crypto::decode_compact_target(bits_compact);
            if (v.size() == 32) target_bytes = v;
        }
        ohmy::quantum::cuda::DeviceMemory<uint8_t> d_target(32);
        CUDA_CHECK(cudaMemcpy(d_target.get(), target_bytes.data(), 32, cudaMemcpyHostToDevice));

        while (!should_stop_.load()) {
            // Reset result count
            d_count.memset(0);

            // Launch kernel
            launch_fused_qhash_kernel(
                d_states.get(),
                d_header.get(),
                nTime,
                nonce_start,
                d_target.get(),
                d_results.get(),
                d_count.get(),
                batch_size_,
                block_size_,
                nullptr /* default stream */);

            // Synchronize to check results
            CUDA_CHECK(cudaDeviceSynchronize());

            // Read back results
            uint32_t h_count = 0;
            CUDA_CHECK(cudaMemcpy(&h_count, d_count.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (h_count > 0) {
                std::vector<uint32_t> h_nonces(h_count);
                CUDA_CHECK(cudaMemcpy(h_nonces.data(), d_results.get(), h_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

                for (uint32_t nonce : h_nonces) {
                    ohmy::pool::ShareResult share;
                    share.job_id = job.job_id;
                    share.nonce = nonce;
                    share.ntime = job.time;
                    share.extranonce2 = job_extranonce2;
                    share.difficulty = (job.share_difficulty > 0.0) ? job.share_difficulty : 1.0;
                    share.accepted = true; // submission happens upstream
                    if (share_callback_) {
                        share_callback_(share);
                    }
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        stats_.shares_found++;
                    }
                }
            }

            // Update stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.hashes_computed += batch_size_;
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_hashrate_update_);
                if (elapsed.count() >= 5) {
                    uint64_t hashes_done = stats_.hashes_computed - last_hash_count_;
                    stats_.hashrate = static_cast<double>(hashes_done) / elapsed.count();
                    last_hashrate_update_ = now;
                    last_hash_count_ = stats_.hashes_computed;
                }
            }

            // Advance
            nonce_start += static_cast<uint64_t>(batch_size_);
            if (nonce_start == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    } catch (const std::exception& e) {
        ohmy::log::line("Worker {} error: {}", worker_id_, e.what());
    }

    is_working_.store(false);
}

} // namespace mining
} // namespace ohmy
