/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace asio { class io_context; }

namespace ohmy {

class PoolConnection; // fwd

class Miner {
public:
    Miner(asio::io_context& io,
          PoolConnection& pool,
          int num_qubits,
          int batch_size,
          std::function<bool()> stop_requested);

    void run();

private:
    asio::io_context& io_;
    PoolConnection& pool_;
    int num_qubits_;
    int batch_size_;
    std::function<bool()> stop_requested_;

    // stats
    std::atomic<uint64_t> total_hashes_{0};
    std::atomic<uint64_t> shares_found_{0};
    std::atomic<uint64_t> shares_accepted_{0};
    std::atomic<uint64_t> shares_rejected_{0};
};

} // namespace ohmy
