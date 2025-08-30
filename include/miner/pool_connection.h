// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef MINER_NET_POOL_CONNECTION_H_
#define MINER_NET_POOL_CONNECTION_H_

#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <functional>
#include <memory>
#include <string>
#include <atomic>
#include <cstdint>
#include <random>

#include "mining_job.h"

namespace asio = boost::asio;

class PoolConnection : public std::enable_shared_from_this<PoolConnection> {
public:
    explicit PoolConnection(asio::io_context& io_context);
    ~PoolConnection();

    // Callbacks
    std::function<void()> on_connected;
    std::function<void(const std::string& reason)> on_disconnected;
    std::function<void(const MiningJob& job)> on_new_job;
    std::function<void(uint64_t request_id, bool accepted, const std::string& error_reason)> on_submit_result;

    // API
    void connect(const std::string& host, const std::string& port, const std::string& user, const std::string& pass);
    void disconnect();
    void submit(const std::string& job_id, const std::string& extranonce2, const std::string& ntime, const std::string& nonce_hex);

    bool is_connected() const { return connected_.load(); }

private:
    void do_resolve();
    void on_resolve(const boost::system::error_code& ec, asio::ip::tcp::resolver::results_type results);
    
    void do_connect(const asio::ip::tcp::resolver::results_type& results);
    void on_connect(const boost::system::error_code& ec);

    void do_read();
    void on_read(const boost::system::error_code& ec, std::size_t bytes_transferred);
    
    void do_write(const std::string& message);

    void process_line(std::string_view line);
    void send_login();
    void clean_up(const std::string& reason, bool is_user_request = false);

    void schedule_reconnect();

    // Asio & State
    asio::io_context& io_context_;
    asio::ip::tcp::resolver resolver_;
    asio::ip::tcp::socket socket_;
    asio::steady_timer connect_timer_;
    asio::steady_timer read_timer_;
    asio::steady_timer reconnect_timer_;
    asio::streambuf buffer_;
    
    std::atomic<bool> connected_{false};
    std::atomic<bool> shutting_down_{false};

    // Reconnection logic
    uint32_t reconnect_attempts_{0};
    std::mt19937 rng_;

    // Connection details
    std::string host_;
    std::string port_;
    std::string user_;
    std::string pass_;
    std::string extranonce1_;
    int extranonce2_size_ = 0;

    std::atomic<uint64_t> request_id_{1};
};
#endif // MINER_NET_POOL_CONNECTION_H_