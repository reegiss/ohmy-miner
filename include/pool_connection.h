/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 */
#ifndef POOL_CONNECTION_H
#define POOL_CONNECTION_H

#include <string>
#include <memory>
#include <atomic>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include "thread_safe_queue.h"
#include "mining_job.h"
#include "found_share.h"

namespace asio = boost::asio;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

class PoolConnection : public std::enable_shared_from_this<PoolConnection> {
public:
    PoolConnection(asio::io_context& io_context, const std::string& host, uint16_t port,
                   ThreadSafeQueue<MiningJob>& job_queue,
                   ThreadSafeQueue<FoundShare>& result_queue);

    void start(std::string user, std::string pass);
    void close();

private:
    enum class State {
        DISCONNECTED,
        RESOLVING,
        CONNECTING,
        SUBSCRIBING,
        AUTHORIZING,
        MINING,
        CLOSING
    };

    // Connection Lifecycle
    void do_resolve();
    void on_resolve(const boost::system::error_code& ec, tcp::resolver::results_type endpoints);
    void do_connect(tcp::resolver::results_type endpoints);
    void on_connect(const boost::system::error_code& ec);
    void do_handshake();
    void do_reconnect(const std::string& reason);

    // IO Operations
    void do_read();
    void on_read(const boost::system::error_code& ec, std::size_t bytes_transferred);
    void do_write(const std::string& message);
    void on_write(const boost::system::error_code& ec, std::size_t bytes_transferred);

    // Message Processing
    void process_pool_message(const json& msg);
    void handle_response(const json& msg);
    void handle_notification(const json& msg);
    bool handle_subscribe_response(const json& result);
    bool handle_authorize_response(const json& result);
    void handle_submit_response(const json& result, const json& error);
    void handle_notify(const json& params);
    void handle_set_difficulty(const json& params);

    // Share Submission
    void check_submit_queue();
    
    // Member Variables
    asio::io_context& io_context_;
    tcp::socket socket_;
    tcp::resolver resolver_;
    asio::steady_timer timer_;
    asio::streambuf buffer_;

    std::string host_;
    uint16_t port_;
    std::string user_;
    std::string pass_;

    ThreadSafeQueue<MiningJob>& job_queue_;
    ThreadSafeQueue<FoundShare>& result_queue_;
    
    std::atomic<State> state_;
    std::string extranonce1_;
    int extranonce2_size_;
    uint32_t extranonce2_counter_;
    uint32_t reconnect_delay_ms_;
};

#endif // POOL_CONNECTION_H