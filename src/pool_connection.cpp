/*
 * Copyright (c) 2025 Regis Araujo Melo
 * License: MIT
 */
#include "pool_connection.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

extern std::atomic<bool> g_shutdown;

PoolConnection::PoolConnection(asio::io_context& io_context, const std::string& host, uint16_t port,
                               ThreadSafeQueue<MiningJob>& job_queue,
                               ThreadSafeQueue<FoundShare>& result_queue)
    : io_context_(io_context),
      socket_(io_context),
      resolver_(io_context),
      timer_(io_context),
      host_(host),
      port_(port),
      job_queue_(job_queue),
      result_queue_(result_queue),
      state_(State::DISCONNECTED),
      extranonce2_size_(4), // Default to a safe value
      extranonce2_counter_(0),
      reconnect_delay_ms_(5000) {}

void PoolConnection::start(std::string user, std::string pass) {
    user_ = std::move(user);
    pass_ = std::move(pass);
    do_resolve();
}

void PoolConnection::close() {
    if (state_ == State::CLOSING) return;
    state_ = State::CLOSING;
    asio::post(io_context_, [self = shared_from_this()]() {
        self->timer_.cancel();
        if (self->socket_.is_open()) {
            boost::system::error_code ec;
            self->socket_.shutdown(tcp::socket::shutdown_both, ec);
            self->socket_.close(ec);
        }
    });
}

void PoolConnection::do_resolve() {
    if (g_shutdown) return;
    state_ = State::RESOLVING;
    std::cout << "[NETWORK] Resolving " << host_ << "..." << std::endl;
    resolver_.async_resolve(host_, std::to_string(port_),
        [self = shared_from_this()](const auto& ec, auto endpoints) { self->on_resolve(ec, endpoints); });
}

void PoolConnection::on_resolve(const boost::system::error_code& ec, tcp::resolver::results_type endpoints) {
    if (g_shutdown || state_ == State::CLOSING) return;
    if (ec) {
        std::cerr << "[NETWORK] Resolve failed: " << ec.message() << std::endl;
        do_reconnect("resolve failed");
        return;
    }
    do_connect(endpoints);
}

void PoolConnection::do_connect(tcp::resolver::results_type endpoints) {
    if (g_shutdown) return;
    state_ = State::CONNECTING;
    std::cout << "[NETWORK] Connecting to " << host_ << ":" << port_ << "..." << std::endl;

    timer_.expires_after(std::chrono::seconds(15));
    timer_.async_wait([self = shared_from_this()](const auto& ec) {
        if (!ec) { self->socket_.cancel(); }
    });

    asio::async_connect(socket_, endpoints,
        [self = shared_from_this()](const auto& ec, const auto& /*ep*/) {
            self->timer_.cancel();
            self->on_connect(ec);
        });
}

void PoolConnection::on_connect(const boost::system::error_code& ec) {
    if (g_shutdown || state_ == State::CLOSING) return;
    if (ec) {
        std::cerr << "[NETWORK] Connect failed: " << ec.message() << std::endl;
        do_reconnect("connect failed");
        return;
    }
    std::cout << "✅ [NETWORK] Connected to pool." << std::endl;
    reconnect_delay_ms_ = 5000;
    do_handshake();
}

void PoolConnection::do_handshake() {
    if (g_shutdown) return;
    state_ = State::SUBSCRIBING;
    json subscribe_req = {{"id", 1}, {"method", "mining.subscribe"}, {"params", {"qtc-miner/1.0"}}};
    do_write(subscribe_req.dump() + "\n");
    do_read();
}

void PoolConnection::do_read() {
    if (g_shutdown || state_ == State::CLOSING) return;
    asio::async_read_until(socket_, buffer_, '\n',
        [self = shared_from_this()](const auto& ec, auto bytes) { self->on_read(ec, bytes); });
}

void PoolConnection::on_read(const boost::system::error_code& ec, std::size_t /*bytes*/) {
    if (g_shutdown || state_ == State::CLOSING) return;
    if (ec) {
        if (ec != asio::error::eof) std::cerr << "[NETWORK] Read error: " << ec.message() << std::endl;
        do_reconnect("read error");
        return;
    }
    std::istream is(&buffer_);
    std::string line;
    std::getline(is, line);
    try {
        process_pool_message(json::parse(line));
    } catch (const json::parse_error& e) {
        std::cerr << "[NETWORK] JSON parse error: " << e.what() << std::endl;
    }
    do_read();
}

void PoolConnection::do_write(const std::string& message) {
    if (g_shutdown || state_ == State::CLOSING) return;
    asio::async_write(socket_, asio::buffer(message),
        [self = shared_from_this()](const auto& ec, auto bytes) { self->on_write(ec, bytes); });
}

void PoolConnection::on_write(const boost::system::error_code& ec, std::size_t /*bytes*/) {
    if (g_shutdown || state_ == State::CLOSING) return;
    if (ec) {
        std::cerr << "[NETWORK] Write error: " << ec.message() << std::endl;
        do_reconnect("write error");
    }
}

void PoolConnection::check_submit_queue() {
    if (g_shutdown || state_ != State::MINING) return;
    FoundShare share;
    if (result_queue_.try_pop(share)) {
        std::cout << "\n[NETWORK] Submitting share for job " << share.job_id << "..." << std::endl;
        json submit_req = {{"method", "mining.submit"}, {"params", {user_, share.job_id, share.extranonce2, share.ntime, share.nonce_hex}}, {"id", 4}};
        do_write(submit_req.dump() + "\n");
    }
    timer_.expires_after(std::chrono::milliseconds(50));
    timer_.async_wait([self = shared_from_this()](const auto& ec) { if (!ec) self->check_submit_queue(); });
}

void PoolConnection::process_pool_message(const json& msg) {
    if (msg.contains("id")) {
        handle_response(msg);
    } else if (msg.contains("method")) {
        handle_notification(msg);
    }
}

void PoolConnection::handle_response(const json& msg) {
    const int id = msg.value("id", 0);
    const json result = msg.value("result", json());
    const json error = msg.value("error", json());

    switch (id) {
        case 1: // Subscribe
            if (handle_subscribe_response(result)) {
                state_ = State::AUTHORIZING;
                json auth_req = {{"id", 2}, {"method", "mining.authorize"}, {"params", {user_, pass_}}};
                do_write(auth_req.dump() + "\n");
            } else {
                std::cerr << "❌ [NETWORK] Subscribe failed: " << error.dump() << std::endl;
                do_reconnect("subscribe failed");
            }
            break;
        case 2: // Authorize
            if (handle_authorize_response(result)) {
                state_ = State::MINING;
                check_submit_queue();
            } else {
                std::cerr << "❌ [NETWORK] Authorization failed: " << error.dump() << std::endl;
                g_shutdown = true;
                close();
            }
            break;
        case 4: // Submit
            handle_submit_response(result, error);
            break;
    }
}

bool PoolConnection::handle_subscribe_response(const json& result) {
    if (!result.is_array() || result.size() < 3) return false;
    extranonce1_ = result[1].get<std::string>();
    if (result[2].is_number()) {
        extranonce2_size_ = result[2].get<int>();
    } else {
        std::cout << "[NETWORK] Warning: Pool sent non-standard extranonce2_size. Defaulting to 4." << std::endl;
        extranonce2_size_ = 4;
    }
    std::cout << "[NETWORK] Subscribed. Extranonce1: " << extranonce1_ << std::endl;
    return true;
}

bool PoolConnection::handle_authorize_response(const json& result) {
    if (result.is_boolean() && result.get<bool>()) {
        std::cout << "✅ [NETWORK] Authorization successful. Stratum active." << std::endl;
        return true;
    }
    return false;
}

void PoolConnection::handle_submit_response(const json& result, const json& error) {
    if (result.is_boolean() && result.get<bool>()) {
        std::cout << "✅ [NETWORK] Share ACCEPTED." << std::endl;
    } else {
        std::cerr << "❌ [NETWORK] Share REJECTED. Reason: " << error.dump() << std::endl;
    }
}

void PoolConnection::handle_notification(const json& msg) {
    const std::string method = msg.value("method", "");
    const json params = msg.value("params", json::array());
    if (method == "mining.notify") {
        handle_notify(params);
    } else if (method == "mining.set_difficulty") {
        handle_set_difficulty(params);
    }
}

void PoolConnection::handle_notify(const json& params) {
    try {
        MiningJob job;
        job.job_id = params[0].get<std::string>();
        job.prev_hash = params[1].get<std::string>();
        job.coinb1 = params[2].get<std::string>();
        job.coinb2 = params[3].get<std::string>();
        job.merkle_branches = params[4].get<std::vector<std::string>>();
        job.version = params[5].get<std::string>();
        job.nbits = params[6].get<std::string>();
        job.ntime = params[7].get<std::string>();
        job.clean_jobs = params[8].get<bool>();
        job.extranonce1 = extranonce1_;
        std::stringstream ss;
        ss << std::hex << std::setw(extranonce2_size_ * 2) << std::setfill('0') << extranonce2_counter_++;
        job.extranonce2 = ss.str();
        job_queue_.push(job);
    } catch (const json::exception& e) {
        std::cerr << "[NETWORK] Error processing mining.notify: " << e.what() << std::endl;
    }
}

void PoolConnection::handle_set_difficulty(const json& params) {
    try {
        double new_diff = params[0].get<double>();
        std::cout << "[NETWORK] Pool new difficulty: " << new_diff << std::endl;
    } catch (const json::exception& e) {
        std::cerr << "[NETWORK] Error processing set_difficulty: " << e.what() << std::endl;
    }
}

void PoolConnection::do_reconnect(const std::string& reason) {
    if (g_shutdown || state_ == State::CLOSING || state_ == State::CONNECTING || state_ == State::RESOLVING) return;
    state_ = State::DISCONNECTED;
    boost::system::error_code ec;
    socket_.close(ec);
    std::cout << "[NETWORK] Disconnected (" << reason << "). Reconnecting in " << reconnect_delay_ms_ / 1000 << "s..." << std::endl;
    timer_.expires_after(std::chrono::milliseconds(reconnect_delay_ms_));
    timer_.async_wait([self = shared_from_this()](const auto& ec) { if (!ec) self->start(self->user_, self->pass_); });
    reconnect_delay_ms_ = std::min(reconnect_delay_ms_ * 2, 30000u);
}