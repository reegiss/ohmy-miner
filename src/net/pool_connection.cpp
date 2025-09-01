// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#include "miner/pool_connection.h"
#include "miner/target_util.h"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace asio = boost::asio;
using boost::system::error_code;

PoolConnection::PoolConnection(asio::io_context& io_context, std::shared_ptr<MiningContext> context)
    : io_context_(io_context),
      context_(context),
      resolver_(io_context),
      socket_(io_context),
      connect_timer_(io_context),
      read_timer_(io_context),
      reconnect_timer_(io_context),
      rng_(std::random_device{}()) {}

PoolConnection::~PoolConnection() {}

bool PoolConnection::is_connected() const { return connected_.load(); }

void PoolConnection::connect(const std::string& host, const std::string& port, const std::string& user, const std::string& pass) {
    if (connected_.load() || shutting_down_.load()) return;
    shutting_down_ = false; host_ = host; port_ = port; user_ = user; pass_ = pass;
    asio::post(io_context_, [self = shared_from_this()]() { self->do_resolve(); });
}

void PoolConnection::disconnect() {
    shutting_down_ = true;
    asio::post(io_context_, [self = shared_from_this()]() { self->clean_up("User requested disconnect.", true); });
}

void PoolConnection::clean_up(const std::string& reason, bool is_user_request) {
    if (!socket_.is_open() && !connected_.load()) return;
    bool was_connected = connected_.exchange(false);
    error_code ec;
    reconnect_timer_.cancel(ec); connect_timer_.cancel(ec); read_timer_.cancel(ec);
    if (socket_.is_open()) { socket_.shutdown(asio::ip::tcp::socket::shutdown_both, ec); socket_.close(ec); }
    if (was_connected && on_disconnected) { on_disconnected(reason); }
    std::cout << "[PoolConnection] INFO: Disconnected. Reason: " << reason << std::endl;
    if (!is_user_request && !shutting_down_.load()) { schedule_reconnect(); }
}

void PoolConnection::schedule_reconnect() {
    reconnect_attempts_++;
    auto delay_seconds = std::min(60, 1 << std::min(6, (int)reconnect_attempts_ - 1));
    std::uniform_int_distribution<int> dist(-150, 150);
    auto jitter_ms = (delay_seconds * dist(rng_));
    auto delay_ms = std::chrono::milliseconds(delay_seconds * 1000 + jitter_ms);
    std::cout << "[PoolConnection] INFO: Scheduling reconnect attempt #" << reconnect_attempts_ << " in " << delay_ms.count() << "ms." << std::endl;
    reconnect_timer_.expires_after(delay_ms);
    reconnect_timer_.async_wait([self = shared_from_this()](const error_code& ec) { if (!ec) { self->do_resolve(); } });
}

void PoolConnection::do_resolve() {
    std::cout << "[PoolConnection] INFO: Resolving " << host_ << ":" << port_ << std::endl;
    resolver_.async_resolve(host_, port_, [self = shared_from_this()](const error_code& ec, asio::ip::tcp::resolver::results_type results) { self->on_resolve(ec, results); });
}

void PoolConnection::on_resolve(const error_code& ec, asio::ip::tcp::resolver::results_type results) {
    if (ec) { clean_up("Resolve failed: " + ec.message()); return; }
    do_connect(results);
}

void PoolConnection::do_connect(const asio::ip::tcp::resolver::results_type& results) {
    std::cout << "[PoolConnection] INFO: Connecting..." << std::endl;
    connect_timer_.expires_after(std::chrono::seconds(10));
    connect_timer_.async_wait([self = shared_from_this()](const error_code& ec) { if (ec != asio::error::operation_aborted) { self->socket_.cancel(); } });
    asio::async_connect(socket_, results, [self = shared_from_this()](const error_code& ec, const asio::ip::tcp::endpoint&) { self->on_connect(ec); });
}

void PoolConnection::on_connect(const error_code& ec) {
    connect_timer_.cancel();
    if (ec) { clean_up("Connect failed: " + ec.message()); return; }
    std::cout << "[PoolConnection] INFO: Connection established." << std::endl;
    connected_ = true; reconnect_attempts_ = 0;
    if (on_connected) on_connected();
    send_login(); do_read();
}

void PoolConnection::send_login() {
    do_write(nlohmann::json{{"id", 1}, {"method", "mining.subscribe"}, {"params", {"qtcminer/0.1.1"}}}.dump() + "\n");
    do_write(nlohmann::json{{"id", 2}, {"method", "mining.authorize"}, {"params", {user_, pass_}}}.dump() + "\n");
}

void PoolConnection::do_read() {
    if (!is_connected()) return;
    read_timer_.expires_after(std::chrono::seconds(90));
    read_timer_.async_wait([self = shared_from_this()](const error_code& ec) { if (ec != asio::error::operation_aborted) { self->clean_up("Pool read timeout."); } });
    asio::async_read_until(socket_, buffer_, '\n', [self = shared_from_this()](const error_code& ec, std::size_t bytes) { self->on_read(ec, bytes); });
}

void PoolConnection::on_read(const error_code& ec, std::size_t) {
    read_timer_.cancel();
    if (ec) { clean_up("Read failed: " + ec.message()); return; }
    std::istream is(&buffer_);
    std::string line;
    if (std::getline(is, line) && !line.empty()) { std::cout << "[POOL] <- " << line << std::endl; process_line(line); }
    if (is_connected()) { do_read(); }
}

void PoolConnection::do_write(const std::string& message) {
    if (!is_connected()) return;
    std::cout << "[CLIENT] -> " << message;
    asio::async_write(socket_, asio::buffer(message), [self = shared_from_this()](const error_code& ec, std::size_t) { if (ec) { self->clean_up("Write failed: " + ec.message()); } });
}

void PoolConnection::submit(const std::string& job_id, const std::string& extranonce2, const std::string& ntime, const std::string& nonce_hex) {
    uint64_t current_req_id = request_id_.fetch_add(1);
    nlohmann::json req = { {"id", current_req_id}, {"method", "mining.submit"}, {"params", {user_, job_id, extranonce2, ntime, nonce_hex}} };
    { std::lock_guard<std::mutex> lock(submission_mutex_); pending_submissions_.insert(current_req_id); }
    do_write(req.dump() + "\n");
}

void PoolConnection::process_line(std::string_view line) {
    try {
        nlohmann::json rpc = nlohmann::json::parse(line);
        if (rpc.contains("method")) {
            const std::string method = rpc.value("method", "");
            if (method == "mining.notify") {
                auto params = rpc["params"]; MiningJob job;
                job.job_id = params[0].get<std::string>(); job.prev_hash = params[1].get<std::string>();
                job.coinb1 = params[2].get<std::string>(); job.coinb2 = params[3].get<std::string>();
                job.merkle_branches = params[4].get<std::vector<std::string>>();
                job.version = params[5].get<std::string>(); job.nbits = params[6].get<std::string>();
                job.ntime = params[7].get<std::string>(); job.clean_jobs = params[8].get<bool>();
                job.extranonce1 = extranonce1_; job.extranonce2 = "";
                {
                    std::lock_guard<std::mutex> lock(context_->mtx);
                    context_->current_job = job;
                    if (job.clean_jobs) { target_from_difficulty(context_->difficulty, context_->share_target); }
                }
                if (on_new_job) on_new_job(job);
            } else if (method == "mining.set_difficulty") {
                double new_diff = rpc["params"][0].get<double>();
                std::cout << "[PoolConnection] INFO: New pool difficulty set to " << new_diff << ". Target updated." << std::endl;
                {
                    std::lock_guard<std::mutex> lock(context_->mtx);
                    context_->difficulty = new_diff;
                    target_from_difficulty(new_diff, context_->share_target);
                }
            }
        } else if (rpc.contains("id")) {
            uint64_t id = rpc.value("id", 0);
            if (id == 1) { auto result = rpc["result"]; extranonce1_ = result[1].get<std::string>(); extranonce2_size_ = result[2].get<int>(); return; }
            if (id == 2) { if (!rpc.value("result", false)) { std::cerr << "[PoolConnection] ERROR: Worker authorization failed." << std::endl; } return; }
            
            bool is_submission_reply;
            { std::lock_guard<std::mutex> lock(submission_mutex_); is_submission_reply = pending_submissions_.erase(id); }
            
            if (is_submission_reply && on_submit_result) {
                // --- FIX: Safely check the 'result' field's type before getting value ---
                bool accepted = false;
                if (rpc.contains("result") && rpc["result"].is_boolean()) {
                    accepted = rpc["result"].get<bool>();
                }
                
                std::string err_str = "";
                if (rpc.contains("error") && !rpc["error"].is_null()) {
                    err_str = rpc["error"].dump();
                }
                on_submit_result(id, accepted, err_str);
            }
        }
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[PoolConnection] ERROR: JSON parse failed: " << e.what() << std::endl;
    }
}