/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/pool/stratum.hpp"
#include "ohmy/pool/messages.hpp"
#include <fmt/format.h>
#include <iostream>

namespace ohmy {
namespace pool {

StratumClient::StratumClient(
    asio::io_context& io_context,
    const std::string& url,
    const std::string& worker_name,
    const std::string& password
)   : io_context_(io_context)
    , url_(url)
    , worker_name_(worker_name)
    , password_(password)
{
}

void StratumClient::connect() {
    // Parse URL into host and port
    size_t colon_pos = url_.find(':');
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("Invalid pool URL format. Expected hostname:port");
    }

    std::string host = url_.substr(0, colon_pos);
    std::string port = url_.substr(colon_pos + 1);

    // Resolve endpoint
    asio::ip::tcp::resolver resolver(io_context_);
    auto endpoints = resolver.resolve(host, port);

    // Create and connect socket
    socket_ = std::make_unique<asio::ip::tcp::socket>(io_context_);
    
    asio::async_connect(*socket_, endpoints,
        [this](std::error_code ec, asio::ip::tcp::endpoint) {
            if (!ec) {
                // Connection successful, subscribe to mining
                fmt::print("Connected to pool {}\n", url_);
                subscribe();
                start_reading();
            } else {
                fmt::print("Connection failed: {}\n", ec.message());
                // Schedule reconnect after delay
                auto timer = std::make_shared<asio::steady_timer>(
                    io_context_, std::chrono::seconds(5));
                timer->async_wait([this, timer](std::error_code) {
                    connect();
                });
            }
        });
}

void StratumClient::disconnect() {
    if (socket_ && socket_->is_open()) {
        socket_->close();
    }
    subscribed_ = false;
    authorized_ = false;
}

bool StratumClient::is_connected() const {
    return socket_ && socket_->is_open() && subscribed_ && authorized_;
}

void StratumClient::start_reading() {
    // Read until newline delimiter
    asio::async_read_until(*socket_, receive_buffer_, '\n',
        [this](std::error_code ec, [[maybe_unused]] std::size_t length) {
            if (!ec) {
                // Extract message from buffer
                std::string message;
                std::istream is(&receive_buffer_);
                std::getline(is, message);

                // Handle the message
                handle_message(message);

                // Continue reading
                start_reading();
            } else {
                fmt::print("Read error: {}\n", ec.message());
                disconnect();
                
                // Schedule reconnect
                auto timer = std::make_shared<asio::steady_timer>(
                    io_context_, std::chrono::seconds(5));
                timer->async_wait([this, timer](std::error_code) {
                    connect();
                });
            }
        });
}

void StratumClient::send_message(const json& message) {
    if (!socket_ || !socket_->is_open()) {
        return;
    }

    std::string message_str = message.dump() + "\n";
    asio::async_write(*socket_, asio::buffer(message_str),
        [this](std::error_code ec, std::size_t) {
            if (ec) {
                fmt::print("Write error: {}\n", ec.message());
                disconnect();
            }
        });
}

void StratumClient::handle_message(const std::string& message) {
    try {
        // Log all incoming messages for debugging
        fmt::print("Received message: {}\n", message.substr(0, 200) + 
                  (message.length() > 200 ? "..." : ""));
        
        json j = json::parse(message);

        if (StratumMessages::is_notification(j)) {
            // Handle notifications from server
            const std::string& method = j["method"];
            
            fmt::print("Processing notification: {}\n", method);
            
            if (method == "mining.notify") {
                handle_mining_notify(j["params"]);
            } else if (method == "mining.set_difficulty") {
                handle_set_difficulty(j["params"]);
            } else {
                fmt::print("Unknown notification method: {}\n", method);
            }
        }
        else if (StratumMessages::is_response(j)) {
            // Handle method responses
            uint64_t id = j["id"];
            
            fmt::print("Processing response for ID: {}\n", id);
            
            if (id == 1) {  // Response to subscribe
                if (!StratumMessages::is_error(j)) {
                    subscribed_ = true;
                    fmt::print("Subscription successful\n");
                    // Now authorize
                    authorize();
                } else {
                    fmt::print("Subscription failed: {}\n", j["error"].dump());
                }
            }
            else if (id == 2) {  // Response to authorize
                if (!StratumMessages::is_error(j)) {
                    authorized_ = true;
                    fmt::print("Successfully authorized worker: {}\n", worker_name_);
                } else {
                    fmt::print("Authorization failed: {}\n", j["error"].dump());
                }
            }
            else {  // Response to share submission
                bool accepted = !StratumMessages::is_error(j);
                handle_submit_result(j, accepted);
            }
        } else {
            fmt::print("Unknown message type: {}\n", j.dump());
        }

    } catch (const std::exception& e) {
        fmt::print("Error parsing message: {} - Raw message: {}\n", 
                  e.what(), message.substr(0, 100));
    }
}

void StratumClient::subscribe() {
    fmt::print("Sending mining.subscribe request...\n");
    send_message(StratumMessages::subscribe(1));
}

void StratumClient::authorize() {
    fmt::print("Sending mining.authorize request for worker: {}\n", worker_name_);
    send_message(StratumMessages::authorize(2, worker_name_, password_));
}

void StratumClient::submit_share(const ShareResult& share) {
    // Format share submission
    auto msg = StratumMessages::submit(
        ++message_id_,
        worker_name_,
        share.job_id,
        share.extranonce2,
        share.ntime,
        fmt::format("{:08x}", share.nonce)
    );
    
    send_message(msg);
}

void StratumClient::handle_mining_notify(const json& params) {
    try {
        // Extract work package from notification
        WorkPackage work;
        
        // Array of parameters as per Stratum spec
        work.job_id = params[0];
        work.previous_hash = params[1];
        work.coinbase1 = params[2];
        work.coinbase2 = params[3];
        work.merkle_branch = params[4];
        work.version = params[5];
        work.bits = params[6];
        work.time = params[7];
        work.clean_jobs = params[8];

        fmt::print("Received mining.notify:\n");
        fmt::print("  Job ID: {}\n", work.job_id);
        fmt::print("  Previous Hash: {}\n", work.previous_hash.substr(0, 16) + "...");
        fmt::print("  Version: {}\n", work.version);
        fmt::print("  Bits: {}\n", work.bits);
        fmt::print("  Time: {}\n", work.time);
        fmt::print("  Clean Jobs: {}\n", work.clean_jobs ? "true" : "false");

        // Notify miner of new work
        if (work_callback_) {
            work_callback_(work);
        }
    } catch (const std::exception& e) {
        fmt::print("Error parsing mining.notify: {}\n", e.what());
    }
}

void StratumClient::handle_set_difficulty(const json& params) {
    try {
        current_difficulty_ = params[0];
        fmt::print("Received mining.set_difficulty: {}\n", current_difficulty_);
        
        // If we have a difficulty callback, notify the work manager
        // This will be set up in main.cpp
    } catch (const std::exception& e) {
        fmt::print("Error parsing mining.set_difficulty: {}\n", e.what());
    }
}

void StratumClient::handle_submit_result(const json& result, bool accepted) {
    ShareResult share;
    share.accepted = accepted;
    
    if (!accepted && StratumMessages::is_error(result)) {
        auto error = result["error"];
        if (!error.is_null()) {
            int code = error[0];
            share.reason = get_error_message(static_cast<StratumError>(code));
        }
    }

    if (share_callback_) {
        share_callback_(share);
    }
}

void StratumClient::set_work_callback(WorkCallback callback) {
    work_callback_ = std::move(callback);
}

void StratumClient::set_share_callback(ShareCallback callback) {
    share_callback_ = std::move(callback);
}

} // namespace pool
} // namespace ohmy