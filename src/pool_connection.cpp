/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "pool_connection.hpp"
#include <fmt/core.h>
#include <fmt/color.h>
#include <iostream>
#include <sstream>
#include <chrono>

namespace ohmy {

PoolConnection::PoolConnection(asio::io_context& io_context,
                               const std::string& pool_url,
                               const std::string& username,
                               const std::string& password)
    : io_context_(io_context)
    , pool_url_(pool_url)
    , username_(username)
    , password_(password) {
}

PoolConnection::~PoolConnection() {
    disconnect();
}

bool PoolConnection::parse_url(const std::string& url, std::string& host, std::string& port) {
    size_t colon_pos = url.find(':');
    if (colon_pos == std::string::npos) {
        fmt::print(fg(fmt::color::red), "Error: Invalid pool URL format. Expected host:port\n");
        return false;
    }
    
    host = url.substr(0, colon_pos);
    port = url.substr(colon_pos + 1);
    
    if (host.empty() || port.empty()) {
        fmt::print(fg(fmt::color::red), "Error: Empty host or port in URL\n");
        return false;
    }
    
    return true;
}

bool PoolConnection::connect() {
    if (connected_) {
        fmt::print(fg(fmt::color::yellow), "Already connected to pool\n");
        return true;
    }

    if (!parse_url(pool_url_, host_, port_)) {
        return false;
    }

    fmt::print(fg(fmt::color::cyan), "Connecting to pool {}...\n", pool_url_);

    try {
        // Create socket
        socket_ = std::make_unique<asio::ip::tcp::socket>(io_context_);

        // Resolve host
        asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(host_, port_);

        // Connect with timeout
        asio::error_code ec;
        asio::connect(*socket_, endpoints, ec);

        if (ec) {
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "Connection failed: {}\n", ec.message());
            return false;
        }

        connected_ = true;
        fmt::print(fg(fmt::color::green), "✓ Connected to {}:{}\n", host_, port_);
        
        return true;

    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Connection error: {}\n", e.what());
        return false;
    }
}

void PoolConnection::disconnect() {
    if (!connected_) {
        return;
    }

    try {
        if (socket_ && socket_->is_open()) {
            socket_->close();
        }
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::yellow), "Disconnect error: {}\n", e.what());
    }

    connected_ = false;
    authorized_ = false;
    subscribed_ = false;
    
    fmt::print(fg(fmt::color::yellow), "Disconnected from pool\n");
}

bool PoolConnection::send_request(const std::string& method, const json& params, int id) {
    if (!connected_) {
        fmt::print(fg(fmt::color::red), "Error: Not connected to pool\n");
        return false;
    }

    try {
        json request = {
            {"id", id},
            {"method", method},
            {"params", params}
        };

        std::string message = request.dump() + "\n";

        std::lock_guard<std::mutex> lock(send_mutex_);
        asio::write(*socket_, asio::buffer(message));

    fmt::print(fg(fmt::color::gray), "→ Sent: {}\n", method);
        
        return true;

    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Send error: {}\n", e.what());
        return false;
    }
}

json PoolConnection::receive_response([[maybe_unused]] int timeout_seconds) {
    if (!connected_) {
        return json::object();
    }

    try {
        // Set socket timeout
        asio::error_code ec;
        std::string line;

        // Read until newline
    [[maybe_unused]] size_t bytes = asio::read_until(*socket_, receive_buffer_, '\n', ec);

        if (ec) {
            if (ec != asio::error::eof) {
                fmt::print(fg(fmt::color::red), "Receive error: {}\n", ec.message());
            }
            return json::object();
        }

        // Extract line from buffer
        std::istream is(&receive_buffer_);
        std::getline(is, line);

        if (line.empty()) {
            return json::object();
        }

        // Parse JSON
        json response = json::parse(line);
        
        return response;

    } catch (const json::parse_error& e) {
        fmt::print(fg(fmt::color::red), "JSON parse error: {}\n", e.what());
        return json::object();
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Receive error: {}\n", e.what());
        return json::object();
    }
}

json PoolConnection::wait_for_response_for_id(int expected_id, [[maybe_unused]] int timeout_seconds) {
    // Keep reading lines until we find a JSON with matching id; notifications will be handled
    while (connected_) {
        try {
            asio::error_code ec;
            [[maybe_unused]] size_t bytes = asio::read_until(*socket_, receive_buffer_, '\n', ec);
            if (ec) {
                if (ec != asio::error::eof) {
                    fmt::print(fg(fmt::color::red), "Receive error: {}\n", ec.message());
                }
                return json::object();
            }

            std::istream is(&receive_buffer_);
            std::string line;
            std::getline(is, line);
            if (line.empty()) {
                continue;
            }

            json msg = json::parse(line);
            if (msg.contains("id") && !msg["id"].is_null()) {
                int id_val = 0;
                // Some pools use integer ids; be tolerant
                try {
                    id_val = msg["id"].get<int>();
                } catch (...) {
                    // Ignore non-integer ids for now
                }
                if (id_val == expected_id) {
                    return msg;
                }
            }

            // If it's a notification, handle it
            if (msg.contains("method")) {
                handle_message(line);
            }

        } catch (const json::parse_error& e) {
            fmt::print(fg(fmt::color::red), "JSON parse error: {}\n", e.what());
        } catch (const std::exception& e) {
            fmt::print(fg(fmt::color::red), "Receive error: {}\n", e.what());
            return json::object();
        }
    }
    return json::object();
}

bool PoolConnection::subscribe() {
    if (!connected_) {
        fmt::print(fg(fmt::color::red), "Error: Not connected to pool\n");
        return false;
    }

    fmt::print(fg(fmt::color::cyan), "Subscribing to pool...\n");

    // Send mining.subscribe (Stratum v1-compatible minimal form)
    json params = json::array({
        "ohmy-miner/0.1.0"  // User agent
    });

    int req_id = request_id_++;
    if (!send_request("mining.subscribe", params, req_id)) {
        return false;
    }

    // Wait for response
    json response = wait_for_response_for_id(req_id);
    
    if (response.empty() || !response.contains("result")) {
        fmt::print(fg(fmt::color::red), "Error: Invalid subscribe response\n");
        return false;
    }

    // Parse subscription result
    auto result = response["result"];
    if (result.is_array() && result.size() >= 2) {
        // Extract subscription details
        if (result[1].is_string()) {
            extra_nonce1_ = result[1].get<std::string>();
        }
        if (result.size() >= 3 && result[2].is_number()) {
            extra_nonce2_size_ = result[2].get<int>();
        }
    }

    subscribed_ = true;
    fmt::print(fg(fmt::color::green), 
        "✓ Subscribed (ExtraNonce1: {}, ExtraNonce2 Size: {})\n",
        extra_nonce1_, extra_nonce2_size_);

    return true;
}

bool PoolConnection::authorize() {
    if (!connected_ || !subscribed_) {
        fmt::print(fg(fmt::color::red), 
            "Error: Must be connected and subscribed before authorization\n");
        return false;
    }

    fmt::print(fg(fmt::color::cyan), "Authorizing with pool...\n");

    // Helpful pre-checks on username format (wallet.worker)
    std::string wallet_part = username_;
    if (auto pos = username_.find('.'); pos != std::string::npos) {
        wallet_part = username_.substr(0, pos);
    }
    // Warn on common mistake: using Bitcoin bech32 address instead of QTC wallet
    if (wallet_part.rfind("bc1", 0) == 0 || wallet_part.rfind("1", 0) == 0 || wallet_part.rfind("3", 0) == 0) {
        fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
            "Warning: Username '{}' looks like a Bitcoin address. Most QTC pools require a valid Qubitcoin (QTC) wallet address.\n",
            wallet_part);
        fmt::print(fg(fmt::color::yellow),
            "Hint: Use 'WALLET.worker' as username and typically 'x' as password.\n");
    }

    // Send mining.authorize
    json params = json::array({username_, password_});

    int req_id = request_id_++;
    if (!send_request("mining.authorize", params, req_id)) {
        return false;
    }

    // Wait for response
    json response = wait_for_response_for_id(req_id);
    
    if (response.empty()) {
        fmt::print(fg(fmt::color::red), "Error: No authorization response\n");
        return false;
    }

    // Check if authorized
    std::string pool_error_msg;
    if (response.contains("result") && response["result"].is_boolean()) {
        authorized_ = response["result"].get<bool>();
        if (!authorized_) {
            // Some pools return result=false without error object
            pool_error_msg = "Pool returned result=false";
        }
    }
    if (response.contains("error") && !response["error"].is_null()) {
        pool_error_msg = response["error"].dump();
    }

    if (authorized_) {
        fmt::print(fg(fmt::color::green), "✓ Authorized as {}\n", username_);
    } else {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold, "Authorization denied\n");
        if (!pool_error_msg.empty()) {
            fmt::print(fg(fmt::color::red), "Pool error: {}\n", pool_error_msg);
        }
        // Also print raw response for troubleshooting
        if (!response.is_null()) {
            fmt::print(fg(fmt::color::gray), "Auth response: {}\n", response.dump());
        }
        fmt::print(fg(fmt::color::yellow),
            "Checklist to fix authorization:\n"
            "- Ensure the username is your QTC wallet address (not Bitcoin) optionally with .worker suffix.\n"
            "- Keep password as 'x' unless the pool requires otherwise.\n"
            "- Verify the pool URL/port and coin match (Qubitcoin).\n");
    }

    return authorized_;
}

bool PoolConnection::submit_share(const std::string& job_id,
                                  const std::string& nonce,
                                  const std::string& ntime,
                                  const std::string& extra_nonce2) {
    if (!connected_ || !authorized_) {
        fmt::print(fg(fmt::color::red), "Error: Not authorized to submit shares\n");
        return false;
    }

    // Send mining.submit
    json params = json::array({
        username_,
        job_id,
        extra_nonce2,
        ntime,
        nonce
    });

    int req_id = request_id_++;
    if (!send_request("mining.submit", params, req_id)) {
        return false;
    }

    // Wait for response
    json response = receive_response();
    
    if (response.empty()) {
        fmt::print(fg(fmt::color::red), "Error: No submit response\n");
        return false;
    }

    // Check if accepted
    bool accepted = false;
    if (response.contains("result") && response["result"].is_boolean()) {
        accepted = response["result"].get<bool>();
    }

    if (accepted) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ Share ACCEPTED!\n");
    } else {
        std::string error_msg = "Unknown error";
        if (response.contains("error") && !response["error"].is_null()) {
            error_msg = response["error"].dump();
        }
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "✗ Share REJECTED: {}\n", error_msg);
    }

    return accepted;
}

void PoolConnection::handle_message(const std::string& message) {
    try {
        json msg = json::parse(message);

        // Check if it's a method call (notification)
        if (msg.contains("method")) {
            std::string method = msg["method"].get<std::string>();

            if (method == "mining.notify") {
                handle_mining_notify(msg["params"]);
            } else if (method == "mining.set_difficulty") {
                handle_set_difficulty(msg["params"]);
            } else {
                fmt::print(fg(fmt::color::gray), "← Received: {}\n", method);
            }
        }

    } catch (const json::parse_error& e) {
        fmt::print(fg(fmt::color::red), "JSON parse error: {}\n", e.what());
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Message handling error: {}\n", e.what());
    }
}

void PoolConnection::handle_mining_notify(const json& params) {
    if (!params.is_array() || params.size() < 9) {
        fmt::print(fg(fmt::color::red), "Error: Invalid mining.notify params\n");
        return;
    }

    MiningJob job;
    job.job_id = params[0].get<std::string>();
    job.prev_hash = params[1].get<std::string>();
    job.coinbase1 = params[2].get<std::string>();
    job.coinbase2 = params[3].get<std::string>();
    
    if (params[4].is_array()) {
        for (const auto& branch : params[4]) {
            job.merkle_branch.push_back(branch.get<std::string>());
        }
    }
    
    job.version = params[5].get<std::string>();
    job.nbits = params[6].get<std::string>();
    job.ntime = params[7].get<std::string>();
    job.clean_jobs = params[8].get<bool>();

    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        current_job_ = job;
    }

    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "← New Job: {} (Clean: {})\n", job.job_id, job.clean_jobs ? "Yes" : "No");

    // Call job callback if set
    if (job_callback_) {
        job_callback_(job);
    }
}

void PoolConnection::handle_set_difficulty(const json& params) {
    if (!params.is_array() || params.empty()) {
        return;
    }

    double new_difficulty = params[0].get<double>();
    difficulty_ = new_difficulty;

    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
        "← Difficulty changed: {}\n", new_difficulty);

    // Call difficulty callback if set
    if (difficulty_callback_) {
        difficulty_callback_(new_difficulty);
    }
}

void PoolConnection::start_receive_loop() {
    if (!connected_) {
        return;
    }
    do_async_read();
}

void PoolConnection::do_async_read() {
    asio::async_read_until(*socket_, receive_buffer_, '\n',
        [this](const asio::error_code& error, std::size_t bytes_transferred) {
            async_read_handler(error, bytes_transferred);
        });
}

void PoolConnection::async_read_handler(const asio::error_code& error, 
                                        [[maybe_unused]] std::size_t bytes_transferred) {
    if (error) {
        if (error != asio::error::eof) {
            fmt::print(fg(fmt::color::red), "Async read error: {}\n", error.message());
            if (error_callback_) {
                error_callback_(error.message());
            }
        }
        connected_ = false;
        return;
    }

    // Extract message
    std::istream is(&receive_buffer_);
    std::string line;
    std::getline(is, line);

    if (!line.empty()) {
        handle_message(line);
    }

    // Continue reading
    if (connected_) {
        do_async_read();
    }
}

} // namespace ohmy
