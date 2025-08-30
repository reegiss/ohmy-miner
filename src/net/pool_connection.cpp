// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#include "miner/pool_connection.h"
#include <iostream>
#include <iomanip>
#include <sstream>

// Adicionando alias para clareza
namespace asio = boost::asio;
using boost::system::error_code;

PoolConnection::PoolConnection(asio::io_context& io_context)
    : io_context_(io_context),
      resolver_(io_context),
      socket_(io_context),
      connect_timer_(io_context),
      read_timer_(io_context) {}

// ... O resto do construtor e métodos permanece o mesmo,
// mas as assinaturas dos callbacks agora usam boost::system::error_code implicitamente ...

void PoolConnection::on_resolve(const error_code& ec, asio::ip::tcp::resolver::results_type results) {
    if (ec) {
        clean_up("Resolve failed: " + ec.message());
        return;
    }
    do_connect(results);
}

void PoolConnection::on_connect(const error_code& ec) {
    connect_timer_.cancel();
    if (ec) {
        clean_up("Connect failed: " + ec.message());
        return;
    }
    // ...
}

void PoolConnection::do_read() {
    if (!is_connected()) return;
    read_timer_.expires_after(std::chrono::seconds(90));
    auto self = shared_from_this();
    read_timer_.async_wait([self](const error_code& ec) {
        if (ec != asio::error::operation_aborted) {
            self->clean_up("Pool read timeout (no data received).");
        }
    });

    asio::async_read_until(socket_, buffer_, '\n',
        [self](const error_code& ec, std::size_t bytes_transferred) {
            self->on_read(ec, bytes_transferred);
        });
}

void PoolConnection::on_read(const error_code& ec, std::size_t) {
    read_timer_.cancel();
    if (ec) {
        clean_up("Read failed: " + ec.message());
        return;
    }
    // ...
}

// O resto do arquivo (lógica de negócio, parsing de JSON, etc.) não precisa de mudanças.
// Apenas as funções que lidam diretamente com callbacks do Asio são afetadas pela mudança do tipo de error_code.
// Cole o resto do arquivo .cpp da etapa anterior aqui.
// ... (implementation of the rest of the methods) ...
PoolConnection::~PoolConnection() {
    disconnect();
}

void PoolConnection::connect(const std::string& host, const std::string& port, const std::string& user, const std::string& pass) {
    if (connected_.load()) return;
    host_ = host;
    port_ = port;
    user_ = user;
    pass_ = pass;
    do_resolve();
}

void PoolConnection::disconnect() {
    asio::post(io_context_, [self = shared_from_this()]() {
        self->clean_up("User requested disconnect.");
    });
}

void PoolConnection::clean_up(const std::string& reason) {
    if (!connected_.exchange(false)) return;
    
    error_code ec;
    connect_timer_.cancel(ec);
    read_timer_.cancel(ec);
    if (socket_.is_open()) {
        socket_.shutdown(asio::ip::tcp::socket::shutdown_both, ec);
        socket_.close(ec);
    }

    if (on_disconnected) {
        on_disconnected(reason);
    }
    std::cout << "[PoolConnection] INFO: Disconnected. Reason: " << reason << std::endl;
}

void PoolConnection::do_resolve() {
    std::cout << "[PoolConnection] INFO: Resolving " << host_ << ":" << port_ << std::endl;
    auto self = shared_from_this();
    resolver_.async_resolve(host_, port_,
        [self](const error_code& ec, asio::ip::tcp::resolver::results_type results) {
            self->on_resolve(ec, results);
        });
}

void PoolConnection::do_connect(const asio::ip::tcp::resolver::results_type& results) {
    std::cout << "[PoolConnection] INFO: Connecting..." << std::endl;
    connect_timer_.expires_after(std::chrono::seconds(10));
    auto self = shared_from_this();
    connect_timer_.async_wait([self](const error_code& ec) {
        std::cout << "[PoolConnection] DEBUG: Connect timer fired with code: " << ec.message() << std::endl;
        if (ec != asio::error::operation_aborted) {
            self->socket_.cancel();
        }
    });

    asio::async_connect(socket_, results,
        [self](const error_code& ec, const asio::ip::tcp::endpoint&) {
            self->on_connect(ec);
        });
}


void PoolConnection::send_login() {
    nlohmann::json req_subscribe = {
        {"id", 1}, {"method", "mining.subscribe"}, {"params", {"qtcminer/0.1"}}
    };
    do_write(req_subscribe.dump() + "\n");

    nlohmann::json req_auth = {
        {"id", 2}, {"method", "mining.authorize"}, {"params", {user_, pass_}}
    };
    do_write(req_auth.dump() + "\n");
}


void PoolConnection::do_write(const std::string& message) {
    if (!is_connected()) return;

    std::cout << "[CLIENT] -> " << message;
    auto self = shared_from_this();
    asio::async_write(socket_, asio::buffer(message),
        [self](const error_code& ec, std::size_t) {
            if (ec) {
                self->clean_up("Write failed: " + ec.message());
            }
        });
}

void PoolConnection::submit(const std::string& job_id, const std::string& extranonce2, const std::string& ntime, const std::string& nonce_hex) {
    uint64_t current_req_id = request_id_.fetch_add(1);
    nlohmann::json req = {
        {"id", current_req_id},
        {"method", "mining.submit"},
        {"params", {user_, job_id, extranonce2, ntime, nonce_hex}}
    };
    do_write(req.dump() + "\n");
}

void PoolConnection::process_line(std::string_view line) {
    try {
        nlohmann::json rpc = nlohmann::json::parse(line);
        
        if (rpc.contains("method")) {
            const std::string method = rpc.value("method", "");
            if (method == "mining.notify") {
                if (on_new_job) {
                    auto params = rpc["params"];
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
                    ss << std::hex << std::setw(extranonce2_size_ * 2) << std::setfill('0') << 0;
                    job.extranonce2 = ss.str();
                    on_new_job(job);
                }
            }
        } else if (rpc.contains("id")) {
            uint64_t id = rpc.value("id", 0);
            if (id == 1) { 
                auto result = rpc["result"];
                extranonce1_ = result[1].get<std::string>();
                extranonce2_size_ = result[2].get<int>();
            } else { 
                if (on_submit_result) {
                    bool result = rpc.value("result", false);
                    std::string error_str = rpc.contains("error") && !rpc["error"].is_null() ? rpc["error"].dump() : "";
                    on_submit_result(id, result, error_str);
                }
            }
        }
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[PoolConnection] ERROR: JSON parse failed: " << e.what() << std::endl;
    }
}