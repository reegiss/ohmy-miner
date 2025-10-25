// Copyright 2025 Regis Araujo Melo
// SPDX-License-Identifier: MIT

#include "miner/net.hpp"

#include <asio.hpp>
#include <nlohmann/json.hpp>

#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

// Para logging (pode ser substituído por um logger mais avançado)
#include <fmt/core.h>

namespace miner::net {

// --- JSON Serialization/Deserialization for Stratum types ---

// Helper para nlohmann::json
using json = nlohmann::json;

// Deserializa a notificação 'mining.notify'
void from_json(const json& j, Job& job) {
    // A notificação vem como um array de parâmetros
    const auto& params = j.at("params");
    if (!params.is_array() || params.size() < 9) {
        throw std::invalid_argument("Invalid 'mining.notify' parameters");
    }

    // Params layout (per Stratum mining.notify):
    // [ job_id, prev_hash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs ]
    job.job_id = params[0].get<std::string>();
    job.prev_hash = params[1].get<std::string>();
    job.coinb1 = params[2].get<std::string>();
    job.coinb2 = params[3].get<std::string>();
    job.merkle_branch = params[4].get<std::vector<std::string>>();
    job.version = params[5].get<std::string>();
    job.nbits = params[6].get<std::string>();
    job.ntime = params[7].get<std::string>();
    job.clean_jobs = params[8].get<bool>();
}

// Serializa a submissão 'mining.submit'
void to_json(json& j, const Share& share) {
    j = json{
        {"id", 1}, // ID da submissão, pode ser um contador
        {"method", "mining.submit"},
        {"params", {share.job_id, share.extranonce2, share.ntime, share.nonce}}
    };
}


// --- PIMPL Implementation for StratumClient ---

class StratumClient : public IStratumClient {
public:
    StratumClient();
    ~StratumClient() override;

    void connect(const std::string& host, uint16_t port, const std::string& user, const std::string& pass) override;
    void disconnect() override;
    void submit(const Share& share) override;

    void onNewJob(std::function<void(const Job&)> callback) override;
    void onConnectionStateChanged(std::function<void(ConnectionState)> callback) override;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

class StratumClient::Impl {
public:
    Impl();
    ~Impl();

    void connect(std::string host, uint16_t port, std::string user, std::string pass);
    void disconnect();
    void submit(const Share& share);

    // Callbacks
    std::function<void(const Job&)> on_new_job_cb;
    std::function<void(ConnectionState)> on_connection_state_changed_cb;

private:
    void do_resolve();
    void on_resolve(const asio::error_code& ec, asio::ip::tcp::resolver::results_type results);
    void on_connect(const asio::error_code& ec);
    void do_auth();
    void do_read();
    void on_read(const asio::error_code& ec, std::size_t bytes_transferred);
    void process_message(const std::string& msg);
    void do_write(const std::string& msg);
    void on_write(const asio::error_code& ec, std::size_t bytes_transferred);
    void set_state(ConnectionState new_state);

    asio::io_context io_context_;
    asio::ip::tcp::socket socket_;
    asio::ip::tcp::resolver resolver_;
    asio::streambuf read_buffer_;
    std::thread network_thread_;
    std::deque<std::string> write_queue_;

    std::string host_;
    uint16_t port_;
    std::string user_;
    std::string pass_;

    ConnectionState state_ = ConnectionState::Disconnected;
};

// --- StratumClient Implementation ---

StratumClient::StratumClient() : pimpl(std::make_unique<Impl>()) {}
StratumClient::~StratumClient() = default;

void StratumClient::connect(const std::string& host, uint16_t port, const std::string& user, const std::string& pass) {
    pimpl->connect(host, port, user, pass);
}
void StratumClient::disconnect() { pimpl->disconnect(); }
void StratumClient::submit(const Share& share) { pimpl->submit(share); }
void StratumClient::onNewJob(std::function<void(const Job&)> callback) { pimpl->on_new_job_cb = std::move(callback); }
void StratumClient::onConnectionStateChanged(std::function<void(ConnectionState)> callback) { pimpl->on_connection_state_changed_cb = std::move(callback); }


// --- StratumClient::Impl Implementation ---

StratumClient::Impl::Impl()
    : socket_(io_context_), resolver_(io_context_) {
    network_thread_ = std::thread([this]() { io_context_.run(); });
}

StratumClient::Impl::~Impl() {
    disconnect();
    if (network_thread_.joinable()) {
        network_thread_.join();
    }
}

void StratumClient::Impl::set_state(ConnectionState new_state) {
    if (state_ == new_state) return;
    state_ = new_state;
    if (on_connection_state_changed_cb) {
        on_connection_state_changed_cb(state_);
    }
}

void StratumClient::Impl::connect(std::string host, uint16_t port, std::string user, std::string pass) {
    host_ = std::move(host);
    port_ = port;
    user_ = std::move(user);
    pass_ = std::move(pass);

    asio::post(io_context_, [this]() { do_resolve(); });
}

void StratumClient::Impl::disconnect() {
    asio::post(io_context_, [this]() {
        socket_.close();
        set_state(ConnectionState::Disconnected);
    });
}

void StratumClient::Impl::submit(const Share& share) {
    json j = share;
    do_write(j.dump() + "\n");
}

void StratumClient::Impl::do_resolve() {
    set_state(ConnectionState::Connecting);
    resolver_.async_resolve(host_, std::to_string(port_),
        [this](const asio::error_code& ec, asio::ip::tcp::resolver::results_type results) {
            on_resolve(ec, results);
        });
}

void StratumClient::Impl::on_resolve(const asio::error_code& ec, asio::ip::tcp::resolver::results_type results) {
    if (ec) {
        fmt::print(stderr, "Net error: Resolve failed: {}\n", ec.message());
        set_state(ConnectionState::Error);
        return;
    }
    asio::async_connect(socket_, results,
        [this](const asio::error_code& ec, const asio::ip::tcp::endpoint& /*endpoint*/) {
            on_connect(ec);
        });
}

void StratumClient::Impl::on_connect(const asio::error_code& ec) {
    if (ec) {
        fmt::print(stderr, "Net error: Connect failed: {}\n", ec.message());
        set_state(ConnectionState::Error);
        return;
    }
    set_state(ConnectionState::Connected);
    do_auth();
    do_read();
}

void StratumClient::Impl::do_auth() {
    json subscribe = {
        {"id", 1},
        {"method", "mining.subscribe"},
        {"params", {"ohmy-miner/0.1.0"}}
    };
    do_write(subscribe.dump() + "\n");

    json authorize = {
        {"id", 2},
        {"method", "mining.authorize"},
        {"params", {user_, pass_}}
    };
    do_write(authorize.dump() + "\n");
}

void StratumClient::Impl::do_read() {
    asio::async_read_until(socket_, read_buffer_, '\n',
        [this](const asio::error_code& ec, std::size_t bytes_transferred) {
            on_read(ec, bytes_transferred);
        });
}

void StratumClient::Impl::on_read(const asio::error_code& ec, std::size_t /*bytes_transferred*/) {
    if (ec) {
        if (ec!= asio::error::eof) {
            fmt::print(stderr, "Net error: Read failed: {}\n", ec.message());
        }
        socket_.close();
        set_state(ConnectionState::Disconnected);
        return;
    }

    std::istream is(&read_buffer_);
    std::string line;
    std::getline(is, line);

    if (!line.empty()) {
        process_message(line);
    }

    do_read(); // Continue reading for the next message
}

void StratumClient::Impl::process_message(const std::string& msg) {
    try {
        json j = json::parse(msg);
        if (j.contains("method") && j["method"] == "mining.notify") {
            Job job = j.get<Job>();
            if (on_new_job_cb) {
                on_new_job_cb(job);
            }
        }
        // TODO: Handle other messages like mining.set_difficulty, responses to submit, etc.
    } catch (const json::exception& e) {
        fmt::print(stderr, "Net error: Failed to parse JSON message: {}\nMessage: {}\n", e.what(), msg);
    }
}

void StratumClient::Impl::do_write(const std::string& msg) {
    asio::post(io_context_, [this, msg]() {
        bool write_in_progress =!write_queue_.empty();
        write_queue_.push_back(msg);
        if (!write_in_progress) {
            asio::async_write(socket_, asio::buffer(write_queue_.front()),
                [this](const asio::error_code& ec, std::size_t bytes_transferred) {
                    on_write(ec, bytes_transferred);
                });
        }
    });
}

void StratumClient::Impl::on_write(const asio::error_code& ec, std::size_t /*bytes_transferred*/) {
    if (ec) {
        fmt::print(stderr, "Net error: Write failed: {}\n", ec.message());
        socket_.close();
        set_state(ConnectionState::Disconnected);
        return;
    }

    write_queue_.pop_front();
    if (!write_queue_.empty()) {
        asio::async_write(socket_, asio::buffer(write_queue_.front()),
            [this](const asio::error_code& ec, std::size_t bytes_transferred) {
                on_write(ec, bytes_transferred);
            });
    }
}

// Factory function para criar uma instância do cliente
std::unique_ptr<IStratumClient> createStratumClient() {
    return std::make_unique<StratumClient>();
}

} // namespace miner::net