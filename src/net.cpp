// Copyright 2025 Regis Araujo Melo
// Copyright 2025 Regis Araujo Melo
// SPDX-License-Identifier: MIT

#include "miner/net.hpp"

#include <asio.hpp>
#include <nlohmann/json.hpp>
#include "miner/stratum_v1.hpp"

#include <deque>
#include <fmt/core.h>
#include <memory>
#include <thread>

namespace miner::net {

using json = nlohmann::json;

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

    // Callbacks exposed to outer class
    std::function<void(const Job&)> on_new_job_cb;
    std::function<void(ConnectionState)> on_connection_state_changed_cb;

private:
    // Connection lifecycle
    void do_resolve();
    void on_resolve(const asio::error_code& ec, asio::ip::tcp::resolver::results_type results);
    void on_connect(const asio::error_code& ec);
    void start_connection_timer();

    // Protocol
    void do_auth();
    void do_read();
    void on_read(const asio::error_code& ec, std::size_t bytes_transferred);
    void process_message(const std::string& msg);
    void do_write(const std::string& msg);
    void on_write(const asio::error_code& ec, std::size_t bytes_transferred);

    void set_state(ConnectionState new_state);

    // Members
    asio::io_context io_context_;
    asio::ip::tcp::socket socket_{io_context_};
    asio::ip::tcp::resolver resolver_{io_context_};
    asio::streambuf read_buffer_;
    asio::steady_timer connection_timer_{io_context_};
    std::thread network_thread_;
    std::deque<std::string> write_queue_;

    std::string host_;
    uint16_t port_ = 0;
    std::string user_;
    std::string pass_;

    struct SessionInfo {
        std::string extranonce1;
        int extranonce2_size = 0;
        double difficulty = 1.0;
    } session_;

    int reconnect_count_ = 0;
    static constexpr int MAX_RECONNECT_ATTEMPTS = 5;
    static constexpr int CONNECT_TIMEOUT_SECONDS = 10;
    static constexpr int INITIAL_RETRY_DELAY_MS = 1000;

    ConnectionState state_ = ConnectionState::Disconnected;
};

// --- StratumClient (thin pimpl wrapper) ---
StratumClient::StratumClient() : pimpl(std::make_unique<Impl>()) {}
StratumClient::~StratumClient() = default;

void StratumClient::connect(const std::string& host, uint16_t port, const std::string& user, const std::string& pass) {
    pimpl->connect(host, port, user, pass);
}
void StratumClient::disconnect() { pimpl->disconnect(); }
void StratumClient::submit(const Share& share) { pimpl->submit(share); }
void StratumClient::onNewJob(std::function<void(const Job&)> callback) { pimpl->on_new_job_cb = std::move(callback); }
void StratumClient::onConnectionStateChanged(std::function<void(ConnectionState)> callback) { pimpl->on_connection_state_changed_cb = std::move(callback); }

// --- Impl implementation ---

StratumClient::Impl::Impl() {
    // Start the io_context on a dedicated thread
    network_thread_ = std::thread([this]() { io_context_.run(); });
}

StratumClient::Impl::~Impl() {
    // Stop network activity and join thread
    asio::post(io_context_, [this]() {
        connection_timer_.cancel();
        if (socket_.is_open()) socket_.close();
    });
    io_context_.stop();
    if (network_thread_.joinable()) network_thread_.join();
}

void StratumClient::Impl::set_state(ConnectionState new_state) {
    if (state_ == new_state) return;
    state_ = new_state;
    if (on_connection_state_changed_cb) on_connection_state_changed_cb(state_);
}

void StratumClient::Impl::connect(std::string host, uint16_t port, std::string user, std::string pass) {
    host_ = std::move(host);
    port_ = port;
    user_ = std::move(user);
    pass_ = std::move(pass);

    reconnect_count_ = 0;
    asio::post(io_context_, [this]() {
        if (socket_.is_open()) socket_.close();
        socket_ = asio::ip::tcp::socket(io_context_);
        do_resolve();
    });
}

void StratumClient::Impl::disconnect() {
    asio::post(io_context_, [this]() {
        connection_timer_.cancel();
        reconnect_count_ = 0;
        if (socket_.is_open()) socket_.close();
        set_state(ConnectionState::Disconnected);
    });
}

void StratumClient::Impl::submit(const Share& share) {
    auto j = miner::stratum_v1::StratumV1::make_submit(share);
    do_write(j.dump() + "\n");
}

void StratumClient::Impl::start_connection_timer() {
    connection_timer_.cancel();
    connection_timer_.expires_after(std::chrono::seconds(CONNECT_TIMEOUT_SECONDS));
    connection_timer_.async_wait([this](const asio::error_code& ec) {
        if (ec) return; // cancelled
        fmt::print("[Net] Connection attempt timed out after {} seconds\n", CONNECT_TIMEOUT_SECONDS);
        if (socket_.is_open()) socket_.close();
        set_state(ConnectionState::Error);

        if (reconnect_count_ < MAX_RECONNECT_ATTEMPTS) {
            int delay_ms = INITIAL_RETRY_DELAY_MS * (1 << reconnect_count_);
            reconnect_count_++;
            fmt::print("[Net] Retrying connection in {} ms (attempt {}/{}) to {}:{}\n", delay_ms, reconnect_count_, MAX_RECONNECT_ATTEMPTS, host_, port_);
            auto retry_timer = std::make_shared<asio::steady_timer>(io_context_);
            retry_timer->expires_after(std::chrono::milliseconds(delay_ms));
            retry_timer->async_wait([this, retry_timer](const asio::error_code& ec) {
                if (ec) return;
                socket_ = asio::ip::tcp::socket(io_context_);
                do_resolve();
            });
        } else {
            fmt::print("[Net] Max reconnection attempts ({}) reached. Giving up.\n", MAX_RECONNECT_ATTEMPTS);
        }
    });
}

void StratumClient::Impl::do_resolve() {
    set_state(ConnectionState::Connecting);
    start_connection_timer();

    resolver_.async_resolve(host_, std::to_string(port_), [this](const asio::error_code& ec, asio::ip::tcp::resolver::results_type results) {
        if (ec) {
            fmt::print(stderr, "[Net] DNS resolution failed: {}\n", ec.message());
            set_state(ConnectionState::Error);
            return;
        }

        asio::async_connect(socket_, results, [this](const asio::error_code& ec, const asio::ip::tcp::endpoint& /*ep*/) {
            on_connect(ec);
        });
    });
}

void StratumClient::Impl::on_connect(const asio::error_code& ec) {
    if (ec) {
        fmt::print(stderr, "[Net] Connect failed: {}\n", ec.message());
        set_state(ConnectionState::Error);
        if (socket_.is_open()) socket_.close();
        return;
    }

    connection_timer_.cancel();
    reconnect_count_ = 0;
    fmt::print("[Net] Connected to {}:{}\n", host_, port_);
    set_state(ConnectionState::Connected);
    do_auth();
    do_read();
}

void StratumClient::Impl::do_auth() {
    auto subscribe = miner::stratum_v1::StratumV1::make_subscribe("ohmy-miner/0.1.0");
    fmt::print("[Net] Sending subscribe: {}\n", subscribe.dump());
    do_write(subscribe.dump() + "\n");

    auto auth = miner::stratum_v1::StratumV1::make_authorize(user_, pass_);
    fmt::print("[Net] Sending authorize: {}\n", auth.dump());
    do_write(auth.dump() + "\n");
}

void StratumClient::Impl::do_read() {
    asio::async_read_until(socket_, read_buffer_, '\n', [this](const asio::error_code& ec, std::size_t bytes_transferred) {
        on_read(ec, bytes_transferred);
    });
}

void StratumClient::Impl::on_read(const asio::error_code& ec, std::size_t bytes_transferred) {
    (void)bytes_transferred; // may be unused on some builds
    if (ec) {
        if (ec != asio::error::eof) fmt::print(stderr, "[Net] Read failed: {}\n", ec.message());
        if (socket_.is_open()) socket_.close();
        set_state(ConnectionState::Disconnected);
        return;
    }

    std::istream is(&read_buffer_);
    std::string line;
    std::getline(is, line);
    if (!line.empty()) process_message(line);
    if (socket_.is_open()) do_read();
}

void StratumClient::Impl::process_message(const std::string& msg) {
    try {
        json j = json::parse(msg);

        if (j.contains("error") && !j["error"].is_null()) {
            fmt::print(stderr, "[Net] Server error: {}\n", j["error"].dump());
            return;
        }

        if (miner::stratum_v1::StratumV1::is_notify(j)) {
            auto job = miner::stratum_v1::StratumV1::parse_notify(j);
            if (on_new_job_cb) on_new_job_cb(job);
            return;
        }

        if (j.contains("id") && j.contains("result")) {
            int id = j["id"].get<int>();
            const auto& result = j["result"];
            if (id == 1) {
                if (result.is_array() && result.size() >= 3) {
                    session_.extranonce1 = result[1].get<std::string>();
                    session_.extranonce2_size = result[2].get<int>();
                }
            } else if (id == 2) {
                bool ok = false;
                if (result.is_boolean()) ok = result.get<bool>();
                else if (result.is_array() && !result.empty()) ok = true;
                set_state(ok ? ConnectionState::Connected : ConnectionState::AuthenticationFailed);
            } else if (id >= 3) {
                if (result.is_boolean()) {
                    bool accepted = result.get<bool>();
                    fmt::print("[Net] Share submission {} (id={})\n", accepted ? "accepted" : "rejected", id);
                }
            }
            return;
        }

        if (j.contains("method") && j["method"] == "mining.set_difficulty") {
            if (j.contains("params") && j["params"].is_array() && !j["params"].empty()) {
                session_.difficulty = j["params"][0].get<double>();
                fmt::print("[Net] New difficulty set to {}\n", session_.difficulty);
            }
            return;
        }

    } catch (const json::exception& e) {
        fmt::print(stderr, "[Net] JSON parse error: {}\n", e.what());
        fmt::print(stderr, "[Net] Raw message: {}\n", msg);
    } catch (const std::exception& e) {
        fmt::print(stderr, "[Net] Error processing message: {}\n", e.what());
    }
}

void StratumClient::Impl::do_write(const std::string& msg) {
    asio::post(io_context_, [this, msg]() {
        if (!socket_.is_open()) {
            fmt::print("[Net] Attempt to write while socket is closed\n");
            return;
        }
        bool write_in_progress = !write_queue_.empty();
        write_queue_.push_back(msg);
        if (!write_in_progress) {
            asio::async_write(socket_, asio::buffer(write_queue_.front()), [this](const asio::error_code& ec, std::size_t bytes_transferred) {
                on_write(ec, bytes_transferred);
            });
        }
    });
}

void StratumClient::Impl::on_write(const asio::error_code& ec, std::size_t bytes_transferred) {
    (void)bytes_transferred;
    if (ec) {
        fmt::print(stderr, "[Net] Write failed: {}\n", ec.message());
        if (socket_.is_open()) socket_.close();
        set_state(ConnectionState::Disconnected);
        return;
    }

    if (!write_queue_.empty()) write_queue_.pop_front();
    if (!write_queue_.empty() && socket_.is_open()) {
        asio::async_write(socket_, asio::buffer(write_queue_.front()), [this](const asio::error_code& ec, std::size_t bytes_transferred) {
            on_write(ec, bytes_transferred);
        });
    }
}

// Factory
std::unique_ptr<IStratumClient> createStratumClient() {
    return std::make_unique<StratumClient>();
}

} // namespace miner::net