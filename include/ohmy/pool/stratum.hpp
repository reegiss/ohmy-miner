#pragma once

#include <functional>
#include <string>
#include <thread>

#include <ohmy/logging/logger.hpp>

namespace asio { class io_context; }

namespace ohmy::pool {

struct StratumOptions {
    std::string host;
    std::string port;
    std::string user;
    std::string pass;
    std::string client{ "ohmy-miner/" OHMY_MINER_VERSION };
};

// Result of handshake (subscribe + authorize)
struct HandshakeResult {
    bool success{false};
    std::string extranonce1;
    int extranonce2_size{0};
    std::string error_msg;
};

class StratumClient {
public:
    StratumClient(ohmy::logging::Logger& log, StratumOptions opts);
    ~StratumClient();

    // Non-copyable
    StratumClient(const StratumClient&) = delete;
    StratumClient& operator=(const StratumClient&) = delete;

    // Start IO in background thread
    void start();
    // Stop IO and join thread
    void stop();

    // Small step: synchronous probe that attempts TCP resolve+connect and returns success.
    bool probe_connect();

    // Listen mode: connect, handshake, then keep reading mining.notify for a duration (seconds)
    bool listen_mode(int duration_sec);

    // Optional: user callback for mining.notify raw payload (JSON string)
    void on_notify(std::function<void(std::string_view)> cb);

private:
    void run_io_();
    void connect_and_handshake_();

    ohmy::logging::Logger& log_;
    StratumOptions opts_;
    std::unique_ptr<asio::io_context> ioc_;
    std::unique_ptr<std::thread> io_thread_;
    std::function<void(std::string_view)> notify_cb_{};
};

} // namespace ohmy::pool
 
