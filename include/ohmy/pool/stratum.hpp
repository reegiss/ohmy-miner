#pragma once

#include <functional>
#include <string>
#include <thread>

#include <ohmy/logging/logger.hpp>

namespace asio { class io_context; }
namespace asio::ip { class tcp; }

namespace ohmy::pool {

struct StratumOptions {
    std::string host;
    std::string port;
    std::string user;
    std::string pass;
    std::string client{ "ohmy-miner/" OHMY_MINER_VERSION };
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
 
