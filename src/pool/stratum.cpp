#include <ohmy/pool/stratum.hpp>

#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/connect.hpp>
#include <asio/steady_timer.hpp>
#include <asio/error.hpp>

using namespace std::literals;

namespace ohmy::pool {

StratumClient::StratumClient(ohmy::logging::Logger& log, StratumOptions opts)
    : log_(log), opts_(std::move(opts)) {}

StratumClient::~StratumClient() { stop(); }

void StratumClient::start() {
    if (io_thread_) return;
    ioc_ = std::make_unique<asio::io_context>();
    io_thread_ = std::make_unique<std::thread>([this]{ run_io_(); });
}

void StratumClient::stop() {
    if (ioc_) ioc_->stop();
    if (io_thread_ && io_thread_->joinable()) io_thread_->join();
    io_thread_.reset();
    ioc_.reset();
}

void StratumClient::on_notify(std::function<void(std::string_view)> cb) {
    notify_cb_ = std::move(cb);
}

void StratumClient::run_io_() {
    try {
        connect_and_handshake_();
        if (ioc_) ioc_->run();
    } catch (const std::exception& e) {
        log_.error(std::string("Stratum IO error: ") + e.what());
    }
}

void StratumClient::connect_and_handshake_() {
    // Placeholder: networking to be implemented next iteration.
    log_.info("Stratum: connect_and_handshake() not yet implemented");
}

bool StratumClient::probe_connect() {
    // Implement a short timeout (3s) using async resolve/connect + steady_timer.
    try {
        asio::io_context ioc;
        asio::ip::tcp::resolver resolver{ioc};
        asio::ip::tcp::socket socket{ioc};
        asio::steady_timer timer{ioc};

        using namespace std::chrono_literals;
        const auto timeout = 3s;
        std::atomic<bool> timed_out{false};
        std::atomic<bool> success{false};
        std::string err_msg;

        log_.info(std::string("Stratum probe: resolving ") + opts_.host + ":" + opts_.port);

        // Timer handler: on timeout, cancel resolver and close socket to abort connect
        timer.expires_after(timeout);
        timer.async_wait([&](const std::error_code& ec){
            if (ec) return; // timer cancelled
            timed_out.store(true);
            std::error_code ignored;
            resolver.cancel();
            socket.close(ignored);
        });

        // Resolve asynchronously
        resolver.async_resolve(opts_.host, opts_.port,
            [&](const std::error_code& ec, asio::ip::tcp::resolver::results_type results){
                if (ec) {
                    if (!timed_out.load()) err_msg = std::string("resolve: ") + ec.message();
                    return; // let io_context finish
                }
                log_.info("Stratum probe: connecting...");
                asio::async_connect(socket, results,
                    [&](const std::error_code& ec2, const asio::ip::tcp::endpoint&){
                        if (!ec2) {
                            success.store(true);
                            std::error_code ignored;
                            timer.cancel(ignored);
                        } else if (!timed_out.load()) {
                            err_msg = std::string("connect: ") + ec2.message();
                        }
                    }
                );
            }
        );

        ioc.run();

        if (success.load()) {
            log_.info("Stratum probe: connected");
            return true;
        }
        if (timed_out.load()) {
            log_.error("Stratum probe failed: timeout (3s)");
        } else {
            log_.error(std::string("Stratum probe failed: ") + (err_msg.empty() ? "unknown" : err_msg));
        }
        return false;
    } catch (const std::exception& e) {
        log_.error(std::string("Stratum probe failed: ") + e.what());
        return false;
    }
}

} // namespace ohmy::pool
