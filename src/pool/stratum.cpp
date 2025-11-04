#include <ohmy/pool/stratum.hpp>

#include <asio/io_context.hpp>

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

} // namespace ohmy::pool
