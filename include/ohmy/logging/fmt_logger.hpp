#pragma once

#include <ohmy/logging/logger.hpp>

#include <atomic>

namespace ohmy::logging {

class FmtLogger : public Logger {
public:
    explicit FmtLogger(bool enable_debug = false) : enable_debug_(enable_debug) {}
    void info(std::string_view msg) override;
    void warn(std::string_view msg) override;
    void error(std::string_view msg) override;
    void debug(std::string_view msg) override;

    void set_debug(bool v) { enable_debug_.store(v); }

private:
    std::atomic<bool> enable_debug_{false};
};

} // namespace ohmy::logging
