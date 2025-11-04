#pragma once

#include <string_view>

namespace ohmy::logging {

class Logger {
public:
    virtual ~Logger() = default;
    virtual void info(std::string_view msg) = 0;
    virtual void warn(std::string_view msg) = 0;
    virtual void error(std::string_view msg) = 0;
    virtual void debug(std::string_view msg) = 0;
};

} // namespace ohmy::logging
