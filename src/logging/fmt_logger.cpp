#include <ohmy/logging/fmt_logger.hpp>

#include <fmt/core.h>

namespace ohmy::logging {

static inline void print_line_stdout(std::string_view level, std::string_view msg) {
    fmt::print("[{}] {}\n", level, msg);
}

static inline void print_line_stderr(std::string_view level, std::string_view msg) {
    fmt::print(stderr, "[{}] {}\n", level, msg);
}

void FmtLogger::info(std::string_view msg) { print_line_stdout("INFO", msg); }
void FmtLogger::warn(std::string_view msg) { print_line_stdout("WARN", msg); }
void FmtLogger::error(std::string_view msg) { print_line_stderr("ERROR", msg); }
void FmtLogger::debug(std::string_view msg) {
    if (enable_debug_.load()) print_line_stdout("DEBUG", msg);
}

} // namespace ohmy::logging
