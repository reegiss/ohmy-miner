#pragma once

#include <ohmy/config/types.hpp>
#include <ohmy/logging/logger.hpp>

namespace ohmy::cli {

// Parse CLI using cxxopts. Writes help/version through provided logger when requested.
ohmy::config::ParseResult parse(int argc, char** argv, ohmy::logging::Logger& log);

} // namespace ohmy::cli
