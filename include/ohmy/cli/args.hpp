#pragma once

#include <ohmy/config/types.hpp>

namespace ohmy::cli {

// Parse CLI using cxxopts. Returns ParseResult with optional config path.
ohmy::config::ParseResult parse(int argc, char** argv);

} // namespace ohmy::cli
