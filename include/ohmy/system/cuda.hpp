#pragma once

#include <ohmy/logging/logger.hpp>

namespace ohmy::system {

// Prints basic CUDA device information (count, name, SM, memory). Best-effort.
void print_cuda_info(ohmy::logging::Logger& log);

} // namespace ohmy::system