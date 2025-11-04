#pragma once

namespace ohmy::system {

// Prints basic CUDA device information (count, name, SM, memory). Best-effort.
void print_cuda_info();

} // namespace ohmy::system