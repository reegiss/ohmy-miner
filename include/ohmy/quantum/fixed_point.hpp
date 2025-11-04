#pragma once

#include <cstdint>
#include <type_traits>

// Q16.15 storage (int32_t) with 64-bit intermediate
// Using MikeLankamp/fpm types in implementation later.
namespace ohmy::quantum {

using fp_storage_t = std::int32_t; // 32-bit storage
constexpr unsigned FP_FRAC_BITS = 15; // Q16.15

} // namespace ohmy::quantum
