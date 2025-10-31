/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <chrono>
#include <ctime>
#include <string>
#include <fmt/format.h>
#include <string_view>

namespace ohmy {
namespace log {

inline std::string now_hms() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return fmt::format("[{:02d}:{:02d}:{:02d}]", tm.tm_hour, tm.tm_min, tm.tm_sec);
}

template <typename... Args>
inline void line(std::string_view fmtstr, Args&&... args) {
    fmt::print("{} ", now_hms());
    fmt::print(fmt::runtime(fmtstr), std::forward<Args>(args)...);
    fmt::print("\n");
}

} // namespace log
} // namespace ohmy
