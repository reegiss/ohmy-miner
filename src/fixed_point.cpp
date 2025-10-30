/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/fixed_point.hpp"

// Fixed-point is header-only template implementation
// This file exists only to satisfy build system requirements
// All implementations are in fixed_point.hpp

namespace ohmy {
// Explicit instantiation of common types to speed up compilation
template class fixed_point<int32_t, 15>;
template class fixed_point<int64_t, 31>;
} // namespace ohmy
