/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

// Minimal CUDA portability shims for fixed-point on device code.
// Note: The project currently uses ohmy::fixed_point for consensus-critical
// conversions (see include/ohmy/fixed_point.hpp). This header is provided to
// ease future integration with external fixed-point libraries if needed.

#ifdef __CUDACC__
  #define FPM_HOSTDEV __host__ __device__
  #define FPM_INLINE __forceinline__
#else
  #define FPM_HOSTDEV
  #define FPM_INLINE inline
#endif

// If integrating third-party fpm library in the future, include it here.
// #include <fpm/fixed.hpp>
// #include <fpm/math.hpp>

