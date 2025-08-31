// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef MINER_ENDIAN_UTIL_H_
#define MINER_ENDIAN_UTIL_H_

#include <cstdint>
#include <algorithm>

// Performs an in-place byte swap on each 4-byte word (dword) within a buffer.
// This is required for fields like prev_hash and merkle_root, converting them
// from the big-endian format of a hex string to the little-endian dwords
// required by the hashing algorithm.
// Example: 0xAABBCCDD becomes 0xDDCCBBAA.
inline void swap_endian_words(uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; i += 4) {
        std::swap(data[i], data[i + 3]);
        std::swap(data[i + 1], data[i + 2]);
    }
}

#endif // MINER_ENDIAN_UTIL_H_