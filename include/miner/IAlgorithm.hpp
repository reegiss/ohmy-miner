// include/miner/IAlgorithm.hpp

/*
 * Copyright (C) 2025 Regis Araujo Melo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <cstdint>

namespace miner {

// Forward declaration for device context
struct CudaDevice; 

// Abstract interface for a mining algorithm plugin.
// This is a pure C++ interface. The C-style Plugin.h provides the ABI boundary.
class IAlgorithm {
public:
    virtual ~IAlgorithm() = default;

    // Returns the name of the algorithm (e.g., "x11", "sha256d").
    virtual const char* getName() const = 0;

    // Initializes the algorithm for a specific CUDA device.
    // Allocates device memory, copies constants, etc.
    // Returns 0 on success, non-zero on failure.
    virtual int init(const CudaDevice& device) = 0;

    // The main hashing function.
    // This will be called in a tight loop by the core.
    // 'header' is the 80-byte block header.
    // 'nonce_start' is the beginning of the nonce range for this call.
    // 'nonce_end' is the end of the nonce range.
    // 'found_nonce' is an output parameter to store a valid nonce if found.
    // Returns true if a valid nonce was found, false otherwise.
    virtual bool hash(
        const uint8_t* header,
        uint32_t nonce_start,
        uint32_t nonce_end,
        uint32_t* found_nonce
    ) = 0;

    // Releases all resources used by the algorithm on the device.
    virtual void destroy() = 0;
};

} // namespace miner