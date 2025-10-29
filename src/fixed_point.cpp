/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "fixed_point.hpp"
#include <cmath>
#include <stdexcept>
#include <openssl/sha.h>
#include <cstring>

namespace ohmy::quantum {

int16_t FixedPoint::from_double(double value) {
    // Clamp to valid range [-1.0, 1.0] for quantum expectations
    if (value > 1.0) value = 1.0;
    if (value < -1.0) value = -1.0;
    
    // Convert: multiply by 2^15 and round
    // Q15.16 format: 1 sign bit + 15 fractional bits in int16_t
    double scaled = value * static_cast<double>(ONE);
    int32_t rounded = static_cast<int32_t>(std::round(scaled));
    
    // Clamp to int16_t range
    if (rounded > 32767) rounded = 32767;
    if (rounded < -32768) rounded = -32768;
    
    return static_cast<int16_t>(rounded);
}

double FixedPoint::to_double(int16_t fixed) {
    return static_cast<double>(fixed) / static_cast<double>(ONE);
}

std::vector<int16_t> FixedPoint::from_doubles(const std::vector<double>& values) {
    std::vector<int16_t> result;
    result.reserve(values.size());
    
    for (double value : values) {
        result.push_back(from_double(value));
    }
    
    return result;
}

std::array<uint8_t, 2> FixedPoint::int16_to_little_endian(int16_t value) {
    std::array<uint8_t, 2> bytes;
    
    // Convert to little-endian (matching QTC implementation)
    bytes[0] = static_cast<uint8_t>(value & 0xFF);
    bytes[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    
    return bytes;
}

std::vector<uint8_t> FixedPoint::to_bytes(const std::vector<int16_t>& values) {
    std::vector<uint8_t> result;
    result.reserve(values.size() * 2);  // 2 bytes per int16_t
    
    for (int16_t value : values) {
        auto bytes = int16_to_little_endian(value);
        result.insert(result.end(), bytes.begin(), bytes.end());
    }
    
    return result;
}

std::vector<uint8_t> FixedPoint::expectations_to_bytes(const std::vector<double>& expectations) {
    // Critical consensus path: double → fixed-point → bytes (little-endian)
    auto fixed_values = from_doubles(expectations);
    return to_bytes(fixed_values);
}

// ============================================================================
// QHashProcessor Implementation - Matching Qubitcoin's qhash algorithm
// ============================================================================

std::array<uint8_t, 32> QHashProcessor::sha256(const std::vector<uint8_t>& data) {
    std::array<uint8_t, 32> result;
    SHA256(data.data(), data.size(), result.data());
    return result;
}

std::array<uint8_t, 32> QHashProcessor::compute_qhash(
    const std::array<uint8_t, 32>& initial_hash,
    const std::vector<double>& expectations,
    uint32_t nTime
) {
    // CRITICAL: QTC qhash algorithm with temporal fork handling
    // 1. Start with initial_hash (already computed SHA256 of block header)
    // 2. Convert quantum expectations to fixed-point bytes (little-endian)
    // 3. Concatenate: initial_hash + quantum_bytes
    // 4. Check invalidation rule (temporal fork)
    // 5. Apply SHA256 (not SHA3!)
    
    // Step 1: Convert expectations to deterministic bytes
    auto quantum_bytes = FixedPoint::expectations_to_bytes(expectations);
    
    // Step 2: CRITICAL - Invalidation rule for pathological quantum states
    // Count zero bytes in quantum_bytes (fixed-point representation)
    // Reference: qhash.cpp lines 147-153
    size_t zeroes = 0;
    for (uint8_t byte : quantum_bytes) {
        if (byte == 0) ++zeroes;
    }
    
    // Temporal fork: different thresholds at different timestamps
    // nTime >= 1758762000 (Aug 25, 2025): invalidate if ALL bytes zero (32 zeros)
    // nTime >= 1754220531 (Jan 27, 2025): invalidate if 25%+ bytes zero (8+ zeros)
    const size_t total_bytes = quantum_bytes.size();  // 16 qubits × 2 bytes = 32 bytes
    
    bool invalid = false;
    if (nTime >= 1758762000) {
        // Post Aug-2025 fork: invalidate only if completely zero
        invalid = (zeroes >= total_bytes);
    } else if (nTime >= 1754220531) {
        // Post Jan-2025 fork: invalidate if 25%+ zeros
        invalid = (zeroes >= total_bytes / 4);
    }
    
    if (invalid) {
        // Return invalid hash (all 0xFF)
        std::array<uint8_t, 32> invalid_hash;
        invalid_hash.fill(0xFF);
        return invalid_hash;
    }
    
    // Step 3: Build data for final hash: initial_hash + quantum_bytes
    std::vector<uint8_t> combined;
    combined.reserve(32 + quantum_bytes.size());
    combined.insert(combined.end(), initial_hash.begin(), initial_hash.end());
    combined.insert(combined.end(), quantum_bytes.begin(), quantum_bytes.end());
    
    // Step 4: Apply SHA256 for final hash (valid quantum state)
    return sha256(combined);
}

} // namespace ohmy::quantum
