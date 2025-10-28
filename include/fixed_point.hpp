/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <array>

namespace ohmy::quantum {

/**
 * Fixed-point arithmetic for deterministic consensus across different GPUs.
 * 
 * Uses Q15.16 format (16-bit signed integer with 15 bits for fractional part)
 * matching Qubitcoin's fixedFloat implementation with fpm library.
 * 
 * Format: Q15.16 (1-bit sign, 15-bit fractional)
 * Range: [-1.0, ~0.999969]
 * Precision: ~1.5e-5
 */
class FixedPoint {
public:
    static constexpr int FRACTIONAL_BITS = 15;
    static constexpr int32_t ONE = 1 << FRACTIONAL_BITS;  // 1.0 in fixed-point (32768)
    
    /**
     * Convert double to fixed-point (Q15.16).
     * 
     * @param value Double value to convert (should be in range [-1.0, 1.0] for quantum expectations)
     * @return 16-bit fixed-point representation (stored in int16_t)
     */
    static int16_t from_double(double value);
    
    /**
     * Convert fixed-point to double.
     * 
     * @param fixed 16-bit fixed-point value
     * @return Double representation
     */
    static double to_double(int16_t fixed);
    
    /**
     * Convert multiple doubles to fixed-point.
     * 
     * @param values Vector of double values
     * @return Vector of fixed-point representations
     */
    static std::vector<int16_t> from_doubles(const std::vector<double>& values);
    
    /**
     * Serialize fixed-point values to bytes for hashing.
     * Uses little-endian byte order matching QTC implementation.
     * 
     * @param values Vector of fixed-point values
     * @return Byte array suitable for hashing
     */
    static std::vector<uint8_t> to_bytes(const std::vector<int16_t>& values);
    
    /**
     * Convert quantum expectations to deterministic byte representation.
     * This is the critical function for consensus: double → fixed-point → bytes.
     * 
     * @param expectations Vector of quantum expectation values (from GPU)
     * @return Byte array for XOR fusion with initial hash
     */
    static std::vector<uint8_t> expectations_to_bytes(const std::vector<double>& expectations);
    
private:
    /**
     * Convert single 16-bit value to little-endian bytes.
     */
    static std::array<uint8_t, 2> int16_to_little_endian(int16_t value);
};

/**
 * Quantum result processor for qhash algorithm.
 * Handles the full pipeline: expectations → fixed-point → XOR → SHA256 (not SHA3!).
 */
class QHashProcessor {
public:
    /**
     * Process quantum simulation results into qhash output.
     * 
     * @param initial_hash SHA256 hash of block header (32 bytes)
     * @param expectations Quantum expectation values from simulation
     * @return Final qhash result (32 bytes) using SHA256, not SHA3
     */
    static std::array<uint8_t, 32> compute_qhash(
        const std::array<uint8_t, 32>& initial_hash,
        const std::vector<double>& expectations
    );
    
private:
    /**
     * Compute SHA256 hash (not SHA3 - matching QTC implementation).
     */
    static std::array<uint8_t, 32> sha256(const std::vector<uint8_t>& data);
};

} // namespace ohmy::quantum
