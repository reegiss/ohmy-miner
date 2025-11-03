/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

/**
 * @file fpm_consensus_device.cuh
 * @brief Device-side fixed-point conversion for Qubitcoin consensus
 * 
 * CRITICAL: This is the MOST IMPORTANT file for consensus correctness.
 * Any deviation from the reference 'fpm' library will result in 100% rejected shares.
 * 
 * The Qubitcoin qhash algorithm uses Q15 fixed-point format:
 * - Type: int32_t
 * - Fractional bits: 15
 * - Scale factor: 2^15 = 32768
 * - Range: [-65536.0, 65535.999969482421875]
 * 
 * Reference: https://github.com/MikeLankamp/fpm
 */

/**
 * @brief Convert double to Q15 fixed-point (int32_t with 15 fractional bits)
 * 
 * This function MUST be bit-exact with the reference fpm library implementation:
 * 
 * ```cpp
 * using Q15 = fpm::fixed<int32_t, int32_t, 15>;
 * Q15 fixed_val = Q15(double_input);
 * int32_t raw = fixed_val.raw_value();
 * ```
 * 
 * Implementation notes:
 * - Uses round() for banker's rounding (round-to-even)
 * - CUDA's round() should match C++ std::round() behavior
 * - Scale factor: 32768.0 = 2^15
 * 
 * @param val Input double value (typically quantum expectation -1.0 to +1.0)
 * @return int32_t Raw Q15 fixed-point representation
 * 
 * @warning DO NOT MODIFY without validating against golden test!
 */
__device__ __forceinline__ int32_t convert_q15_device(double val)
{
    // Scale by 2^15
    double scaled = val * 32768.0;
    
    // Round to nearest integer (banker's rounding)
    // CUDA's round() implements round-to-even for .5 cases
    double rounded = round(scaled);
    
    // Cast to int32_t (truncation for integer part)
    return static_cast<int32_t>(rounded);
}

/**
 * @brief Batch conversion for multiple expectation values
 * 
 * Processes array of quantum expectations and converts to Q15 format.
 * Used in kernel for converting all <σ_z> measurements.
 * 
 * @param expectations Input array of double expectations [num_qubits]
 * @param fixed_outputs Output array of Q15 fixed-point values [num_qubits]
 * @param num_qubits Number of values to convert
 */
__device__ __forceinline__ void convert_q15_batch_device(
    const double* expectations,
    int32_t* fixed_outputs,
    int num_qubits
) {
    for (int i = 0; i < num_qubits; ++i) {
        fixed_outputs[i] = convert_q15_device(expectations[i]);
    }
}

/**
 * @brief Alternative conversion using float (lower precision)
 * 
 * WARNING: This may NOT be consensus-compatible!
 * Only use for performance comparison or if double version fails validation.
 * 
 * @param val Input float value
 * @return int32_t Q15 representation
 */
__device__ __forceinline__ int32_t convert_q15_device_float(float val)
{
    float scaled = val * 32768.0f;
    return static_cast<int32_t>(roundf(scaled));
}

/**
 * @brief Debug helper: Check if value is in valid Q15 range
 * 
 * @param val Input double
 * @return bool True if value can be represented in Q15 without overflow
 */
__device__ __forceinline__ bool is_valid_q15_range(double val)
{
    // Q15 range with int32_t: [-2^16, 2^16 - 2^-15]
    return (val >= -65536.0) && (val <= 65535.999969482421875);
}

/**
 * @brief Reverse conversion: Q15 back to double (for debugging)
 * 
 * @param raw_value Q15 fixed-point int32_t
 * @return double Floating-point representation
 */
__device__ __forceinline__ double q15_to_double_device(int32_t raw_value)
{
    return static_cast<double>(raw_value) / 32768.0;
}
