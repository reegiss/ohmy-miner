/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_MONOLITHIC_KERNEL_CUH
#define OHMY_MINER_MONOLITHIC_KERNEL_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

namespace ohmy {
namespace quantum {

/**
 * @brief Monolithic QTC circuit kernel
 * 
 * Executes the complete QTC circuit (2 layers × [RY+RZ+CNOT]) in a single kernel launch.
 * This eliminates kernel launch overhead which was the primary bottleneck.
 * 
 * Expected performance: 100-500× faster than gate-by-gate approach
 */
__global__ void qtc_circuit_monolithic(
    cuFloatComplex* states,     // [batch_size][65536]
    const float* ry_angles_l0,  // [batch_size][16]
    const float* rz_angles_l0,  // [batch_size][16]
    const float* ry_angles_l1,  // [batch_size][16]
    const float* rz_angles_l1,  // [batch_size][16]
    int batch_size,
    int num_qubits
);

/**
 * @brief Optimized measurement kernel
 * 
 * Computes ⟨σz⟩ for all qubits using warp-level operations
 */
__global__ void measure_monolithic(
    const cuFloatComplex* states,
    float* expectations,
    int batch_size,
    int num_qubits
);

} // namespace quantum
} // namespace ohmy

#endif // OHMY_MINER_MONOLITHIC_KERNEL_CUH
