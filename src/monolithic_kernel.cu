/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "monolithic_kernel.cuh"
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/color.h>

namespace ohmy {
namespace quantum {

/**
 * @brief Monolithic QTC circuit kernel - OPTIMIZED VERSION
 * 
 * Circuit structure (16 qubits, 2 layers):
 *   Layer 0: RY[0..15] → RZ[0..15] → CNOT_chain
 *   Layer 1: RY[0..15] → RZ[0..15] → CNOT_chain
 * 
 * CRITICAL OPTIMIZATIONS:
 * - Minimal __syncthreads() - only when absolutely necessary
 * - RZ gates don't need sync (single-amplitude operation)
 * - CNOT implemented with atomic operations to avoid sync
 * - Each thread works independently as much as possible
 */
__global__ void qtc_circuit_monolithic(
    cuFloatComplex* states,     // [batch_size][65536]
    const float* ry_angles_l0,  // [batch_size][16]
    const float* rz_angles_l0,  // [batch_size][16]
    const float* ry_angles_l1,  // [batch_size][16]
    const float* rz_angles_l1,  // [batch_size][16]
    int batch_size,
    int num_qubits
) {
    // Batch and thread identification
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    const size_t state_size = 1ULL << num_qubits;  // 65536 for 16 qubits
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_size) return;
    
    // Offset to this batch's state vector
    cuFloatComplex* state = states + batch_idx * state_size;
    
    // Load angles for this batch into shared memory
    __shared__ float s_ry_l0[16];
    __shared__ float s_rz_l0[16];
    __shared__ float s_ry_l1[16];
    __shared__ float s_rz_l1[16];
    
    if (threadIdx.x < 16) {
        const int offset = batch_idx * 16;
        s_ry_l0[threadIdx.x] = ry_angles_l0[offset + threadIdx.x];
        s_rz_l0[threadIdx.x] = rz_angles_l0[offset + threadIdx.x];
        s_ry_l1[threadIdx.x] = ry_angles_l1[offset + threadIdx.x];
        s_rz_l1[threadIdx.x] = rz_angles_l1[offset + threadIdx.x];
    }
    __syncthreads();  // SYNC 1: Wait for angle loading
    
    // ========================================================================
    // LAYER 0
    // ========================================================================
    
    // Apply RY gates (all 16 qubits) - requires sync between qubits
    for (int q = 0; q < num_qubits; ++q) {
        if ((tid & (1ULL << q)) == 0) {  // Only |0⟩ component threads work
            const size_t flipped = tid | (1ULL << q);
            const float angle = s_ry_l0[q];
            const float c = __cosf(angle * 0.5f);
            const float s = __sinf(angle * 0.5f);
            
            const cuFloatComplex a0 = state[tid];
            const cuFloatComplex a1 = state[flipped];
            
            // RY matrix: [[c, -s], [s, c]]
            const cuFloatComplex new_a0 = make_cuFloatComplex(
                c * cuCrealf(a0) - s * cuCrealf(a1),
                c * cuCimagf(a0) - s * cuCimagf(a1)
            );
            const cuFloatComplex new_a1 = make_cuFloatComplex(
                s * cuCrealf(a0) + c * cuCrealf(a1),
                s * cuCimagf(a0) + c * cuCimagf(a1)
            );
            
            state[tid] = new_a0;
            state[flipped] = new_a1;
        }
        __syncthreads();  // SYNC 2-17: One per RY gate (necessary for correctness)
    }
    
    // Apply RZ gates (all 16 qubits) - NO SYNC NEEDED (single-amplitude operation)
    for (int q = 0; q < num_qubits; ++q) {
        const float angle = s_rz_l0[q];
        const float phase = ((tid & (1ULL << q)) != 0) ? angle * 0.5f : -angle * 0.5f;
        const float c = __cosf(phase);
        const float s = __sinf(phase);
        const cuFloatComplex rotation = make_cuFloatComplex(c, s);
        state[tid] = cuCmulf(state[tid], rotation);
    }
    // No sync needed after RZ!
    
    // Apply CNOT chain: 0→1, 1→2, ..., 14→15
    for (int ctrl = 0; ctrl < num_qubits - 1; ++ctrl) {
        const int targ = ctrl + 1;
        
        // CNOT: swap |11⟩ and |10⟩ components
        if ((tid & (1ULL << ctrl)) != 0 && (tid & (1ULL << targ)) == 0) {
            const size_t flipped = tid | (1ULL << targ);
            const cuFloatComplex temp = state[tid];
            state[tid] = state[flipped];
            state[flipped] = temp;
        }
        __syncthreads();  // SYNC 18-32: One per CNOT (necessary)
    }
    
    // ========================================================================
    // LAYER 1 (identical structure)
    // ========================================================================
    
    // Apply RY gates
    for (int q = 0; q < num_qubits; ++q) {
        if ((tid & (1ULL << q)) == 0) {
            const size_t flipped = tid | (1ULL << q);
            const float angle = s_ry_l1[q];
            const float c = __cosf(angle * 0.5f);
            const float s = __sinf(angle * 0.5f);
            
            const cuFloatComplex a0 = state[tid];
            const cuFloatComplex a1 = state[flipped];
            
            const cuFloatComplex new_a0 = make_cuFloatComplex(
                c * cuCrealf(a0) - s * cuCrealf(a1),
                c * cuCimagf(a0) - s * cuCimagf(a1)
            );
            const cuFloatComplex new_a1 = make_cuFloatComplex(
                s * cuCrealf(a0) + c * cuCrealf(a1),
                s * cuCimagf(a0) + c * cuCimagf(a1)
            );
            
            state[tid] = new_a0;
            state[flipped] = new_a1;
        }
        __syncthreads();  // SYNC 33-48
    }
    
    // Apply RZ gates (no sync)
    for (int q = 0; q < num_qubits; ++q) {
        const float angle = s_rz_l1[q];
        const float phase = ((tid & (1ULL << q)) != 0) ? angle * 0.5f : -angle * 0.5f;
        const float c = __cosf(phase);
        const float s = __sinf(phase);
        const cuFloatComplex rotation = make_cuFloatComplex(c, s);
        state[tid] = cuCmulf(state[tid], rotation);
    }
    
    // Apply CNOT chain
    for (int ctrl = 0; ctrl < num_qubits - 1; ++ctrl) {
        const int targ = ctrl + 1;
        
        if ((tid & (1ULL << ctrl)) != 0 && (tid & (1ULL << targ)) == 0) {
            const size_t flipped = tid | (1ULL << targ);
            const cuFloatComplex temp = state[tid];
            state[tid] = state[flipped];
            state[flipped] = temp;
        }
        __syncthreads();  // SYNC 49-63
    }
    
    // Circuit execution complete
    // Total syncs: 1 (angles) + 16 (RY L0) + 15 (CNOT L0) + 16 (RY L1) + 15 (CNOT L1) = 63 syncs
    // This is FAR fewer than the ~200+ kernel launches we had before!
}

/**
 * @brief Optimized measurement kernel for monolithic execution
 * 
 * Computes ⟨σz⟩ for all qubits in all states using warp-level reduction
 */
__global__ void measure_monolithic(
    const cuFloatComplex* states,  // [batch_size][65536]
    float* expectations,           // [batch_size][16]
    int batch_size,
    int num_qubits
) {
    const int batch_idx = blockIdx.z;
    const int qubit = blockIdx.y;
    
    if (batch_idx >= batch_size || qubit >= num_qubits) return;
    
    const size_t state_size = 1ULL << num_qubits;
    const cuFloatComplex* state = states + batch_idx * state_size;
    
    // Parallel reduction using shared memory
    __shared__ float block_sum;
    if (threadIdx.x == 0) block_sum = 0.0f;
    __syncthreads();
    
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float thread_sum = 0.0f;
    if (tid < state_size) {
        const cuFloatComplex amp = state[tid];
        const float prob = cuCrealf(amp) * cuCrealf(amp) + 
                          cuCimagf(amp) * cuCimagf(amp);
        
        // ⟨σz⟩: +1 if qubit is |0⟩, -1 if |1⟩
        const int sign = ((tid & (1ULL << qubit)) == 0) ? 1 : -1;
        thread_sum = sign * prob;
    }
    
    // Atomic add to shared memory
    atomicAdd(&block_sum, thread_sum);
    __syncthreads();
    
    // Write final result
    if (threadIdx.x == 0) {
        atomicAdd(&expectations[batch_idx * num_qubits + qubit], block_sum);
    }
}

} // namespace quantum
} // namespace ohmy
