/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/cuda_types.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace ohmy {
namespace quantum {
namespace cuda {

/**
 * Fused Single-Qubit Gates Kernel
 * 
 * Applies all RY and RZ rotations for a circuit layer in a single kernel launch.
 * This dramatically reduces kernel launch overhead (32 launches → 1 launch).
 * 
 * Strategy:
 * - Each thread block processes one qubit across all states in batch
 * - Shared memory holds rotation parameters for the qubit
 * - Coalesced memory access to state vectors
 * 
 * Performance targets:
 * - Memory bandwidth: >80% of theoretical (350 GB/s on GTX 1660 SUPER)
 * - Occupancy: >75%
 * - Speedup: 15-20× vs sequential gates
 * 
 * @param states        State vectors [batch_size × state_size]
 * @param ry_angles     RY rotation angles [batch_size × num_qubits]
 * @param rz_angles     RZ rotation angles [batch_size × num_qubits]
 * @param batch_size    Number of states being processed
 * @param num_qubits    Number of qubits (16 for qhash)
 * @param state_size    Size of each state vector (2^num_qubits)
 */
__global__ void fused_single_qubit_gates_kernel(
    Complex* states,
    const float* ry_angles,
    const float* rz_angles,
    int batch_size,
    int num_qubits,
    size_t state_size
) {
    // Shared memory for rotation parameters (one set per block)
    __shared__ float shared_ry_sin;
    __shared__ float shared_ry_cos;
    __shared__ float shared_rz_sin;
    __shared__ float shared_rz_cos;
    
    // Block processes one qubit, thread processes one amplitude pair
    int qubit = blockIdx.y;
    if (qubit >= num_qubits) return;
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    // Load rotation angles to shared memory (once per block)
    if (threadIdx.x == 0) {
        float angle_ry = ry_angles[batch_idx * num_qubits + qubit];
        float angle_rz = rz_angles[batch_idx * num_qubits + qubit];

        float ry_sin_local, ry_cos_local;
        __sincosf(angle_ry * 0.5f, &ry_sin_local, &ry_cos_local);

        float rz_sin_local, rz_cos_local;
        __sincosf(angle_rz * 0.5f, &rz_sin_local, &rz_cos_local);

        shared_ry_sin = ry_sin_local;
        shared_ry_cos = ry_cos_local;
        shared_rz_sin = rz_sin_local;
        shared_rz_cos = rz_cos_local;
    }
    __syncthreads();
    
    // Calculate which amplitude pair this thread handles
    size_t num_pairs = state_size / 2;
    size_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx >= num_pairs) return;
    
    // Compute indices for amplitude pair (differ in target qubit bit)
    size_t low_bits = pair_idx & ((1ULL << qubit) - 1);
    size_t high_bits = (pair_idx >> qubit) << (qubit + 1);
    size_t idx0 = high_bits | low_bits;
    size_t idx1 = idx0 | (1ULL << qubit);
    
    // Base offset for this state in the batch
    size_t state_offset = batch_idx * state_size;
    
    // Load amplitudes from global memory
    Complex alpha = states[state_offset + idx0];
    Complex beta = states[state_offset + idx1];
    
    // ========== Apply RY Rotation ==========
    // R_Y(θ) = [[cos(θ/2), -sin(θ/2)],
    //           [sin(θ/2),  cos(θ/2)]]
    float ry_sin = shared_ry_sin;
    float ry_cos = shared_ry_cos;
    
    Complex temp_alpha = cuCsubf(
        cuCmulf(make_cuFloatComplex(ry_cos, 0.0f), alpha),
        cuCmulf(make_cuFloatComplex(ry_sin, 0.0f), beta)
    );
    
    Complex temp_beta = cuCaddf(
        cuCmulf(make_cuFloatComplex(ry_sin, 0.0f), alpha),
        cuCmulf(make_cuFloatComplex(ry_cos, 0.0f), beta)
    );
    
    // ========== Apply RZ Rotation ==========
    // R_Z(θ) = [[e^(-iθ/2), 0        ],
    //           [0,         e^(iθ/2)]]
    float rz_sin = shared_rz_sin;
    float rz_cos = shared_rz_cos;
    
    // e^(-iθ/2) = cos(θ/2) - i*sin(θ/2)
    Complex phase0 = make_cuFloatComplex(rz_cos, -rz_sin);
    // e^(iθ/2) = cos(θ/2) + i*sin(θ/2)  
    Complex phase1 = make_cuFloatComplex(rz_cos, rz_sin);
    
    Complex final_alpha = cuCmulf(phase0, temp_alpha);
    Complex final_beta = cuCmulf(phase1, temp_beta);
    
    // Write results back to global memory
    states[state_offset + idx0] = final_alpha;
    states[state_offset + idx1] = final_beta;
}

/**
 * Optimized CNOT Chain Kernel
 * 
 * Applies the CNOT chain: (0,1), (1,2), (2,3), ..., (14,15)
 * Exploits the linear topology for cache locality and warp efficiency.
 * 
 * Key optimization: Process multiple CNOTs with good data reuse
 * since adjacent qubits in the chain share state vector regions.
 * 
 * Performance targets:
 * - Memory bandwidth: >75% of theoretical
 * - Occupancy: >70%
 * - Speedup: 8-10× vs sequential CNOTs
 * 
 * @param states        State vectors [batch_size × state_size]
 * @param batch_size    Number of states being processed
 * @param num_qubits    Number of qubits (16 for qhash)
 * @param state_size    Size of each state vector (2^num_qubits)
 */
__global__ void cnot_chain_kernel(
    Complex* states,
    int batch_size,
    int num_qubits,
    size_t state_size
) {
    // Each thread handles one amplitude across all CNOTs in sequence
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    size_t state_offset = batch_idx * state_size;
    
    // Apply CNOT chain sequentially (but all amplitudes in parallel)
    // CNOT chain: (0,1), (1,2), (2,3), ..., (num_qubits-2, num_qubits-1)
    for (int control = 0; control < num_qubits - 1; ++control) {
        int target = control + 1;
        
        // Check if control qubit is |1⟩ in this amplitude's basis state
        bool control_is_one = (idx & (1ULL << control)) != 0;
        
        if (control_is_one) {
            // Flip target qubit (swap amplitudes)
            size_t flip_idx = idx ^ (1ULL << target);
            
            // Only perform swap once (from lower index)
            if (idx < flip_idx) {
                Complex temp = states[state_offset + idx];
                states[state_offset + idx] = states[state_offset + flip_idx];
                states[state_offset + flip_idx] = temp;
            }
        }
        
        // Implicit sync: all threads must complete before next CNOT
        __syncthreads();
    }
}

} // namespace cuda
} // namespace quantum
} // namespace ohmy
