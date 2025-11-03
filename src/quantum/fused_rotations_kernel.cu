/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

/**
 * @file fused_rotations_kernel.cu
 * @brief Fused rotation gate kernels for batched quantum simulation
 * 
 * This file implements gate fusion optimization to reduce cuStateVec API calls.
 * Instead of 64 individual rotation operations (16 qubits × 2 axes × 2 layers),
 * we apply all rotations of the same type in a layer with a single batched kernel.
 * 
 * Performance Impact:
 * - Before: 64 custatevecApplyMatrixBatched calls for rotations
 * - After:  4 custom kernel launches (2 layers × 2 rotation types)
 * - Expected speedup: 10-15× from eliminating kernel launch overhead
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

/**
 * @brief Apply batched Y-rotations to all qubits in a layer
 * 
 * Each thread processes one amplitude pair (|0⟩ and |1⟩ components) for one qubit
 * in one circuit. The rotation matrix for RY(θ) is:
 * 
 * RY(θ) = [ cos(θ/2)  -sin(θ/2) ]
 *         [ sin(θ/2)   cos(θ/2) ]
 * 
 * @param states       Batched state vectors [batch_size × 2^num_qubits]
 * @param angles       Rotation angles [batch_size × num_qubits]
 * @param batch_size   Number of circuits in batch
 * @param num_qubits   Number of qubits per circuit (16 for qhash)
 * @param state_size   Size of each state vector (2^num_qubits = 65536)
 */
__global__ void fused_ry_rotations_kernel(
    cuFloatComplex* states,
    const float* angles,
    const int batch_size,
    const int num_qubits,
    const size_t state_size
) {
    // Grid-stride loop over batch dimension
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    // Each thread handles one qubit's rotation for one batch element
    const int qubit = blockIdx.x;
    if (qubit >= num_qubits) return;
    
    // Thread processes amplitude pairs that differ in this qubit's bit
    const int tid = threadIdx.x;
    const int num_pairs = state_size / 2;
    
    // Offset to this batch's state vector
    cuFloatComplex* batch_state = states + batch_idx * state_size;
    
    // Get rotation angle for this qubit in this batch
    const float theta = angles[batch_idx * num_qubits + qubit];
    const float half_theta = theta * 0.5f;
    const float cos_half = cosf(half_theta);
    const float sin_half = sinf(half_theta);
    
    // Process amplitude pairs in grid-stride pattern
    for (int pair_idx = tid; pair_idx < num_pairs; pair_idx += blockDim.x) {
        // Calculate indices of the two amplitudes affected by this qubit
        // Pattern: pairs differ in the qubit-th bit position
        const size_t lower_mask = (1ULL << qubit) - 1;
        const size_t upper_mask = ~lower_mask;
        
        const size_t idx0 = (pair_idx & lower_mask) | ((pair_idx & upper_mask) << 1);
        const size_t idx1 = idx0 | (1ULL << qubit);
        
        // Load amplitudes
        cuFloatComplex alpha = batch_state[idx0];
        cuFloatComplex beta = batch_state[idx1];
        
        // Apply RY rotation: [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]
        cuFloatComplex new_alpha = make_cuFloatComplex(
            cos_half * alpha.x - sin_half * beta.x,
            cos_half * alpha.y - sin_half * beta.y
        );
        
        cuFloatComplex new_beta = make_cuFloatComplex(
            sin_half * alpha.x + cos_half * beta.x,
            sin_half * alpha.y + cos_half * beta.y
        );
        
        // Store results
        batch_state[idx0] = new_alpha;
        batch_state[idx1] = new_beta;
    }
}

/**
 * @brief Apply batched Z-rotations to all qubits in a layer
 * 
 * RZ(θ) rotation only adds phase to the |1⟩ component:
 * 
 * RZ(θ) = [ e^(-iθ/2)     0      ]
 *         [    0       e^(iθ/2)  ]
 * 
 * In computational basis, this multiplies amplitude by phase factor.
 * We can optimize by only touching |1⟩ components.
 * 
 * @param states       Batched state vectors [batch_size × 2^num_qubits]
 * @param angles       Rotation angles [batch_size × num_qubits]
 * @param batch_size   Number of circuits in batch
 * @param num_qubits   Number of qubits per circuit (16 for qhash)
 * @param state_size   Size of each state vector (2^num_qubits = 65536)
 */
__global__ void fused_rz_rotations_kernel(
    cuFloatComplex* states,
    const float* angles,
    const int batch_size,
    const int num_qubits,
    const size_t state_size
) {
    // Grid-stride loop over batch dimension
    const int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    // Each thread handles one qubit's rotation for one batch element
    const int qubit = blockIdx.x;
    if (qubit >= num_qubits) return;
    
    // Thread processes amplitudes where this qubit is |1⟩
    const int tid = threadIdx.x;
    const int num_ones = state_size / 2;
    
    // Offset to this batch's state vector
    cuFloatComplex* batch_state = states + batch_idx * state_size;
    
    // Get rotation angle for this qubit in this batch
    const float theta = angles[batch_idx * num_qubits + qubit];
    const float half_theta = theta * 0.5f;
    
    // Phase factors: e^(iθ/2) for |1⟩, e^(-iθ/2) for |0⟩
    const float cos_phase = cosf(half_theta);
    const float sin_phase = sinf(half_theta);
    
    // Process amplitudes where qubit is |1⟩
    for (int idx = tid; idx < state_size; idx += blockDim.x) {
        // Check if this qubit is |1⟩ in this basis state
        if (idx & (1ULL << qubit)) {
            // Apply e^(iθ/2) phase
            cuFloatComplex amp = batch_state[idx];
            batch_state[idx] = make_cuFloatComplex(
                cos_phase * amp.x - sin_phase * amp.y,
                sin_phase * amp.x + cos_phase * amp.y
            );
        } else {
            // Apply e^(-iθ/2) phase
            cuFloatComplex amp = batch_state[idx];
            batch_state[idx] = make_cuFloatComplex(
                cos_phase * amp.x + sin_phase * amp.y,
                -sin_phase * amp.x + cos_phase * amp.y
            );
        }
    }
}

// Host wrapper functions (called from custom backend)

extern "C" void launch_fused_ry_rotations(
    cuFloatComplex* d_states,
    const float* d_angles,
    int batch_size,
    int num_qubits,
    size_t state_size,
    cudaStream_t stream
) {
    // Launch configuration: one block per qubit per batch
    dim3 grid(num_qubits, batch_size);
    dim3 block(256); // 256 threads per block for amplitude processing
    
    fused_ry_rotations_kernel<<<grid, block, 0, stream>>>(
        d_states, d_angles, batch_size, num_qubits, state_size
    );
}

extern "C" void launch_fused_rz_rotations(
    cuFloatComplex* d_states,
    const float* d_angles,
    int batch_size,
    int num_qubits,
    size_t state_size,
    cudaStream_t stream
) {
    // Launch configuration: one block per qubit per batch
    dim3 grid(num_qubits, batch_size);
    dim3 block(256); // 256 threads per block for amplitude processing
    
    fused_rz_rotations_kernel<<<grid, block, 0, stream>>>(
        d_states, d_angles, batch_size, num_qubits, state_size
    );
}
