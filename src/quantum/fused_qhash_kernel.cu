/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

/**
 * @file fused_qhash_kernel.cu
 * @brief Monolithic on-the-fly qhash kernel with O(1) VRAM per nonce
 * 
 * ARCHITECTURE PIVOT: This kernel replaces cuStateVec entirely.
 * - Memory: O(1) VRAM per nonce (not O(2^n))
 * - Strategy: 1 Block = 1 Nonce
 * - State vector: 2^16 amplitudes (1MB) in global memory per block
 * - Shared memory: Reductions and intra-block communication
 * 
 * TARGET: 36 MH/s on GTX 1660 SUPER (vs current 3.33 KH/s)
 * 
 * CRITICAL: Uses validated consensus functions:
 * - convert_q15_device() from fpm_consensus_device.cuh (Fase 2)
 * - sha256d_80_bytes() from sha256_device.cuh (Fase 3)
 */

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdint.h>
#include <math.h>

#include "fpm_consensus_device.cuh"
#include "sha256_device.cuh"

// Constants for qhash (16 qubits, 2 layers)
#define QHASH_NUM_QUBITS 16
#define QHASH_STATE_SIZE 65536  // 2^16
#define QHASH_NUM_LAYERS 2
#define QHASH_GATES_PER_LAYER 32  // 16 RY + 16 RZ
#define QHASH_NUM_CNOTS 8

/**
 * @brief Apply single-qubit rotation gate to state vector
 * 
 * Each thread processes a subset of amplitude pairs.
 * RY(θ): [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]
 * RZ(θ): [e^(-iθ/2), 0; 0, e^(iθ/2)]
 */
__device__ void apply_rotation_gate(
    cuDoubleComplex* state,
    int qubit,
    double angle,
    bool is_y_axis,
    int tid,
    int block_size
) {
    const double half_angle = angle * 0.5;
    const double cos_half = cos(half_angle);
    const double sin_half = sin(half_angle);
    
    if (is_y_axis) {
        // RY rotation
        for (int idx = tid; idx < QHASH_STATE_SIZE; idx += block_size) {
            // Check if this thread handles the lower index of the pair
            if ((idx & (1 << qubit)) == 0) {
                int idx0 = idx;
                int idx1 = idx | (1 << qubit);
                
                cuDoubleComplex alpha = state[idx0];
                cuDoubleComplex beta = state[idx1];
                
                // Apply RY matrix
                state[idx0] = make_cuDoubleComplex(
                    cos_half * alpha.x - sin_half * beta.x,
                    cos_half * alpha.y - sin_half * beta.y
                );
                state[idx1] = make_cuDoubleComplex(
                    sin_half * alpha.x + cos_half * beta.x,
                    sin_half * alpha.y + cos_half * beta.y
                );
            }
        }
    } else {
        // RZ rotation (phase gate)
        const double cos_phase = cos_half;
        const double sin_phase = sin_half;
        
        for (int idx = tid; idx < QHASH_STATE_SIZE; idx += block_size) {
            cuDoubleComplex amp = state[idx];
            
            if (idx & (1 << qubit)) {
                // Apply e^(+iθ/2) phase
                state[idx] = make_cuDoubleComplex(
                    cos_phase * amp.x - sin_phase * amp.y,
                    sin_phase * amp.x + cos_phase * amp.y
                );
            } else {
                // Apply e^(-iθ/2) phase
                state[idx] = make_cuDoubleComplex(
                    cos_phase * amp.x + sin_phase * amp.y,
                    -sin_phase * amp.x + cos_phase * amp.y
                );
            }
        }
    }
}

/**
 * @brief Apply CNOT gate to state vector
 * 
 * CNOT swaps amplitudes when control qubit is |1⟩
 */
__device__ void apply_cnot_gate(
    cuDoubleComplex* state,
    int control,
    int target,
    int tid,
    int block_size
) {
    for (int idx = tid; idx < QHASH_STATE_SIZE; idx += block_size) {
        // Only act when control qubit is |1⟩
        if (idx & (1 << control)) {
            // Check if we're at the lower index of the swap pair
            if ((idx & (1 << target)) == 0) {
                int idx0 = idx;
                int idx1 = idx | (1 << target);
                
                // Swap amplitudes
                cuDoubleComplex temp = state[idx0];
                state[idx0] = state[idx1];
                state[idx1] = temp;
            }
        }
    }
}

/**
 * @brief Extract rotation angles from hash using qhash parametrization
 * 
 * Reference: Qubitcoin qhash specification
 * - Uses nibbles (4-bit chunks) from hash
 * - Fork #4 temporal flag: nTime >= 1758762000
 */
__device__ void extract_angles(
    const uint32_t hash[8],
    uint32_t nTime,
    double angles[QHASH_GATES_PER_LAYER * QHASH_NUM_LAYERS]
) {
    // Convert hash to bytes
    uint8_t hash_bytes[32];
    for (int i = 0; i < 8; i++) {
        hash_bytes[i * 4 + 0] = (hash[i] >> 24) & 0xFF;
        hash_bytes[i * 4 + 1] = (hash[i] >> 16) & 0xFF;
        hash_bytes[i * 4 + 2] = (hash[i] >> 8) & 0xFF;
        hash_bytes[i * 4 + 3] = hash[i] & 0xFF;
    }
    
    // Extract nibbles
    uint8_t nibbles[64];
    for (int i = 0; i < 32; i++) {
        nibbles[i * 2 + 0] = (hash_bytes[i] >> 4) & 0xF;  // High nibble
        nibbles[i * 2 + 1] = hash_bytes[i] & 0xF;          // Low nibble
    }
    
    // Temporal flag for Fork #4
    const int temporal_flag = (nTime >= 1758762000) ? 1 : 0;
    
    // Generate angles for 2 layers × 16 qubits × 2 rotations
    for (int layer = 0; layer < QHASH_NUM_LAYERS; layer++) {
        for (int qubit = 0; qubit < QHASH_NUM_QUBITS; qubit++) {
            // RY angle
            int ry_idx = (2 * layer * QHASH_NUM_QUBITS + qubit) % 64;
            uint8_t ry_nibble = nibbles[ry_idx];
            angles[layer * QHASH_GATES_PER_LAYER + qubit] = 
                -(2.0 * ry_nibble + temporal_flag) * M_PI / 32.0;
            
            // RZ angle
            int rz_idx = ((2 * layer + 1) * QHASH_NUM_QUBITS + qubit) % 64;
            uint8_t rz_nibble = nibbles[rz_idx];
            angles[layer * QHASH_GATES_PER_LAYER + QHASH_NUM_QUBITS + qubit] = 
                -(2.0 * rz_nibble + temporal_flag) * M_PI / 32.0;
        }
    }
}

/**
 * @brief Monolithic qhash kernel - 1 Block = 1 Nonce
 * 
 * Grid: (batch_size, 1, 1)
 * Block: (256, 1, 1) - adjustable for occupancy
 * 
 * Shared memory layout:
 * - 32 bytes: H_initial (SHA256 hash)
 * - 256 bytes: angles (64 doubles)
 * - 32 KB: partial_sums (16 qubits × 256 threads × 8 bytes)
 * - 64 bytes: q15_results (16 × int32_t)
 * Total: ~33 KB per block
 */
__global__ void fused_qhash_kernel(
    cuDoubleComplex* d_state_vectors,  // Global: batch_size × 1MB each
    const uint8_t* d_header_template,  // Template: 76 bytes (before nonce)
    uint32_t nTime,                    // Block timestamp
    uint64_t nonce_start,              // Starting nonce
    uint32_t target_compact,           // Difficulty target (compact form)
    uint32_t* d_result_buffer,         // Output: valid nonces
    uint32_t* d_result_count           // Atomic counter
) {
    // ===== 1. BLOCK SETUP (1 Block = 1 Nonce) =====
    
    const int nonce_idx = blockIdx.x;
    const uint64_t nonce = nonce_start + nonce_idx;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Pointer to this block's state vector (1MB in global memory)
    cuDoubleComplex* my_sv = d_state_vectors + (nonce_idx * QHASH_STATE_SIZE);
    
    // Shared memory allocation
    extern __shared__ uint8_t s_mem[];
    uint32_t* s_h_initial = (uint32_t*)s_mem;                          // 32 bytes
    double* s_angles = (double*)(s_mem + 32);                          // 512 bytes (64 doubles)
    double* s_partial_sums = (double*)(s_mem + 32 + 512);              // 32768 bytes
    int32_t* s_q15_results = (int32_t*)(s_mem + 32 + 512 + 32768);    // 64 bytes
    
    // ===== 2. HASHING & PARAMETRIZATION (Thread 0 only) =====
    
    if (tid == 0) {
        // Build 80-byte header with nonce
        uint8_t header[80];
        
        // Copy template (first 76 bytes)
        for (int i = 0; i < 76; i++) {
            header[i] = d_header_template[i];
        }
        
        // Append nonce (4 bytes, little-endian)
        header[76] = (nonce >> 0) & 0xFF;
        header[77] = (nonce >> 8) & 0xFF;
        header[78] = (nonce >> 16) & 0xFF;
        header[79] = (nonce >> 24) & 0xFF;
        
        // Compute H_initial = SHA256d(header)
        sha256d_80_bytes(header, s_h_initial);
        
        // Extract rotation angles from hash
        extract_angles(s_h_initial, nTime, s_angles);
    }
    __syncthreads();  // Publish H_initial and angles to all threads
    
    // ===== 3. STATE VECTOR INITIALIZATION (Parallel) =====
    
    // Initialize to |0...0⟩ state
    for (int i = tid; i < QHASH_STATE_SIZE; i += block_size) {
        my_sv[i] = (i == 0) ? make_cuDoubleComplex(1.0, 0.0) 
                             : make_cuDoubleComplex(0.0, 0.0);
    }
    __syncthreads();
    
    // ===== 4. QUANTUM GATE APPLICATION (Sequential with sync) =====
    
    // Apply 2 layers of rotations
    for (int layer = 0; layer < QHASH_NUM_LAYERS; layer++) {
        // Apply all RY rotations for this layer
        for (int q = 0; q < QHASH_NUM_QUBITS; q++) {
            double angle = s_angles[layer * QHASH_GATES_PER_LAYER + q];
            apply_rotation_gate(my_sv, q, angle, true, tid, block_size);
            __syncthreads();  // Ensure gate completes before next gate
        }
        
        // Apply all RZ rotations for this layer
        for (int q = 0; q < QHASH_NUM_QUBITS; q++) {
            double angle = s_angles[layer * QHASH_GATES_PER_LAYER + QHASH_NUM_QUBITS + q];
            apply_rotation_gate(my_sv, q, angle, false, tid, block_size);
            __syncthreads();
        }
    }
    
    // Apply CNOT ladder (standard qhash pattern)
    for (int i = 0; i < QHASH_NUM_CNOTS; i++) {
        int control = i;
        int target = i + 1;
        apply_cnot_gate(my_sv, control, target, tid, block_size);
        __syncthreads();
    }
    
    // ===== 5. EXPECTATION VALUE CALCULATION (Parallel reduction) =====
    
    for (int q = 0; q < QHASH_NUM_QUBITS; q++) {
        double my_sum = 0.0;
        
        // Each thread computes partial sum
        for (int i = tid; i < QHASH_STATE_SIZE; i += block_size) {
            cuDoubleComplex amp = my_sv[i];
            double mag_sq = amp.x * amp.x + amp.y * amp.y;
            
            // Sign based on qubit state: + for |0⟩, - for |1⟩
            if (i & (1 << q)) {
                my_sum -= mag_sq;
            } else {
                my_sum += mag_sq;
            }
        }
        
        // Store in shared memory
        s_partial_sums[q * block_size + tid] = my_sum;
        __syncthreads();
        
        // ===== 6. REDUCTION & CONSENSUS (Thread 0 only) =====
        
        if (tid == 0) {
            double total_sum = 0.0;
            for (int j = 0; j < block_size; j++) {
                total_sum += s_partial_sums[q * block_size + j];
            }
            
            // Apply validated consensus conversion (Fase 2)
            s_q15_results[q] = convert_q15_device(total_sum);
        }
        __syncthreads();
    }
    
    // ===== 7. FINAL HASHING & DIFFICULTY CHECK (Thread 0 only) =====
    
    if (tid == 0) {
        // Form S_quantum from Q15 results (64 bytes)
        uint8_t s_quantum[64];
        for (int i = 0; i < QHASH_NUM_QUBITS; i++) {
            int32_t q15_val = s_q15_results[i];
            s_quantum[i * 4 + 0] = (q15_val >> 0) & 0xFF;
            s_quantum[i * 4 + 1] = (q15_val >> 8) & 0xFF;
            s_quantum[i * 4 + 2] = (q15_val >> 16) & 0xFF;
            s_quantum[i * 4 + 3] = (q15_val >> 24) & 0xFF;
        }
        
        // XOR: Result = H_initial ⊕ S_quantum
        uint32_t result_xor[8];
        for (int i = 0; i < 8; i++) {
            uint32_t s_quantum_word = 
                ((uint32_t)s_quantum[i * 4 + 0] << 0) |
                ((uint32_t)s_quantum[i * 4 + 1] << 8) |
                ((uint32_t)s_quantum[i * 4 + 2] << 16) |
                ((uint32_t)s_quantum[i * 4 + 3] << 24);
            result_xor[i] = s_h_initial[i] ^ s_quantum_word;
        }
        
        // Final SHA256 (TODO: implement or use existing)
        // For now, use Result_XOR directly for difficulty check
        
        // Check difficulty (simple target comparison for now)
        // TODO: Implement proper compact target expansion and comparison
        bool passes_difficulty = (result_xor[7] <= target_compact);
        
        if (passes_difficulty) {
            // Atomically claim output slot
            uint32_t result_idx = atomicAdd(d_result_count, 1);
            
            // Write winning nonce
            if (result_idx < 1024) {  // Buffer overflow protection
                d_result_buffer[result_idx] = (uint32_t)nonce;
            }
        }
    }
}

// Host wrapper function
extern "C" void launch_fused_qhash_kernel(
    cuDoubleComplex* d_state_vectors,
    const uint8_t* d_header_template,
    uint32_t nTime,
    uint64_t nonce_start,
    uint32_t target_compact,
    uint32_t* d_result_buffer,
    uint32_t* d_result_count,
    int batch_size,
    int block_size,
    cudaStream_t stream
) {
    // Calculate shared memory requirement
    size_t shared_mem_size = 
        32 +                                    // H_initial
        512 +                                   // angles (64 doubles)
        (QHASH_NUM_QUBITS * block_size * 8) +  // partial_sums
        64;                                     // q15_results
    
    dim3 grid(batch_size, 1, 1);
    dim3 block(block_size, 1, 1);
    
    fused_qhash_kernel<<<grid, block, shared_mem_size, stream>>>(
        d_state_vectors,
        d_header_template,
        nTime,
        nonce_start,
        target_compact,
        d_result_buffer,
        d_result_count
    );
}

/**
 * @brief Debug version of kernel with intermediate value outputs
 * 
 * This kernel runs ONLY for blockIdx.x == 0 and writes all intermediate
 * computational steps to debug buffers for golden value validation.
 */
__global__ void fused_qhash_kernel_debug(
    cuDoubleComplex* d_state_vectors,
    const uint8_t* d_header_template,
    uint32_t nTime,
    uint64_t nonce,
    uint32_t* d_result_buffer,
    uint32_t* d_result_count,
    // Debug outputs (1 nonce only)
    uint32_t* d_debug_h_initial,
    double* d_debug_angles,
    double* d_debug_expectations,
    int32_t* d_debug_q15_results,
    uint32_t* d_debug_result_xor
) {
    // Only process first block
    if (blockIdx.x > 0) return;
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    cuDoubleComplex* my_sv = d_state_vectors;
    
    // Shared memory
    extern __shared__ uint8_t s_mem[];
    uint32_t* s_h_initial = (uint32_t*)s_mem;
    double* s_angles = (double*)(s_mem + 32);
    double* s_partial_sums = (double*)(s_mem + 32 + 512);
    int32_t* s_q15_results = (int32_t*)(s_mem + 32 + 512 + 32768);
    
    // === 2. HASHING & PARAMETRIZATION ===
    
    if (tid == 0) {
        uint8_t header[80];
        for (int i = 0; i < 76; i++) {
            header[i] = d_header_template[i];
        }
        header[76] = (nonce >> 0) & 0xFF;
        header[77] = (nonce >> 8) & 0xFF;
        header[78] = (nonce >> 16) & 0xFF;
        header[79] = (nonce >> 24) & 0xFF;
        
        sha256d_80_bytes(header, s_h_initial);
        extract_angles(s_h_initial, nTime, s_angles);
        
        // **DEBUG WRITE**: SHA256 and angles
        for (int i = 0; i < 8; i++) d_debug_h_initial[i] = s_h_initial[i];
        for (int i = 0; i < 64; i++) d_debug_angles[i] = s_angles[i];
    }
    __syncthreads();
    
    // === 3. STATE INITIALIZATION ===
    
    for (int i = tid; i < QHASH_STATE_SIZE; i += block_size) {
        my_sv[i] = (i == 0) ? make_cuDoubleComplex(1.0, 0.0) 
                             : make_cuDoubleComplex(0.0, 0.0);
    }
    __syncthreads();
    
    // === 4. GATE APPLICATION ===
    
    for (int layer = 0; layer < QHASH_NUM_LAYERS; layer++) {
        for (int q = 0; q < QHASH_NUM_QUBITS; q++) {
            double angle = s_angles[layer * QHASH_GATES_PER_LAYER + q];
            apply_rotation_gate(my_sv, q, angle, true, tid, block_size);
            __syncthreads();
        }
        
        for (int q = 0; q < QHASH_NUM_QUBITS; q++) {
            double angle = s_angles[layer * QHASH_GATES_PER_LAYER + QHASH_NUM_QUBITS + q];
            apply_rotation_gate(my_sv, q, angle, false, tid, block_size);
            __syncthreads();
        }
    }
    
    for (int i = 0; i < QHASH_NUM_CNOTS; i++) {
        apply_cnot_gate(my_sv, i, i + 1, tid, block_size);
        __syncthreads();
    }
    
    // === 5. EXPECTATION CALCULATION ===
    
    for (int q = 0; q < QHASH_NUM_QUBITS; q++) {
        double my_sum = 0.0;
        
        for (int i = tid; i < QHASH_STATE_SIZE; i += block_size) {
            cuDoubleComplex amp = my_sv[i];
            double mag_sq = amp.x * amp.x + amp.y * amp.y;
            
            if (i & (1 << q)) {
                my_sum -= mag_sq;
            } else {
                my_sum += mag_sq;
            }
        }
        
        s_partial_sums[q * block_size + tid] = my_sum;
        __syncthreads();
        
        // === 6. REDUCTION & CONSENSUS ===
        
        if (tid == 0) {
            double total_sum = 0.0;
            for (int j = 0; j < block_size; j++) {
                total_sum += s_partial_sums[q * block_size + j];
            }
            
            // **DEBUG WRITE**: Expectation value before Q15 conversion
            d_debug_expectations[q] = total_sum;
            
            s_q15_results[q] = convert_q15_device(total_sum);
            
            // **DEBUG WRITE**: Q15 result after conversion
            d_debug_q15_results[q] = s_q15_results[q];
        }
        __syncthreads();
    }
    
    // === 7. FINAL HASHING ===
    
    if (tid == 0) {
        uint8_t s_quantum[64];
        for (int i = 0; i < QHASH_NUM_QUBITS; i++) {
            int32_t q15_val = s_q15_results[i];
            s_quantum[i * 4 + 0] = (q15_val >> 0) & 0xFF;
            s_quantum[i * 4 + 1] = (q15_val >> 8) & 0xFF;
            s_quantum[i * 4 + 2] = (q15_val >> 16) & 0xFF;
            s_quantum[i * 4 + 3] = (q15_val >> 24) & 0xFF;
        }
        
        uint32_t result_xor[8];
        for (int i = 0; i < 8; i++) {
            uint32_t s_quantum_word = 
                ((uint32_t)s_quantum[i * 4 + 0] << 0) |
                ((uint32_t)s_quantum[i * 4 + 1] << 8) |
                ((uint32_t)s_quantum[i * 4 + 2] << 16) |
                ((uint32_t)s_quantum[i * 4 + 3] << 24);
            result_xor[i] = s_h_initial[i] ^ s_quantum_word;
        }
        
        // **DEBUG WRITE**: Final XOR result
        for (int i = 0; i < 8; i++) d_debug_result_xor[i] = result_xor[i];
    }
}

// Host wrapper for debug kernel
extern "C" void launch_fused_qhash_kernel_debug(
    cuDoubleComplex* d_state_vectors,
    const uint8_t* d_header_template,
    uint32_t nTime,
    uint64_t nonce,
    uint32_t* d_result_buffer,
    uint32_t* d_result_count,
    uint32_t* d_debug_h_initial,
    double* d_debug_angles,
    double* d_debug_expectations,
    int32_t* d_debug_q15_results,
    uint32_t* d_debug_result_xor,
    int block_size,
    cudaStream_t stream
) {
    size_t shared_mem_size = 
        32 +
        512 +
        (QHASH_NUM_QUBITS * block_size * 8) +
        64;
    
    dim3 grid(1, 1, 1);  // Only 1 block for debugging
    dim3 block(block_size, 1, 1);
    
    fused_qhash_kernel_debug<<<grid, block, shared_mem_size, stream>>>(
        d_state_vectors,
        d_header_template,
        nTime,
        nonce,
        d_result_buffer,
        d_result_count,
        d_debug_h_initial,
        d_debug_angles,
        d_debug_expectations,
        d_debug_q15_results,
        d_debug_result_xor
    );
}
