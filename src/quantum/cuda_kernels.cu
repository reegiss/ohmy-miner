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
 * CUDA Kernel: Initialize quantum state to |0...0⟩
 * 
 * State vector has 2^n amplitudes, all set to 0 except amplitude[0] = 1
 * 
 * @param state     Output state vector (device memory)
 * @param state_size Number of amplitudes (2^num_qubits)
 */
__global__ void init_zero_state_kernel(Complex* state, size_t state_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        // First amplitude is 1+0i (all qubits in |0⟩)
        state[0] = make_cuFloatComplex(1.0f, 0.0f);
    } else if (idx < state_size) {
        // All other amplitudes are 0+0i
        state[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

/**
 * CUDA Kernel: Apply R_Y rotation gate to single qubit
 * 
 * R_Y(θ) = [[cos(θ/2), -sin(θ/2)],
 *           [sin(θ/2),  cos(θ/2)]]
 * 
 * Each thread processes a pair of amplitudes that differ in the target qubit bit.
 * 
 * Decision: Use float32 trigonometry (__sincosf) for performance
 * 
 * @param state       State vector to modify (device memory)
 * @param target_qubit Qubit index to apply gate (0-based)
 * @param angle       Rotation angle in radians
 * @param state_size  Number of amplitudes
 */
__global__ void apply_rotation_y_kernel(
    Complex* state,
    int target_qubit,
    float angle,
    size_t state_size
) {
    // Each thread handles pair of amplitudes affected by gate
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_pairs = state_size / 2;
    
    if (idx >= num_pairs) return;
    
    // Calculate indices that differ in target_qubit bit
    // Strategy: Iterate through pairs, skipping higher bits
    size_t low_bits = idx & ((1ULL << target_qubit) - 1);  // Bits below target
    size_t high_bits = (idx >> target_qubit) << (target_qubit + 1);  // Bits above target
    
    size_t idx0 = high_bits | low_bits;           // target_qubit = 0
    size_t idx1 = idx0 | (1ULL << target_qubit);  // target_qubit = 1
    
    // Load amplitudes
    Complex alpha = state[idx0];  // Amplitude when qubit is |0⟩
    Complex beta = state[idx1];   // Amplitude when qubit is |1⟩
    
    // Compute rotation matrix elements
    float cos_half, sin_half;
    __sincosf(angle * 0.5f, &sin_half, &cos_half);  // Fast float32 sincos
    
    // Apply R_Y gate transformation
    // |0⟩ →  cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    // |1⟩ → -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
    
    Complex new_alpha = cuCsubf(
        cuCmulf(make_cuFloatComplex(cos_half, 0.0f), alpha),
        cuCmulf(make_cuFloatComplex(sin_half, 0.0f), beta)
    );
    
    Complex new_beta = cuCaddf(
        cuCmulf(make_cuFloatComplex(sin_half, 0.0f), alpha),
        cuCmulf(make_cuFloatComplex(cos_half, 0.0f), beta)
    );
    
    // Write back results
    state[idx0] = new_alpha;
    state[idx1] = new_beta;
}

/**
 * CUDA Kernel: Apply R_Z rotation gate to single qubit
 * 
 * R_Z(θ) = [[e^(-iθ/2),     0    ],
 *           [    0,      e^(iθ/2)]]
 * 
 * This is a diagonal gate - only applies phase to |1⟩ amplitude.
 * More efficient than R_Y since no mixing of amplitudes required.
 * 
 * Decision: Use __sincosf for phase calculation
 * 
 * @param state       State vector to modify (device memory)
 * @param target_qubit Qubit index to apply gate (0-based)
 * @param angle       Rotation angle in radians
 * @param state_size  Number of amplitudes
 */
__global__ void apply_rotation_z_kernel(
    Complex* state,
    int target_qubit,
    float angle,
    size_t state_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= state_size) return;
    
    // Check if target qubit is |1⟩ in this amplitude
    bool qubit_is_one = (idx & (1ULL << target_qubit)) != 0;
    
    if (qubit_is_one) {
        // Apply phase e^(iθ/2) = cos(θ/2) + i*sin(θ/2)
        float cos_half, sin_half;
        __sincosf(angle * 0.5f, &sin_half, &cos_half);
        
        Complex phase = make_cuFloatComplex(cos_half, sin_half);
        state[idx] = cuCmulf(state[idx], phase);
    } else {
        // Apply phase e^(-iθ/2) = cos(θ/2) - i*sin(θ/2)
        float cos_half, sin_half;
        __sincosf(angle * 0.5f, &sin_half, &cos_half);
        
        Complex phase = make_cuFloatComplex(cos_half, -sin_half);
        state[idx] = cuCmulf(state[idx], phase);
    }
}

/**
 * CUDA Kernel: Apply CNOT gate (controlled-NOT)
 * 
 * CNOT flips the target qubit if and only if control qubit is |1⟩
 * 
 * Decision: Simple conditional swap approach
 * - Check each amplitude's control bit
 * - If control=1, swap amplitudes that differ in target bit
 * - Each thread handles one amplitude independently
 * 
 * @param state         State vector to modify (device memory)
 * @param control_qubit Control qubit index (0-based)
 * @param target_qubit  Target qubit index (0-based)
 * @param state_size    Number of amplitudes
 */
__global__ void apply_cnot_kernel(
    Complex* state,
    int control_qubit,
    int target_qubit,
    size_t state_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= state_size) return;
    
    // Check if control qubit is |1⟩
    bool control_is_one = (idx & (1ULL << control_qubit)) != 0;
    
    if (!control_is_one) {
        // Control is |0⟩ - no operation needed
        return;
    }
    
    // Control is |1⟩ - need to flip target qubit
    // Calculate paired index (differs in target_qubit bit)
    size_t paired_idx = idx ^ (1ULL << target_qubit);
    
    // Only process if idx < paired_idx to avoid double-processing
    if (idx < paired_idx) {
        // Swap amplitudes
        Complex temp = state[idx];
        state[idx] = state[paired_idx];
        state[paired_idx] = temp;
    }
}

/**
 * CUDA Kernel: Compute Z-basis expectation value for single qubit
 * 
 * Expectation ⟨Z⟩ = P(|0⟩) - P(|1⟩)
 *               = Σ|α_i|² (qubit=0) - Σ|α_i|² (qubit=1)
 * 
 * Phase 1 (this kernel): Block-level reduction using shared memory
 * - Each block computes partial sum for its chunk of state
 * - Results stored in intermediate buffer for Phase 2
 * 
 * Decision: Three-phase hierarchical reduction
 * - Phase 1 (block-level): 256 threads → 1 partial sum per block
 * - Phase 2 (warp-level): Warp shuffle operations  
 * - Phase 3 (atomic): Final reduction
 * 
 * @param state          State vector (device memory)
 * @param target_qubit   Qubit to measure (0-based)
 * @param state_size     Number of amplitudes
 * @param partial_sums   Output buffer for partial sums (device memory)
 */
__global__ void compute_z_expectation_phase1_kernel(
    const Complex* state,
    int target_qubit,
    size_t state_size,
    float* partial_sums
) {
    // Shared memory for block-level reduction
    __shared__ float shared_sum[256];  // Match DEFAULT_BLOCK_SIZE
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    
    // Each thread accumulates its portion
    if (idx < state_size) {
        Complex amplitude = state[idx];
        float probability = cuCrealf(amplitude) * cuCrealf(amplitude) +
                           cuCimagf(amplitude) * cuCimagf(amplitude);
        
        // Check if target qubit is |0⟩ or |1⟩
        bool qubit_is_zero = ((idx & (1ULL << target_qubit)) == 0);
        
        // ⟨Z⟩ contribution: +prob if |0⟩, -prob if |1⟩
        local_sum = qubit_is_zero ? probability : -probability;
    }
    
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

/**
 * CUDA Kernel: Final reduction of partial sums
 * 
 * Phase 2 (this kernel): Reduce all block partial sums to single value
 * Launched with single block, completes reduction using shared memory
 * 
 * @param partial_sums  Input partial sums from Phase 1 (device memory)
 * @param num_blocks    Number of partial sums
 * @param result        Output: final expectation value (device memory)
 */
__global__ void compute_z_expectation_phase2_kernel(
    const float* partial_sums,
    int num_blocks,
    float* result
) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    
    // Load partial sums (may need multiple loads if num_blocks > blockDim.x)
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_sum += partial_sums[i];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[0] = shared_sum[0];
    }
}

//
// === BATCHED KERNELS FOR PARALLEL NONCE PROCESSING ===
//

/**
 * CUDA Kernel: Initialize multiple quantum states to |0...0⟩ in parallel
 * 
 * Batch layout: [batch_size][state_size]
 * Each state is 65,536 amplitudes apart in memory
 * 
 * @param states     Batched state vectors (device memory)
 * @param batch_size Number of states to initialize
 * @param state_size Number of amplitudes per state (2^num_qubits)
 */
__global__ void init_zero_state_batch_kernel(
    Complex* states,
    int batch_size,
    size_t state_size
) {
    // Global thread ID across all states
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = batch_size * state_size;
    
    if (global_idx >= total_elements) return;
    
    // Determine which amplitude (batch_idx not needed since we index directly)
    size_t amplitude_idx = global_idx % state_size;
    
    if (amplitude_idx == 0) {
        // First amplitude is 1+0i
        states[global_idx] = make_cuFloatComplex(1.0f, 0.0f);
    } else {
        // All other amplitudes are 0+0i
        states[global_idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

/**
 * CUDA Kernel: Apply R_Y rotation to batch of states in parallel
 * 
 * Each state processed independently - perfect parallelization
 * 
 * @param states      Batched state vectors (device memory)
 * @param batch_size  Number of states
 * @param target_qubit Qubit index to apply gate
 * @param angle       Rotation angle in radians
 * @param state_size  Number of amplitudes per state
 */
__global__ void apply_rotation_y_batch_kernel(
    Complex* states,
    int batch_size,
    int target_qubit,
    float angle,
    size_t state_size
) {
    // Global thread ID
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pairs_per_state = state_size / 2;
    size_t total_pairs = static_cast<size_t>(batch_size) * pairs_per_state;
    
    if (global_idx >= total_pairs) {
        return;
    }
    
    __shared__ float s_sin_half;
    __shared__ float s_cos_half;
    if (threadIdx.x == 0) {
        float sin_half, cos_half;
        __sincosf(angle * 0.5f, &sin_half, &cos_half);
        s_sin_half = sin_half;
        s_cos_half = cos_half;
    }
    __syncthreads();

    // Determine which state and which pair
    size_t batch_idx = global_idx / pairs_per_state;
    size_t pair_idx = global_idx % pairs_per_state;
    
    // Calculate amplitude indices within this state
    size_t state_offset = batch_idx * state_size;
    size_t low_bits = pair_idx & ((1ULL << target_qubit) - 1);
    size_t high_bits = (pair_idx >> target_qubit) << (target_qubit + 1);
    
    size_t idx0 = state_offset + high_bits + low_bits;
    size_t idx1 = idx0 + (1ULL << target_qubit);
    
    // Load amplitudes
    Complex alpha = states[idx0];
    Complex beta = states[idx1];
    
    // Apply gate
    Complex new_alpha = cuCsubf(
        cuCmulf(make_cuFloatComplex(s_cos_half, 0.0f), alpha),
        cuCmulf(make_cuFloatComplex(s_sin_half, 0.0f), beta)
    );
    
    Complex new_beta = cuCaddf(
        cuCmulf(make_cuFloatComplex(s_sin_half, 0.0f), alpha),
        cuCmulf(make_cuFloatComplex(s_cos_half, 0.0f), beta)
    );
    
    states[idx0] = new_alpha;
    states[idx1] = new_beta;
}

/**
 * CUDA Kernel: Apply R_Z rotation to batch of states
 */
__global__ void apply_rotation_z_batch_kernel(
    Complex* states,
    int batch_size,
    int target_qubit,
    float angle,
    size_t state_size
) {
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = static_cast<size_t>(batch_size) * state_size;
    
    if (global_idx >= total_elements) {
        return;
    }

    __shared__ Complex s_phase_zero;
    __shared__ Complex s_phase_one;
    if (threadIdx.x == 0) {
        float sin_half, cos_half;
        __sincosf(angle * 0.5f, &sin_half, &cos_half);
        s_phase_zero = make_cuFloatComplex(cos_half, -sin_half);
        s_phase_one = make_cuFloatComplex(cos_half, sin_half);
    }
    __syncthreads();

    // Check if target qubit is |1⟩
    size_t amplitude_idx = global_idx % state_size;
    bool qubit_is_one = (amplitude_idx & (1ULL << target_qubit)) != 0;

    Complex phase;
    if (qubit_is_one) {
        phase = s_phase_one;
    } else {
        phase = s_phase_zero;
    }

    states[global_idx] = cuCmulf(states[global_idx], phase);
}

/**
 * CUDA Kernel: Apply CNOT to batch of states
 */
__global__ void apply_cnot_batch_kernel(
    Complex* states,
    int batch_size,
    int control_qubit,
    int target_qubit,
    size_t state_size
) {
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = batch_size * state_size;
    
    if (global_idx >= total_elements) return;
    
    size_t amplitude_idx = global_idx % state_size;
    bool control_is_one = (amplitude_idx & (1ULL << control_qubit)) != 0;
    
    if (!control_is_one) return;
    
    // Calculate paired index
    size_t paired_idx = global_idx ^ (1ULL << target_qubit);
    
    // Only process if idx < paired_idx to avoid double-processing
    if (global_idx < paired_idx) {
        Complex temp = states[global_idx];
        states[global_idx] = states[paired_idx];
        states[paired_idx] = temp;
    }
}

/**
 * CUDA Kernel: Measure Z-expectations for batch of states
 * 
 * Each state produces one expectation value
 * Uses block-level reduction per state
 * 
 * @param states         Batched state vectors (device memory)
 * @param batch_size     Number of states
 * @param target_qubit   Qubit to measure
 * @param state_size     Amplitudes per state
 * @param expectations   Output expectations [batch_size] (device memory)
 */
__global__ void compute_z_expectation_batch_kernel(
    const Complex* states,
    int batch_size,
    int target_qubit,
    size_t state_size,
    float* expectations
) {
    __shared__ float shared_sum[256];
    
    // Determine which state this block is processing
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    size_t state_offset = batch_idx * state_size;
    float local_sum = 0.0f;
    
    // Each thread accumulates part of this state
    for (size_t idx = threadIdx.x; idx < state_size; idx += blockDim.x) {
        Complex amplitude = states[state_offset + idx];
        float probability = cuCrealf(amplitude) * cuCrealf(amplitude) +
                           cuCimagf(amplitude) * cuCimagf(amplitude);
        
        bool qubit_is_zero = ((idx & (1ULL << target_qubit)) == 0);
        local_sum += qubit_is_zero ? probability : -probability;
    }
    
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        expectations[batch_idx] = shared_sum[0];
    }
}

} // namespace cuda
} // namespace quantum
} // namespace ohmy
