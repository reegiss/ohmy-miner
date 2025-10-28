/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum_kernel.cuh"
#include <fmt/core.h>
#include <fmt/color.h>
#include <cmath>

namespace ohmy {
namespace quantum {

// ============================================================================
// CUDA Device Helper Functions
// ============================================================================

/**
 * @brief Complex multiplication
 */
__device__ __forceinline__ Complex complex_mul(Complex a, Complex b) {
    return make_cuDoubleComplex(
        cuCreal(a) * cuCreal(b) - cuCimag(a) * cuCimag(b),
        cuCreal(a) * cuCimag(b) + cuCimag(a) * cuCreal(b)
    );
}

/**
 * @brief Complex addition
 */
__device__ __forceinline__ Complex complex_add(Complex a, Complex b) {
    return make_cuDoubleComplex(
        cuCreal(a) + cuCreal(b),
        cuCimag(a) + cuCimag(b)
    );
}

/**
 * @brief Check if qubit is set in basis state index
 */
__device__ __forceinline__ bool is_qubit_set(size_t index, int qubit) {
    return (index & (1ULL << qubit)) != 0;
}

/**
 * @brief Flip qubit bit in basis state index
 */
__device__ __forceinline__ size_t flip_qubit(size_t index, int qubit) {
    return index ^ (1ULL << qubit);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void init_quantum_state(Complex* state, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Initialize to |0⟩^⊗n: first element = 1, rest = 0
        if (idx == 0) {
            state[idx] = make_cuDoubleComplex(1.0, 0.0);
        } else {
            state[idx] = make_cuDoubleComplex(0.0, 0.0);
        }
    }
}

__global__ void apply_rx_gate(Complex* state, int target_qubit, double angle, int num_qubits) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t state_size = 1ULL << num_qubits;
    
    if (idx >= state_size) return;
    
    // Only process pairs where target qubit is |0⟩
    if (is_qubit_set(idx, target_qubit)) return;
    
    size_t idx_flipped = flip_qubit(idx, target_qubit);
    
    // Rx(θ) = cos(θ/2)*I - i*sin(θ/2)*X
    double cos_half = cos(angle / 2.0);
    double sin_half = sin(angle / 2.0);
    
    Complex state0 = state[idx];
    Complex state1 = state[idx_flipped];
    
    // New |0⟩ component: cos(θ/2)|0⟩ - i*sin(θ/2)|1⟩
    state[idx] = make_cuDoubleComplex(
        cos_half * cuCreal(state0) + sin_half * cuCimag(state1),
        cos_half * cuCimag(state0) - sin_half * cuCreal(state1)
    );
    
    // New |1⟩ component: -i*sin(θ/2)|0⟩ + cos(θ/2)|1⟩
    state[idx_flipped] = make_cuDoubleComplex(
        cos_half * cuCreal(state1) + sin_half * cuCimag(state0),
        cos_half * cuCimag(state1) - sin_half * cuCreal(state0)
    );
}

__global__ void apply_ry_gate(Complex* state, int target_qubit, double angle, int num_qubits) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t state_size = 1ULL << num_qubits;
    
    if (idx >= state_size) return;
    
    // Only process pairs where target qubit is |0⟩
    if (is_qubit_set(idx, target_qubit)) return;
    
    size_t idx_flipped = flip_qubit(idx, target_qubit);
    
    // Ry(θ) = cos(θ/2)*I - i*sin(θ/2)*Y
    double cos_half = cos(angle / 2.0);
    double sin_half = sin(angle / 2.0);
    
    Complex state0 = state[idx];
    Complex state1 = state[idx_flipped];
    
    // New |0⟩ component: cos(θ/2)|0⟩ - sin(θ/2)|1⟩
    state[idx] = make_cuDoubleComplex(
        cos_half * cuCreal(state0) - sin_half * cuCreal(state1),
        cos_half * cuCimag(state0) - sin_half * cuCimag(state1)
    );
    
    // New |1⟩ component: sin(θ/2)|0⟩ + cos(θ/2)|1⟩
    state[idx_flipped] = make_cuDoubleComplex(
        sin_half * cuCreal(state0) + cos_half * cuCreal(state1),
        sin_half * cuCimag(state0) + cos_half * cuCimag(state1)
    );
}

__global__ void apply_rz_gate(Complex* state, int target_qubit, double angle, int num_qubits) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t state_size = 1ULL << num_qubits;
    
    if (idx >= state_size) return;
    
    // Rz(θ) = e^(-iθ/2)|0⟩⟨0| + e^(iθ/2)|1⟩⟨1|
    double half_angle = angle / 2.0;
    
    Complex phase;
    if (is_qubit_set(idx, target_qubit)) {
        // |1⟩: multiply by e^(iθ/2) = cos(θ/2) + i*sin(θ/2)
        phase = make_cuDoubleComplex(cos(half_angle), sin(half_angle));
    } else {
        // |0⟩: multiply by e^(-iθ/2) = cos(θ/2) - i*sin(θ/2)
        phase = make_cuDoubleComplex(cos(half_angle), -sin(half_angle));
    }
    
    state[idx] = complex_mul(state[idx], phase);
}

__global__ void apply_cnot_gate(Complex* state, int control_qubit, int target_qubit, int num_qubits) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t state_size = 1ULL << num_qubits;
    
    if (idx >= state_size) return;
    
    // CNOT flips target only when control is |1⟩
    if (!is_qubit_set(idx, control_qubit)) return;
    if (is_qubit_set(idx, target_qubit)) return;
    
    size_t idx_flipped = flip_qubit(idx, target_qubit);
    
    // Swap amplitudes
    Complex temp = state[idx];
    state[idx] = state[idx_flipped];
    state[idx_flipped] = temp;
}

/**
 * @brief OPTIMIZED: Fused RY+RZ layer kernel
 * 
 * This kernel applies RY(θy) followed by RZ(θz) to all qubits in a single pass.
 * Instead of 32 separate kernel launches (16 RY + 16 RZ), we do everything in
 * one kernel, dramatically reducing launch overhead.
 * 
 * Mathematical operation per qubit q:
 *   |ψ'⟩ = Rz(θz[q]) * Ry(θy[q]) * |ψ⟩
 * 
 * Combined rotation matrix:
 *   Rz*Ry = [ cos(θy/2)*e^(-iθz/2)    -sin(θy/2)*e^(-iθz/2) ]
 *           [ sin(θy/2)*e^(iθz/2)      cos(θy/2)*e^(iθz/2)  ]
 * 
 * Strategy:
 * - Each thread processes one basis state |i⟩
 * - For each qubit q, compute local 2x2 rotation for that qubit's subspace
 * - Apply rotations sequentially to maintain determinism
 */
/**
 * @brief OPTIMIZED: Fused RY+RZ kernel for a SINGLE qubit
 * 
 * This kernel applies RY(θy) followed by RZ(θz) to ONE qubit in a single pass.
 * Mathematical operation: |ψ'⟩ = Rz(θz) * Ry(θy) * |ψ⟩
 * 
 * We still need to call this 16 times (once per qubit), but each call fuses
 * 2 operations that would normally require 2 separate kernels.
 * 
 * Net result: 32 kernel launches (16 RY + 16 RZ) → 16 kernel launches (2x reduction)
 * 
 * This is the CORRECT approach that maintains determinism and avoids race conditions.
 */
__global__ void apply_ry_rz_fused_single_qubit(
    Complex* state,
    double theta_y,
    double theta_z,
    int qubit,
    int num_qubits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t state_size = 1ULL << num_qubits;
    
    if (idx >= state_size) return;
    
    // Only process pairs where this basis state has qubit = |0⟩
    if (is_qubit_set(idx, qubit)) return;
    
    size_t idx_flipped = flip_qubit(idx, qubit);
    
    // Precompute rotation matrix elements
    double cos_y_half = cos(theta_y / 2.0);
    double sin_y_half = sin(theta_y / 2.0);
    double cos_z_half = cos(theta_z / 2.0);
    double sin_z_half = sin(theta_z / 2.0);
    
    // Get current amplitudes
    Complex amp0 = state[idx];
    Complex amp1 = state[idx_flipped];
    
    // Apply fused Rz*Ry rotation
    // Matrix elements: [[ cos(y/2)*e^(-iz/2), -sin(y/2)*e^(-iz/2) ],
    //                   [ sin(y/2)*e^(iz/2),   cos(y/2)*e^(iz/2)  ]]
    
    Complex e_minus = make_cuDoubleComplex(cos_z_half, -sin_z_half);
    Complex e_plus = make_cuDoubleComplex(cos_z_half, sin_z_half);
    
    // |0⟩ component
    Complex term0_0 = complex_mul(make_cuDoubleComplex(cos_y_half, 0), e_minus);
    Complex term0_1 = complex_mul(make_cuDoubleComplex(-sin_y_half, 0), e_minus);
    Complex new_amp0 = complex_add(
        complex_mul(term0_0, amp0),
        complex_mul(term0_1, amp1)
    );
    
    // |1⟩ component
    Complex term1_0 = complex_mul(make_cuDoubleComplex(sin_y_half, 0), e_plus);
    Complex term1_1 = complex_mul(make_cuDoubleComplex(cos_y_half, 0), e_plus);
    Complex new_amp1 = complex_add(
        complex_mul(term1_0, amp0),
        complex_mul(term1_1, amp1)
    );
    
    // Write back (no race condition - each thread owns its idx pair)
    state[idx] = new_amp0;
    state[idx_flipped] = new_amp1;
}

__global__ void measure_expectations(const Complex* state, double* expectations, int num_qubits) {
    int qubit = blockIdx.x;
    if (qubit >= num_qubits) return;
    
    size_t state_size = 1ULL << num_qubits;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for reduction
    __shared__ double shared_expectation[256];
    
    double local_sum = 0.0;
    
    // Each thread processes multiple basis states
    for (size_t idx = tid; idx < state_size; idx += block_size) {
        double prob = cuCreal(state[idx]) * cuCreal(state[idx]) + 
                     cuCimag(state[idx]) * cuCimag(state[idx]);
        
        // ⟨σz⟩ = P(|0⟩) - P(|1⟩)
        if (is_qubit_set(idx, qubit)) {
            local_sum -= prob;  // |1⟩ contributes -1
        } else {
            local_sum += prob;  // |0⟩ contributes +1
        }
    }
    
    shared_expectation[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_expectation[tid] += shared_expectation[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        expectations[qubit] = shared_expectation[0];
    }
}

// ============================================================================
// Host-side QuantumSimulator Implementation
// ============================================================================

QuantumSimulator::QuantumSimulator(int num_qubits)
    : num_qubits_(num_qubits)
    , state_size_(1ULL << num_qubits)
    , d_state_(nullptr)
    , d_expectations_(nullptr) {
    
    // Allocate device memory for state vector
    size_t state_bytes = state_size_ * sizeof(Complex);
    cudaError_t err = cudaMalloc(&d_state_, state_bytes);
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Failed to allocate GPU memory for quantum state: {}\n",
            cudaGetErrorString(err));
        throw std::runtime_error("CUDA allocation failed");
    }
    
    // Allocate device memory for expectations
    err = cudaMalloc(&d_expectations_, num_qubits_ * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_state_);
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "Failed to allocate GPU memory for expectations: {}\n",
            cudaGetErrorString(err));
        throw std::runtime_error("CUDA allocation failed");
    }
    
    fmt::print(fg(fmt::color::green),
        "Quantum simulator initialized: {} qubits, state size = {}\n",
        num_qubits_, state_size_);
}

QuantumSimulator::~QuantumSimulator() {
    if (d_state_) {
        cudaFree(d_state_);
    }
    if (d_expectations_) {
        cudaFree(d_expectations_);
    }
}

bool QuantumSimulator::initialize_state() {
    // Launch kernel to initialize state to |0⟩^⊗n
    int block_size = 256;
    int num_blocks = (state_size_ + block_size - 1) / block_size;
    
    init_quantum_state<<<num_blocks, block_size>>>(d_state_, state_size_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Kernel launch error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Kernel execution error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

bool QuantumSimulator::apply_circuit(const QuantumCircuit& circuit) {
    if (circuit.num_qubits != num_qubits_) {
        fmt::print(fg(fmt::color::red),
            "Circuit qubit count mismatch: {} vs {}\n",
            circuit.num_qubits, num_qubits_);
        return false;
    }
    
    int block_size = 256;
    int num_blocks = (state_size_ + block_size - 1) / block_size;
    
    // Apply gates sequentially
    for (const auto& gate : circuit.gates) {
        switch (gate.type) {
            case GateType::RX:
                apply_rx_gate<<<num_blocks, block_size>>>(
                    d_state_, gate.target_qubit, gate.angle, num_qubits_);
                break;
                
            case GateType::RY:
                apply_ry_gate<<<num_blocks, block_size>>>(
                    d_state_, gate.target_qubit, gate.angle, num_qubits_);
                break;
                
            case GateType::RZ:
                apply_rz_gate<<<num_blocks, block_size>>>(
                    d_state_, gate.target_qubit, gate.angle, num_qubits_);
                break;
                
            case GateType::CNOT:
                apply_cnot_gate<<<num_blocks, block_size>>>(
                    d_state_, gate.control_qubit, gate.target_qubit, num_qubits_);
                break;
        }
        
        // Check for errors after each gate
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fmt::print(fg(fmt::color::red),
                "Gate kernel launch error: {}\n", cudaGetErrorString(err));
            return false;
        }
    }
    
    // Synchronize after all gates
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Circuit execution error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

bool QuantumSimulator::apply_circuit_optimized(const QuantumCircuit& circuit) {
    if (circuit.num_qubits != num_qubits_) {
        fmt::print(fg(fmt::color::red),
            "Circuit qubit count mismatch: {} vs {}\n",
            circuit.num_qubits, num_qubits_);
        return false;
    }
    
    cudaError_t err;
    int block_size = 256;
    int num_blocks = (state_size_ + block_size - 1) / block_size;
    
    // Process circuit gates with optimizations
    // QTC circuit structure: [RY_all → RZ_all → CNOT_chain] × 2 layers
    // OPTIMIZATIONS:
    // 1. Fuse adjacent RY+RZ pairs per qubit (32 launches → 16 launches)
    // 2. Detect and optimize CNOT chains with shared memory (15 launches → 1 launch per chain)
    
    size_t gate_idx = 0;
    const size_t total_gates = circuit.gates.size();
    
    while (gate_idx < total_gates) {
        // OPTIMIZATION 1: Try to detect RY-RZ pair on the same qubit
        if (gate_idx + 1 < total_gates &&
            circuit.gates[gate_idx].type == GateType::RY &&
            circuit.gates[gate_idx + 1].type == GateType::RZ &&
            circuit.gates[gate_idx].target_qubit == circuit.gates[gate_idx + 1].target_qubit) {
            
            // Found fusible pair!
            int qubit = circuit.gates[gate_idx].target_qubit;
            double theta_y = circuit.gates[gate_idx].angle;
            double theta_z = circuit.gates[gate_idx + 1].angle;
            
            // Launch fused kernel
            apply_ry_rz_fused_single_qubit<<<num_blocks, block_size>>>(
                d_state_, theta_y, theta_z, qubit, num_qubits_);
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fmt::print(fg(fmt::color::red),
                    "Fused RY+RZ kernel error: {}\n", cudaGetErrorString(err));
                return false;
            }
            
            gate_idx += 2;  // Skip both gates
        }
        // Apply CNOT gates (no chain optimization - complexity not worth it)
        else if (circuit.gates[gate_idx].type == GateType::CNOT) {
            const auto& gate = circuit.gates[gate_idx];
            apply_cnot_gate<<<num_blocks, block_size>>>(
                d_state_, gate.control_qubit, gate.target_qubit, num_qubits_);
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fmt::print(fg(fmt::color::red),
                    "CNOT kernel error: {}\n", cudaGetErrorString(err));
                return false;
            }
            
            gate_idx++;
        }
        else {
            // Non-fusible gate - apply individually
            const auto& gate = circuit.gates[gate_idx];
            
            switch (gate.type) {
                case GateType::RX:
                    apply_rx_gate<<<num_blocks, block_size>>>(
                        d_state_, gate.target_qubit, gate.angle, num_qubits_);
                    break;
                case GateType::RY:
                    apply_ry_gate<<<num_blocks, block_size>>>(
                        d_state_, gate.target_qubit, gate.angle, num_qubits_);
                    break;
                case GateType::RZ:
                    apply_rz_gate<<<num_blocks, block_size>>>(
                        d_state_, gate.target_qubit, gate.angle, num_qubits_);
                    break;
                default:
                    break;
            }
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fmt::print(fg(fmt::color::red),
                    "Gate kernel error: {}\n", cudaGetErrorString(err));
                return false;
            }
            
            gate_idx++;
        }
    }
    
    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Circuit execution error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

bool QuantumSimulator::measure(std::vector<double>& expectations) {
    expectations.resize(num_qubits_);
    
    int block_size = 256;
    
    // Launch one block per qubit
    measure_expectations<<<num_qubits_, block_size>>>(
        d_state_, d_expectations_, num_qubits_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Measure kernel launch error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Measure execution error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    // Copy results back to host
    err = cudaMemcpy(expectations.data(), d_expectations_,
                     num_qubits_ * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Measure memcpy error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

} // namespace quantum
} // namespace ohmy
