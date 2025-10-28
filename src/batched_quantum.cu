/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "batched_quantum.cuh"
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/color.h>

namespace ohmy {
namespace quantum {

// ============================================================================
// BATCHED KERNELS - Process multiple state vectors in parallel
// ============================================================================

/**
 * @brief Initialize all states in batch to |0⟩^⊗n
 * 
 * Grid: 2D with blockIdx.y = batch index
 */
__global__ void initialize_states_batched(
    Complex* states,      // [batch_size][state_size]
    int batch_size,
    size_t state_size
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    // Offset to this batch's state vector
    Complex* state = states + batch_idx * state_size;
    
    if (idx == 0) {
        state[0] = make_cuDoubleComplex(1.0, 0.0);  // |0⟩
    } else {
        state[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

/**
 * @brief Batched RY gate
 */
__global__ void apply_ry_gate_batched(
    Complex* states,
    int batch_size,
    int target_qubit,
    const double* angles,  // [batch_size] - one angle per batch
    int num_qubits
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    size_t state_size = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    Complex* state = states + batch_idx * state_size;
    double angle = angles[batch_idx];
    
    // Standard RY gate logic (same as non-batched)
    if ((idx & (1ULL << target_qubit)) != 0) return;
    
    size_t flipped_idx = idx | (1ULL << target_qubit);
    
    double cos_half = cos(angle / 2.0);
    double sin_half = sin(angle / 2.0);
    
    Complex a0 = state[idx];
    Complex a1 = state[flipped_idx];
    
    state[idx] = cuCadd(
        make_cuDoubleComplex(cos_half * cuCreal(a0), cos_half * cuCimag(a0)),
        make_cuDoubleComplex(-sin_half * cuCreal(a1), -sin_half * cuCimag(a1))
    );
    
    state[flipped_idx] = cuCadd(
        make_cuDoubleComplex(sin_half * cuCreal(a0), sin_half * cuCimag(a0)),
        make_cuDoubleComplex(cos_half * cuCreal(a1), cos_half * cuCimag(a1))
    );
}

/**
 * @brief Batched RZ gate
 */
__global__ void apply_rz_gate_batched(
    Complex* states,
    int batch_size,
    int target_qubit,
    const double* angles,  // [batch_size]
    int num_qubits
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    size_t state_size = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    Complex* state = states + batch_idx * state_size;
    double angle = angles[batch_idx];
    
    double phase = ((idx & (1ULL << target_qubit)) != 0) ? angle / 2.0 : -angle / 2.0;
    
    Complex rotation = make_cuDoubleComplex(cos(phase), sin(phase));
    state[idx] = cuCmul(state[idx], rotation);
}

/**
 * @brief Batched CNOT gate
 */
__global__ void apply_cnot_gate_batched(
    Complex* states,
    int batch_size,
    int control_qubit,
    int target_qubit,
    int num_qubits
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    size_t state_size = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    Complex* state = states + batch_idx * state_size;
    
    // CNOT: flip target if control is |1⟩
    if ((idx & (1ULL << control_qubit)) == 0) return;
    if ((idx & (1ULL << target_qubit)) != 0) return;
    
    size_t flipped_idx = idx | (1ULL << target_qubit);
    
    if (idx < flipped_idx) {
        Complex temp = state[idx];
        state[idx] = state[flipped_idx];
        state[flipped_idx] = temp;
    }
}

/**
 * @brief Batched fused RY+RZ layer (OPTIMIZED)
 * 
 * Applies RY followed by RZ to all qubits in one kernel.
 * Reduces 32 kernel launches to 16 per batch.
 */
__global__ void apply_fused_ry_rz_layer_batched(
    Complex* states,
    int batch_size,
    const double* ry_angles,  // [batch_size][16]
    const double* rz_angles,  // [batch_size][16]
    int num_qubits
) {
    int batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) return;
    
    size_t state_size = 1ULL << num_qubits;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    Complex* state = states + batch_idx * state_size;
    const double* ry = ry_angles + batch_idx * num_qubits;
    const double* rz = rz_angles + batch_idx * num_qubits;
    
    // Apply all 16 RY gates sequentially, then all 16 RZ gates
    // This maintains determinism while processing entire layer
    
    // RY layer
    for (int q = 0; q < num_qubits; q++) {
        if ((idx & (1ULL << q)) != 0) continue;
        
        size_t flipped_idx = idx | (1ULL << q);
        double angle = ry[q];
        double cos_half = cos(angle / 2.0);
        double sin_half = sin(angle / 2.0);
        
        Complex a0 = state[idx];
        Complex a1 = state[flipped_idx];
        
        Complex new_a0 = cuCadd(
            make_cuDoubleComplex(cos_half * cuCreal(a0), cos_half * cuCimag(a0)),
            make_cuDoubleComplex(-sin_half * cuCreal(a1), -sin_half * cuCimag(a1))
        );
        
        Complex new_a1 = cuCadd(
            make_cuDoubleComplex(sin_half * cuCreal(a0), sin_half * cuCimag(a0)),
            make_cuDoubleComplex(cos_half * cuCreal(a1), cos_half * cuCimag(a1))
        );
        
        state[idx] = new_a0;
        state[flipped_idx] = new_a1;
    }
    
    // RZ layer
    for (int q = 0; q < num_qubits; q++) {
        double angle = rz[q];
        double phase = ((idx & (1ULL << q)) != 0) ? angle / 2.0 : -angle / 2.0;
        Complex rotation = make_cuDoubleComplex(cos(phase), sin(phase));
        state[idx] = cuCmul(state[idx], rotation);
    }
}

/**
 * @brief Batched measurement kernel
 * 
 * Computes ⟨σz⟩ for all qubits in all states
 */
__global__ void measure_expectations_batched(
    const Complex* states,
    int batch_size,
    double* expectations,  // [batch_size][num_qubits]
    int num_qubits
) {
    int batch_idx = blockIdx.z;  // Use 3D grid for measurement
    int qubit = blockIdx.y;
    
    if (batch_idx >= batch_size || qubit >= num_qubits) return;
    
    size_t state_size = 1ULL << num_qubits;
    const Complex* state = states + batch_idx * state_size;
    
    __shared__ double block_sum;
    if (threadIdx.x == 0) block_sum = 0.0;
    __syncthreads();
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double thread_sum = 0.0;
    if (idx < state_size) {
        Complex amplitude = state[idx];
        double prob = cuCreal(amplitude) * cuCreal(amplitude) +
                     cuCimag(amplitude) * cuCimag(amplitude);
        
        // Expectation: +1 if qubit is |0⟩, -1 if qubit is |1⟩
        int sign = ((idx & (1ULL << qubit)) == 0) ? 1 : -1;
        thread_sum = sign * prob;
    }
    
    // Atomic add to shared memory
    atomicAdd(&block_sum, thread_sum);
    __syncthreads();
    
    // Write result
    if (threadIdx.x == 0) {
        atomicAdd(&expectations[batch_idx * num_qubits + qubit], block_sum);
    }
}

// ============================================================================
// BatchedQuantumSimulator Implementation
// ============================================================================

BatchedQuantumSimulator::BatchedQuantumSimulator(int num_qubits, int batch_size)
    : num_qubits_(num_qubits)
    , batch_size_(batch_size)
    , state_size_(1ULL << num_qubits)
    , d_states_(nullptr)
    , d_expectations_(nullptr)
{
    // Allocate GPU memory for all state vectors
    size_t total_state_size = batch_size_ * state_size_ * sizeof(Complex);
    size_t total_expectations_size = batch_size_ * num_qubits_ * sizeof(double);
    
    cudaError_t err = cudaMalloc(&d_states_, total_state_size);
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Failed to allocate {} MB for batched states: {}\n",
            total_state_size / (1024 * 1024), cudaGetErrorString(err));
        throw std::runtime_error("GPU memory allocation failed");
    }
    
    err = cudaMalloc(&d_expectations_, total_expectations_size);
    if (err != cudaSuccess) {
        cudaFree(d_states_);
        fmt::print(fg(fmt::color::red),
            "Failed to allocate expectations memory: {}\n", cudaGetErrorString(err));
        throw std::runtime_error("GPU memory allocation failed");
    }
    
    fmt::print("Batched quantum simulator initialized: {} qubits, batch size = {}\n",
        num_qubits_, batch_size_);
    fmt::print("  GPU memory: {} MB states + {} KB expectations = {} MB total\n",
        total_state_size / (1024 * 1024),
        total_expectations_size / 1024,
        (total_state_size + total_expectations_size) / (1024 * 1024));
}

BatchedQuantumSimulator::~BatchedQuantumSimulator() {
    if (d_states_) cudaFree(d_states_);
    if (d_expectations_) cudaFree(d_expectations_);
}

bool BatchedQuantumSimulator::initialize_states() {
    int block_size = 256;
    int num_blocks = (state_size_ + block_size - 1) / block_size;
    
    dim3 grid(num_blocks, batch_size_);
    dim3 block(block_size);
    
    initialize_states_batched<<<grid, block>>>(d_states_, batch_size_, state_size_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Batched initialization error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    return true;
}

bool BatchedQuantumSimulator::apply_circuits_optimized(
    const std::vector<QuantumCircuit>& circuits)
{
    if (circuits.size() != static_cast<size_t>(batch_size_)) {
        fmt::print(fg(fmt::color::red),
            "Circuit count mismatch: expected {}, got {}\n",
            batch_size_, circuits.size());
        return false;
    }
    
    // For now, use simple approach: apply gates one at a time across batch
    // Future optimization: group gates and apply in parallel
    
    int block_size = 256;
    int num_blocks = (state_size_ + block_size - 1) / block_size;
    dim3 grid(num_blocks, batch_size_);
    dim3 block(block_size);
    
    // Assume all circuits have same structure (QTC property)
    const auto& reference_circuit = circuits[0];
    
    for (size_t gate_idx = 0; gate_idx < reference_circuit.gates.size(); gate_idx++) {
        const auto& gate = reference_circuit.gates[gate_idx];
        
        // Extract angles for this gate from all circuits
        std::vector<double> angles(batch_size_);
        for (int b = 0; b < batch_size_; b++) {
            angles[b] = circuits[b].gates[gate_idx].angle;
        }
        
        // Upload angles to GPU
        double* d_angles = nullptr;
        cudaMalloc(&d_angles, batch_size_ * sizeof(double));
        cudaMemcpy(d_angles, angles.data(), batch_size_ * sizeof(double),
                   cudaMemcpyHostToDevice);
        
        // Apply gate to all states in batch
        switch (gate.type) {
            case GateType::RY:
                apply_ry_gate_batched<<<grid, block>>>(
                    d_states_, batch_size_, gate.target_qubit, d_angles, num_qubits_);
                break;
            case GateType::RZ:
                apply_rz_gate_batched<<<grid, block>>>(
                    d_states_, batch_size_, gate.target_qubit, d_angles, num_qubits_);
                break;
            case GateType::CNOT:
                apply_cnot_gate_batched<<<grid, block>>>(
                    d_states_, batch_size_, gate.control_qubit,
                    gate.target_qubit, num_qubits_);
                break;
            default:
                break;
        }
        
        cudaFree(d_angles);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fmt::print(fg(fmt::color::red),
                "Batched gate error at index {}: {}\n",
                gate_idx, cudaGetErrorString(err));
            return false;
        }
    }
    
    cudaDeviceSynchronize();
    return true;
}

bool BatchedQuantumSimulator::measure_all(
    std::vector<std::vector<double>>& expectations)
{
    expectations.resize(batch_size_);
    for (int b = 0; b < batch_size_; b++) {
        expectations[b].resize(num_qubits_, 0.0);
    }
    
    // Zero out device expectations
    cudaMemset(d_expectations_, 0, batch_size_ * num_qubits_ * sizeof(double));
    
    int block_size = 256;
    int num_blocks = (state_size_ + block_size - 1) / block_size;
    
    dim3 grid(num_blocks, num_qubits_, batch_size_);
    dim3 block(block_size);
    
    measure_expectations_batched<<<grid, block>>>(
        d_states_, batch_size_, d_expectations_, num_qubits_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fmt::print(fg(fmt::color::red),
            "Batched measurement error: {}\n", cudaGetErrorString(err));
        return false;
    }
    
    // Copy results to host
    std::vector<double> flat_expectations(batch_size_ * num_qubits_);
    cudaMemcpy(flat_expectations.data(), d_expectations_,
               batch_size_ * num_qubits_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    
    // Reshape to 2D
    for (int b = 0; b < batch_size_; b++) {
        for (int q = 0; q < num_qubits_; q++) {
            expectations[b][q] = flat_expectations[b * num_qubits_ + q];
        }
    }
    
    return true;
}

} // namespace quantum
} // namespace ohmy
