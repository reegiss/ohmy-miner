/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/cuda_simulator.hpp"
#include <fmt/format.h>

namespace ohmy {
namespace quantum {
namespace cuda {

// Forward declarations of CUDA kernels (defined in cuda_kernels.cu)
extern __global__ void init_zero_state_kernel(Complex* state, size_t state_size);

extern __global__ void apply_rotation_y_kernel(
    Complex* state, int target_qubit, float angle, size_t state_size);

extern __global__ void apply_rotation_z_kernel(
    Complex* state, int target_qubit, float angle, size_t state_size);

extern __global__ void apply_cnot_kernel(
    Complex* state, int control_qubit, int target_qubit, size_t state_size);

extern __global__ void compute_z_expectation_phase1_kernel(
    const Complex* state, int target_qubit, size_t state_size, float* partial_sums);

extern __global__ void compute_z_expectation_phase2_kernel(
    const float* partial_sums, int num_blocks, float* result);

// Batched kernel declarations
extern __global__ void init_zero_state_batch_kernel(
    Complex* states, int batch_size, size_t state_size);

extern __global__ void apply_rotation_y_batch_kernel(
    Complex* states, int batch_size, int target_qubit, float angle, size_t state_size);

extern __global__ void apply_rotation_z_batch_kernel(
    Complex* states, int batch_size, int target_qubit, float angle, size_t state_size);

extern __global__ void apply_cnot_batch_kernel(
    Complex* states, int batch_size, int control_qubit, int target_qubit, size_t state_size);

extern __global__ void compute_z_expectation_batch_kernel(
    const Complex* states, int batch_size, int target_qubit, size_t state_size, float* expectations);

// --- Constructor ---

CudaQuantumSimulator::CudaQuantumSimulator(int max_qubits, int device_id)
    : max_qubits_(max_qubits)
    , device_id_(device_id)
    , state_size_(1ULL << max_qubits)
    , block_size_(DEFAULT_BLOCK_SIZE)
    , d_state_(state_size_)
    , d_workspace_(state_size_)
    , d_partial_sums_(calculate_grid_size(state_size_).x)  // One per block
    , d_expectation_(1)  // Single output value
    , h_state_(state_size_)
    , compute_stream_()
    , transfer_stream_()
{
    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Query device capabilities
    device_info_ = DeviceInfo::query(device_id_);
    
    // Validate device compatibility
    if (!device_info_.is_compatible()) {
        throw std::runtime_error(fmt::format(
            "GPU compute capability {}.{} is below minimum 7.5 (Turing required)",
            device_info_.compute_capability_major,
            device_info_.compute_capability_minor
        ));
    }
    
    // Calculate memory requirements
    memory_reqs_ = MemoryRequirements::calculate(max_qubits_);
    
    // Check available memory
    if (device_info_.free_memory < memory_reqs_.total_bytes * 1.2) {
        throw std::runtime_error(fmt::format(
            "Insufficient GPU memory: need {} (with 20% overhead), have {} free",
            MemoryRequirements::format_bytes(memory_reqs_.total_bytes * 1.2),
            MemoryRequirements::format_bytes(device_info_.free_memory)
        ));
    }
    
    // Log initialization success
    fmt::print("[CUDA] Initialized {} with {} qubits\n",
               device_info_.name, max_qubits_);
    fmt::print("[CUDA] State size: {} ({} amplitudes)\n",
               MemoryRequirements::format_bytes(memory_reqs_.state_bytes),
               state_size_);
    fmt::print("[CUDA] Device memory: {} / {} available\n",
               MemoryRequirements::format_bytes(device_info_.free_memory),
               MemoryRequirements::format_bytes(device_info_.total_memory));
    
    // Initialize state to |0...0⟩
    reset();
}

// --- Destructor ---

CudaQuantumSimulator::~CudaQuantumSimulator() {
    // RAII handles all cleanup automatically
    // No manual cudaFree needed - DeviceMemory/StreamHandle destructors handle it
}

// --- State Management ---

void CudaQuantumSimulator::reset() {
    // Launch kernel to initialize state to |0...0⟩
    dim3 grid = calculate_grid_size(state_size_);
    dim3 block(block_size_);
    
    init_zero_state_kernel<<<grid, block, 0, compute_stream_.get()>>>(
        d_state_.get(), state_size_);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize to ensure state is ready
    compute_stream_.synchronize();
}

// --- Circuit Simulation ---

void CudaQuantumSimulator::validate_circuit(const QuantumCircuit& circuit) const {
    if (circuit.num_qubits() > max_qubits_) {
        throw std::runtime_error(fmt::format(
            "Circuit has {} qubits but simulator supports maximum {}",
            circuit.num_qubits(), max_qubits_
        ));
    }
}

void CudaQuantumSimulator::simulate(const QuantumCircuit& circuit) {
    validate_circuit(circuit);
    
    // Apply all rotation gates
    for (const auto& gate : circuit.rotation_gates()) {
        apply_rotation(gate.qubit, static_cast<float>(gate.angle), gate.axis);
    }
    
    // Apply all CNOT gates
    for (const auto& gate : circuit.cnot_gates()) {
        apply_cnot(gate.control, gate.target);
    }
    
    // Synchronize to ensure all gates applied
    compute_stream_.synchronize();
}

// --- Gate Applications ---

void CudaQuantumSimulator::apply_rotation(int qubit, float angle, RotationAxis axis) {
    dim3 grid = calculate_grid_size(state_size_ / 2);  // Each thread handles pair
    dim3 block(block_size_);
    
    if (axis == RotationAxis::Y) {
        apply_rotation_y_kernel<<<grid, block, 0, compute_stream_.get()>>>(
            d_state_.get(), qubit, angle, state_size_);
    } else {
        apply_rotation_z_kernel<<<grid, block, 0, compute_stream_.get()>>>(
            d_state_.get(), qubit, angle, state_size_);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void CudaQuantumSimulator::apply_cnot(int control, int target) {
    dim3 grid = calculate_grid_size(state_size_);
    dim3 block(block_size_);
    
    apply_cnot_kernel<<<grid, block, 0, compute_stream_.get()>>>(
        d_state_.get(), control, target, state_size_);
    
    CUDA_CHECK(cudaGetLastError());
}

// --- Measurement ---

float CudaQuantumSimulator::compute_z_expectation(int qubit) {
    // Phase 1: Block-level reduction
    dim3 grid1 = calculate_grid_size(state_size_);
    dim3 block1(block_size_);
    int num_blocks = grid1.x;
    
    compute_z_expectation_phase1_kernel<<<grid1, block1, 0, compute_stream_.get()>>>(
        d_state_.get(), qubit, state_size_, d_partial_sums_.get());
    
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 2: Final reduction
    dim3 grid2(1);  // Single block for final reduction
    dim3 block2(block_size_);
    
    compute_z_expectation_phase2_kernel<<<grid2, block2, 0, compute_stream_.get()>>>(
        d_partial_sums_.get(), num_blocks, d_expectation_.get());
    
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    float expectation;
    CUDA_CHECK(cudaMemcpy(&expectation, d_expectation_.get(), sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    return expectation;
}

std::vector<Q15> CudaQuantumSimulator::measure_expectations(const std::vector<int>& qubits) {
    std::vector<Q15> expectations;
    expectations.reserve(qubits.size());
    
    for (int qubit : qubits) {
        if (qubit >= max_qubits_) {
            throw std::runtime_error(fmt::format(
                "Qubit index {} exceeds maximum {}", qubit, max_qubits_ - 1
            ));
        }
        
        float expectation = compute_z_expectation(qubit);
        expectations.push_back(Q15::from_float(expectation));
    }
    
    return expectations;
}

// --- Batch Operations (Phase 2 - TODO) ---

void CudaQuantumSimulator::simulate_batch(const std::vector<QuantumCircuit>& circuits) {
    // Phase 2 feature - not yet implemented
    // For now, fall back to sequential simulation
    for (const auto& circuit : circuits) {
        reset();
        simulate(circuit);
    }
}

std::vector<std::vector<Q15>> CudaQuantumSimulator::measure_batch_expectations(
    const std::vector<std::vector<int>>& qubit_sets
) {
    // Phase 2 feature - not yet implemented
    // For now, fall back to sequential measurement
    std::vector<std::vector<Q15>> results;
    results.reserve(qubit_sets.size());
    
    for (const auto& qubits : qubit_sets) {
        results.push_back(measure_expectations(qubits));
    }
    
    return results;
}

} // namespace cuda
} // namespace quantum
} // namespace ohmy
