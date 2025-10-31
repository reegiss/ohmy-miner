/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/batched_cuda_simulator.hpp"
#include <fmt/format.h>

namespace ohmy {
namespace quantum {
namespace cuda {

// Forward declarations of batched kernels
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

BatchedCudaSimulator::BatchedCudaSimulator(int num_qubits, int batch_size, int device_id)
    : num_qubits_(num_qubits)
    , batch_size_(batch_size)
    , device_id_(device_id)
    , state_size_(1ULL << num_qubits)
    , block_size_(DEFAULT_BLOCK_SIZE)
    , d_batch_states_(batch_size * state_size_)
    , d_batch_expectations_(batch_size)
    , compute_stream_()
{
    // Set device
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Query device
    device_info_ = DeviceInfo::query(device_id_);
    
    if (!device_info_.is_compatible()) {
        throw std::runtime_error(fmt::format(
            "GPU compute capability {}.{} below minimum 7.5",
            device_info_.compute_capability_major,
            device_info_.compute_capability_minor
        ));
    }
    
    // Check memory requirements
    size_t total_memory_needed = batch_size * state_size_ * sizeof(Complex);
    if (device_info_.free_memory < total_memory_needed * 1.2) {
        throw std::runtime_error(fmt::format(
            "Insufficient GPU memory for batch size {}: need {}, have {} free",
            batch_size,
            MemoryRequirements::format_bytes(total_memory_needed * 1.2),
            MemoryRequirements::format_bytes(device_info_.free_memory)
        ));
    }
    
    fmt::print("[CUDA Batch] Initialized {} with {} qubits, batch size {}\n",
               device_info_.name, num_qubits_, batch_size_);
    fmt::print("[CUDA Batch] Memory: {} ({} per state × {} states)\n",
               MemoryRequirements::format_bytes(total_memory_needed),
               MemoryRequirements::format_bytes(state_size_ * sizeof(Complex)),
               batch_size_);
    
    reset_batch();
}

BatchedCudaSimulator::~BatchedCudaSimulator() {
    // RAII handles cleanup
}

// --- Batch Operations ---

void BatchedCudaSimulator::reset_batch() {
    // Initialize all states to |0...0⟩
    dim3 grid = calculate_batch_grid_size(state_size_);
    dim3 block(block_size_);
    
    init_zero_state_batch_kernel<<<grid, block, 0, compute_stream_.get()>>>(
        d_batch_states_.get(), batch_size_, state_size_);
    
    CUDA_CHECK(cudaGetLastError());
    compute_stream_.synchronize();
}

void BatchedCudaSimulator::apply_rotation_batch(int qubit, float angle, RotationAxis axis) {
    if (axis == RotationAxis::Y) {
        // R_Y processes pairs of amplitudes
        dim3 grid = calculate_batch_grid_size(state_size_ / 2, true);
        dim3 block(block_size_);
        
        apply_rotation_y_batch_kernel<<<grid, block, 0, compute_stream_.get()>>>(
            d_batch_states_.get(), batch_size_, qubit, angle, state_size_);
    } else {
        // R_Z processes all amplitudes
        dim3 grid = calculate_batch_grid_size(state_size_);
        dim3 block(block_size_);
        
        apply_rotation_z_batch_kernel<<<grid, block, 0, compute_stream_.get()>>>(
            d_batch_states_.get(), batch_size_, qubit, angle, state_size_);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void BatchedCudaSimulator::apply_cnot_batch(int control, int target) {
    dim3 grid = calculate_batch_grid_size(state_size_);
    dim3 block(block_size_);
    
    apply_cnot_batch_kernel<<<grid, block, 0, compute_stream_.get()>>>(
        d_batch_states_.get(), batch_size_, control, target, state_size_);
    
    CUDA_CHECK(cudaGetLastError());
}

std::vector<float> BatchedCudaSimulator::measure_batch(const std::vector<int>& qubits) {
    std::vector<float> all_expectations;
    all_expectations.reserve(batch_size_ * qubits.size());
    
    for (int qubit : qubits) {
        // Each state gets its own block for measurement
        dim3 grid(batch_size_);
        dim3 block(block_size_);
        
        compute_z_expectation_batch_kernel<<<grid, block, 0, compute_stream_.get()>>>(
            d_batch_states_.get(), batch_size_, qubit, state_size_, d_batch_expectations_.get());
        
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results to host
        std::vector<float> qubit_expectations(batch_size_);
        CUDA_CHECK(cudaMemcpy(qubit_expectations.data(), d_batch_expectations_.get(),
                              batch_size_ * sizeof(float), cudaMemcpyDeviceToHost));
        
        all_expectations.insert(all_expectations.end(),
                               qubit_expectations.begin(),
                               qubit_expectations.end());
    }
    
    return all_expectations;
}

std::vector<std::vector<Q15>> BatchedCudaSimulator::simulate_and_measure_batch(
    const std::vector<QuantumCircuit>& circuits,
    const std::vector<int>& qubits_to_measure
) {
    if (circuits.empty()) {
        return {};
    }
    
    if (circuits.size() != static_cast<size_t>(batch_size_)) {
        throw std::runtime_error(fmt::format(
            "Circuit count {} != batch size {}",
            circuits.size(), batch_size_
        ));
    }
    
    // Validate all circuits have same structure
    const auto& ref_circuit = circuits[0];
    for (size_t i = 1; i < circuits.size(); i++) {
        if (circuits[i].rotation_gates().size() != ref_circuit.rotation_gates().size() ||
            circuits[i].cnot_gates().size() != ref_circuit.cnot_gates().size()) {
            throw std::runtime_error("All circuits in batch must have same structure");
        }
    }
    
    // Reset all states
    reset_batch();
    
    // Apply gates from reference circuit
    // (In qhash, all circuits have same gates, only nonce differs in hash seed)
    for (const auto& gate : ref_circuit.rotation_gates()) {
        apply_rotation_batch(gate.qubit, static_cast<float>(gate.angle), gate.axis);
    }
    
    for (const auto& gate : ref_circuit.cnot_gates()) {
        apply_cnot_batch(gate.control, gate.target);
    }
    
    // Synchronize before measurement
    compute_stream_.synchronize();
    
    // Measure all qubits for all states
    auto flat_expectations = measure_batch(qubits_to_measure);
    
    // Reshape results: [batch_size][num_qubits]
    std::vector<std::vector<Q15>> results;
    results.reserve(batch_size_);
    
    size_t num_qubits = qubits_to_measure.size();
    for (int batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
        std::vector<Q15> state_expectations;
        state_expectations.reserve(num_qubits);
        
        for (size_t q = 0; q < num_qubits; q++) {
            float expectation = flat_expectations[q * batch_size_ + batch_idx];
            state_expectations.push_back(Q15::from_float(expectation));
        }
        
        results.push_back(std::move(state_expectations));
    }
    
    return results;
}

// --- Helper Methods ---

dim3 BatchedCudaSimulator::calculate_batch_grid_size(
    size_t elements_per_state,
    bool /* is_pair_kernel */  // Unused for now, reserved for future optimization
) const {
    size_t total_elements = batch_size_ * elements_per_state;
    size_t num_blocks = (total_elements + block_size_ - 1) / block_size_;
    return dim3(static_cast<unsigned int>(num_blocks));
}

int BatchedCudaSimulator::get_optimal_batch_size() const {
    size_t memory_per_state = state_size_ * sizeof(Complex);
    size_t usable_memory = device_info_.free_memory * 0.8;  // Leave 20% headroom
    int optimal = static_cast<int>(usable_memory / memory_per_state);
    
    // Clamp to reasonable range
    return std::min(std::max(optimal, 100), MAX_BATCH_SIZE);
}

} // namespace cuda
} // namespace quantum
} // namespace ohmy
