/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "ohmy/quantum/simulator.hpp"
#include "ohmy/quantum/cuda_types.hpp"
#include <memory>
#include <vector>

namespace ohmy {
namespace quantum {
namespace cuda {

/**
 * CUDA-accelerated quantum circuit simulator
 * 
 * Implements IQuantumSimulator interface using GPU kernels for quantum gates
 * and measurements. Designed for 16-qubit qhash circuits with batched nonce
 * processing for maximum mining hashrate.
 * 
 * Memory Layout (16 qubits):
 * - State vector: 2^16 amplitudes × 8 bytes (float32 complex) = 512 KB
 * - Workspace: 512 KB (for intermediate results)
 * - Total per nonce: ~1 MB device memory
 * 
 * Performance Target:
 * - Single nonce: 500-1,000 H/s (Phase 1)
 * - Batched (1000+ nonces): 5,000-15,000 H/s (Phase 2)
 */
class CudaQuantumSimulator : public IQuantumSimulator {
public:
    /**
     * Constructor: Initialize GPU and allocate memory
     * 
     * @param max_qubits Maximum number of qubits (default: 16)
     * @param device_id  CUDA device ID to use (default: 0)
     * 
     * @throws std::runtime_error if GPU initialization fails or insufficient memory
     */
    explicit CudaQuantumSimulator(int max_qubits = 16, int device_id = 0);
    
    /**
     * Destructor: Free GPU resources
     * 
     * Uses RAII pattern - all CUDA resources cleaned up automatically
     */
    ~CudaQuantumSimulator() override;
    
    // --- IQuantumSimulator interface implementation ---
    
    /**
     * Simulate quantum circuit on GPU
     * 
     * Applies all rotation gates followed by all CNOT gates to state vector.
     * State must be reset() before calling simulate().
     * 
     * @param circuit Circuit to simulate (gates applied in order)
     */
    void simulate(const QuantumCircuit& circuit) override;
    
    /**
     * Measure Z-basis expectation values for multiple qubits
     * 
     * Returns ⟨Z⟩ for each qubit in fixed-point Q15 format for consensus safety.
     * 
     * @param qubits List of qubit indices to measure
     * @return Fixed-point expectation values (one per qubit)
     */
    std::vector<Q15> measure_expectations(const std::vector<int>& qubits) override;
    
    /**
     * Reset state to |0...0⟩
     * 
     * Must be called before each circuit simulation.
     */
    void reset() override;
    
    /**
     * Batch simulation (Phase 2 - not yet implemented)
     * 
     * Currently falls back to sequential simulation.
     * Future: Process multiple circuits in parallel on GPU.
     */
    void simulate_batch(const std::vector<QuantumCircuit>& circuits) override;
    
    /**
     * Batch measurement (Phase 2 - not yet implemented)
     * 
     * Currently falls back to sequential measurement.
     */
    std::vector<std::vector<Q15>> measure_batch_expectations(
        const std::vector<std::vector<int>>& qubit_sets) override;
    
    // --- Simulator properties ---
    
    int max_qubits() const override { return max_qubits_; }
    bool supports_batch() const override { return false; }  // Phase 2 feature
    std::string backend_name() const override { return "CUDA_CUSTOM"; }
    
    /**
     * Get GPU device information
     */
    DeviceInfo get_device_info() const { return device_info_; }
    
    /**
     * Get memory usage statistics
     */
    MemoryRequirements get_memory_usage() const { return memory_reqs_; }

private:
    // --- Configuration ---
    int max_qubits_;              // Maximum supported qubits (typically 16)
    int device_id_;               // CUDA device ID
    size_t state_size_;           // Number of amplitudes (2^max_qubits)
    int block_size_;              // Threads per block (default: 256)
    
    // --- GPU Resources (RAII-managed) ---
    DeviceMemory<Complex> d_state_;         // Main quantum state vector
    DeviceMemory<Complex> d_workspace_;     // Scratch space for operations
    DeviceMemory<float> d_partial_sums_;    // Intermediate reduction results
    DeviceMemory<float> d_expectation_;     // Final expectation value
    
    PinnedMemory<Complex> h_state_;         // Host pinned memory for transfers
    
    StreamHandle compute_stream_;           // CUDA stream for computation
    StreamHandle transfer_stream_;          // CUDA stream for memory transfers
    
    // --- Device Info ---
    DeviceInfo device_info_;
    MemoryRequirements memory_reqs_;
    
    // --- Helper methods ---
    
    /**
     * Apply single-qubit rotation gate (R_Y or R_Z)
     */
    void apply_rotation(int qubit, float angle, RotationAxis axis);
    
    /**
     * Apply two-qubit CNOT gate
     */
    void apply_cnot(int control, int target);
    
    /**
     * Compute Z-basis expectation value for single qubit
     * 
     * Uses two-phase hierarchical reduction:
     * 1. Block-level reduction → partial sums
     * 2. Final reduction → single value
     */
    float compute_z_expectation(int qubit);
    
    /**
     * Calculate grid dimensions for kernel launch
     */
    dim3 calculate_grid_size(size_t num_threads) const {
        size_t num_blocks = (num_threads + block_size_ - 1) / block_size_;
        return dim3(static_cast<unsigned int>(num_blocks));
    }
    
    /**
     * Validate circuit fits within max_qubits
     */
    void validate_circuit(const QuantumCircuit& circuit) const;
};

} // namespace cuda
} // namespace quantum
} // namespace ohmy
