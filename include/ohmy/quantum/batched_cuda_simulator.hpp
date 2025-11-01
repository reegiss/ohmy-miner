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

#ifdef OHMY_WITH_CUQUANTUM
// Forward declaration
class CuQuantumSimulator;
#endif

namespace cuda {

/**
 * Batched CUDA quantum simulator
 * 
 * Processes multiple quantum circuits in parallel on GPU
 * for maximum mining hashrate. Uses memory layout [batch_size][state_size]
 * with all states contiguous in device memory for coalesced access.
 * 
 * Target Performance:
 * - 1000 nonces: 10-20x speedup vs single nonce
 * - GTX 1660 Super: 10-50 KH/s
 * - RTX 4090: 50-200 KH/s
 */
class BatchedCudaSimulator {
public:
    /**
     * Constructor
     * 
     * @param num_qubits Maximum qubits per circuit
     * @param batch_size Number of circuits to process in parallel
     * @param device_id  CUDA device ID
     */
    explicit BatchedCudaSimulator(
        int num_qubits = 16,
        int batch_size = DEFAULT_BATCH_SIZE,
        int device_id = 0
    );
    
    ~BatchedCudaSimulator();
    
    /**
     * Process batch of circuits in parallel
     * 
     * All circuits must have same structure (same gates in same order).
     * This is the case for qhash mining where only nonce changes.
     * 
     * @param circuits Vector of circuits (all must be identical structure)
     * @return Measurement results for each circuit
     */
    std::vector<std::vector<Q15>> simulate_and_measure_batch(
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<int>& qubits_to_measure
    );
    
    /**
     * Get optimal batch size for current GPU
     * 
     * Calculates based on available VRAM and leaves 20% headroom
     */
    int get_optimal_batch_size() const;
    
    /**
     * Get current batch size
     */
    int batch_size() const { return batch_size_; }
    
    /**
     * Get device info
     */
    DeviceInfo get_device_info() const { return device_info_; }

#ifdef OHMY_WITH_CUQUANTUM
    /**
     * Get cuQuantum backend (for direct pipeline access)
     */
    CuQuantumSimulator* get_cuquantum_backend() const { return cuquantum_backend_.get(); }
#endif

private:
    // Configuration
    int num_qubits_;
    int batch_size_;
    int device_id_;
    size_t state_size_;
    int block_size_;
    
    // GPU resources
    DeviceMemory<Complex> d_batch_states_;      // [batch_size][state_size]
    DeviceMemory<float> d_batch_expectations_;  // [batch_size]
    StreamHandle compute_stream_;
    
    // Device info
    DeviceInfo device_info_;
    
#ifdef OHMY_WITH_CUQUANTUM
    // cuQuantum backend for optimal performance
    std::unique_ptr<CuQuantumSimulator> cuquantum_backend_;
    bool use_cuquantum_ = true;
#endif
    
    // Helper methods
    void reset_batch();
    void apply_rotation_batch(int qubit, float angle, RotationAxis axis);
    void apply_cnot_batch(int control, int target);
    std::vector<float> measure_batch(const std::vector<int>& qubits);
    
    dim3 calculate_batch_grid_size(size_t elements_per_state, bool is_pair_kernel = false) const;
};

} // namespace cuda
} // namespace quantum
} // namespace ohmy
