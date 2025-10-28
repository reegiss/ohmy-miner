/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_BATCHED_QUANTUM_CUH
#define OHMY_MINER_BATCHED_QUANTUM_CUH

#include "quantum_kernel.cuh"
#include <vector>

namespace ohmy {
namespace quantum {

/**
 * @brief Batched quantum simulator for parallel nonce processing
 * 
 * This class extends QuantumSimulator to process multiple nonces (work units)
 * in parallel on the GPU. By batching work, we achieve better GPU utilization
 * and amortize memory transfer overhead.
 * 
 * ARCHITECTURE:
 * - Each nonce gets its own state vector (1MB for 16 qubits)
 * - All state vectors stored contiguously in GPU memory
 * - Kernels process batch_size states in parallel using 2D grid
 * 
 * MEMORY LAYOUT:
 * - batch_size = 64 → 64 MB GPU memory (64 × 1MB state vectors)
 * - Each state: 65536 complex amplitudes × 16 bytes = 1 MB
 * - Modern GPUs have 8-24 GB → can handle batch_size = 1000+
 * 
 * EXPECTED PERFORMANCE:
 * - Single nonce: ~3ms (326 H/s) with 30% GPU utilization
 * - Batched (64 nonces): ~64ms (1000 H/s) with 95% GPU utilization
 * - Expected speedup: 2-3× from better resource utilization
 * 
 * USAGE:
 *   BatchedQuantumSimulator batch_sim(16, 64);  // 16 qubits, 64 batch size
 *   batch_sim.initialize_states();
 *   
 *   // Process 64 different nonces
 *   for (int i = 0; i < 64; i++) {
 *       circuits[i] = create_qtc_circuit(angles[i]);
 *   }
 *   
 *   batch_sim.apply_circuits_optimized(circuits);
 *   batch_sim.measure_all(expectations);  // expectations[64][16]
 */
class BatchedQuantumSimulator {
public:
    /**
     * @brief Constructor
     * @param num_qubits Number of qubits per state vector
     * @param batch_size Number of states to process in parallel
     */
    BatchedQuantumSimulator(int num_qubits, int batch_size);
    
    /**
     * @brief Destructor - cleanup GPU memory
     */
    ~BatchedQuantumSimulator();
    
    /**
     * @brief Initialize all states to |0⟩^⊗n
     */
    bool initialize_states();
    
    /**
     * @brief Apply circuits to all states in batch (OPTIMIZED)
     * 
     * Processes batch_size circuits in parallel using 2D grid:
     * - blockIdx.x: amplitude index within state vector
     * - blockIdx.y: batch index (which nonce/circuit)
     * 
     * @param circuits Vector of circuits (size = batch_size)
     */
    bool apply_circuits_optimized(const std::vector<QuantumCircuit>& circuits);
    
    /**
     * @brief Measure all states in batch
     * @param expectations Output matrix [batch_size][num_qubits]
     */
    bool measure_all(std::vector<std::vector<double>>& expectations);
    
    /**
     * @brief Get batch size
     */
    int get_batch_size() const { return batch_size_; }
    
    /**
     * @brief Get number of qubits per state
     */
    int get_num_qubits() const { return num_qubits_; }
    
    /**
     * @brief Get GPU memory usage in bytes
     */
    size_t get_memory_usage() const {
        return batch_size_ * state_size_ * sizeof(Complex) +
               batch_size_ * num_qubits_ * sizeof(double);
    }

private:
    int num_qubits_;
    int batch_size_;
    size_t state_size_;  // 2^num_qubits
    
    // GPU memory: batch_size contiguous state vectors
    Complex* d_states_;  // [batch_size][state_size]
    
    // GPU memory: expectations for all states
    double* d_expectations_;  // [batch_size][num_qubits]
    
    // Disable copy
    BatchedQuantumSimulator(const BatchedQuantumSimulator&) = delete;
    BatchedQuantumSimulator& operator=(const BatchedQuantumSimulator&) = delete;
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_MINER_BATCHED_QUANTUM_CUH
