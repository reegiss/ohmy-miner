/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "simulator.hpp"
#include "custatevec_backend.hpp"
#include <cuda_runtime.h>
#include <custatevec.h>
#include <memory>
#include <vector>

namespace ohmy::quantum {

/**
 * @brief High-performance quantum simulator with gate fusion optimization
 * 
 * This backend combines custom fused rotation kernels with cuStateVec for CNOTs:
 * - Custom kernels: All Y/Z rotations per layer (4 kernel launches total)
 * - cuStateVec: CNOT gates only (8 custatevecApplyMatrixBatched calls)
 * 
 * Performance improvement over pure cuStateVec:
 * - Before: 72 API calls per circuit (64 rotations + 8 CNOTs)
 * - After:  12 operations per circuit (4 custom + 8 cuStateVec)
 * - Expected: 10-15× speedup from reduced kernel launch overhead
 */
class FusedCuQuantumBackend : public QuantumSimulator {
public:
    FusedCuQuantumBackend(int num_qubits, int batch_size, int device_id = 0);
    ~FusedCuQuantumBackend() override;

    // Disable copy/move (CUDA resources)
    FusedCuQuantumBackend(const FusedCuQuantumBackend&) = delete;
    FusedCuQuantumBackend& operator=(const FusedCuQuantumBackend&) = delete;

    /**
     * @brief Simulate circuits with gate fusion optimization
     * 
     * Uses hybrid approach:
     * 1. Custom fused kernels for all rotations in a layer (4 launches)
     * 2. cuStateVec batched API for CNOTs (8 calls)
     * 3. Custom measurement kernel for Z-expectations
     */
    std::vector<std::vector<Q15>> simulate_and_measure_batched(
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<int>& qubits_to_measure
    ) override;

    /**
     * @brief Async version with external buffers (for triple-buffering)
     */
    std::vector<std::vector<Q15>> simulate_and_measure_batched_async(
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<int>& qubits_to_measure,
        GpuBatchBuffers& buffers,
        HostPinnedBuffers& host_buffers,
        GpuPipelineStreams& streams
    );

    int batch_size() const override { return batch_size_; }
    int num_qubits() const override { return num_qubits_; }
    int device_id() const override { return device_id_; }

private:
    int num_qubits_;
    int batch_size_;
    int device_id_;
    size_t state_size_;  // 2^num_qubits

    // cuStateVec resources (for CNOTs only)
    custatevecHandle_t handle_;
    void* d_custatevec_workspace_;
    size_t workspace_size_;

    // Persistent device memory pools
    cuFloatComplex* d_batched_states_pool_;  // Reused across calls
    float* d_angles_pool_;                    // Staging for rotation angles
    
    // CUDA stream management
    cudaStream_t internal_stream_;  // Used when not in async mode

    /**
     * @brief Initialize state vectors to |0...0⟩
     */
    void initialize_states_batch(
        cuFloatComplex* d_states,
        int batch_size,
        cudaStream_t stream
    );

    /**
     * @brief Apply all Y-rotations for a layer using fused kernel
     */
    void apply_fused_ry_layer(
        cuFloatComplex* d_states,
        const std::vector<QuantumCircuit>& circuits,
        int layer_idx,
        cudaStream_t stream
    );

    /**
     * @brief Apply all Z-rotations for a layer using fused kernel
     */
    void apply_fused_rz_layer(
        cuFloatComplex* d_states,
        const std::vector<QuantumCircuit>& circuits,
        int layer_idx,
        cudaStream_t stream
    );

    /**
     * @brief Apply CNOT gates using cuStateVec (broadcast mode)
     */
    void apply_cnot_gates(
        cuFloatComplex* d_states,
        const std::vector<CNOTGate>& cnots,
        int batch_size,
        cudaStream_t stream
    );

    /**
     * @brief Compute Z-expectations for all qubits
     */
    void compute_z_expectations_batch(
        const cuFloatComplex* d_states,
        float* d_results,
        const std::vector<int>& qubits,
        int batch_size,
        cudaStream_t stream
    );
};

} // namespace ohmy::quantum
