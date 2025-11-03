/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#ifdef OHMY_WITH_CUQUANTUM

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <cuComplex.h>

#include "ohmy/quantum/simulator.hpp"

namespace ohmy {
namespace quantum {

/**
 * Triple-buffered GPU pipeline resources
 * 
 * These structures support asynchronous pipelined execution:
 * - 3 buffer sets allow overlap of H2D (batch N), Compute (batch N-1), D2H (batch N-2)
 * - Separate streams for each stage enable concurrent execution
 * - Events manage dependencies between pipeline stages
 */

/**
 * GPU device buffers for one batch of quantum circuit simulations
 */
struct GpuBatchBuffers {
    cuComplex* d_batched_states;    // State vectors [nSVs * state_size]
    float* d_angles_buf[2];         // Double-buffered angles for rotations
    cuComplex* d_mats_buf[2];       // Double-buffered matrices for rotations
    int32_t* d_indices;             // Sequential indices for matrix indexing
    int32_t* d_qubits;              // Qubits to measure
    double* d_outZ;                 // Measurement results
    void* d_workspace;              // cuQuantum workspace
    size_t workspace_size;          // Workspace size in bytes
    
    GpuBatchBuffers() = default;
    
    // Allocate all buffers for given batch size
    void allocate(size_t nSVs, size_t state_size, int num_qubits, size_t workspace_sz);
    
    // Free all buffers
    void free();
};

/**
 * CUDA streams and events for triple-buffered pipeline
 */
struct GpuPipelineStreams {
    cudaStream_t h2d_stream;        // Host-to-Device transfers
    cudaStream_t compute_stream;    // Kernel execution
    cudaStream_t d2h_stream;        // Device-to-Host transfers
    
    cudaEvent_t h2d_done;           // H2D completion marker
    cudaEvent_t compute_done;       // Compute completion marker
    
    GpuPipelineStreams() = default;
    
    // Create streams and events
    void create();
    
    // Destroy streams and events
    void destroy();
};

/**
 * Host pinned memory buffers for async transfers
 */
struct HostPinnedBuffers {
    float* h_angles_pinned[2];      // Double-buffered angles for H2D
    double* h_results_pinned;       // Results buffer for D2H
    
    HostPinnedBuffers() = default;
    
    // Allocate pinned memory
    void allocate(size_t nSVs, int num_measurements);
    
    // Free pinned memory
    void free();
};

/**
 * cuQuantum-backed simulator (custatevec)
 *
 * NOTE: Initial skeleton implementation. Gate application will be filled in
 * progressively; reset and basic metadata are functional.
 */
class CuQuantumSimulator final : public IQuantumSimulator {
public:
    explicit CuQuantumSimulator(int max_qubits);
    ~CuQuantumSimulator() override;

    // IQuantumSimulator
    void simulate(const QuantumCircuit& circuit) override;
    std::vector<Q15> measure_expectations(const std::vector<int>& qubits) override;
    void reset() override;

    void simulate_batch(const std::vector<QuantumCircuit>& circuits) override;
    std::vector<std::vector<Q15>> measure_batch_expectations(
        const std::vector<std::vector<int>>& qubit_sets) override;

    // Non-interface convenience API for batched processing:
    // Simulate a batch of circuits (identical structure, angles may differ) and measure
    // the same set of qubits for all states. Returns [batch][num_qubits] expectations (Q15).
    std::vector<std::vector<Q15>> simulate_and_measure_batched(
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<int>& qubits_to_measure);

    // New async pipeline API: externally-managed buffers and streams
    // This version allows triple-buffered pipeline execution with overlapped H2D/Compute/D2H
    // Returns results immediately (sync wrapper for phase 1, will be async in phase 2)
    std::vector<std::vector<Q15>> simulate_and_measure_batched_async(
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<int>& qubits_to_measure,
        GpuBatchBuffers& buffers,
        HostPinnedBuffers& host_buffers,
        GpuPipelineStreams& streams);

    int max_qubits() const override { return max_qubits_; }
    bool supports_batch() const override { return false; }
    std::string backend_name() const override { return "CUQUANTUM"; }

private:
    void init_resources();
    void free_resources();

    // Allow internal batched helper access to private members
    friend std::vector<std::vector<Q15>> cuquantum_simulate_and_measure_batched(
        CuQuantumSimulator& self,
        const std::vector<QuantumCircuit>& circuits,
        const std::vector<int>& qubits_to_measure);

private:
    int max_qubits_;
    size_t state_size_{};           // number of amplitudes = 2^max_qubits
    custatevecHandle_t handle_{};   // cuQuantum handle
    cuComplex* d_state_{};          // device state vector (float32 complex)
    cuComplex* d_gate2x2_{};        // reusable device buffer for 2x2 gate matrices
    void* d_workspace_{};           // reusable cuStateVec workspace
    size_t workspace_size_{};       // workspace size in bytes
    cudaStream_t stream_{};         // dedicated compute stream for cuStateVec operations

    // Persistent pools for batched async path (avoid per-call alloc/free)
    cuComplex* d_batched_states_pool_{}; // [nSVs * state_size]
    size_t d_batched_states_bytes_{0};
    void* d_ws_batched_pool_{};
    size_t ws_batched_pool_size_{0};
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_WITH_CUQUANTUM
