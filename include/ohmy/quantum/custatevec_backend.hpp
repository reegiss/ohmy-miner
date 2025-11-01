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
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_WITH_CUQUANTUM
