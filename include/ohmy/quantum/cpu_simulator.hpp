/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "ohmy/quantum/simulator.hpp"

namespace ohmy {
namespace quantum {

/**
 * Basic CPU quantum simulator implementation
 */
class CPUSimulator : public IQuantumSimulator {
public:
    CPUSimulator(int max_qubits);

    void simulate(const QuantumCircuit& circuit) override;
    std::vector<Q15> measure_expectations(const std::vector<int>& qubits) override;
    void reset() override;
    
    void simulate_batch(const std::vector<QuantumCircuit>& circuits) override;
    std::vector<std::vector<Q15>> measure_batch_expectations(
        const std::vector<std::vector<int>>& qubit_sets) override;
    
    int max_qubits() const override { return max_qubits_; }
    bool supports_batch() const override { return false; }
    std::string backend_name() const override { return "CPU_BASIC"; }

private:
    void apply_rotation(int qubit, double angle);
    void apply_cnot(int control, int target);
    double compute_z_expectation(int qubit);

    int max_qubits_;
    size_t state_size_;
    StateVector state_;
};

} // namespace quantum
} // namespace ohmy
