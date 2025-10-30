/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#include <fmt/format.h>
#include <cmath>

namespace ohmy {
namespace quantum {

/**
 * Basic CPU quantum simulator implementation
 */
class CPUSimulator : public IQuantumSimulator {
public:
    CPUSimulator(int max_qubits) 
        : max_qubits_(max_qubits)
        , state_size_(1ULL << max_qubits) {
        state_.resize(state_size_, Complex(0.0, 0.0));
        reset();
    }

    void simulate(const QuantumCircuit& circuit) override {
        if (circuit.num_qubits() > max_qubits_) {
            throw std::runtime_error("Circuit too large for simulator");
        }

        // Apply rotation gates
        for (const auto& gate : circuit.rotation_gates()) {
            apply_rotation(gate.qubit, gate.angle);
        }

        // Apply CNOT gates
        for (const auto& gate : circuit.cnot_gates()) {
            apply_cnot(gate.control, gate.target);
        }
    }

    std::vector<Q15> measure_expectations(const std::vector<int>& qubits) override {
        std::vector<Q15> expectations;
        expectations.reserve(qubits.size());

        for (int qubit : qubits) {
            double expectation = compute_z_expectation(qubit);
            expectations.push_back(Q15::from_float(expectation));
        }

        return expectations;
    }

    void reset() override {
        // Initialize to |00...0⟩ state
        std::fill(state_.begin(), state_.end(), Complex(0.0, 0.0));
        state_[0] = Complex(1.0, 0.0);
    }

    void simulate_batch(const std::vector<QuantumCircuit>& circuits) override {
        // For CPU simulator, just simulate sequentially
        for (const auto& circuit : circuits) {
            simulate(circuit);
        }
    }

    std::vector<std::vector<Q15>> measure_batch_expectations(
        const std::vector<std::vector<int>>& qubit_sets) override {
        std::vector<std::vector<Q15>> results;
        for (const auto& qubits : qubit_sets) {
            results.push_back(measure_expectations(qubits));
        }
        return results;
    }

    int max_qubits() const override { return max_qubits_; }
    bool supports_batch() const override { return false; }
    std::string backend_name() const override { return "CPU_BASIC"; }

private:
    void apply_rotation(int qubit, double angle) {
        double cos_half = std::cos(angle / 2.0);
        double sin_half = std::sin(angle / 2.0);

        size_t qubit_mask = 1ULL << qubit;

        for (size_t i = 0; i < state_size_; ++i) {
            if ((i & qubit_mask) == 0) {  // qubit is 0
                size_t j = i | qubit_mask;  // corresponding index with qubit = 1
                
                Complex alpha = state_[i];
                Complex beta = state_[j];
                
                state_[i] = cos_half * alpha - Complex(0, sin_half) * beta;
                state_[j] = cos_half * beta - Complex(0, sin_half) * alpha;
            }
        }
    }

    void apply_cnot(int control, int target) {
        size_t control_mask = 1ULL << control;
        size_t target_mask = 1ULL << target;

        for (size_t i = 0; i < state_size_; ++i) {
            if ((i & control_mask) != 0) {  // control qubit is 1
                size_t j = i ^ target_mask;  // flip target qubit
                if (i < j) {  // avoid double swap
                    std::swap(state_[i], state_[j]);
                }
            }
        }
    }

    double compute_z_expectation(int qubit) {
        size_t qubit_mask = 1ULL << qubit;
        double expectation = 0.0;

        for (size_t i = 0; i < state_size_; ++i) {
            double probability = std::norm(state_[i]);
            if ((i & qubit_mask) == 0) {
                expectation += probability;  // |0⟩ contributes +1
            } else {
                expectation -= probability;  // |1⟩ contributes -1
            }
        }

        return expectation;
    }

    int max_qubits_;
    size_t state_size_;
    StateVector state_;
};

// Factory implementation
std::unique_ptr<IQuantumSimulator> SimulatorFactory::create(Backend backend, int max_qubits) {
    switch (backend) {
        case Backend::CPU_BASIC:
            return std::make_unique<CPUSimulator>(max_qubits);
        
        case Backend::CUDA_CUSTOM:
            throw std::runtime_error("CUDA backend not yet implemented");
        
        case Backend::CUQUANTUM:
            throw std::runtime_error("cuQuantum backend not yet implemented");
        
        default:
            throw std::invalid_argument("Unknown backend");
    }
}

std::vector<SimulatorFactory::Backend> SimulatorFactory::available_backends() {
    return {Backend::CPU_BASIC};
}

std::string SimulatorFactory::backend_name(Backend backend) {
    switch (backend) {
        case Backend::CPU_BASIC: return "CPU_BASIC";
        case Backend::CUDA_CUSTOM: return "CUDA_CUSTOM";
        case Backend::CUQUANTUM: return "CUQUANTUM";
        default: return "UNKNOWN";
    }
}

} // namespace quantum
} // namespace ohmy