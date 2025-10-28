/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum/simulator.hpp"
#include "quantum_kernel.cuh"
#include <memory>

#if defined(OHMY_WITH_CUQUANTUM)
#include "quantum/custatevec_backend.hpp"
#endif

namespace ohmy {
namespace quantum {

// Lightweight adapter to reuse existing QuantumSimulator behind the ISimulator interface
class CustomSimulatorAdapter final : public ISimulator {
public:
    explicit CustomSimulatorAdapter(int n) : impl_(n) {}

    bool initialize_state() override { return impl_.initialize_state(); }
    bool apply_circuit(const QuantumCircuit& circuit) override { return impl_.apply_circuit(circuit); }
    bool apply_circuit_optimized(const QuantumCircuit& circuit) override { return impl_.apply_circuit_optimized(circuit); }
    bool measure(std::vector<double>& expectations) override { return impl_.measure(expectations); }

    int get_num_qubits() const override { return impl_.get_num_qubits(); }
    size_t get_state_size() const override { return impl_.get_state_size(); }

    const char* backend_name() const override { return "custom"; }

private:
    QuantumSimulator impl_;
};

std::unique_ptr<ISimulator> create_simulator(int num_qubits) {
#if defined(OHMY_WITH_CUQUANTUM)
    // Prefer cuQuantum backend when compiled with support
    try {
        return std::make_unique<CuQuantumSimulator>(num_qubits);
    } catch (...) {
        // Fallback to custom backend if cuQuantum construction fails
    }
#endif
    return std::make_unique<CustomSimulatorAdapter>(num_qubits);
}

} // namespace quantum
} // namespace ohmy
