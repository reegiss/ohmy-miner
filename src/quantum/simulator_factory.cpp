/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum/simulator_factory.hpp"
#include "circuit_types.hpp"
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
    try {
        // cuQuantum single-state is fastest for individual nonce processing
        return std::make_unique<CuQuantumSimulator>(num_qubits);
    } catch (const std::exception&) {
        // Fall back to custom if cuQuantum fails
        return std::make_unique<CustomSimulatorAdapter>(num_qubits);
    }
#else
    return std::make_unique<CustomSimulatorAdapter>(num_qubits);
#endif
}

} // namespace quantum
} // namespace ohmy
