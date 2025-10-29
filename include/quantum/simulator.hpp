/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_QUANTUM_SIMULATOR_IFACE_HPP
#define OHMY_MINER_QUANTUM_SIMULATOR_IFACE_HPP

#include <memory>
#include <vector>
#include "../circuit_types.hpp"

namespace ohmy {
namespace quantum {

class ISimulator {
public:
    virtual ~ISimulator() = default;

    virtual bool initialize_state() = 0;
    virtual bool apply_circuit(const QuantumCircuit& circuit) = 0;
    virtual bool apply_circuit_optimized(const QuantumCircuit& circuit) = 0;
    virtual bool measure(std::vector<double>& expectations) = 0;

    virtual int get_num_qubits() const = 0;
    virtual size_t get_state_size() const = 0;

    // Human-readable backend name (e.g., "custom", "cuquantum")
    virtual const char* backend_name() const = 0;
};

// Factory: returns best available simulator backend
std::unique_ptr<ISimulator> create_simulator(int num_qubits);

} // namespace quantum
} // namespace ohmy

#endif // OHMY_MINER_QUANTUM_SIMULATOR_IFACE_HPP
