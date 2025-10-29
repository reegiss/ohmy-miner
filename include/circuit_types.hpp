/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_CIRCUIT_TYPES_HPP
#define OHMY_MINER_CIRCUIT_TYPES_HPP

#include <vector>
#include <cstdint>

namespace ohmy {
namespace quantum {

/**
 * @brief Quantum gate types
 */
enum class GateType {
    RX,    // Rotation around X axis
    RY,    // Rotation around Y axis
    RZ,    // Rotation around Z axis
    CNOT   // Controlled-NOT
};

/**
 * @brief Single quantum gate operation
 */
struct QuantumGate {
    GateType type;
    int target_qubit;      // Qubit index this gate acts on
    int control_qubit;     // For CNOT: control qubit index (-1 if not applicable)
    double angle;          // Rotation angle in radians (for Rx, Ry, Rz)
    
    QuantumGate(GateType t, int target, double a = 0.0, int control = -1)
        : type(t), target_qubit(target), control_qubit(control), angle(a) {}
};

/**
 * @brief Quantum circuit configuration
 */
struct QuantumCircuit {
    int num_qubits;
    std::vector<QuantumGate> gates;
    
    QuantumCircuit(int n_qubits) : num_qubits(n_qubits) {}
    
    void add_gate(GateType type, int target, double angle = 0.0, int control = -1) {
        gates.emplace_back(type, target, angle, control);
    }
    
    size_t state_vector_size() const {
        return 1ULL << num_qubits;  // 2^n
    }
};

} // namespace quantum
} // namespace ohmy

#endif // OHMY_MINER_CIRCUIT_TYPES_HPP
