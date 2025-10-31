/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#include <stdexcept>

namespace ohmy {
namespace quantum {

QuantumCircuit::QuantumCircuit(int num_qubits) 
    : num_qubits_(num_qubits) {
    if (num_qubits <= 0 || num_qubits > 16) {
        throw std::invalid_argument("Invalid number of qubits");
    }
}

void QuantumCircuit::add_rotation(int qubit, double angle, RotationAxis axis) {
    if (qubit < 0 || qubit >= num_qubits_) {
        throw std::out_of_range("Qubit index out of range");
    }
    rotation_gates_.push_back({qubit, angle, axis});
}

void QuantumCircuit::add_cnot(int control, int target) {
    if (control < 0 || control >= num_qubits_ || 
        target < 0 || target >= num_qubits_ || 
        control == target) {
        throw std::invalid_argument("Invalid CNOT gate qubits");
    }
    cnot_gates_.push_back({control, target});
}

void QuantumCircuit::clear() {
    rotation_gates_.clear();
    cnot_gates_.clear();
}

} // namespace quantum
} // namespace ohmy