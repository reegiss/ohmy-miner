/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "circuit_generator.hpp"
#include <cmath>

namespace ohmy::quantum {

uint8_t CircuitGenerator::extract_nibble_idx(const std::array<uint8_t, 32>& hash, int nibble_idx) {
    // Each byte contributes two nibbles: high then low
    // nibble_idx in [0..63]
    int byte_index = nibble_idx / 2;
    bool high = (nibble_idx % 2 == 0);
    uint8_t byte = (byte_index >= 0 && byte_index < 32) ? hash[byte_index] : 0;
    return high ? static_cast<uint8_t>((byte >> 4) & 0x0F) : static_cast<uint8_t>(byte & 0x0F);
}

double CircuitGenerator::nibble_to_angle_qhash(uint8_t nibble) {
    // Reference: angle = -nibble * π/16
    return -static_cast<double>(nibble) * (M_PI / 16.0);
}

void CircuitGenerator::add_cnot_chain(QuantumCircuit& circuit, int num_qubits) {
    // Nearest-neighbor controlled-X chain: (0->1), (1->2), ..., (n-2->n-1)
    for (int i = 0; i < num_qubits - 1; ++i) {
        circuit.add_gate(GateType::CNOT, /*target=*/i + 1, /*angle=*/0.0, /*control=*/i);
    }
}

QuantumCircuit CircuitGenerator::build_from_hash(const std::array<uint8_t, 32>& hash, int num_qubits) {
    QuantumCircuit circuit(num_qubits);

    // QTC reference qhash circuit:
    // NUM_LAYERS = 2
    // For each layer:
    //   - Apply RY(theta_y[q]) for each qubit q
    //   - Apply RZ(theta_z[q]) for each qubit q
    //   - Apply nearest-neighbor CNOT chain (0->1, 1->2, ..., n-2->n-1)
    // where theta = -nibble * π/16, consuming exactly 64 nibbles.

    int nibble_idx = 0; // 0..63
    for (int layer = 0; layer < 2; ++layer) {
        // RY layer
        for (int q = 0; q < num_qubits; ++q) {
            uint8_t n = extract_nibble_idx(hash, nibble_idx++);
            double angle = nibble_to_angle_qhash(n);
            circuit.add_gate(GateType::RY, q, angle);
        }
        // RZ layer
        for (int q = 0; q < num_qubits; ++q) {
            uint8_t n = extract_nibble_idx(hash, nibble_idx++);
            double angle = nibble_to_angle_qhash(n);
            circuit.add_gate(GateType::RZ, q, angle);
        }
        // CNOT chain
        add_cnot_chain(circuit, num_qubits);
    }

    return circuit;
}

std::vector<QuantumCircuit> CircuitGenerator::build_from_hash_batch(
    const std::vector<std::array<uint8_t, 32>>& hashes,
    int num_qubits)
{
    const size_t batch_size = hashes.size();
    std::vector<QuantumCircuit> circuits;
    circuits.reserve(batch_size);

    // Pre-allocate all circuits with same structure
    for (size_t b = 0; b < batch_size; ++b) {
        circuits.emplace_back(num_qubits);
    }

    // Vectorized angle extraction: process all hashes in parallel-friendly loops
    // (compiler can auto-vectorize or we can later use OpenMP/SIMD)
    
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& hash = hashes[b];
        auto& circuit = circuits[b];
        
        int nibble_idx = 0;
        for (int layer = 0; layer < 2; ++layer) {
            // RY layer
            for (int q = 0; q < num_qubits; ++q) {
                uint8_t n = extract_nibble_idx(hash, nibble_idx++);
                double angle = nibble_to_angle_qhash(n);
                circuit.add_gate(GateType::RY, q, angle);
            }
            // RZ layer
            for (int q = 0; q < num_qubits; ++q) {
                uint8_t n = extract_nibble_idx(hash, nibble_idx++);
                double angle = nibble_to_angle_qhash(n);
                circuit.add_gate(GateType::RZ, q, angle);
            }
            // CNOT chain (same for all batches)
            add_cnot_chain(circuit, num_qubits);
        }
    }

    return circuits;
}

} // namespace ohmy::quantum
