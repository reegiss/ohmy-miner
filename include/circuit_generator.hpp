/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include "quantum_kernel.cuh"
#include <array>
#include <vector>
#include <cstdint>

namespace ohmy::quantum {

/**
 * Generates quantum circuits from SHA256 hashes for qhash algorithm.
 * 
 * The qhash algorithm converts SHA256 256-bit output into quantum circuit parameters:
 * - 4-bit segments define rotation angles (16 possible values per gate)
 * - Circuit structure: layers of single-qubit rotations + entangling CNOTs
 */
class CircuitGenerator {
public:
    /**
     * Build a quantum circuit from SHA256 hash output.
     * 
     * @param hash 32-byte SHA256 hash (256 bits)
     * @param num_qubits Number of qubits in the circuit (typically 8-12)
     * @return QuantumCircuit with parameterized gates
     */
    static QuantumCircuit build_from_hash(const std::array<uint8_t, 32>& hash, int num_qubits);
    
private:
    // Extract nibble by index [0..63] from 32-byte hash (big-endian nibble order)
    static inline uint8_t extract_nibble_idx(const std::array<uint8_t, 32>& hash, int nibble_idx);

    // Map nibble (0..15) to angle = -nibble * pi/16.0
    static inline double nibble_to_angle_qhash(uint8_t nibble);

    // Add nearest-neighbor CNOT chain: (0->1), (1->2), ..., (n-2->n-1)
    static void add_cnot_chain(QuantumCircuit& circuit, int num_qubits);
};

} // namespace ohmy::quantum
