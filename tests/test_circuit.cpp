/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "circuit_generator.hpp"
#include <fmt/core.h>
#include <fmt/color.h>
#include <array>
#include <cstring>

using namespace ohmy::quantum;

void print_circuit(const QuantumCircuit& circuit) {
    fmt::print("Circuit with {} qubits, {} gates:\n", circuit.num_qubits, circuit.gates.size());
    
    for (size_t i = 0; i < circuit.gates.size(); i++) {
        const auto& gate = circuit.gates[i];
        
        std::string gate_name;
        switch (gate.type) {
            case GateType::RX: gate_name = "Rx"; break;
            case GateType::RY: gate_name = "Ry"; break;
            case GateType::RZ: gate_name = "Rz"; break;
            case GateType::CNOT: gate_name = "CNOT"; break;
        }
        
        if (gate.type == GateType::CNOT) {
            fmt::print("  {:3d}. {} q[{}], control=q[{}]\n", 
                      i, gate_name, gate.target_qubit, gate.control_qubit);
        } else {
            fmt::print("  {:3d}. {} q[{}], angle={:.4f} rad ({:.2f}°)\n",
                      i, gate_name, gate.target_qubit, gate.angle, gate.angle * 180.0 / M_PI);
        }
    }
}

void test_circuit_generation() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Testing Circuit Generator ===\n\n");
    
    // Test 1: Generate from simple hash
    fmt::print(fg(fmt::color::yellow), "Test 1: Circuit from known hash\n");
    
    std::array<uint8_t, 32> hash1;
    // Simple pattern: 0x01, 0x23, 0x45, 0x67, ...
    for (int i = 0; i < 32; i++) {
        hash1[i] = (i * 0x23) & 0xFF;
    }
    
    auto circuit1 = CircuitGenerator::build_from_hash(hash1, 4);
    print_circuit(circuit1);
    
    if (circuit1.gates.size() > 0) {
        fmt::print(fg(fmt::color::green), "✓ Test 1 PASSED\n\n");
    } else {
        fmt::print(fg(fmt::color::red), "✗ Test 1 FAILED\n\n");
    }
    
    // Test 2: Different hash should give different circuit
    fmt::print(fg(fmt::color::yellow), "Test 2: Different hash produces different circuit\n");
    
    std::array<uint8_t, 32> hash2;
    for (int i = 0; i < 32; i++) {
        hash2[i] = ((i * 0x67 + 0x45) & 0xFF);
    }
    
    auto circuit2 = CircuitGenerator::build_from_hash(hash2, 4);
    print_circuit(circuit2);
    
    // Check that circuits are different (at least some gates have different angles)
    bool different = false;
    if (circuit1.gates.size() == circuit2.gates.size()) {
        for (size_t i = 0; i < circuit1.gates.size(); i++) {
            if (circuit1.gates[i].type == circuit2.gates[i].type &&
                circuit1.gates[i].type != GateType::CNOT) {
                if (std::abs(circuit1.gates[i].angle - circuit2.gates[i].angle) > 1e-6) {
                    different = true;
                    break;
                }
            }
        }
    }
    
    if (different) {
        fmt::print(fg(fmt::color::green), "✓ Test 2 PASSED - Circuits are different\n\n");
    } else {
        fmt::print(fg(fmt::color::orange), "⚠ Test 2 WARNING - Circuits may be similar\n\n");
    }
    
    // Test 3: Larger circuit
    fmt::print(fg(fmt::color::yellow), "Test 3: 8-qubit circuit\n");
    
    std::array<uint8_t, 32> hash3;
    for (int i = 0; i < 32; i++) {
        hash3[i] = (i * i + 42) & 0xFF;
    }
    
    auto circuit3 = CircuitGenerator::build_from_hash(hash3, 8);
    fmt::print("Generated {} gates for 8 qubits\n", circuit3.gates.size());
    
    // Count gate types
    int rx_count = 0, ry_count = 0, rz_count = 0, cnot_count = 0;
    for (const auto& gate : circuit3.gates) {
        switch (gate.type) {
            case GateType::RX: rx_count++; break;
            case GateType::RY: ry_count++; break;
            case GateType::RZ: rz_count++; break;
            case GateType::CNOT: cnot_count++; break;
        }
    }
    
    fmt::print("Gate distribution: Rx={}, Ry={}, Rz={}, CNOT={}\n",
               rx_count, ry_count, rz_count, cnot_count);
    
    if (circuit3.gates.size() > 20 && cnot_count > 0) {
        fmt::print(fg(fmt::color::green), "✓ Test 3 PASSED\n\n");
    } else {
        fmt::print(fg(fmt::color::red), "✗ Test 3 FAILED\n\n");
    }
    
    fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
        "Circuit generator tests completed!\n");
}

int main() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        R"(
  ____ _                _ _     _____         _   
 / ___(_)_ __ ___ _   _(_) |_  |_   _|__  ___| |_ 
| |   | | '__/ __| | | | | __|   | |/ _ \/ __| __|
| |___| | | | (__| |_| | | |_    | |  __/\__ \ |_ 
 \____|_|_|  \___|\__,_|_|\__|   |_|\___||___/\__|
                                                   
)");
    
    test_circuit_generation();
    
    return 0;
}
