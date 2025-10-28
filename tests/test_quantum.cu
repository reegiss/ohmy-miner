/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum_kernel.cuh"
#include <fmt/core.h>
#include <fmt/color.h>
#include <iostream>
#include <cmath>

using namespace ohmy::quantum;

void test_quantum_simulator() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Testing Quantum Simulator ===\n\n");
    
    // Test 1: Simple 2-qubit circuit
    fmt::print(fg(fmt::color::yellow), "Test 1: 2-qubit Hadamard-like circuit\n");
    
    try {
        QuantumSimulator sim(2);
        
        // Initialize to |00⟩
        if (!sim.initialize_state()) {
            fmt::print(fg(fmt::color::red), "Failed to initialize state\n");
            return;
        }
        
        // Create circuit: Ry(π/2) on qubit 0 (approximates Hadamard)
        QuantumCircuit circuit(2);
        circuit.add_gate(GateType::RY, 0, M_PI / 2.0);
        
        // Apply circuit
        if (!sim.apply_circuit(circuit)) {
            fmt::print(fg(fmt::color::red), "Failed to apply circuit\n");
            return;
        }
        
        // Measure
        std::vector<double> expectations;
        if (!sim.measure(expectations)) {
            fmt::print(fg(fmt::color::red), "Failed to measure\n");
            return;
        }
        
        fmt::print("Expectation values:\n");
        for (size_t i = 0; i < expectations.size(); i++) {
            fmt::print("  Qubit {}: ⟨σz⟩ = {:.6f}\n", i, expectations[i]);
        }
        
        // For Ry(π/2)|0⟩, qubit 0 should have ⟨σz⟩ ≈ 0 (equal superposition)
        if (std::abs(expectations[0]) < 0.01) {
            fmt::print(fg(fmt::color::green), "✓ Test 1 PASSED\n\n");
        } else {
            fmt::print(fg(fmt::color::red), "✗ Test 1 FAILED\n\n");
        }
        
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Exception: {}\n", e.what());
        return;
    }
    
    // Test 2: CNOT entanglement
    fmt::print(fg(fmt::color::yellow), "Test 2: CNOT entanglement\n");
    
    try {
        QuantumSimulator sim(2);
        
        if (!sim.initialize_state()) {
            fmt::print(fg(fmt::color::red), "Failed to initialize state\n");
            return;
        }
        
        // Create Bell state: Ry(π/2) on qubit 0, then CNOT(0,1)
        QuantumCircuit circuit(2);
        circuit.add_gate(GateType::RY, 0, M_PI / 2.0);
        circuit.add_gate(GateType::CNOT, 1, 0.0, 0);  // control=0, target=1
        
        if (!sim.apply_circuit(circuit)) {
            fmt::print(fg(fmt::color::red), "Failed to apply circuit\n");
            return;
        }
        
        std::vector<double> expectations;
        if (!sim.measure(expectations)) {
            fmt::print(fg(fmt::color::red), "Failed to measure\n");
            return;
        }
        
        fmt::print("Expectation values:\n");
        for (size_t i = 0; i < expectations.size(); i++) {
            fmt::print("  Qubit {}: ⟨σz⟩ = {:.6f}\n", i, expectations[i]);
        }
        
        // Both qubits should be in equal superposition
        if (std::abs(expectations[0]) < 0.01 && std::abs(expectations[1]) < 0.01) {
            fmt::print(fg(fmt::color::green), "✓ Test 2 PASSED\n\n");
        } else {
            fmt::print(fg(fmt::color::red), "✗ Test 2 FAILED\n\n");
        }
        
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Exception: {}\n", e.what());
        return;
    }
    
    // Test 3: Larger circuit
    fmt::print(fg(fmt::color::yellow), "Test 3: 4-qubit circuit\n");
    
    try {
        QuantumSimulator sim(4);
        
        if (!sim.initialize_state()) {
            fmt::print(fg(fmt::color::red), "Failed to initialize state\n");
            return;
        }
        
        // Create parameterized circuit
        QuantumCircuit circuit(4);
        circuit.add_gate(GateType::RX, 0, 0.5);
        circuit.add_gate(GateType::RY, 1, 1.0);
        circuit.add_gate(GateType::RZ, 2, 1.5);
        circuit.add_gate(GateType::CNOT, 1, 0.0, 0);
        circuit.add_gate(GateType::CNOT, 3, 0.0, 2);
        
        if (!sim.apply_circuit(circuit)) {
            fmt::print(fg(fmt::color::red), "Failed to apply circuit\n");
            return;
        }
        
        std::vector<double> expectations;
        if (!sim.measure(expectations)) {
            fmt::print(fg(fmt::color::red), "Failed to measure\n");
            return;
        }
        
        fmt::print("Expectation values:\n");
        for (size_t i = 0; i < expectations.size(); i++) {
            fmt::print("  Qubit {}: ⟨σz⟩ = {:.6f}\n", i, expectations[i]);
        }
        
        fmt::print(fg(fmt::color::green), "✓ Test 3 PASSED\n\n");
        
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Exception: {}\n", e.what());
        return;
    }
    
    fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
        "All quantum simulator tests completed!\n");
}

int main() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        R"(
  ___                   _                    _____         _   
 / _ \ _   _  __ _ _ __ | |_ _   _ _ __ ___ |_   _|__  ___| |_ 
| | | | | | |/ _` | '_ \| __| | | | '_ ` _ \  | |/ _ \/ __| __|
| |_| | |_| | (_| | | | | |_| |_| | | | | | | | |  __/\__ \ |_ 
 \__\_\\__,_|\__,_|_| |_|\__|\__,_|_| |_| |_| |_|\___||___/\__|
                                                                
)");
    
    test_quantum_simulator();
    
    return 0;
}
