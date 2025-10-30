/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <memory>

using namespace ohmy::quantum;

void test_simulator_creation() {
    std::cout << "Testing simulator creation..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    assert(sim != nullptr);
    
    std::cout << "  ✓ Simulator creation test passed" << std::endl;
}

void test_single_qubit_state() {
    std::cout << "Testing single qubit state..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(1);
    // No gates - initial state |0⟩
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0});
    
    // Expectation of Z on |0⟩ should be +1
    assert(result.size() == 1);
    assert(std::abs(result[0].to_double() - 1.0) < 0.01);
    
    std::cout << "  ✓ Single qubit test passed" << std::endl;
}

void test_rotation_gate() {
    std::cout << "Testing rotation gate..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(1);
    // Rotate by π to flip state |0⟩ → |1⟩
    circuit.add_rotation(0, M_PI);
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0});
    
    // Expectation of Z on |1⟩ should be -1
    assert(result.size() == 1);
    assert(std::abs(result[0].to_double() - (-1.0)) < 0.01);
    
    std::cout << "  ✓ Rotation gate test passed" << std::endl;
}

void test_half_rotation() {
    std::cout << "Testing half rotation (superposition)..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(1);
    // Rotate by π/2 to create superposition
    circuit.add_rotation(0, M_PI / 2.0);
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0});
    
    // Expectation of Z on (|0⟩ + i|1⟩)/√2 should be ~0
    assert(result.size() == 1);
    assert(std::abs(result[0].to_double()) < 0.1);
    
    std::cout << "  ✓ Half rotation test passed" << std::endl;
}

void test_two_qubit_cnot() {
    std::cout << "Testing two-qubit CNOT..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(2);
    // Prepare control in |1⟩
    circuit.add_rotation(0, M_PI);
    // CNOT should flip target
    circuit.add_cnot(0, 1);
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0, 1});
    
    // Both qubits should be in |1⟩
    assert(result.size() == 2);
    assert(std::abs(result[0].to_double() - (-1.0)) < 0.01);
    assert(std::abs(result[1].to_double() - (-1.0)) < 0.01);
    
    std::cout << "  ✓ CNOT test passed" << std::endl;
}

void test_multi_qubit_circuit() {
    std::cout << "Testing multi-qubit circuit..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(3);
    circuit.add_rotation(0, M_PI / 4.0);
    circuit.add_rotation(1, M_PI / 3.0);
    circuit.add_cnot(0, 1);
    circuit.add_rotation(2, M_PI / 6.0);
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0, 1, 2});
    
    // Should return 3 expectation values
    assert(result.size() == 3);
    
    // All values should be in [-1, 1]
    for (const auto& val : result) {
        [[maybe_unused]] double d = val.to_double();
        assert(d >= -1.0 && d <= 1.0);
    }
    
    std::cout << "  ✓ Multi-qubit circuit test passed" << std::endl;
}

void test_circuit_determinism() {
    std::cout << "Testing circuit simulation determinism..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(2);
    circuit.add_rotation(0, 0.123);
    circuit.add_cnot(0, 1);
    circuit.add_rotation(1, 0.321);
    
    sim->simulate(circuit);
    auto result1 = sim->measure_expectations({0, 1});
    
    sim->reset();
    sim->simulate(circuit);
    auto result2 = sim->measure_expectations({0, 1});
    
    assert(result1.size() == result2.size());
    for (size_t i = 0; i < result1.size(); i++) {
        assert(std::abs(result1[i].to_double() - result2[i].to_double()) < 1e-10);
    }
    
    std::cout << "  ✓ Determinism test passed" << std::endl;
}

void test_empty_circuit() {
    std::cout << "Testing empty circuit..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(2);
    // No gates - all qubits in |0⟩
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0, 1});
    
    // All expectations should be +1
    assert(result.size() == 2);
    for (const auto& val : result) {
        [[maybe_unused]] auto v = val;  // Suppress unused warning
        assert(std::abs(val.to_double() - 1.0) < 0.01);
    }
    
    std::cout << "  ✓ Empty circuit test passed" << std::endl;
}

void test_large_circuit() {
    std::cout << "Testing larger circuit (5 qubits)..." << std::endl;
    
    auto sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 10);
    
    QuantumCircuit circuit(5);
    for (int i = 0; i < 5; i++) {
        circuit.add_rotation(i, M_PI / (i + 2));
    }
    for (int i = 0; i < 4; i++) {
        circuit.add_cnot(i, i + 1);
    }
    
    sim->simulate(circuit);
    auto result = sim->measure_expectations({0, 1, 2, 3, 4});
    
    assert(result.size() == 5);
    
    std::cout << "  ✓ Large circuit test passed" << std::endl;
}

int main() {
    std::cout << "\n=== Quantum Simulator Tests ===" << std::endl;
    
    try {
        test_simulator_creation();
        test_single_qubit_state();
        test_rotation_gate();
        test_half_rotation();
        test_two_qubit_cnot();
        test_multi_qubit_circuit();
        test_circuit_determinism();
        test_empty_circuit();
        test_large_circuit();
        
        std::cout << "\n✅ All quantum simulator tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
