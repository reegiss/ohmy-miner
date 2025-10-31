/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#include "ohmy/quantum/cuda_simulator.hpp"
#include <iostream>
#include <fmt/format.h>
#include <cmath>

using namespace ohmy::quantum;

/**
 * Simple test to validate CUDA backend initialization and basic operations
 */
int main() {
    try {
        fmt::print("=== CUDA Backend Validation Test ===\n\n");
        
        // Test 1: Device detection
        fmt::print("Test 1: GPU Detection\n");
        auto device_info = cuda::DeviceInfo::query(0);
        fmt::print("  {}\n", device_info.to_string());
        
        if (!device_info.is_compatible()) {
            fmt::print("  ❌ FAILED: GPU compute capability too low\n");
            return 1;
        }
        fmt::print("  ✓ GPU is compatible\n\n");
        
        // Test 2: Memory requirements
        fmt::print("Test 2: Memory Requirements\n");
        auto mem_reqs = cuda::MemoryRequirements::calculate(16);
        fmt::print("  State vector: {}\n", cuda::MemoryRequirements::format_bytes(mem_reqs.state_bytes));
        fmt::print("  Workspace: {}\n", cuda::MemoryRequirements::format_bytes(mem_reqs.workspace_bytes));
        fmt::print("  Total needed: {}\n", cuda::MemoryRequirements::format_bytes(mem_reqs.total_bytes));
        fmt::print("  Available: {}\n", cuda::MemoryRequirements::format_bytes(device_info.free_memory));
        
        if (device_info.free_memory < mem_reqs.total_bytes * 1.2) {
            fmt::print("  ❌ FAILED: Insufficient GPU memory\n");
            return 1;
        }
        fmt::print("  ✓ Sufficient memory available\n\n");
        
        // Test 3: Simulator initialization
        fmt::print("Test 3: Simulator Initialization\n");
        auto simulator = std::make_unique<cuda::CudaQuantumSimulator>(16);
        fmt::print("  Backend: {}\n", simulator->backend_name());
        fmt::print("  Max qubits: {}\n", simulator->max_qubits());
        fmt::print("  ✓ Simulator created successfully\n\n");
        
        // Test 4: Simple circuit simulation
        fmt::print("Test 4: Simple Circuit Simulation\n");
        QuantumCircuit circuit(16);
        
        // Create a simple circuit: R_Y(π/2) on qubit 0
        circuit.add_rotation(0, M_PI / 2.0, RotationAxis::Y);
        
        // Simulate
        simulator->reset();
        simulator->simulate(circuit);
        
        // Measure qubit 0
        auto expectations = simulator->measure_expectations({0});
        double expectation = expectations[0].to_double();
        
        fmt::print("  Circuit: R_Y(π/2) on qubit 0\n");
        fmt::print("  Expected ⟨Z⟩: ~0.0 (superposition)\n");
        fmt::print("  Measured ⟨Z⟩: {:.6f}\n", expectation);
        
        // For R_Y(π/2), we expect |+⟩ state: (|0⟩ + |1⟩)/√2
        // ⟨Z⟩ = P(0) - P(1) = 0.5 - 0.5 = 0.0
        if (std::abs(expectation) > 0.01) {
            fmt::print("  ❌ FAILED: Expectation value incorrect\n");
            return 1;
        }
        fmt::print("  ✓ Circuit simulation correct\n\n");
        
        // Test 5: Multi-qubit circuit
        fmt::print("Test 5: Multi-Qubit Circuit\n");
        QuantumCircuit circuit2(16);
        
        // Apply R_Y(0) to qubit 0 → stays |0⟩
        circuit2.add_rotation(0, 0.0, RotationAxis::Y);
        // Apply R_Y(π) to qubit 1 → flips to |1⟩
        circuit2.add_rotation(1, M_PI, RotationAxis::Y);
        
        simulator->reset();
        simulator->simulate(circuit2);
        
        auto expectations2 = simulator->measure_expectations({0, 1});
        double exp0 = expectations2[0].to_double();
        double exp1 = expectations2[1].to_double();
        
        fmt::print("  Circuit: R_Y(0) on q0, R_Y(π) on q1\n");
        fmt::print("  Expected: q0 ⟨Z⟩ ≈ +1.0 (|0⟩), q1 ⟨Z⟩ ≈ -1.0 (|1⟩)\n");
        fmt::print("  Measured: q0 ⟨Z⟩ = {:.6f}, q1 ⟨Z⟩ = {:.6f}\n", exp0, exp1);
        
        if (std::abs(exp0 - 1.0) > 0.01 || std::abs(exp1 + 1.0) > 0.01) {
            fmt::print("  ❌ FAILED: Multi-qubit expectations incorrect\n");
            return 1;
        }
        fmt::print("  ✓ Multi-qubit simulation correct\n\n");
        
        // Test 6: CNOT gate
        fmt::print("Test 6: CNOT Gate\n");
        QuantumCircuit circuit3(16);
        
        // Prepare |01⟩: q0 stays |0⟩, q1 → |1⟩
        circuit3.add_rotation(1, M_PI, RotationAxis::Y);
        // CNOT(0,1): control=q0, target=q1
        // Since q0=|0⟩, q1 should stay |1⟩
        circuit3.add_cnot(0, 1);
        
        simulator->reset();
        simulator->simulate(circuit3);
        
        auto expectations3 = simulator->measure_expectations({0, 1});
        double exp0_cnot = expectations3[0].to_double();
        double exp1_cnot = expectations3[1].to_double();
        
        fmt::print("  Circuit: q1←X, CNOT(q0→q1)\n");
        fmt::print("  Initial: |01⟩, Control=0 → no flip\n");
        fmt::print("  Expected: q0 ⟨Z⟩ ≈ +1.0, q1 ⟨Z⟩ ≈ -1.0\n");
        fmt::print("  Measured: q0 ⟨Z⟩ = {:.6f}, q1 ⟨Z⟩ = {:.6f}\n", exp0_cnot, exp1_cnot);
        
        if (std::abs(exp0_cnot - 1.0) > 0.01 || std::abs(exp1_cnot + 1.0) > 0.01) {
            fmt::print("  ❌ FAILED: CNOT gate incorrect\n");
            return 1;
        }
        fmt::print("  ✓ CNOT gate correct\n\n");
        
        fmt::print("=== All Tests Passed! ===\n");
        fmt::print("\n🎉 CUDA backend is fully functional!\n");
        
        return 0;
        
    } catch (const std::exception& e) {
        fmt::print("❌ FATAL ERROR: {}\n", e.what());
        return 1;
    }
}
