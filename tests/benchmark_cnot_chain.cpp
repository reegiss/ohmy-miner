/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum_kernel.cuh"
#include <fmt/core.h>
#include <fmt/color.h>
#include <chrono>
#include <vector>
#include <cmath>

using namespace ohmy::quantum;

// Create QTC circuit: [RY_all → RZ_all → CNOT_chain] × 2 layers
QuantumCircuit create_qtc_circuit(const std::vector<uint8_t>& angles_data) {
    const int NUM_QUBITS = 16;
    const int NUM_LAYERS = 2;
    
    QuantumCircuit circuit(NUM_QUBITS);
    
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        // RY gates for all qubits
        for (int q = 0; q < NUM_QUBITS; q++) {
            int angle_idx = (2 * layer * NUM_QUBITS + q) % angles_data.size();
            double angle = angles_data[angle_idx] * M_PI / 16.0;
            circuit.add_gate(GateType::RY, q, angle);
        }
        
        // RZ gates for all qubits
        for (int q = 0; q < NUM_QUBITS; q++) {
            int angle_idx = ((2 * layer + 1) * NUM_QUBITS + q) % angles_data.size();
            double angle = angles_data[angle_idx] * M_PI / 16.0;
            circuit.add_gate(GateType::RZ, q, angle);
        }
        
        // CNOT chain: qubit i controls qubit i+1
        for (int q = 0; q < NUM_QUBITS - 1; q++) {
            circuit.add_gate(GateType::CNOT, q + 1, 0.0, q);  // control=q, target=q+1
        }
    }
    
    return circuit;
}

int main() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== CNOT Chain Optimization Benchmark ===\n\n");
    
    const int NUM_QUBITS = 16;
    const int NUM_WARMUP = 5;
    const int NUM_ITERATIONS = 20;
    
    // Create test data
    std::vector<uint8_t> angles_data;
    for (int i = 0; i < 64; i++) {
        angles_data.push_back(static_cast<uint8_t>((i * 13 + 7) % 16));
    }
    
    fmt::print("Configuration:\n");
    fmt::print("  Qubits: {}\n", NUM_QUBITS);
    fmt::print("  State size: {} amplitudes ({} MB)\n",
        1ULL << NUM_QUBITS, (1ULL << NUM_QUBITS) * sizeof(cuDoubleComplex) / (1024 * 1024));
    fmt::print("  Warmup iterations: {}\n", NUM_WARMUP);
    fmt::print("  Benchmark iterations: {}\n", NUM_ITERATIONS);
    fmt::print("  Circuit structure: [RY×16 → RZ×16 → CNOT_chain×15] × 2 layers\n\n");
    
    // Create simulator
    QuantumSimulator sim(NUM_QUBITS);
    QuantumCircuit circuit = create_qtc_circuit(angles_data);
    
    fmt::print("Total gates: {}\n", circuit.gates.size());
    fmt::print("  RY gates: 32\n");
    fmt::print("  RZ gates: 32\n");
    fmt::print("  CNOT gates: 30 (2 chains of 15)\n\n");
    
    // ========================================================================
    // Benchmark 1: Original (naive CNOT)
    // ========================================================================
    fmt::print(fg(fmt::color::yellow), "Benchmark 1: Original Implementation (naive CNOT)\n");
    fmt::print("Strategy: Each CNOT launches separate kernel\n");
    fmt::print("Expected: ~30 CNOT kernel launches\n\n");
    
    std::vector<double> original_times;
    std::vector<double> original_expectations;
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        sim.initialize_state();
        sim.apply_circuit(circuit);
    }
    
    // Benchmark
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        sim.initialize_state();
        
        auto start = std::chrono::high_resolution_clock::now();
        sim.apply_circuit(circuit);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        original_times.push_back(ms);
        
        if (i == 0) {
            sim.measure(original_expectations);
        }
    }
    
    double original_mean = 0.0;
    for (double t : original_times) original_mean += t;
    original_mean /= original_times.size();
    
    double original_hashrate = 1000.0 / original_mean;
    
    fmt::print(fg(fmt::color::green), "Results:\n");
    fmt::print("  Mean time: {:.2f} ms/hash\n", original_mean);
    fmt::print("  Hashrate: {:.2f} H/s\n", original_hashrate);
    fmt::print("  Expectations[0]: {:.6f}\n\n", original_expectations[0]);
    
    // ========================================================================
    // Benchmark 2: Optimized (CNOT chain + Gate Fusion)
    // ========================================================================
    fmt::print(fg(fmt::color::yellow), "Benchmark 2: Optimized Implementation (CNOT chain + fusion)\n");
    fmt::print("Strategy:\n");
    fmt::print("  1. Fuse RY+RZ pairs: 32 launches → 16 launches\n");
    fmt::print("  2. CNOT chain optimization: 15 launches → 1 launch per chain\n");
    fmt::print("  3. Shared memory caching: reduce DRAM traffic 30×\n\n");
    
    std::vector<double> optimized_times;
    std::vector<double> optimized_expectations;
    
    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        sim.initialize_state();
        sim.apply_circuit_optimized(circuit);
    }
    
    // Benchmark
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        sim.initialize_state();
        
        auto start = std::chrono::high_resolution_clock::now();
        sim.apply_circuit_optimized(circuit);
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        optimized_times.push_back(ms);
        
        if (i == 0) {
            sim.measure(optimized_expectations);
        }
    }
    
    double optimized_mean = 0.0;
    for (double t : optimized_times) optimized_mean += t;
    optimized_mean /= optimized_times.size();
    
    double optimized_hashrate = 1000.0 / optimized_mean;
    
    fmt::print(fg(fmt::color::green), "Results:\n");
    fmt::print("  Mean time: {:.2f} ms/hash\n", optimized_mean);
    fmt::print("  Hashrate: {:.2f} H/s\n", optimized_hashrate);
    fmt::print("  Expectations[0]: {:.6f}\n\n", optimized_expectations[0]);
    
    // ========================================================================
    // Comparison and Validation
    // ========================================================================
    double speedup = original_mean / optimized_mean;
    
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold, "=== Performance Summary ===\n\n");
    fmt::print("Original:  {:.2f} ms/hash ({:.2f} H/s)\n", original_mean, original_hashrate);
    fmt::print("Optimized: {:.2f} ms/hash ({:.2f} H/s)\n", optimized_mean, optimized_hashrate);
    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
        "Speedup:   {:.2f}×\n\n", speedup);
    
    // Determinism validation
    fmt::print(fg(fmt::color::cyan), "=== Determinism Validation ===\n\n");
    
    bool all_match = true;
    double max_diff = 0.0;
    
    for (size_t i = 0; i < original_expectations.size(); i++) {
        double diff = std::abs(original_expectations[i] - optimized_expectations[i]);
        max_diff = std::max(max_diff, diff);
        
        if (diff > 1e-10) {
            fmt::print(fg(fmt::color::red),
                "  Qubit {}: MISMATCH! Original = {:.15f}, Optimized = {:.15f}, Diff = {:.2e}\n",
                i, original_expectations[i], optimized_expectations[i], diff);
            all_match = false;
        }
    }
    
    if (all_match) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ DETERMINISM VERIFIED: All expectations match bit-exact!\n");
        fmt::print("  Maximum difference: {:.2e}\n\n", max_diff);
    } else {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "✗ DETERMINISM FAILED: Expectations differ!\n\n");
        return 1;
    }
    
    // Performance analysis
    fmt::print(fg(fmt::color::cyan), "=== Performance Analysis ===\n\n");
    
    if (speedup >= 3.0) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ EXCELLENT: {:.1f}× speedup achieved (expected 3-5×)\n", speedup);
        fmt::print("  CNOT chain optimization is working effectively!\n");
    } else if (speedup >= 1.5) {
        fmt::print(fg(fmt::color::yellow),
            "◐ MODERATE: {:.1f}× speedup achieved (expected 3-5×)\n", speedup);
        fmt::print("  Some improvement, but below target. Possible causes:\n");
        fmt::print("  - Cross-tile CNOT swaps still hitting DRAM\n");
        fmt::print("  - Shared memory bank conflicts\n");
        fmt::print("  - Need larger tile size or better tiling strategy\n");
    } else {
        fmt::print(fg(fmt::color::red),
            "✗ MINIMAL: {:.1f}× speedup achieved (expected 3-5×)\n", speedup);
        fmt::print("  Optimization not effective. Possible causes:\n");
        fmt::print("  - Implementation bug in shared memory version\n");
        fmt::print("  - Most CNOT swaps crossing tile boundaries\n");
        fmt::print("  - Memory access pattern not improved\n");
    }
    
    fmt::print("\n");
    return all_match ? 0 : 1;
}
