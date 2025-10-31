/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/simulator.hpp"
#include "ohmy/quantum/cuda_simulator.hpp"
#include <iostream>
#include <chrono>
#include <fmt/format.h>
#include <cmath>
#include <vector>
#include <numeric>

using namespace ohmy::quantum;

/**
 * Performance baseline test: Compare CUDA vs CPU on qhash-like circuits
 * 
 * Simulates realistic qhash workload:
 * - 16 qubits
 * - 3 layers of gates (R_Y, R_Z, CNOT pattern)
 * - Measurement of all 16 qubits
 */

struct BenchmarkResult {
    std::string backend_name;
    double total_time_ms;
    double circuits_per_second;
    double gates_per_second;
    
    // Detailed breakdown
    double gate_time_ms;
    double measurement_time_ms;
    
    void print() const {
        fmt::print("\n{} Performance:\n", backend_name);
        fmt::print("  Total time: {:.3f} ms per circuit\n", total_time_ms);
        fmt::print("  Hashrate: {:.2f} H/s ({:.2f} KH/s)\n", 
                   circuits_per_second, circuits_per_second / 1000.0);
        fmt::print("  Gate simulation: {:.3f} ms ({:.1f}%)\n", 
                   gate_time_ms, 100.0 * gate_time_ms / total_time_ms);
        fmt::print("  Measurement: {:.3f} ms ({:.1f}%)\n", 
                   measurement_time_ms, 100.0 * measurement_time_ms / total_time_ms);
        fmt::print("  Gates/second: {:.2f} M gates/s\n", gates_per_second / 1e6);
    }
};

/**
 * Create a realistic qhash circuit:
 * 3 layers × 16 qubits × (R_Y + R_Z) + 3 layers × 15 CNOT = 96 gates + 45 CNOT
 */
QuantumCircuit create_qhash_circuit() {
    QuantumCircuit circuit(16);
    
    // 3 layers of quantum gates (similar to qhash)
    for (int layer = 0; layer < 3; layer++) {
        // Rotation gates on all qubits
        for (int q = 0; q < 16; q++) {
            // Angles derived from "hash" - use varying angles for realism
            double angle_y = (layer * 16 + q) * 0.1;
            double angle_z = (layer * 16 + q) * 0.15;
            
            circuit.add_rotation(q, angle_y, RotationAxis::Y);
            circuit.add_rotation(q, angle_z, RotationAxis::Z);
        }
        
        // CNOT chain (entanglement layer)
        for (int q = 0; q < 15; q++) {
            circuit.add_cnot(q, q + 1);
        }
    }
    
    return circuit;
}

/**
 * Benchmark a simulator on single-nonce workload
 */
BenchmarkResult benchmark_simulator(
    IQuantumSimulator* simulator,
    int num_iterations = 100
) {
    auto circuit = create_qhash_circuit();
    std::vector<int> qubits_to_measure(16);
    std::iota(qubits_to_measure.begin(), qubits_to_measure.end(), 0);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        simulator->reset();
        simulator->simulate(circuit);
        simulator->measure_expectations(qubits_to_measure);
    }
    
    // Timed runs
    std::vector<double> times_ms;
    std::vector<double> gate_times_ms;
    std::vector<double> measurement_times_ms;
    
    for (int i = 0; i < num_iterations; i++) {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        simulator->reset();
        
        auto start_gates = std::chrono::high_resolution_clock::now();
        simulator->simulate(circuit);
        auto end_gates = std::chrono::high_resolution_clock::now();
        
        auto start_measure = std::chrono::high_resolution_clock::now();
        auto expectations = simulator->measure_expectations(qubits_to_measure);
        auto end_measure = std::chrono::high_resolution_clock::now();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        
        double total_ms = std::chrono::duration<double, std::milli>(
            end_total - start_total).count();
        double gate_ms = std::chrono::duration<double, std::milli>(
            end_gates - start_gates).count();
        double measure_ms = std::chrono::duration<double, std::milli>(
            end_measure - start_measure).count();
        
        times_ms.push_back(total_ms);
        gate_times_ms.push_back(gate_ms);
        measurement_times_ms.push_back(measure_ms);
    }
    
    // Calculate statistics (median for robustness)
    std::sort(times_ms.begin(), times_ms.end());
    std::sort(gate_times_ms.begin(), gate_times_ms.end());
    std::sort(measurement_times_ms.begin(), measurement_times_ms.end());
    
    double median_total = times_ms[num_iterations / 2];
    double median_gates = gate_times_ms[num_iterations / 2];
    double median_measure = measurement_times_ms[num_iterations / 2];
    
    int total_gates = circuit.rotation_gates().size() + circuit.cnot_gates().size();
    
    return BenchmarkResult{
        .backend_name = simulator->backend_name(),
        .total_time_ms = median_total,
        .circuits_per_second = 1000.0 / median_total,
        .gates_per_second = total_gates * 1000.0 / median_total,
        .gate_time_ms = median_gates,
        .measurement_time_ms = median_measure
    };
}

int main() {
    try {
        fmt::print("=== Performance Baseline Test ===\n");
        fmt::print("Workload: 16-qubit qhash-like circuit\n");
        fmt::print("  - 3 layers × 16 qubits × 2 rotations = 96 rotation gates\n");
        fmt::print("  - 3 layers × 15 CNOT gates = 45 CNOT gates\n");
        fmt::print("  - Measure all 16 qubits\n");
        fmt::print("  - Total: 141 gate operations + 16 measurements\n\n");
        
        // Benchmark CPU
        fmt::print("Benchmarking CPU backend (100 iterations)...\n");
        auto cpu_sim = SimulatorFactory::create(SimulatorFactory::Backend::CPU_BASIC, 16);
        auto cpu_result = benchmark_simulator(cpu_sim.get(), 100);
        cpu_result.print();
        
        // Benchmark CUDA
        fmt::print("\nBenchmarking CUDA backend (100 iterations)...\n");
        auto cuda_sim = SimulatorFactory::create(SimulatorFactory::Backend::CUDA_CUSTOM, 16);
        auto cuda_result = benchmark_simulator(cuda_sim.get(), 100);
        cuda_result.print();
        
        // Speedup analysis
        fmt::print("\n=== Speedup Analysis ===\n");
        double speedup = cpu_result.circuits_per_second / cuda_result.circuits_per_second;
        fmt::print("CUDA vs CPU speedup: {:.2f}x\n", 1.0 / speedup);
        
        if (speedup < 1.0) {
            fmt::print("✅ CUDA is {:.2f}x FASTER than CPU\n", 1.0 / speedup);
        } else {
            fmt::print("⚠️  CPU is {:.2f}x faster than CUDA (unexpected!)\n", speedup);
            fmt::print("    This suggests GPU overhead dominates for single-nonce workload.\n");
            fmt::print("    Batching will be critical for performance.\n");
        }
        
        // Bottleneck analysis
        fmt::print("\n=== Bottleneck Analysis ===\n");
        
        fmt::print("CPU bottleneck: ");
        if (cpu_result.gate_time_ms > cpu_result.measurement_time_ms) {
            fmt::print("Gate simulation ({:.1f}% of time)\n", 
                       100.0 * cpu_result.gate_time_ms / cpu_result.total_time_ms);
        } else {
            fmt::print("Measurement ({:.1f}% of time)\n",
                       100.0 * cpu_result.measurement_time_ms / cpu_result.total_time_ms);
        }
        
        fmt::print("CUDA bottleneck: ");
        if (cuda_result.gate_time_ms > cuda_result.measurement_time_ms) {
            fmt::print("Gate simulation ({:.1f}% of time)\n",
                       100.0 * cuda_result.gate_time_ms / cuda_result.total_time_ms);
        } else {
            fmt::print("Measurement ({:.1f}% of time)\n",
                       100.0 * cuda_result.measurement_time_ms / cuda_result.total_time_ms);
        }
        
        // Recommendations
        fmt::print("\n=== Recommendations ===\n");
        if (speedup > 0.5) {  // CPU competitive
            fmt::print("⚠️  Single-nonce GPU performance not significantly better than CPU.\n");
            fmt::print("✅ Solution: Implement batching to amortize GPU overhead.\n");
            fmt::print("   Target: Process 1000+ nonces in parallel.\n");
            fmt::print("   Expected: 10-100x speedup with batching.\n");
        } else {
            fmt::print("✅ Good single-nonce GPU performance!\n");
            fmt::print("   Batching will provide additional 10-20x speedup.\n");
        }
        
        // Mining hashrate projection
        fmt::print("\n=== Mining Hashrate Projections ===\n");
        fmt::print("Single-nonce (current):\n");
        fmt::print("  CPU: {:.2f} H/s\n", cpu_result.circuits_per_second);
        fmt::print("  CUDA: {:.2f} H/s\n", cuda_result.circuits_per_second);
        
        fmt::print("\nWith batching (1000 nonces, estimated):\n");
        double batch_overhead = 1.2;  // 20% overhead for batching
        fmt::print("  CUDA: {:.2f} KH/s\n", 
                   cuda_result.circuits_per_second * 1000 / batch_overhead / 1000.0);
        
        fmt::print("\nTarget for GTX 1660 Super: 5-10 KH/s\n");
        
        return 0;
        
    } catch (const std::exception& e) {
        fmt::print("❌ ERROR: {}\n", e.what());
        return 1;
    }
}
