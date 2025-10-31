/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/quantum/batched_cuda_simulator.hpp"
#include <iostream>
#include <chrono>
#include <fmt/format.h>
#include <cmath>
#include <vector>
#include <numeric>

using namespace ohmy::quantum;

/**
 * Create realistic qhash circuit
 */
QuantumCircuit create_qhash_circuit() {
    QuantumCircuit circuit(16);
    
    for (int layer = 0; layer < 3; layer++) {
        for (int q = 0; q < 16; q++) {
            double angle_y = (layer * 16 + q) * 0.1;
            double angle_z = (layer * 16 + q) * 0.15;
            
            circuit.add_rotation(q, angle_y, RotationAxis::Y);
            circuit.add_rotation(q, angle_z, RotationAxis::Z);
        }
        
        for (int q = 0; q < 15; q++) {
            circuit.add_cnot(q, q + 1);
        }
    }
    
    return circuit;
}

int main() {
    try {
        fmt::print("=== Batched CUDA Performance Test ===\n\n");
        
        // Test different batch sizes
        std::vector<int> batch_sizes = {100, 500, 1000, 2000};
        
        for (int batch_size : batch_sizes) {
            fmt::print("Testing batch size: {}\n", batch_size);
            
            // Create simulator
            cuda::BatchedCudaSimulator simulator(16, batch_size);
            
            // Create batch of identical circuits (like qhash mining)
            std::vector<QuantumCircuit> circuits;
            auto circuit = create_qhash_circuit();
            for (int i = 0; i < batch_size; i++) {
                circuits.push_back(circuit);
            }
            
            std::vector<int> qubits_to_measure(16);
            std::iota(qubits_to_measure.begin(), qubits_to_measure.end(), 0);
            
            // Warmup
            for (int i = 0; i < 3; i++) {
                simulator.simulate_and_measure_batch(circuits, qubits_to_measure);
            }
            
            // Timed runs
            const int num_iterations = 10;
            std::vector<double> times_ms;
            
            for (int i = 0; i < num_iterations; i++) {
                auto start = std::chrono::high_resolution_clock::now();
                
                auto results = simulator.simulate_and_measure_batch(circuits, qubits_to_measure);
                
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                times_ms.push_back(time_ms);
            }
            
            // Calculate median
            std::sort(times_ms.begin(), times_ms.end());
            double median_time_ms = times_ms[num_iterations / 2];
            double time_per_circuit_ms = median_time_ms / batch_size;
            double hashrate = (batch_size * 1000.0) / median_time_ms;
            
            fmt::print("  Total time: {:.2f} ms for {} circuits\n", median_time_ms, batch_size);
            fmt::print("  Time per circuit: {:.3f} ms\n", time_per_circuit_ms);
            fmt::print("  Hashrate: {:.2f} H/s ({:.2f} KH/s)\n", 
                       hashrate, hashrate / 1000.0);
            fmt::print("  Throughput: {:.2f} M gates/s\n\n", 
                       141 * hashrate / 1e6);
        }
        
        // Compare with single-nonce baseline
        fmt::print("=== Comparison with Single-Nonce ===\n");
        fmt::print("Single-nonce (from baseline test): 1,446.8 H/s (1.45 KH/s)\n");
        fmt::print("Expected improvement with batching: 10-20×\n");
        fmt::print("Target for GTX 1660 Super: 5-10 KH/s\n\n");
        
        // Recommend optimal batch size
        cuda::BatchedCudaSimulator temp_sim(16, 100);
        int optimal = temp_sim.get_optimal_batch_size();
        fmt::print("Recommended optimal batch size: {}\n", optimal);
        
        return 0;
        
    } catch (const std::exception& e) {
        fmt::print("❌ ERROR: {}\n", e.what());
        return 1;
    }
}
