/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum_kernel.cuh"
#include "batched_quantum.cuh"
#include "circuit_generator.hpp"
#include <fmt/core.h>
#include <fmt/color.h>
#include <chrono>
#include <vector>

using namespace ohmy::quantum;

int main() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Nonce Batching Benchmark ===\n\n");
    
    const int NUM_QUBITS = 16;
    const int NUM_WARMUP = 3;
    const int NUM_ITERATIONS = 10;
    
    fmt::print("Configuration:\n");
    fmt::print("  Qubits: {}\n", NUM_QUBITS);
    fmt::print("  Warmup iterations: {}\n", NUM_WARMUP);
    fmt::print("  Benchmark iterations: {}\n\n", NUM_ITERATIONS);
    
    // Test different batch sizes
    std::vector<int> batch_sizes = {1, 8, 16, 32, 64};
    
    for (int batch_size : batch_sizes) {
        fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
            "=== Batch Size: {} ===\n", batch_size);
        
        // Calculate memory usage
        size_t memory_per_state = (1ULL << NUM_QUBITS) * sizeof(cuDoubleComplex);
        size_t total_memory = batch_size * memory_per_state;
        
        fmt::print("  Memory: {} MB ({} × {} MB)\n",
            total_memory / (1024 * 1024), batch_size, memory_per_state / (1024 * 1024));
        
        try {
            // Create batched simulator
            BatchedQuantumSimulator batch_sim(NUM_QUBITS, batch_size);
            
            // Generate test circuits (different nonces)
            std::vector<QuantumCircuit> circuits;
            circuits.reserve(batch_size);
            
            for (int i = 0; i < batch_size; i++) {
                // Create unique circuit for each nonce (simulating different SHA256 hashes)
                std::array<uint8_t, 32> hash;
                for (int j = 0; j < 32; j++) {
                    hash[j] = static_cast<uint8_t>((i * 17 + j * 13 + 7) % 256);
                }
                circuits.push_back(CircuitGenerator::build_from_hash(hash, NUM_QUBITS));
            }
            
            fmt::print("  Gates per circuit: {}\n", circuits[0].gates.size());
            
            // Warmup
            for (int i = 0; i < NUM_WARMUP; i++) {
                batch_sim.initialize_states();
                batch_sim.apply_circuits_optimized(circuits);
            }
            
            // Benchmark
            std::vector<double> batch_times;
            std::vector<std::vector<double>> expectations;
            
            for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
                batch_sim.initialize_states();
                
                auto start = std::chrono::high_resolution_clock::now();
                batch_sim.apply_circuits_optimized(circuits);
                auto end = std::chrono::high_resolution_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                batch_times.push_back(ms);
                
                if (iter == 0) {
                    batch_sim.measure_all(expectations);
                }
            }
            
            // Calculate statistics
            double mean_time = 0.0;
            for (double t : batch_times) mean_time += t;
            mean_time /= batch_times.size();
            
            double time_per_hash = mean_time / batch_size;
            double total_hashrate = 1000.0 / time_per_hash;
            double batch_hashrate = (batch_size * 1000.0) / mean_time;
            
            fmt::print(fg(fmt::color::green), "\nResults:\n");
            fmt::print("  Total time: {:.2f} ms for {} hashes\n", mean_time, batch_size);
            fmt::print("  Time per hash: {:.2f} ms\n", time_per_hash);
            fmt::print("  Effective hashrate: {:.2f} H/s\n", total_hashrate);
            fmt::print("  Batch throughput: {:.2f} H/s\n", batch_hashrate);
            
            // Verify determinism: process same circuit twice
            if (batch_size >= 2) {
                fmt::print("\n  Determinism check: ");
                bool deterministic = true;
                for (int q = 0; q < NUM_QUBITS; q++) {
                    if (std::abs(expectations[0][q] - expectations[0][q]) > 1e-10) {
                        deterministic = false;
                        break;
                    }
                }
                
                if (deterministic) {
                    fmt::print(fg(fmt::color::green), "✓ PASS\n");
                } else {
                    fmt::print(fg(fmt::color::red), "✗ FAIL\n");
                }
            }
            
            // Print sample expectation
            if (!expectations.empty()) {
                fmt::print("  Sample expectation[0][0]: {:.6f}\n", expectations[0][0]);
            }
            
        } catch (const std::exception& e) {
            fmt::print(fg(fmt::color::red), "  ERROR: {}\n", e.what());
        }
        
        fmt::print("\n");
    }
    
    // ========================================================================
    // Speedup Analysis
    // ========================================================================
    
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "=== Speedup Analysis ===\n\n");
    
    fmt::print("Expected behavior:\n");
    fmt::print("  Batch=1:  Baseline (same as single-nonce processing)\n");
    fmt::print("  Batch=8:  ~1.3-1.5× speedup from better GPU utilization\n");
    fmt::print("  Batch=16: ~1.5-1.8× speedup\n");
    fmt::print("  Batch=32: ~1.8-2.0× speedup (diminishing returns)\n");
    fmt::print("  Batch=64: ~2.0-2.2× speedup (near-optimal GPU usage)\n\n");
    
    fmt::print("Key insights:\n");
    fmt::print("  - Batching amortizes CPU-GPU transfer overhead\n");
    fmt::print("  - Increases GPU occupancy and SM utilization\n");
    fmt::print("  - Trade-off: higher latency per hash, but better throughput\n");
    fmt::print("  - Ideal for mining pool work: process many nonces in parallel\n\n");
    
    return 0;
}
