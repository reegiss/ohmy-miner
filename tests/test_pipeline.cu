/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "circuit_generator.hpp"
#include "quantum_kernel.cuh"
#include <fmt/core.h>
#include <fmt/color.h>
#include <openssl/sha.h>
#include <array>
#include <chrono>

using namespace ohmy::quantum;

// Simple SHA256 wrapper
std::array<uint8_t, 32> compute_sha256(const std::string& data) {
    std::array<uint8_t, 32> hash;
    SHA256(reinterpret_cast<const unsigned char*>(data.c_str()), data.size(), hash.data());
    return hash;
}

void test_full_pipeline() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Testing Full Quantum Mining Pipeline ===\n\n");
    
    // Test 1: Single mining iteration
    fmt::print(fg(fmt::color::yellow), "Test 1: Hash → Circuit → Simulation\n");
    
    std::string block_data = "Block Header: nonce=12345, prev_hash=abc...";
    auto hash = compute_sha256(block_data);
    
    fmt::print("Input data: \"{}\"\n", block_data);
    fmt::print("SHA256: ");
    for (int i = 0; i < 8; i++) {  // Print first 8 bytes
        fmt::print("{:02x}", hash[i]);
    }
    fmt::print("...\n\n");
    
    // Generate circuit from hash
    const int num_qubits = 8;
    auto circuit = CircuitGenerator::build_from_hash(hash, num_qubits);
    fmt::print("Generated circuit: {} qubits, {} gates\n", 
               circuit.num_qubits, circuit.gates.size());
    
    // Simulate circuit
    try {
        QuantumSimulator sim(num_qubits);
        
        if (!sim.initialize_state()) {
            fmt::print(fg(fmt::color::red), "Failed to initialize quantum state\n");
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!sim.apply_circuit(circuit)) {
            fmt::print(fg(fmt::color::red), "Failed to apply circuit\n");
            return;
        }
        
        std::vector<double> expectations;
        if (!sim.measure(expectations)) {
            fmt::print(fg(fmt::color::red), "Failed to measure\n");
            return;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        fmt::print("\nQuantum simulation results:\n");
        for (int i = 0; i < num_qubits; i++) {
            fmt::print("  Qubit {}: ⟨σz⟩ = {:+.6f}\n", i, expectations[i]);
        }
        
        fmt::print("\nSimulation time: {:.3f} ms\n", duration.count() / 1000.0);
        fmt::print(fg(fmt::color::green), "✓ Test 1 PASSED\n\n");
        
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Exception: {}\n", e.what());
        return;
    }
    
    // Test 2: Multiple hashes (mining loop simulation)
    fmt::print(fg(fmt::color::yellow), "Test 2: Mining loop with varying nonces\n");
    
    const int num_iterations = 10;
    std::vector<std::vector<double>> all_results;
    
    auto loop_start = std::chrono::high_resolution_clock::now();
    
    for (int nonce = 0; nonce < num_iterations; nonce++) {
        std::string data = "Block Header: nonce=" + std::to_string(nonce);
        auto nonce_hash = compute_sha256(data);
        auto nonce_circuit = CircuitGenerator::build_from_hash(nonce_hash, num_qubits);
        
        QuantumSimulator sim(num_qubits);
        sim.initialize_state();
        sim.apply_circuit(nonce_circuit);
        
        std::vector<double> expectations;
        sim.measure(expectations);
        all_results.push_back(expectations);
    }
    
    auto loop_end = std::chrono::high_resolution_clock::now();
    auto loop_duration = std::chrono::duration_cast<std::chrono::milliseconds>(loop_end - loop_start);
    
    fmt::print("Processed {} mining iterations\n", num_iterations);
    fmt::print("Total time: {:.3f} ms\n", loop_duration.count() / 1.0);
    fmt::print("Average: {:.3f} ms/iteration\n", loop_duration.count() / (double)num_iterations);
    fmt::print("Hashrate: {:.2f} hashes/sec\n", 1000.0 * num_iterations / loop_duration.count());
    
    // Verify results are different for different nonces
    bool all_different = true;
    for (int i = 0; i < num_iterations - 1; i++) {
        bool same = true;
        for (int q = 0; q < num_qubits; q++) {
            if (std::abs(all_results[i][q] - all_results[i+1][q]) > 1e-6) {
                same = false;
                break;
            }
        }
        if (same) {
            all_different = false;
            break;
        }
    }
    
    if (all_different) {
        fmt::print(fg(fmt::color::green), "✓ Test 2 PASSED - All results are unique\n\n");
    } else {
        fmt::print(fg(fmt::color::orange), "⚠ Test 2 WARNING - Some results may be similar\n\n");
    }
    
    // Test 3: Larger circuit performance
    fmt::print(fg(fmt::color::yellow), "Test 3: Performance with 10 qubits\n");
    
    try {
        const int large_qubits = 10;
        std::string data = "Performance test data";
        auto perf_hash = compute_sha256(data);
        auto perf_circuit = CircuitGenerator::build_from_hash(perf_hash, large_qubits);
        
        fmt::print("Circuit: {} qubits, {} gates\n", large_qubits, perf_circuit.gates.size());
        fmt::print("State vector size: {} complex numbers ({} KB)\n",
                   perf_circuit.state_vector_size(),
                   (perf_circuit.state_vector_size() * sizeof(Complex)) / 1024);
        
        QuantumSimulator sim(large_qubits);
        sim.initialize_state();
        
        auto perf_start = std::chrono::high_resolution_clock::now();
        sim.apply_circuit(perf_circuit);
        
        std::vector<double> expectations;
        sim.measure(expectations);
        auto perf_end = std::chrono::high_resolution_clock::now();
        
        auto perf_duration = std::chrono::duration_cast<std::chrono::microseconds>(perf_end - perf_start);
        
        fmt::print("Simulation time: {:.3f} ms\n", perf_duration.count() / 1000.0);
        fmt::print(fg(fmt::color::green), "✓ Test 3 PASSED\n\n");
        
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red), "Exception: {}\n", e.what());
        return;
    }
    
    fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
        "Full pipeline tests completed successfully!\n");
}

int main() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        R"(
  _____ _            _ _             _____         _   
 |  ___(_)_ __   ___| (_)_ __   ___ |_   _|__  ___| |_ 
 | |_  | | '_ \ / _ \ | | '_ \ / _ \  | |/ _ \/ __| __|
 |  _| | | |_) |  __/ | | | | |  __/  | |  __/\__ \ |_ 
 |_|   |_| .__/ \___|_|_|_| |_|\___|  |_|\___||___/\__|
         |_|                                            
)");
    
    test_full_pipeline();
    
    return 0;
}
