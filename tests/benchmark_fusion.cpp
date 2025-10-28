/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "quantum_kernel.cuh"
#include "circuit_generator.hpp"
#include "fixed_point.hpp"
#include <fmt/core.h>
#include <fmt/color.h>
#include <openssl/sha.h>
#include <chrono>
#include <array>
#include <vector>
#include <cstring>

using namespace ohmy::quantum;
using namespace std::chrono;

// Test header
static std::array<uint8_t, 80> get_test_header() {
    std::array<uint8_t, 80> header{};
    
    header[0] = 0x01; header[1] = 0x00; header[2] = 0x00; header[3] = 0x00;
    
    for (int i = 0; i < 32; i++) {
        header[4 + i] = static_cast<uint8_t>(i * 7 + 13);
    }
    
    for (int i = 0; i < 32; i++) {
        header[36 + i] = static_cast<uint8_t>(i * 11 + 17);
    }
    
    header[68] = 0x00; header[69] = 0x10; header[70] = 0x20; header[71] = 0x30;
    header[72] = 0xff; header[73] = 0xff; header[74] = 0x00; header[75] = 0x1d;
    header[76] = 0x42; header[77] = 0x13; header[78] = 0x37; header[79] = 0x00;
    
    return header;
}

// Benchmark qhash with original implementation
double benchmark_original(int iterations) {
    auto header = get_test_header();
    
    std::array<uint8_t, 32> initial_hash;
    SHA256(header.data(), header.size(), initial_hash.data());
    
    const int num_qubits = 16;
    auto circuit = CircuitGenerator::build_from_hash(initial_hash, num_qubits);
    
    QuantumSimulator sim(num_qubits);
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        sim.initialize_state();
        sim.apply_circuit(circuit);
        
        std::vector<double> expectations;
        sim.measure(expectations);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    
    return static_cast<double>(duration) / iterations;
}

// Benchmark qhash with optimized fused kernel
double benchmark_optimized(int iterations) {
    auto header = get_test_header();
    
    std::array<uint8_t, 32> initial_hash;
    SHA256(header.data(), header.size(), initial_hash.data());
    
    const int num_qubits = 16;
    auto circuit = CircuitGenerator::build_from_hash(initial_hash, num_qubits);
    
    QuantumSimulator sim(num_qubits);
    
    auto start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        sim.initialize_state();
        sim.apply_circuit_optimized(circuit);
        
        std::vector<double> expectations;
        sim.measure(expectations);
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    
    return static_cast<double>(duration) / iterations;
}

// Verify determinism
bool verify_determinism() {
    auto header = get_test_header();
    
    std::array<uint8_t, 32> initial_hash;
    SHA256(header.data(), header.size(), initial_hash.data());
    
    const int num_qubits = 16;
    auto circuit = CircuitGenerator::build_from_hash(initial_hash, num_qubits);
    
    // Run original version
    QuantumSimulator sim1(num_qubits);
    sim1.initialize_state();
    sim1.apply_circuit(circuit);
    std::vector<double> exp1;
    sim1.measure(exp1);
    auto hash1 = QHashProcessor::compute_qhash(initial_hash, exp1);
    
    // Run optimized version
    QuantumSimulator sim2(num_qubits);
    sim2.initialize_state();
    sim2.apply_circuit_optimized(circuit);
    std::vector<double> exp2;
    sim2.measure(exp2);
    auto hash2 = QHashProcessor::compute_qhash(initial_hash, exp2);
    
    // Compare bit-exact
    bool identical = (std::memcmp(hash1.data(), hash2.data(), 32) == 0);
    
    if (!identical) {
        fmt::print(fg(fmt::color::red), "Results differ!\n");
        fmt::print("Original:  ");
        for (int i = 0; i < 8; i++) fmt::print("{:02x}", hash1[i]);
        fmt::print("...\n");
        fmt::print("Optimized: ");
        for (int i = 0; i < 8; i++) fmt::print("{:02x}", hash2[i]);
        fmt::print("...\n");
    }
    
    return identical;
}

int main() {
    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
        R"(
╔═══════════════════════════════════════════════════════════╗
║     GATE FUSION BENCHMARK                                 ║
║     Comparing Original vs. Optimized Implementation       ║
╚═══════════════════════════════════════════════════════════╝
)");
    
    fmt::print("\n");
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "=== Step 1: Determinism Verification ===\n");
    
    fmt::print("Verifying that optimized version produces identical results...\n");
    
    bool deterministic = verify_determinism();
    
    if (!deterministic) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "✗ FAILED: Optimized version produces different results!\n");
        fmt::print(fg(fmt::color::red),
            "  Cannot proceed with benchmark. Fix determinism first.\n\n");
        return 1;
    }
    
    fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
        "✓ PASSED: Optimized version is bit-exact identical!\n\n");
    
    // Warmup
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "=== Step 2: GPU Warmup ===\n");
    fmt::print("Running warmup iterations...\n");
    benchmark_original(5);
    benchmark_optimized(5);
    fmt::print(fg(fmt::color::green), "✓ Warmup complete\n\n");
    
    // Benchmark
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "=== Step 3: Performance Benchmark ===\n");
    
    const int iterations = 50;
    fmt::print("Running {} iterations per version...\n\n", iterations);
    
    fmt::print(fg(fmt::color::yellow), "Benchmarking ORIGINAL implementation...\n");
    double time_original = benchmark_original(iterations);
    fmt::print("  Average time per hash: {:.2f} ms\n", time_original);
    fmt::print("  Estimated hashrate: {:.2f} H/s\n\n", 1000.0 / time_original);
    
    fmt::print(fg(fmt::color::yellow), "Benchmarking OPTIMIZED implementation...\n");
    double time_optimized = benchmark_optimized(iterations);
    fmt::print("  Average time per hash: {:.2f} ms\n", time_optimized);
    fmt::print("  Estimated hashrate: {:.2f} H/s\n\n", 1000.0 / time_optimized);
    
    // Results
    double speedup = time_original / time_optimized;
    
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "╔═══════════════════════════════════════════════════════════╗\n");
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "║ RESULTS                                                   ║\n");
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "╚═══════════════════════════════════════════════════════════╝\n");
    
    fmt::print("\n");
    fmt::print("Original implementation:  {:.2f} ms/hash ({:.2f} H/s)\n", 
        time_original, 1000.0 / time_original);
    fmt::print("Optimized implementation: {:.2f} ms/hash ({:.2f} H/s)\n", 
        time_optimized, 1000.0 / time_optimized);
    fmt::print("\n");
    
    if (speedup >= 3.0) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ EXCELLENT: {:.2f}x speedup achieved!\n", speedup);
        fmt::print(fg(fmt::color::green),
            "  Gate fusion optimization is working as expected.\n\n");
    } else if (speedup >= 2.0) {
        fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
            "⚠ GOOD: {:.2f}x speedup achieved.\n", speedup);
        fmt::print(fg(fmt::color::yellow),
            "  Expected 3-5x. Consider further tuning.\n\n");
    } else {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "✗ POOR: Only {:.2f}x speedup achieved.\n", speedup);
        fmt::print(fg(fmt::color::red),
            "  Expected 3-5x. Implementation may need review.\n\n");
    }
    
    return 0;
}
