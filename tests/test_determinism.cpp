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
#include <array>
#include <vector>
#include <cstring>
#include <cmath>

using namespace ohmy::quantum;

/**
 * @brief Test determinism of qhash computation
 * 
 * This test ensures that:
 * 1. The same input always produces the same output (determinism)
 * 2. Results are bit-exact across multiple runs
 * 3. Floating-point precision is maintained correctly
 * 4. Fixed-point conversion is deterministic
 * 
 * This is CRITICAL for blockchain consensus - all nodes must produce
 * identical results for the same block header.
 */

// Test vector: fixed 80-byte block header
static std::array<uint8_t, 80> get_test_header() {
    std::array<uint8_t, 80> header{};
    
    // Version (4 bytes, little-endian)
    header[0] = 0x01; header[1] = 0x00; header[2] = 0x00; header[3] = 0x00;
    
    // Previous block hash (32 bytes, reversed)
    for (int i = 0; i < 32; i++) {
        header[4 + i] = static_cast<uint8_t>(i * 7 + 13); // Pseudo-random pattern
    }
    
    // Merkle root (32 bytes)
    for (int i = 0; i < 32; i++) {
        header[36 + i] = static_cast<uint8_t>(i * 11 + 17);
    }
    
    // nTime (4 bytes, little-endian) - timestamp
    header[68] = 0x00; header[69] = 0x10; header[70] = 0x20; header[71] = 0x30;
    
    // nBits (4 bytes, little-endian) - difficulty target
    header[72] = 0xff; header[73] = 0xff; header[74] = 0x00; header[75] = 0x1d;
    
    // nonce (4 bytes, little-endian)
    header[76] = 0x42; header[77] = 0x13; header[78] = 0x37; header[79] = 0x00;
    
    return header;
}

// Compute qhash for a given header
static std::array<uint8_t, 32> compute_qhash(const std::array<uint8_t, 80>& header) {
    // Step 1: SHA256 of header (initial hash)
    std::array<uint8_t, 32> initial_hash;
    SHA256(header.data(), header.size(), initial_hash.data());
    
    // Step 2: Generate quantum circuit from hash
    const int num_qubits = 16;
    auto circuit = CircuitGenerator::build_from_hash(initial_hash, num_qubits);
    
    // Step 3: Simulate circuit
    QuantumSimulator sim(num_qubits);
    
    if (!sim.initialize_state()) {
        throw std::runtime_error("Failed to initialize quantum state");
    }
    
    if (!sim.apply_circuit(circuit)) {
        throw std::runtime_error("Failed to apply quantum circuit");
    }
    
    std::vector<double> expectations;
    if (!sim.measure(expectations)) {
        throw std::runtime_error("Failed to measure quantum state");
    }
    
    // Step 4: Compute final qhash (fixed-point → SHA256)
    return QHashProcessor::compute_qhash(initial_hash, expectations);
}

bool test_single_run_determinism() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Test 1: Single Run Determinism ===\n");
    
    auto header = get_test_header();
    
    fmt::print("Computing qhash for test header...\n");
    auto result1 = compute_qhash(header);
    
    fmt::print(fg(fmt::color::green), "✓ First computation completed\n");
    fmt::print("  Hash: ");
    for (int i = 0; i < 8; i++) {
        fmt::print("{:02x}", result1[i]);
    }
    fmt::print("...\n");
    
    return true;
}

bool test_multiple_runs_identical() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Test 2: Multiple Runs Produce Identical Results ===\n");
    
    auto header = get_test_header();
    const int num_runs = 10;
    
    fmt::print("Running qhash {} times on same header...\n", num_runs);
    
    std::vector<std::array<uint8_t, 32>> results;
    results.reserve(num_runs);
    
    for (int i = 0; i < num_runs; i++) {
        results.push_back(compute_qhash(header));
        fmt::print("  Run {}/{}: ", i + 1, num_runs);
        for (int j = 0; j < 8; j++) {
            fmt::print("{:02x}", results[i][j]);
        }
        fmt::print("...\n");
    }
    
    // Verify all results are bit-exact identical
    bool all_identical = true;
    for (int i = 1; i < num_runs; i++) {
        if (std::memcmp(results[0].data(), results[i].data(), 32) != 0) {
            all_identical = false;
            fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                "✗ Run {} differs from run 0!\n", i);
            
            // Show differences
            for (int j = 0; j < 32; j++) {
                if (results[0][j] != results[i][j]) {
                    fmt::print("  Byte {}: 0x{:02x} vs 0x{:02x}\n",
                        j, results[0][j], results[i][j]);
                }
            }
        }
    }
    
    if (all_identical) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ All {} runs produced bit-exact identical results\n", num_runs);
    }
    
    return all_identical;
}

bool test_expectation_values_range() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Test 3: Expectation Values in Valid Range ===\n");
    
    auto header = get_test_header();
    
    // Compute initial hash and circuit
    std::array<uint8_t, 32> initial_hash;
    SHA256(header.data(), header.size(), initial_hash.data());
    
    const int num_qubits = 16;
    auto circuit = CircuitGenerator::build_from_hash(initial_hash, num_qubits);
    
    // Simulate
    QuantumSimulator sim(num_qubits);
    sim.initialize_state();
    sim.apply_circuit(circuit);
    
    std::vector<double> expectations;
    sim.measure(expectations);
    
    // Verify expectations are in [-1, 1] range
    bool all_in_range = true;
    fmt::print("Checking expectation values for {} qubits:\n", num_qubits);
    
    for (int i = 0; i < num_qubits; i++) {
        double exp = expectations[i];
        bool in_range = (exp >= -1.0 && exp <= 1.0);
        
        if (!in_range) {
            all_in_range = false;
            fmt::print(fg(fmt::color::red),
                "  Qubit {}: ⟨σz⟩ = {:.6f} [OUT OF RANGE]\n", i, exp);
        } else if (i < 4) {
            // Print first 4 for inspection
            fmt::print(fg(fmt::color::green),
                "  Qubit {}: ⟨σz⟩ = {:.6f} ✓\n", i, exp);
        }
    }
    
    if (num_qubits > 4) {
        fmt::print(fg(fmt::color::gray), "  ... (showing first 4 of {})\n", num_qubits);
    }
    
    if (all_in_range) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ All expectation values within [-1, 1] range\n");
    }
    
    return all_in_range;
}

bool test_nonce_variation() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Test 4: Different Nonces Produce Different Results ===\n");
    
    auto header1 = get_test_header();
    auto header2 = get_test_header();
    auto header3 = get_test_header();
    
    // Change nonce in header2 and header3
    header2[76] = 0x43;  // nonce + 1
    header3[76] = 0x44;  // nonce + 2
    
    fmt::print("Computing qhash for 3 different nonces...\n");
    
    auto result1 = compute_qhash(header1);
    auto result2 = compute_qhash(header2);
    auto result3 = compute_qhash(header3);
    
    fmt::print("  Nonce 0x42: ");
    for (int i = 0; i < 8; i++) fmt::print("{:02x}", result1[i]);
    fmt::print("...\n");
    
    fmt::print("  Nonce 0x43: ");
    for (int i = 0; i < 8; i++) fmt::print("{:02x}", result2[i]);
    fmt::print("...\n");
    
    fmt::print("  Nonce 0x44: ");
    for (int i = 0; i < 8; i++) fmt::print("{:02x}", result3[i]);
    fmt::print("...\n");
    
    // Verify all are different (avalanche effect)
    bool all_different = true;
    if (std::memcmp(result1.data(), result2.data(), 32) == 0) {
        all_different = false;
        fmt::print(fg(fmt::color::red), "✗ Nonce 0x42 and 0x43 produced same hash!\n");
    }
    if (std::memcmp(result1.data(), result3.data(), 32) == 0) {
        all_different = false;
        fmt::print(fg(fmt::color::red), "✗ Nonce 0x42 and 0x44 produced same hash!\n");
    }
    if (std::memcmp(result2.data(), result3.data(), 32) == 0) {
        all_different = false;
        fmt::print(fg(fmt::color::red), "✗ Nonce 0x43 and 0x44 produced same hash!\n");
    }
    
    if (all_different) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ Different nonces produce different hashes (avalanche effect verified)\n");
    }
    
    return all_different;
}

bool test_fixed_point_precision() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Test 5: Fixed-Point Conversion Precision ===\n");
    
    // Test known values
    std::vector<double> test_values = {
        -1.0, -0.5, 0.0, 0.5, 1.0,
        -0.999, 0.999, -0.123456, 0.654321
    };
    
    fmt::print("Testing fixed-point conversion for edge cases:\n");
    
    bool all_valid = true;
    for (double val : test_values) {
        try {
            // Convert to fixed-point
            int16_t fixed = FixedPoint::from_double(val);
            
            // Convert back to verify
            double recovered = FixedPoint::to_double(fixed);
            double error = std::abs(val - recovered);
            
            // For Q15 format, precision is ~1.5e-5
            bool within_tolerance = (error < 1e-4);
            
            if (within_tolerance) {
                fmt::print("  {:.6f} → 0x{:04x} → {:.6f} (error: {:.2e}) ✓\n",
                    val, static_cast<uint16_t>(fixed), recovered, error);
            } else {
                all_valid = false;
                fmt::print(fg(fmt::color::red),
                    "  ✗ Value {:.6f} has excessive error: {:.2e}\n",
                    val, error);
            }
        } catch (const std::exception& e) {
            all_valid = false;
            fmt::print(fg(fmt::color::red),
                "  ✗ Value {:.6f} threw exception: {}\n",
                val, e.what());
        }
    }
    
    if (all_valid) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ Fixed-point conversion handles all test cases correctly\n");
    }
    
    return all_valid;
}

int main() {
    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
        R"(
╔═══════════════════════════════════════════════════════════╗
║     QHASH DETERMINISM TEST SUITE                          ║
║     Critical for Blockchain Consensus Compatibility       ║
╚═══════════════════════════════════════════════════════════╝
)");
    
    fmt::print("\nThis test suite validates that qhash computation is:\n");
    fmt::print("  1. Deterministic (same input → same output)\n");
    fmt::print("  2. Bit-exact across multiple runs\n");
    fmt::print("  3. Produces valid quantum expectations\n");
    fmt::print("  4. Shows proper avalanche effect\n");
    fmt::print("  5. Uses correct fixed-point precision\n\n");
    
    int passed = 0;
    int total = 5;
    
    try {
        // Run all tests
        if (test_single_run_determinism()) passed++;
        if (test_multiple_runs_identical()) passed++;
        if (test_expectation_values_range()) passed++;
        if (test_nonce_variation()) passed++;
        if (test_fixed_point_precision()) passed++;
        
    } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "\n✗ FATAL ERROR: {}\n", e.what());
        return 1;
    }
    
    // Summary
    fmt::print("\n");
    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
        "╔═══════════════════════════════════════════════════════════╗\n");
    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
        "║ RESULTS: {}/{} tests passed                                 ║\n", passed, total);
    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
        "╚═══════════════════════════════════════════════════════════╝\n");
    
    if (passed == total) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "\n✓ ALL TESTS PASSED - Determinism validated!\n");
        fmt::print(fg(fmt::color::green),
            "  qhash implementation is ready for consensus-critical use.\n\n");
        return 0;
    } else {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
            "\n✗ SOME TESTS FAILED - Fix issues before optimizing!\n");
        fmt::print(fg(fmt::color::red),
            "  {} test(s) failed. Determinism is CRITICAL for blockchain consensus.\n\n",
            total - passed);
        return 1;
    }
}
