/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "fixed_point.hpp"
#include "circuit_generator.hpp"
#include "quantum_kernel.cuh"
#include <fmt/core.h>
#include <fmt/color.h>
#include <openssl/sha.h>
#include <cmath>
#include <iomanip>

using namespace ohmy::quantum;

void print_hex(const std::vector<uint8_t>& data, size_t max_bytes = 16) {
    for (size_t i = 0; i < std::min(data.size(), max_bytes); i++) {
        fmt::print("{:02x}", data[i]);
    }
    if (data.size() > max_bytes) {
        fmt::print("...");
    }
}

void print_hex_array(const std::array<uint8_t, 32>& data, size_t max_bytes = 16) {
    for (size_t i = 0; i < std::min(static_cast<size_t>(32), max_bytes); i++) {
        fmt::print("{:02x}", data[i]);
    }
    if (32 > max_bytes) {
        fmt::print("...");
    }
}

void test_fixed_point_conversion() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Testing Fixed-Point Arithmetic ===\n\n");
    
    // Test 1: Basic conversions
    fmt::print(fg(fmt::color::yellow), "Test 1: Double ↔ Fixed-Point conversions\n");
    
    struct TestCase {
        double value;
        const char* description;
    };
    
    TestCase test_cases[] = {
        {0.0, "Zero"},
        {1.0, "One"},
        {-1.0, "Negative one"},
        {0.5, "Half"},
        {-0.5, "Negative half"},
        {0.123456789, "Small positive"},
        {-0.987654321, "Small negative"},
        {0.0000001, "Very small"},
    };
    
    bool all_passed = true;
    for (const auto& tc : test_cases) {
        int64_t fixed = FixedPoint::from_double(tc.value);
        double recovered = FixedPoint::to_double(fixed);
        double error = std::abs(recovered - tc.value);
        
        fmt::print("  {:<20} {: .10f} → 0x{:016x} → {: .10f}  (error: {:.2e})\n",
                   tc.description, tc.value, 
                   static_cast<uint64_t>(fixed), 
                   recovered, error);
        
        // Error should be less than precision (2^-32 ≈ 2.3e-10)
        if (error > 1e-9) {
            all_passed = false;
        }
    }
    
    if (all_passed) {
        fmt::print(fg(fmt::color::green), "✓ Test 1 PASSED\n\n");
    } else {
        fmt::print(fg(fmt::color::red), "✗ Test 1 FAILED\n\n");
    }
    
    // Test 2: Determinism across multiple conversions
    fmt::print(fg(fmt::color::yellow), "Test 2: Determinism test\n");
    
    double test_value = 0.123456789;
    std::vector<int64_t> results;
    
    for (int i = 0; i < 100; i++) {
        int64_t fixed = FixedPoint::from_double(test_value);
        results.push_back(fixed);
    }
    
    bool deterministic = true;
    for (size_t i = 1; i < results.size(); i++) {
        if (results[i] != results[0]) {
            deterministic = false;
            break;
        }
    }
    
    if (deterministic) {
        fmt::print("  100 conversions of {} → 0x{:016x} (all identical)\n", 
                   test_value, static_cast<uint64_t>(results[0]));
        fmt::print(fg(fmt::color::green), "✓ Test 2 PASSED - Deterministic\n\n");
    } else {
        fmt::print(fg(fmt::color::red), "✗ Test 2 FAILED - Non-deterministic!\n\n");
    }
    
    // Test 3: Byte serialization
    fmt::print(fg(fmt::color::yellow), "Test 3: Byte serialization (big-endian)\n");
    
    std::vector<double> expectations = {0.5, -0.5, 0.0, 1.0};
    auto bytes = FixedPoint::expectations_to_bytes(expectations);
    
    fmt::print("  Expectations: [");
    for (size_t i = 0; i < expectations.size(); i++) {
        fmt::print("{:.3f}", expectations[i]);
        if (i < expectations.size() - 1) fmt::print(", ");
    }
    fmt::print("]\n");
    
    fmt::print("  Bytes ({} total): ", bytes.size());
    print_hex(bytes, 32);
    fmt::print("\n");
    
    if (bytes.size() == expectations.size() * 8) {
        fmt::print(fg(fmt::color::green), "✓ Test 3 PASSED\n\n");
    } else {
        fmt::print(fg(fmt::color::red), "✗ Test 3 FAILED\n\n");
    }
}

void test_qhash_algorithm() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Testing Complete qhash Algorithm ===\n\n");
    
    // Test: Full qhash pipeline
    fmt::print(fg(fmt::color::yellow), "Test: SHA256 → Circuit → Simulation → Fixed-Point → XOR → SHA3\n");
    
    // Generate initial hash
    std::string block_data = "Block: nonce=99999, prev=000000...";
    std::array<uint8_t, 32> initial_hash;
    SHA256(reinterpret_cast<const unsigned char*>(block_data.c_str()), 
           block_data.size(), initial_hash.data());
    
    fmt::print("Input: \"{}\"\n", block_data);
    fmt::print("SHA256: ");
    print_hex_array(initial_hash, 16);
    fmt::print("\n\n");
    
    // Generate and simulate quantum circuit
    const int num_qubits = 8;
    auto circuit = CircuitGenerator::build_from_hash(initial_hash, num_qubits);
    
    fmt::print("Circuit: {} qubits, {} gates\n", circuit.num_qubits, circuit.gates.size());
    
    QuantumSimulator sim(num_qubits);
    sim.initialize_state();
    sim.apply_circuit(circuit);
    
    std::vector<double> expectations;
    sim.measure(expectations);
    
    fmt::print("Quantum expectations: [");
    for (size_t i = 0; i < expectations.size(); i++) {
        fmt::print("{:+.6f}", expectations[i]);
        if (i < expectations.size() - 1) fmt::print(", ");
    }
    fmt::print("]\n\n");
    
    // Convert to fixed-point bytes
    auto quantum_bytes = FixedPoint::expectations_to_bytes(expectations);
    fmt::print("Fixed-point bytes ({} total): ", quantum_bytes.size());
    print_hex(quantum_bytes, 16);
    fmt::print("\n\n");
    
    // Compute final qhash
    auto final_hash = QHashProcessor::compute_qhash(initial_hash, expectations);
    
    fmt::print("Final qhash: ");
    print_hex_array(final_hash, 32);
    fmt::print("\n");
    
    fmt::print(fg(fmt::color::green) | fmt::emphasis::bold, 
        "\n✓ Complete qhash algorithm executed successfully!\n");
    
    // Test determinism
    fmt::print(fg(fmt::color::yellow), "\nDeterminism test: Same input → Same output\n");
    
    auto final_hash2 = QHashProcessor::compute_qhash(initial_hash, expectations);
    
    bool identical = (final_hash == final_hash2);
    
    if (identical) {
        fmt::print(fg(fmt::color::green), "✓ qhash is deterministic\n\n");
    } else {
        fmt::print(fg(fmt::color::red), "✗ qhash is NOT deterministic!\n\n");
    }
}

void test_mining_scenario() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        "\n=== Testing Mining Scenario ===\n\n");
    
    fmt::print(fg(fmt::color::yellow), "Simulating mining with difficulty check\n\n");
    
    // Mock difficulty target (first N bytes must be zero)
    uint32_t difficulty_zeros = 2;  // First 2 bytes must be 0x00
    
    fmt::print("Difficulty: First {} bytes must be 0x00\n", difficulty_zeros);
    fmt::print("Mining attempts:\n\n");
    
    bool found_solution = false;
    int attempts = 0;
    const int max_attempts = 100;
    
    for (uint32_t nonce = 0; nonce < max_attempts; nonce++) {
        std::string block_data = "Block: nonce=" + std::to_string(nonce);
        
        // SHA256 initial hash
        std::array<uint8_t, 32> initial_hash;
        SHA256(reinterpret_cast<const unsigned char*>(block_data.c_str()),
               block_data.size(), initial_hash.data());
        
        // Generate circuit and simulate
        auto circuit = CircuitGenerator::build_from_hash(initial_hash, 8);
        QuantumSimulator sim(8);
        sim.initialize_state();
        sim.apply_circuit(circuit);
        
        std::vector<double> expectations;
        sim.measure(expectations);
        
        // Compute qhash
        auto qhash = QHashProcessor::compute_qhash(initial_hash, expectations);
        
        attempts++;
        
        // Check difficulty
        bool meets_difficulty = true;
        for (uint32_t i = 0; i < difficulty_zeros; i++) {
            if (qhash[i] != 0x00) {
                meets_difficulty = false;
                break;
            }
        }
        
        if (nonce < 5 || meets_difficulty) {
            fmt::print("  Nonce {:5d}: qhash=", nonce);
            print_hex_array(qhash, 8);
            if (meets_difficulty) {
                fmt::print(fg(fmt::color::green), " ✓ VALID SHARE!");
                found_solution = true;
            }
            fmt::print("\n");
        }
        
        if (meets_difficulty) {
            break;
        }
    }
    
    fmt::print("\nAttempts: {}\n", attempts);
    
    if (found_solution) {
        fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
            "✓ Found valid share!\n");
    } else {
        fmt::print(fg(fmt::color::orange),
            "⚠ No valid share found in {} attempts (expected with difficulty {})\n",
            max_attempts, difficulty_zeros);
    }
}

int main() {
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
        R"(
  _____ _              _ ____       _       _     _____         _   
 |  ___(_)_  _____  __| |  _ \ ___ (_)_ __ | |_  |_   _|__  ___| |_ 
 | |_  | \ \/ / _ \/ _` | |_) / _ \| | '_ \| __|   | |/ _ \/ __| __|
 |  _| | |>  <  __/ (_| |  __/ (_) | | | | | |_    | |  __/\__ \ |_ 
 |_|   |_/_/\_\___|\__,_|_|   \___/|_|_| |_|\__|   |_|\___||___/\__|
                                                                     
)");
    
    test_fixed_point_conversion();
    test_qhash_algorithm();
    test_mining_scenario();
    
    fmt::print(fg(fmt::color::green) | fmt::emphasis::bold,
        "\nAll fixed-point and qhash tests completed!\n");
    
    return 0;
}
