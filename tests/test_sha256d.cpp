/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/crypto/sha256d.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace ohmy::crypto;

// Helper to convert hex string to bytes
std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte_str = hex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::strtol(byte_str.c_str(), nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

// Helper to convert bytes to hex string
std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::ostringstream ss;
    for (uint8_t b : bytes) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return ss.str();
}

void test_sha256d_empty() {
    std::cout << "Testing SHA256d on empty input..." << std::endl;
    
    std::vector<uint8_t> input;
    std::vector<uint8_t> hash = sha256d(input);
    
    // SHA256d("") = SHA256(SHA256(""))
    // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    // SHA256(e3b0c442...) = 5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456
    std::string expected = "5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456";
    std::string result = bytes_to_hex(hash);
    
    assert(result == expected);
    std::cout << "  ✓ Empty input test passed" << std::endl;
}

void test_sha256d_hello_world() {
    std::cout << "Testing SHA256d on 'hello world'..." << std::endl;
    
    std::string input_str = "hello world";
    std::vector<uint8_t> input(input_str.begin(), input_str.end());
    std::vector<uint8_t> hash = sha256d(input);
    
    // SHA256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
    // SHA256(b94d27b9...) = bc62d4b80d9e36da29c16c5d4d9f11731f36052c72401a76c23c0fb5a9b74423
    std::string expected = "bc62d4b80d9e36da29c16c5d4d9f11731f36052c72401a76c23c0fb5a9b74423";
    std::string result = bytes_to_hex(hash);
    
    assert(result == expected);
    std::cout << "  ✓ 'hello world' test passed" << std::endl;
}

void test_sha256d_determinism() {
    std::cout << "Testing SHA256d determinism..." << std::endl;
    
    std::string input_str = "deterministic test";
    std::vector<uint8_t> input(input_str.begin(), input_str.end());
    
    std::vector<uint8_t> hash1 = sha256d(input);
    std::vector<uint8_t> hash2 = sha256d(input);
    
    assert(hash1.size() == hash2.size());
    assert(hash1 == hash2);
    
    std::cout << "  ✓ Determinism test passed" << std::endl;
}

void test_sha256d_bitcoin_genesis() {
    std::cout << "Testing SHA256d on Bitcoin genesis block header..." << std::endl;
    
    // Bitcoin genesis block header (80 bytes)
    std::string header_hex = 
        "0100000000000000000000000000000000000000000000000000000000000000"
        "000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
        "4b1e5e4a29ab5f49ffff001d1dac2b7c";
    
    std::vector<uint8_t> header = hex_to_bytes(header_hex);
    std::vector<uint8_t> hash = sha256d(header);
    
    // Reverse for little-endian (Bitcoin convention)
    std::reverse(hash.begin(), hash.end());
    std::string result = bytes_to_hex(hash);
    
    // Bitcoin genesis block hash
    std::string expected = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";
    
    assert(result == expected);
    std::cout << "  ✓ Bitcoin genesis block test passed" << std::endl;
}

void test_sha256d_binary_data() {
    std::cout << "Testing SHA256d on binary data..." << std::endl;
    
    // Test with binary data containing null bytes
    std::vector<uint8_t> input = {0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD, 0xFC};
    std::vector<uint8_t> hash = sha256d(input);
    
    // Should produce 32-byte hash
    assert(hash.size() == 32);
    
    // Should be consistent
    std::vector<uint8_t> hash2 = sha256d(input);
    assert(hash == hash2);
    
    std::cout << "  ✓ Binary data test passed" << std::endl;
}

void test_sha256d_large_input() {
    std::cout << "Testing SHA256d on large input..." << std::endl;
    
    // Create 1KB of data
    std::vector<uint8_t> input(1024);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<uint8_t>(i & 0xFF);
    }
    
    std::vector<uint8_t> hash = sha256d(input);
    
    // Should produce 32-byte hash
    assert(hash.size() == 32);
    
    // Should be deterministic
    std::vector<uint8_t> hash2 = sha256d(input);
    assert(hash == hash2);
    
    std::cout << "  ✓ Large input test passed" << std::endl;
}

int main() {
    std::cout << "\n=== SHA256d Cryptographic Tests ===" << std::endl;
    
    try {
        test_sha256d_empty();
        test_sha256d_hello_world();
        test_sha256d_determinism();
        test_sha256d_bitcoin_genesis();
        test_sha256d_binary_data();
        test_sha256d_large_input();
        
        std::cout << "\n✅ All SHA256d tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
