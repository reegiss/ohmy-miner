/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/crypto/difficulty.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

using namespace ohmy::crypto;

void test_compact_target_decode() {
    std::cout << "Testing compact target decoding..." << std::endl;
    
    // Bitcoin difficulty 1 target: 0x1d00ffff
    uint32_t bits = 0x1d00ffff;
    std::vector<uint8_t> target = decode_compact_target(bits);
    
    // Should be 32 bytes
    assert(target.size() == 32);
    
    // Bitcoin diff 1: 0x00000000ffff0000000000000000000000000000000000000000000000000000
    assert(target[0] == 0x00);
    assert(target[1] == 0x00);
    assert(target[2] == 0x00);
    assert(target[3] == 0x00);
    assert(target[4] == 0xff);
    assert(target[5] == 0xff);
    
    std::cout << "  ✓ Compact target decode test passed" << std::endl;
}

void test_compact_target_various() {
    std::cout << "Testing various compact targets..." << std::endl;
    
    // Test easy target (0x2007ffff)
    uint32_t easy_bits = 0x2007ffff;
    std::vector<uint8_t> easy_target = decode_compact_target(easy_bits);
    assert(easy_target.size() == 32);
    
    // Test hard target (0x1b0404cb)
    uint32_t hard_bits = 0x1b0404cb;
    std::vector<uint8_t> hard_target = decode_compact_target(hard_bits);
    assert(hard_target.size() == 32);
    
    std::cout << "  ✓ Various compact target tests passed" << std::endl;
}

void test_hash_meets_target_easy() {
    std::cout << "Testing hash meets easy target..." << std::endl;
    
    // Very easy target (lots of leading zeros required)
    [[maybe_unused]] uint32_t bits = 0x20ffffff;
    
    // Hash with many leading zeros (meets target)
    std::vector<uint8_t> good_hash(32, 0);
    good_hash[31] = 0x01;  // Small value
    
    assert(hash_meets_target(good_hash.data(), 0x20ffffff));
    
    // Hash with no leading zeros (doesn't meet target)
    std::vector<uint8_t> bad_hash(32, 0xff);
    
    assert(!hash_meets_target(bad_hash.data(), 0x20ffffff));
    
    std::cout << "  ✓ Easy target test passed" << std::endl;
}

void test_hash_meets_target_boundary() {
    std::cout << "Testing hash meets target boundary..." << std::endl;
    
    uint32_t bits = 0x1d00ffff;  // Bitcoin difficulty 1
    std::vector<uint8_t> target = decode_compact_target(bits);
    
    // Hash exactly at target should meet it
    assert(hash_meets_target(target.data(), bits));
    
    // Hash slightly below target should meet it
    std::vector<uint8_t> below_target = target;
    if (below_target[31] > 0) {
        below_target[31]--;
    }
    assert(hash_meets_target(below_target.data(), bits));
    
    // Hash above target should not meet it
    std::vector<uint8_t> above_target = target;
    if (above_target[4] < 0xff) {
        above_target[4]++;
    }
    assert(!hash_meets_target(above_target.data(), bits));
    
    std::cout << "  ✓ Boundary test passed" << std::endl;
}

void test_hash_meets_target_zero() {
    std::cout << "Testing all-zero hash..." << std::endl;
    
    // All-zero hash should meet any target
    std::vector<uint8_t> zero_hash(32, 0);
    
    assert(hash_meets_target(zero_hash.data(), 0x1d00ffff));
    assert(hash_meets_target(zero_hash.data(), 0x1b0404cb));
    assert(hash_meets_target(zero_hash.data(), 0x20ffffff));
    
    std::cout << "  ✓ Zero hash test passed" << std::endl;
}

void test_hash_meets_target_max() {
    std::cout << "Testing max hash..." << std::endl;
    
    // All-ones hash should not meet any reasonable target
    std::vector<uint8_t> max_hash(32, 0xff);
    
    assert(!hash_meets_target(max_hash.data(), 0x1d00ffff));
    assert(!hash_meets_target(max_hash.data(), 0x1b0404cb));
    
    std::cout << "  ✓ Max hash test passed" << std::endl;
}

void test_target_determinism() {
    std::cout << "Testing target decoding determinism..." << std::endl;
    
    uint32_t bits = 0x1d00ffff;
    
    std::vector<uint8_t> target1 = decode_compact_target(bits);
    std::vector<uint8_t> target2 = decode_compact_target(bits);
    
    assert(target1 == target2);
    
    std::cout << "  ✓ Determinism test passed" << std::endl;
}

int main() {
    std::cout << "\n=== Difficulty Target Tests ===" << std::endl;
    
    try {
        test_compact_target_decode();
        test_compact_target_various();
        test_hash_meets_target_easy();
        test_hash_meets_target_boundary();
        test_hash_meets_target_zero();
        test_hash_meets_target_max();
        test_target_determinism();
        
        std::cout << "\n✅ All difficulty tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
