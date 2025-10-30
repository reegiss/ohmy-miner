/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/fixed_point.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ohmy;

void test_fixed_point_construction() {
    std::cout << "Testing fixed-point construction..." << std::endl;
    
    // Test from_raw
    Q15 a = Q15::from_raw(32768);
    (void)a;  // Mark as intentionally unused for this test
    assert(a.raw() == 32768);
    
    // Test from_float
    Q15 b = Q15::from_float(1.0);
    (void)b;
    assert(std::abs(b.to_double() - 1.0) < 0.001);
    
    Q15 c = Q15::from_float(0.5);
    (void)c;
    assert(std::abs(c.to_double() - 0.5) < 0.001);
    
    // Test from_int
    Q15 d = Q15::from_int(5);
    (void)d;
    assert(d.to_int() == 5);
    
    std::cout << "  ✓ Construction tests passed" << std::endl;
}

void test_fixed_point_arithmetic() {
    std::cout << "Testing fixed-point arithmetic..." << std::endl;
    
    Q15 a = Q15::from_float(2.0);
    Q15 b = Q15::from_float(3.0);
    
    // Addition
    Q15 sum = a + b;
    (void)sum;
    assert(std::abs(sum.to_double() - 5.0) < 0.001);
    
    // Subtraction
    Q15 diff = b - a;
    (void)diff;
    assert(std::abs(diff.to_double() - 1.0) < 0.001);
    
    // Multiplication
    Q15 prod = a * b;
    (void)prod;
    assert(std::abs(prod.to_double() - 6.0) < 0.01);
    
    // Division
    Q15 quot = b / a;
    (void)quot;
    assert(std::abs(quot.to_double() - 1.5) < 0.01);
    
    std::cout << "  ✓ Arithmetic tests passed" << std::endl;
}

void test_fixed_point_comparison() {
    std::cout << "Testing fixed-point comparison..." << std::endl;
    
    Q15 a = Q15::from_float(1.5);
    Q15 b = Q15::from_float(2.0);
    Q15 c = Q15::from_float(1.5);
    (void)a; (void)b; (void)c;
    
    assert(a < b);
    assert(b > a);
    assert(a == c);
    assert(a != b);
    assert(a <= c);
    assert(b >= a);
    
    std::cout << "  ✓ Comparison tests passed" << std::endl;
}

void test_fixed_point_determinism() {
    std::cout << "Testing fixed-point determinism..." << std::endl;
    
    // Same operations should give same raw values
    Q15 a1 = Q15::from_float(0.123456);
    Q15 a2 = Q15::from_float(0.123456);
    assert(a1.raw() == a2.raw());
    
    Q15 b1 = Q15::from_float(0.5);
    Q15 b2 = Q15::from_float(0.5);
    Q15 result1 = a1 * b1;
    Q15 result2 = a2 * b2;
    (void)result1; (void)result2;
    assert(result1.raw() == result2.raw());
    
    std::cout << "  ✓ Determinism tests passed" << std::endl;
}

void test_fixed_point_range() {
    std::cout << "Testing fixed-point range..." << std::endl;
    
    // Test values near bounds
    Q15 small = Q15::from_float(0.0001);
    (void)small;
    assert(small.to_double() > 0.0);
    
    Q15 large = Q15::from_float(1000.0);
    (void)large;
    assert(large.to_double() > 999.0);
    
    Q15 negative = Q15::from_float(-10.5);
    (void)negative;
    assert(negative.to_double() < -10.0);
    
    std::cout << "  ✓ Range tests passed" << std::endl;
}

int main() {
    std::cout << "\n=== Fixed-Point Arithmetic Tests ===" << std::endl;
    
    try {
        test_fixed_point_construction();
        test_fixed_point_arithmetic();
        test_fixed_point_comparison();
        test_fixed_point_determinism();
        test_fixed_point_range();
        
        std::cout << "\n✅ All fixed-point tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
