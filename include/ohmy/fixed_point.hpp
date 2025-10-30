/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <string>

namespace ohmy {

/**
 * Fixed-point arithmetic for deterministic cross-platform consensus
 * Template parameters:
 * - IntType: underlying integer type (int32_t, int64_t)
 * - FracBits: number of fractional bits
 */
template<typename IntType, int FracBits>
class fixed_point {
public:
    static_assert(FracBits > 0 && FracBits < sizeof(IntType) * 8, "Invalid fractional bits");
    
    using value_type = IntType;
    static constexpr int fractional_bits = FracBits;
    static constexpr IntType scale_factor = IntType(1) << FracBits;
    
    // Constructors
    constexpr fixed_point() : value_(0) {}
    constexpr explicit fixed_point(IntType raw_value) : value_(raw_value) {}
    
    // Factory methods for type safety
    static constexpr fixed_point from_raw(IntType raw) {
        return fixed_point(raw);
    }
    
    static fixed_point from_float(double f) {
        return fixed_point(static_cast<IntType>(f * scale_factor + (f >= 0 ? 0.5 : -0.5)));
    }
    
    static fixed_point from_int(IntType i) {
        return fixed_point(i << FracBits);
    }
    
    // Conversions
    constexpr IntType raw() const { return value_; }
    
    double to_double() const {
        return static_cast<double>(value_) / scale_factor;
    }
    
    IntType to_int() const {
        return value_ >> FracBits;
    }
    
    // Arithmetic operators
    fixed_point operator+(const fixed_point& other) const {
        return fixed_point(value_ + other.value_);
    }
    
    fixed_point operator-(const fixed_point& other) const {
        return fixed_point(value_ - other.value_);
    }
    
    fixed_point operator*(const fixed_point& other) const {
        // Use 64-bit intermediate to prevent overflow
        using WideType = typename std::conditional<
            sizeof(IntType) == 4, int64_t, __int128_t>::type;
        
        WideType wide_result = static_cast<WideType>(value_) * other.value_;
        return fixed_point(static_cast<IntType>(wide_result >> FracBits));
    }
    
    fixed_point operator/(const fixed_point& other) const {
        using WideType = typename std::conditional<
            sizeof(IntType) == 4, int64_t, __int128_t>::type;
        
        WideType wide_value = static_cast<WideType>(value_) << FracBits;
        return fixed_point(static_cast<IntType>(wide_value / other.value_));
    }
    
    // Assignment operators
    fixed_point& operator+=(const fixed_point& other) {
        value_ += other.value_;
        return *this;
    }
    
    fixed_point& operator-=(const fixed_point& other) {
        value_ -= other.value_;
        return *this;
    }
    
    fixed_point& operator*=(const fixed_point& other) {
        *this = *this * other;
        return *this;
    }
    
    fixed_point& operator/=(const fixed_point& other) {
        *this = *this / other;
        return *this;
    }
    
    // Comparison operators
    bool operator==(const fixed_point& other) const { return value_ == other.value_; }
    bool operator!=(const fixed_point& other) const { return value_ != other.value_; }
    bool operator<(const fixed_point& other) const { return value_ < other.value_; }
    bool operator<=(const fixed_point& other) const { return value_ <= other.value_; }
    bool operator>(const fixed_point& other) const { return value_ > other.value_; }
    bool operator>=(const fixed_point& other) const { return value_ >= other.value_; }
    
    // Unary operators
    fixed_point operator-() const { return fixed_point(-value_); }
    fixed_point operator+() const { return *this; }
    
    // String representation
    std::string to_string() const {
        return std::to_string(to_double());
    }

private:
    IntType value_;
};

// Common fixed-point types for quantum computations
using Q15 = fixed_point<int32_t, 15>;  // 16.15 format, range ±65536
using Q31 = fixed_point<int64_t, 31>;  // 32.31 format, range ±2^32

// Mathematical functions for fixed-point
template<typename IntType, int FracBits>
fixed_point<IntType, FracBits> abs(const fixed_point<IntType, FracBits>& x) {
    return x.raw() < 0 ? -x : x;
}

template<typename IntType, int FracBits>
fixed_point<IntType, FracBits> sqrt(const fixed_point<IntType, FracBits>& x) {
    if (x.raw() <= 0) return fixed_point<IntType, FracBits>::from_raw(0);
    
    // Newton-Raphson iteration for fixed-point square root
    auto guess = fixed_point<IntType, FracBits>::from_raw(x.raw() >> 1);
    auto half = fixed_point<IntType, FracBits>::from_float(0.5);
    
    for (int i = 0; i < 10; ++i) {  // Fixed iterations for determinism
        auto new_guess = half * (guess + x / guess);
        if (abs(new_guess - guess).raw() < 2) break;  // Converged
        guess = new_guess;
    }
    
    return guess;
}

} // namespace ohmy