/*
 * Copyright (c) 2025 Regis Araujo Melo
 * (License header as above)
 */
#include "gtest/gtest.h"
#include "qhash_algorithm.h" // Test the concrete algorithm class
#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>

// Helper class to access private members for testing
class QHashAlgorithmTester : public QHashAlgorithm {
public:
    // Expose the hex_to_bytes method for testing
    std::vector<uint8_t> test_hex_to_bytes(const std::string& hex) {
        return hex_to_bytes(hex);
    }
};

// Helper function to format byte arrays for easy comparison
static std::string format_hash(const uint8_t* hash) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) {
        ss << std::setw(2) << static_cast<int>(hash[i]);
    }
    return ss.str();
}

// Test fixture for qhash algorithm
class QHashAlgorithmTest : public ::testing::Test {
protected:
    std::unique_ptr<QHashAlgorithmTester> algo;

    void SetUp() override {
        algo = std::make_unique<QHashAlgorithmTester>();
        
        // Set the CUDA device context before initializing the algorithm
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device to 0. Error: " << cudaGetErrorString(err);
        
        ASSERT_TRUE(algo->thread_init(0));
    }

    void TearDown() override {
        algo->thread_destroy();
    }
};

TEST_F(QHashAlgorithmTest, HexToBytesConversion) {
    std::string hex1 = "deadbeef";
    std::vector<uint8_t> expected1 = {0xde, 0xad, 0xbe, 0xef};
    EXPECT_EQ(algo->test_hex_to_bytes(hex1), expected1);

    std::string hex2 = "";
    std::vector<uint8_t> expected2 = {};
    EXPECT_EQ(algo->test_hex_to_bytes(hex2), expected2);

    std::string hex3 = "1A2b3c4D";
    std::vector<uint8_t> expected3 = {0x1a, 0x2b, 0x3c, 0x4d};
    EXPECT_EQ(algo->test_hex_to_bytes(hex3), expected3);
}

TEST_F(QHashAlgorithmTest, RegressionVector) {
    const uint8_t input_header[80] = {
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3b, 0xa3, 0xed, 0xfd,
        0x7a, 0x7b, 0x12, 0xb2, 0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
        0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32, 0x3a, 0x9f, 0xb8, 0xaa,
        0x4b, 0x1e, 0x5e, 0x4a, 0x29, 0xab, 0x5f, 0x49, 0xff, 0xff, 0x00, 0x1d,
        0x4c, 0x86, 0x04, 0x19
    };
    const std::string expected_hash_str = "4122d1059b025341203b9543e0c053c89365e4b6c38221c0e6a815a5f1e16f7f";

    uint8_t actual_hash[32];
    qhash_hash(actual_hash, input_header, 0);

    EXPECT_EQ(format_hash(actual_hash), expected_hash_str);
}