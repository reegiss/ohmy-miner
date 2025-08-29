/*
 * Copyright (c) 2025 Regis Araujo Melo
 * (License header as above)
 */
#include "gtest/gtest.h"
#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h> // <-- ADD THIS LINE

// Forward declare the C function we are testing
extern "C" {
#include "qhash_miner.h"
}

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
    // This method is called before each test.
    void SetUp() override {
        // First, explicitly set the CUDA device for this thread's context.
        // This is crucial in a multi-GPU system.
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device to 0. Error: " << cudaGetErrorString(err);
        
        // Now, initialize the CUDA/qhash resources for this thread
        ASSERT_TRUE(qhash_thread_init(0));
    }

    // This method is called after each test.
    void TearDown() override {
        // Clean up resources
        qhash_thread_destroy();
    }
};

TEST_F(QHashAlgorithmTest, RegressionVector) {
    // This is a test vector generated from the current implementation.
    // If the hashing logic changes, this test will fail, indicating a regression.
    
    // Input: A simplified 80-byte block header (based on Bitcoin genesis block)
    const uint8_t input_header[80] = {
        0x01, 0x00, 0x00, 0x00, // version
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // prev_hash (null)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2, 0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61, // merkle_root
        0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32, 0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
        0x29, 0xab, 0x5f, 0x49, // ntime (1231006505)
        0xff, 0xff, 0x00, 0x1d, // nbits
        0x4c, 0x86, 0x04, 0x19  // nonce (420042060)
    };

    // Expected Output Hash (pre-calculated with this exact code)
    const std::string expected_hash_str = "4122d1059b025341203b9543e0c053c89365e4b6c38221c0e6a815a5f1e16f7f";

    uint8_t actual_hash[32];
    qhash_hash(actual_hash, input_header, 0);

    EXPECT_EQ(format_hash(actual_hash), expected_hash_str);
}