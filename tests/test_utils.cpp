/*
 * Copyright (c) 2025 Regis Araujo Melo
 * (License header as above)
 */
#include "gtest/gtest.h"
#include "miner_bridge.h" // To access hex_to_bytes
#include <vector>
#include <string>

// Test fixture for MinerBridge utility functions
class MinerBridgeUtilsTest : public ::testing::Test {};

TEST_F(MinerBridgeUtilsTest, HexToBytesConversion) {
    // Test case 1: Basic conversion
    std::string hex1 = "deadbeef";
    std::vector<uint8_t> expected1 = {0xde, 0xad, 0xbe, 0xef};
    EXPECT_EQ(MinerBridge::hex_to_bytes(hex1), expected1);

    // Test case 2: Empty string
    std::string hex2 = "";
    std::vector<uint8_t> expected2 = {};
    EXPECT_EQ(MinerBridge::hex_to_bytes(hex2), expected2);

    // Test case 3: String with mixed case
    std::string hex3 = "1A2b3c4D";
    std::vector<uint8_t> expected3 = {0x1a, 0x2b, 0x3c, 0x4d};
    EXPECT_EQ(MinerBridge::hex_to_bytes(hex3), expected3);
}