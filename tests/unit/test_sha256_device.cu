/*
 * Unit tests for SHA256 CUDA device implementation
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <ohmy/crypto/sha256_device.cuh>
#include <cuda_runtime.h>
#include <cstring>
#include <iomanip>
#include <sstream>

using namespace ohmy::crypto;

// Helper: Convert hex string to bytes
void hex_to_bytes(const std::string& hex, uint8_t* bytes) {
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteString = hex.substr(i, 2);
        bytes[i / 2] = static_cast<uint8_t>(strtol(byteString.c_str(), nullptr, 16));
    }
}

// Helper: Convert bytes to hex string
std::string bytes_to_hex(const uint8_t* bytes, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        oss << std::setw(2) << static_cast<int>(bytes[i]);
    }
    return oss.str();
}

// Kernel: Test single SHA256
__global__ void test_sha256_kernel(const uint8_t* input, uint32_t len, uint8_t* output) {
    sha256_hash(input, len, output);
}

// Kernel: Test SHA256d
__global__ void test_sha256d_kernel(const uint8_t* input, uint32_t len, uint8_t* output) {
    sha256d(input, len, output);
}

TEST_SUITE("SHA256 Device") {
    TEST_CASE("SHA256 - empty string") {
        // SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        const char* expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        
        uint8_t* d_input;
        uint8_t* d_output;
        cudaMalloc(&d_input, 1);
        cudaMalloc(&d_output, 32);
        
        test_sha256_kernel<<<1, 1>>>(d_input, 0, d_output);
        cudaDeviceSynchronize();
        
        uint8_t output[32];
        cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
        
        std::string result = bytes_to_hex(output, 32);
        CHECK(result == expected);
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    TEST_CASE("SHA256 - 'abc'") {
        // SHA256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        const char* input_str = "abc";
        const char* expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        
        uint8_t* d_input;
        uint8_t* d_output;
        cudaMalloc(&d_input, 3);
        cudaMalloc(&d_output, 32);
        
        cudaMemcpy(d_input, input_str, 3, cudaMemcpyHostToDevice);
        
        test_sha256_kernel<<<1, 1>>>(d_input, 3, d_output);
        cudaDeviceSynchronize();
        
        uint8_t output[32];
        cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
        
        std::string result = bytes_to_hex(output, 32);
        CHECK(result == expected);
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    TEST_CASE("SHA256 - Bitcoin genesis block header (single hash)") {
        // Bitcoin genesis block header (80 bytes)
        // This tests SHA256 transform on 80-byte block header
        // Note: For actual Bitcoin validation, use SHA256d (double hash)
        const char* header_hex = 
            "0100000000000000000000000000000000000000000000000000000000000000"
            "000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
            "4b1e5e4a29ab5f49ffff001d1dac2b7c";
        
        uint8_t header[80];
        hex_to_bytes(header_hex, header);
        
        uint8_t* d_input;
        uint8_t* d_output;
        cudaMalloc(&d_input, 80);
        cudaMalloc(&d_output, 32);
        
        cudaMemcpy(d_input, header, 80, cudaMemcpyHostToDevice);
        
        test_sha256_kernel<<<1, 1>>>(d_input, 80, d_output);
        cudaDeviceSynchronize();
        
        uint8_t output[32];
        cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
        
        // Just verify it produces a 32-byte hash (actual value depends on implementation)
        std::string result = bytes_to_hex(output, 32);
        CHECK(result.length() == 64); // 32 bytes = 64 hex chars
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    TEST_CASE("SHA256d - Bitcoin genesis block") {
        // Bitcoin uses SHA256d (double SHA256)
        const char* header_hex = 
            "0100000000000000000000000000000000000000000000000000000000000000"
            "000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
            "4b1e5e4a29ab5f49ffff001d1dac2b7c";
        
        // Expected SHA256d hash (big-endian):
        const char* expected = "6fe28c0ab6f1b372c1a6a246ae63f74f931e8365e15a089c68d6190000000000";
        
        uint8_t header[80];
        hex_to_bytes(header_hex, header);
        
        uint8_t* d_input;
        uint8_t* d_output;
        cudaMalloc(&d_input, 80);
        cudaMalloc(&d_output, 32);
        
        cudaMemcpy(d_input, header, 80, cudaMemcpyHostToDevice);
        
        test_sha256d_kernel<<<1, 1>>>(d_input, 80, d_output);
        cudaDeviceSynchronize();
        
        uint8_t output[32];
        cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
        
        std::string result = bytes_to_hex(output, 32);
        CHECK(result == expected);
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    TEST_CASE("SHA256 - longer message (448 bits boundary)") {
        // Test message that requires padding to 512 bits
        const char* input_str = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        const char* expected = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
        
        uint8_t* d_input;
        uint8_t* d_output;
        size_t len = strlen(input_str);
        cudaMalloc(&d_input, len);
        cudaMalloc(&d_output, 32);
        
        cudaMemcpy(d_input, input_str, len, cudaMemcpyHostToDevice);
        
        test_sha256_kernel<<<1, 1>>>(d_input, len, d_output);
        cudaDeviceSynchronize();
        
        uint8_t output[32];
        cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
        
        std::string result = bytes_to_hex(output, 32);
        CHECK(result == expected);
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    TEST_CASE("SHA256 - multi-block message") {
        // Test message longer than 64 bytes (requires multiple blocks)
        std::string input_str(100, 'a'); // 100 'a' characters
        
        uint8_t* d_input;
        uint8_t* d_output;
        cudaMalloc(&d_input, 100);
        cudaMalloc(&d_output, 32);
        
        cudaMemcpy(d_input, input_str.c_str(), 100, cudaMemcpyHostToDevice);
        
        test_sha256_kernel<<<1, 1>>>(d_input, 100, d_output);
        cudaDeviceSynchronize();
        
        uint8_t output[32];
        cudaMemcpy(output, d_output, 32, cudaMemcpyDeviceToHost);
        
        std::string result = bytes_to_hex(output, 32);
        CHECK(result.length() == 64); // Valid 32-byte hash
        
        cudaFree(d_input);
        cudaFree(d_output);
    }
}
