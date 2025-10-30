/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/crypto/sha256d.hpp"
#include <openssl/evp.h>
#include <stdexcept>

namespace ohmy {
namespace crypto {

std::vector<uint8_t> sha256d(const std::vector<uint8_t>& data) {
    // First SHA256
    std::vector<uint8_t> first_hash(32);
    unsigned int len = 32;
    
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }
    
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to initialize SHA256");
    }
    
    if (EVP_DigestUpdate(ctx, data.data(), data.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to update SHA256");
    }
    
    if (EVP_DigestFinal_ex(ctx, first_hash.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize first SHA256");
    }
    
    // Second SHA256
    std::vector<uint8_t> second_hash(32);
    len = 32;
    
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to initialize second SHA256");
    }
    
    if (EVP_DigestUpdate(ctx, first_hash.data(), first_hash.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to update second SHA256");
    }
    
    if (EVP_DigestFinal_ex(ctx, second_hash.data(), &len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize second SHA256");
    }
    
    EVP_MD_CTX_free(ctx);
    return second_hash;
}

std::vector<uint8_t> sha256d(const std::string& data) {
    std::vector<uint8_t> bytes(data.begin(), data.end());
    return sha256d(bytes);
}

} // namespace crypto
} // namespace ohmy
