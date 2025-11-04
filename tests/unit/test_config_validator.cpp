/*
 * Unit tests for config validation (hostname, host:port)
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <ohmy/config/validator.hpp>

using namespace ohmy::config;

TEST_SUITE("Config Validator") {
    TEST_CASE("is_valid_hostname - valid hostnames") {
        std::string err;
        
        CHECK(is_valid_hostname("localhost", err));
        CHECK(is_valid_hostname("example.com", err));
        CHECK(is_valid_hostname("sub.example.com", err));
        CHECK(is_valid_hostname("a", err));
        CHECK(is_valid_hostname("a1-b2.c3-d4.example.org", err));
        CHECK(is_valid_hostname("192.168.1.1", err)); // IP-like is valid hostname
    }
    
    TEST_CASE("is_valid_hostname - invalid hostnames") {
        std::string err;
        
        CHECK_FALSE(is_valid_hostname("", err));
        CHECK(err.find("vazio") != std::string::npos);
        
        CHECK_FALSE(is_valid_hostname("-start.com", err));
        bool has_start_or_end = (err.find("começar") != std::string::npos) || (err.find("terminar") != std::string::npos);
        CHECK(has_start_or_end);
        
        CHECK_FALSE(is_valid_hostname("end-.com", err));
        
        CHECK_FALSE(is_valid_hostname("invalid_char.com", err));
        CHECK(err.find("inválidos") != std::string::npos);
        
        CHECK_FALSE(is_valid_hostname(".example.com", err));
        CHECK(err.find("vazia") != std::string::npos);
        
        CHECK_FALSE(is_valid_hostname("example..com", err));
        
        // Hostname > 253 chars
        std::string long_host(254, 'a');
        CHECK_FALSE(is_valid_hostname(long_host, err));
        CHECK(err.find("longo") != std::string::npos);
        
        // Label > 63 chars
        std::string long_label(64, 'a');
        long_label += ".com";
        CHECK_FALSE(is_valid_hostname(long_label, err));
    }
    
    TEST_CASE("validate_host_port - valid URLs") {
        std::string err;
        
        CHECK(validate_host_port("example.com:8080", err));
        CHECK(validate_host_port("localhost:3000", err));
        CHECK(validate_host_port("sub.domain.com:443", err));
        CHECK(validate_host_port("192.168.1.1:80", err));
        CHECK(validate_host_port("pool.example.org:8610", err));
    }
    
    TEST_CASE("validate_host_port - invalid URLs") {
        std::string err;
        
        // Missing port
        CHECK_FALSE(validate_host_port("example.com", err));
        CHECK(err.find("host:port") != std::string::npos);
        
        // Empty port
        CHECK_FALSE(validate_host_port("example.com:", err));
        
        // Non-numeric port
        CHECK_FALSE(validate_host_port("example.com:abc", err));
        CHECK(err.find("dígitos") != std::string::npos);
        
        // Port out of range
        CHECK_FALSE(validate_host_port("example.com:0", err));
        CHECK(err.find("intervalo") != std::string::npos);
        
        CHECK_FALSE(validate_host_port("example.com:70000", err));
        CHECK(err.find("intervalo") != std::string::npos);
        
        // Invalid hostname
        CHECK_FALSE(validate_host_port("invalid_host:8080", err));
    }
    
    TEST_CASE("validate_host_port - edge cases") {
        std::string err;
        
        CHECK(validate_host_port("a:1", err));
        CHECK(validate_host_port("example.com:65535", err));
        CHECK_FALSE(validate_host_port("example.com:65536", err));
    }
}
