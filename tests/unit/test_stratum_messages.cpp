/*
 * Unit tests for Stratum message builders
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <ohmy/pool/stratum_messages.hpp>
#include <nlohmann/json.hpp>

using namespace ohmy::pool::stratum_messages;
using json = nlohmann::json;

TEST_SUITE("Stratum Messages") {
    TEST_CASE("build_subscribe - basic structure") {
        std::string msg = build_subscribe("test-client", 1);
        
        // Should end with newline
        REQUIRE_FALSE(msg.empty());
        CHECK(msg.back() == '\n');
        
        // Parse as JSON
        json j = json::parse(msg);
        
        // Check structure
        REQUIRE(j.contains("id"));
        REQUIRE(j.contains("method"));
        REQUIRE(j.contains("params"));
        
        CHECK(j["id"].get<int>() == 1);
        CHECK(j["method"].get<std::string>() == "mining.subscribe");
        
        // Params should be array with client name
        REQUIRE(j["params"].is_array());
        REQUIRE(j["params"].size() >= 1);
        CHECK(j["params"][0].get<std::string>() == "test-client");
    }
    
    TEST_CASE("build_subscribe - different IDs") {
        auto msg1 = build_subscribe("client", 1);
        auto msg2 = build_subscribe("client", 99);
        
        json j1 = json::parse(msg1);
        json j2 = json::parse(msg2);
        
        CHECK(j1["id"].get<int>() == 1);
        CHECK(j2["id"].get<int>() == 99);
    }
    
    TEST_CASE("build_subscribe - different clients") {
        auto msg1 = build_subscribe("ohmy-miner/0.1", 1);
        auto msg2 = build_subscribe("another-miner", 1);
        
        json j1 = json::parse(msg1);
        json j2 = json::parse(msg2);
        
        CHECK(j1["params"][0].get<std::string>() == "ohmy-miner/0.1");
        CHECK(j2["params"][0].get<std::string>() == "another-miner");
    }
    
    TEST_CASE("build_authorize - basic structure") {
        std::string msg = build_authorize("user.worker", "password", 2);
        
        // Should end with newline
        REQUIRE_FALSE(msg.empty());
        CHECK(msg.back() == '\n');
        
        // Parse as JSON
        json j = json::parse(msg);
        
        // Check structure
        REQUIRE(j.contains("id"));
        REQUIRE(j.contains("method"));
        REQUIRE(j.contains("params"));
        
        CHECK(j["id"].get<int>() == 2);
        CHECK(j["method"].get<std::string>() == "mining.authorize");
        
        // Params should be array with [user, pass]
        REQUIRE(j["params"].is_array());
        REQUIRE(j["params"].size() == 2);
        CHECK(j["params"][0].get<std::string>() == "user.worker");
        CHECK(j["params"][1].get<std::string>() == "password");
    }
    
    TEST_CASE("build_authorize - different credentials") {
        auto msg1 = build_authorize("bc1q...wallet.RIG1", "x", 2);
        auto msg2 = build_authorize("another_user", "secret", 5);
        
        json j1 = json::parse(msg1);
        json j2 = json::parse(msg2);
        
        CHECK(j1["id"].get<int>() == 2);
        CHECK(j1["params"][0].get<std::string>() == "bc1q...wallet.RIG1");
        CHECK(j1["params"][1].get<std::string>() == "x");
        
        CHECK(j2["id"].get<int>() == 5);
        CHECK(j2["params"][0].get<std::string>() == "another_user");
        CHECK(j2["params"][1].get<std::string>() == "secret");
    }
    
    TEST_CASE("Messages are valid JSON-RPC") {
        auto subscribe = build_subscribe("client", 1);
        auto authorize = build_authorize("user", "pass", 2);
        
        // Should not throw when parsing
        CHECK_NOTHROW(json::parse(subscribe));
        CHECK_NOTHROW(json::parse(authorize));
        
        // Verify JSON-RPC version implied (no version field = 1.0)
        json j_sub = json::parse(subscribe);
        json j_auth = json::parse(authorize);
        
        // Should have method string
        CHECK(j_sub["method"].is_string());
        CHECK(j_auth["method"].is_string());
        
        // Should have integer id
        CHECK(j_sub["id"].is_number_integer());
        CHECK(j_auth["id"].is_number_integer());
    }
}
