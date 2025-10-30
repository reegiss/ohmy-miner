/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/pool/messages.hpp"
#include <nlohmann/json.hpp>
#include <cassert>
#include <iostream>
#include <vector>

using namespace ohmy::pool;
using json = nlohmann::json;

void test_subscribe_request() {
    std::cout << "Testing subscribe request..." << std::endl;
    
    std::string request = StratumMessages::subscribe_request(1, "OhMyMiner/0.1.0");
    json j = json::parse(request);
    
    assert(j["jsonrpc"] == "2.0");
    assert(j["id"] == 1);
    assert(j["method"] == "mining.subscribe");
    assert(j["params"].is_array());
    assert(j["params"].size() == 1);
    assert(j["params"][0] == "OhMyMiner/0.1.0");
    
    std::cout << "  ✓ Subscribe request test passed" << std::endl;
}

void test_authorize_request() {
    std::cout << "Testing authorize request..." << std::endl;
    
    std::string request = StratumMessages::authorize_request(2, "worker1", "password");
    json j = json::parse(request);
    
    assert(j["jsonrpc"] == "2.0");
    assert(j["id"] == 2);
    assert(j["method"] == "mining.authorize");
    assert(j["params"].is_array());
    assert(j["params"].size() == 2);
    assert(j["params"][0] == "worker1");
    assert(j["params"][1] == "password");
    
    std::cout << "  ✓ Authorize request test passed" << std::endl;
}

void test_submit_request() {
    std::cout << "Testing submit request..." << std::endl;
    
    std::string request = StratumMessages::submit_request(
        3, "worker1", "job123", "extranonce2", "ntime", "nonce");
    json j = json::parse(request);
    
    assert(j["jsonrpc"] == "2.0");
    assert(j["id"] == 3);
    assert(j["method"] == "mining.submit");
    assert(j["params"].is_array());
    assert(j["params"].size() == 5);
    assert(j["params"][0] == "worker1");
    assert(j["params"][1] == "job123");
    assert(j["params"][2] == "extranonce2");
    assert(j["params"][3] == "ntime");
    assert(j["params"][4] == "nonce");
    
    std::cout << "  ✓ Submit request test passed" << std::endl;
}

void test_parse_response_success() {
    std::cout << "Testing parse response (success)..." << std::endl;
    
    std::string response = R"({"jsonrpc":"2.0","id":1,"result":true,"error":null})";
    
    int id;
    json result;
    StratumErrorInfo error;
    
    [[maybe_unused]] bool success = StratumMessages::parse_response(response, id, result, error);
    
    assert(success);
    assert(id == 1);
    assert(result == true);
    assert(error.code == 0);
    
    std::cout << "  ✓ Parse success response test passed" << std::endl;
}

void test_parse_response_error() {
    std::cout << "Testing parse response (error)..." << std::endl;
    
    std::string response = R"({"jsonrpc":"2.0","id":2,"result":null,"error":[20,"Invalid nonce",""]})";
    
    int id;
    json result;
    StratumErrorInfo error;
    
    [[maybe_unused]] bool success = StratumMessages::parse_response(response, id, result, error);
    
    assert(!success);
    assert(id == 2);
    assert(error.code == 20);
    assert(error.message == "Invalid nonce");
    
    std::cout << "  ✓ Parse error response test passed" << std::endl;
}

void test_parse_notify() {
    std::cout << "Testing parse notify..." << std::endl;
    
    std::string notify = R"({
        "jsonrpc": "2.0",
        "method": "mining.notify",
        "params": [
            "job1",
            "prevhash",
            "coinbase1",
            "coinbase2",
            ["merkle1", "merkle2"],
            "version",
            "nbits",
            "ntime",
            true
        ]
    })";
    
    std::string job_id;
    std::string prev_hash;
    std::string coinbase1;
    std::string coinbase2;
    std::vector<std::string> merkle_branch;
    std::string version;
    std::string nbits;
    std::string ntime;
    bool clean_jobs;
    
    [[maybe_unused]] bool success = StratumMessages::parse_notify(
        notify, job_id, prev_hash, coinbase1, coinbase2,
        merkle_branch, version, nbits, ntime, clean_jobs);
    
    assert(success);
    assert(job_id == "job1");
    assert(prev_hash == "prevhash");
    assert(coinbase1 == "coinbase1");
    assert(coinbase2 == "coinbase2");
    assert(merkle_branch.size() == 2);
    assert(merkle_branch[0] == "merkle1");
    assert(merkle_branch[1] == "merkle2");
    assert(version == "version");
    assert(nbits == "nbits");
    assert(ntime == "ntime");
    assert(clean_jobs == true);
    
    std::cout << "  ✓ Parse notify test passed" << std::endl;
}

void test_parse_set_difficulty() {
    std::cout << "Testing parse set_difficulty..." << std::endl;
    
    std::string set_diff = R"({
        "jsonrpc": "2.0",
        "method": "mining.set_difficulty",
        "params": [16.0]
    })";
    
    double difficulty;
    [[maybe_unused]] bool success = StratumMessages::parse_set_difficulty(set_diff, difficulty);
    
    assert(success);
    assert(std::abs(difficulty - 16.0) < 0.001);
    
    std::cout << "  ✓ Parse set_difficulty test passed" << std::endl;
}

void test_json_format() {
    std::cout << "Testing JSON format compliance..." << std::endl;
    
    std::string request = StratumMessages::subscribe_request(1, "test");
    
    // Should be valid JSON
    json j = json::parse(request);
    
    // Should have required fields
    assert(j.contains("jsonrpc"));
    assert(j.contains("id"));
    assert(j.contains("method"));
    assert(j.contains("params"));
    
    std::cout << "  ✓ JSON format test passed" << std::endl;
}

void test_message_determinism() {
    std::cout << "Testing message determinism..." << std::endl;
    
    std::string msg1 = StratumMessages::subscribe_request(1, "test");
    std::string msg2 = StratumMessages::subscribe_request(1, "test");
    
    assert(msg1 == msg2);
    
    std::cout << "  ✓ Message determinism test passed" << std::endl;
}

int main() {
    std::cout << "\n=== Stratum Message Protocol Tests ===" << std::endl;
    
    try {
        test_subscribe_request();
        test_authorize_request();
        test_submit_request();
        test_parse_response_success();
        test_parse_response_error();
        test_parse_notify();
        test_parse_set_difficulty();
        test_json_format();
        test_message_determinism();
        
        std::cout << "\n✅ All Stratum message tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
