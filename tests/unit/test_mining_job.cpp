/*
 * Unit tests for mining job parser
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <ohmy/pool/mining_job.hpp>
#include <nlohmann/json.hpp>

using namespace ohmy::pool;
using json = nlohmann::json;

TEST_SUITE("Mining Job Parser") {
    TEST_CASE("parse_mining_notify - valid complete job") {
        // Realistic mining.notify params from actual pool
        std::vector<json> params = {
            "job123",                                                    // job_id
            "0177921...prev_hash_hex...",                               // prevhash
            "01000000...coinbase1_hex...",                              // coinbase1
            "0d2f6e6f6465...coinbase2_hex...",                         // coinbase2
            json::array({"merkle1_hex", "merkle2_hex"}),               // merkle_branch
            "20000004",                                                  // version
            "1a0ccaa1",                                                  // nbits
            "69098232",                                                  // ntime
            false                                                        // clean_jobs
        };
        
        auto job_opt = parse_mining_notify(params);
        REQUIRE(job_opt.has_value());
        
        const auto& job = *job_opt;
        CHECK(job.job_id == "job123");
        CHECK(job.prev_hash == "0177921...prev_hash_hex...");
        CHECK(job.coinbase1 == "01000000...coinbase1_hex...");
        CHECK(job.coinbase2 == "0d2f6e6f6465...coinbase2_hex...");
        CHECK(job.merkle_branch.size() == 2);
        CHECK(job.merkle_branch[0] == "merkle1_hex");
        CHECK(job.merkle_branch[1] == "merkle2_hex");
        CHECK(job.version == "20000004");
        CHECK(job.nbits == "1a0ccaa1");
        CHECK(job.ntime == "69098232");
        CHECK(job.clean_jobs == false);
    }
    
    TEST_CASE("parse_mining_notify - clean_jobs true") {
        std::vector<json> params = {
            "new_job", "prev", "cb1", "cb2", json::array(),
            "ver", "bits", "time", true
        };
        
        auto job_opt = parse_mining_notify(params);
        REQUIRE(job_opt.has_value());
        CHECK(job_opt->clean_jobs == true);
    }
    
    TEST_CASE("parse_mining_notify - empty merkle branch") {
        std::vector<json> params = {
            "job1", "prev", "cb1", "cb2", 
            json::array(),  // Empty merkle branch
            "ver", "bits", "time", false
        };
        
        auto job_opt = parse_mining_notify(params);
        REQUIRE(job_opt.has_value());
        CHECK(job_opt->merkle_branch.empty());
    }
    
    TEST_CASE("parse_mining_notify - multiple merkle branches") {
        std::vector<json> params = {
            "job_multi", "prev", "cb1", "cb2",
            json::array({"m1", "m2", "m3", "m4"}),
            "ver", "bits", "time", false
        };
        
        auto job_opt = parse_mining_notify(params);
        REQUIRE(job_opt.has_value());
        REQUIRE(job_opt->merkle_branch.size() == 4);
        CHECK(job_opt->merkle_branch[0] == "m1");
        CHECK(job_opt->merkle_branch[3] == "m4");
    }
    
    TEST_CASE("parse_mining_notify - insufficient params") {
        // Only 8 params (need 9)
        std::vector<json> params = {
            "job", "prev", "cb1", "cb2", json::array(),
            "ver", "bits", "time"
        };
        
        auto job_opt = parse_mining_notify(params);
        CHECK_FALSE(job_opt.has_value());
    }
    
    TEST_CASE("parse_mining_notify - empty params") {
        std::vector<json> params;
        
        auto job_opt = parse_mining_notify(params);
        CHECK_FALSE(job_opt.has_value());
    }
    
    TEST_CASE("parse_mining_notify - wrong type in params") {
        std::vector<json> params = {
            "job", "prev", "cb1", "cb2", json::array(),
            123,  // version should be string
            "bits", "time", false
        };
        
        // Parser should handle type errors gracefully
        auto job_opt = parse_mining_notify(params);
        // May return nullopt or throw, depends on implementation
        // Current implementation uses try-catch, so should return nullopt
        CHECK_FALSE(job_opt.has_value());
    }
    
    TEST_CASE("parse_mining_notify - merkle_branch not array") {
        std::vector<json> params = {
            "job", "prev", "cb1", "cb2",
            "not_an_array",  // Should be array
            "ver", "bits", "time", false
        };
        
        auto job_opt = parse_mining_notify(params);
        REQUIRE(job_opt.has_value());
        // If merkle_branch is not array, it should be empty
        CHECK(job_opt->merkle_branch.empty());
    }
    
    TEST_CASE("parse_mining_notify - realistic pool message") {
        // Example from actual qubitcoin pool
        std::vector<json> params = {
            "480f",
            "01779216da70df67be46143a8dd959a4ab209c367e2a38350000006b00000000",
            "010000000100000000000000000000000000000000000000000000000000000000000000"
            "00ffffffff20031df400043282096908",
            "0d2f6e6f64655374726174756d2f00000000030000000000000000266a24aa21a9ed"
            "2671306d68834ddb449e4833143d962c331fd5bcdcd9de9275384d630e8267c67612"
            "0b270100000016001496a15ea9fb6cc5d0d327e5fa88bb4183170f47ceabf0fa0200"
            "000000160014b2d089f1db1205d98baa5b8d9bdeda22813e6fe000000000",
            json::array({
                "1d365c04f2d04ebcd35fbc8da6300cd131a623aa8cdb7dbb6f8a72429f5407ed",
                "49952090aac37b3d8449f15cd7baf0df26ae787bc21ebb5dbc9bfb8a3262aec0"
            }),
            "20000004",
            "1a0ccaa1",
            "69098232",
            false
        };
        
        auto job_opt = parse_mining_notify(params);
        REQUIRE(job_opt.has_value());
        
        CHECK(job_opt->job_id == "480f");
        CHECK(job_opt->version == "20000004");
        CHECK(job_opt->nbits == "1a0ccaa1");
        CHECK(job_opt->ntime == "69098232");
        CHECK(job_opt->merkle_branch.size() == 2);
        CHECK_FALSE(job_opt->clean_jobs);
    }
}
