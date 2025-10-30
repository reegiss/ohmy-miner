/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/pool/messages.hpp"
#include <vector>

namespace ohmy {
namespace pool {

using json = nlohmann::json;

// Helper functions for tests

std::string StratumMessages::subscribe_request(int id, const std::string& user_agent) {
    json j;
    j["jsonrpc"] = "2.0";
    j["id"] = id;
    j["method"] = "mining.subscribe";
    j["params"] = json::array({user_agent});
    return j.dump();
}

std::string StratumMessages::authorize_request(int id, const std::string& worker, const std::string& password) {
    json j;
    j["jsonrpc"] = "2.0";
    j["id"] = id;
    j["method"] = "mining.authorize";
    j["params"] = json::array({worker, password});
    return j.dump();
}

std::string StratumMessages::submit_request(int id, const std::string& worker,
                                           const std::string& job_id,
                                           const std::string& extranonce2,
                                           const std::string& ntime,
                                           const std::string& nonce) {
    json j;
    j["jsonrpc"] = "2.0";
    j["id"] = id;
    j["method"] = "mining.submit";
    j["params"] = json::array({worker, job_id, extranonce2, ntime, nonce});
    return j.dump();
}

bool StratumMessages::parse_response(const std::string& response, int& id, 
                                    json& result, StratumErrorInfo& error) {
    try {
        json j = json::parse(response);
        
        if (!j.contains("id")) return false;
        id = j["id"];
        
        if (j.contains("error") && !j["error"].is_null()) {
            // Parse error array: [code, message, traceback]
            if (j["error"].is_array() && j["error"].size() >= 2) {
                error.code = j["error"][0];
                error.message = j["error"][1];
                if (j["error"].size() > 2) {
                    error.traceback = j["error"][2];
                }
            }
            return false;
        }
        
        if (j.contains("result")) {
            result = j["result"];
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

bool StratumMessages::parse_notify(const std::string& notification,
                                  std::string& job_id,
                                  std::string& prev_hash,
                                  std::string& coinbase1,
                                  std::string& coinbase2,
                                  std::vector<std::string>& merkle_branch,
                                  std::string& version,
                                  std::string& nbits,
                                  std::string& ntime,
                                  bool& clean_jobs) {
    try {
        json j = json::parse(notification);
        
        if (!j.contains("method") || j["method"] != "mining.notify") {
            return false;
        }
        
        if (!j.contains("params") || !j["params"].is_array() || j["params"].size() < 9) {
            return false;
        }
        
        auto params = j["params"];
        job_id = params[0];
        prev_hash = params[1];
        coinbase1 = params[2];
        coinbase2 = params[3];
        
        // Parse merkle branch array
        merkle_branch.clear();
        if (params[4].is_array()) {
            for (const auto& branch : params[4]) {
                merkle_branch.push_back(branch);
            }
        }
        
        version = params[5];
        nbits = params[6];
        ntime = params[7];
        clean_jobs = params[8];
        
        return true;
    } catch (...) {
        return false;
    }
}

bool StratumMessages::parse_set_difficulty(const std::string& notification, double& difficulty) {
    try {
        json j = json::parse(notification);
        
        if (!j.contains("method") || j["method"] != "mining.set_difficulty") {
            return false;
        }
        
        if (!j.contains("params") || !j["params"].is_array() || j["params"].size() < 1) {
            return false;
        }
        
        difficulty = j["params"][0];
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace pool
} // namespace ohmy
