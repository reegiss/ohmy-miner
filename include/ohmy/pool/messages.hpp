/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ohmy {
namespace pool {

/**
 * Stratum error structure for parsing
 */
struct StratumErrorInfo {
    int code = 0;
    std::string message;
    std::string traceback;
};

/**
 * Stratum message types and JSON-RPC helper functions
 */
class StratumMessages {
public:
    using json = nlohmann::json;

    // Client -> Server methods
    static json subscribe(uint64_t id) {
        return make_request(id, "mining.subscribe", json::array());
    }

    static json authorize(uint64_t id, const std::string& worker, const std::string& password) {
        return make_request(id, "mining.authorize", json::array({worker, password}));
    }

    static json submit(uint64_t id, const std::string& worker, const std::string& job_id,
                      const std::string& extranonce2, const std::string& ntime,
                      const std::string& nonce) {
        return make_request(id, "mining.submit",
            json::array({worker, job_id, extranonce2, ntime, nonce}));
    }

    // Server -> Client method parsing
    static bool is_notification(const json& msg) {
        return msg.contains("method") && 
               (!msg.contains("id") || msg["id"].is_null());
    }

    static bool is_response(const json& msg) {
        return msg.contains("id") && !msg["id"].is_null() && !msg.contains("method");
    }

    static bool is_error(const json& msg) {
        return msg.contains("error") && msg["error"] != nullptr;
    }

    // Test helper methods (string-based)
    static std::string subscribe_request(int id, const std::string& user_agent);
    static std::string authorize_request(int id, const std::string& worker, const std::string& password);
    static std::string submit_request(int id, const std::string& worker, const std::string& job_id,
                                     const std::string& extranonce2, const std::string& ntime,
                                     const std::string& nonce);
    static bool parse_response(const std::string& response, int& id, json& result, StratumErrorInfo& error);
    static bool parse_notify(const std::string& notification, std::string& job_id, std::string& prev_hash,
                            std::string& coinbase1, std::string& coinbase2,
                            std::vector<std::string>& merkle_branch, std::string& version,
                            std::string& nbits, std::string& ntime, bool& clean_jobs);
    static bool parse_set_difficulty(const std::string& notification, double& difficulty);

private:
    static json make_request(uint64_t id, const std::string& method, const json& params) {
        json req;
        req["jsonrpc"] = "2.0";
        req["id"] = id;
        req["method"] = method;
        req["params"] = params;
        return req;
    }
};

/**
 * Error codes defined by the Stratum protocol
 */
enum class StratumError {
    UNKNOWN = 20,
    JOB_NOT_FOUND = 21,
    DUPLICATE_SHARE = 22,
    LOW_DIFFICULTY = 23,
    UNAUTHORIZED = 24,
    NOT_SUBSCRIBED = 25
};

/**
 * Helper to convert error codes to human readable messages
 */
inline std::string get_error_message(StratumError code) {
    switch (code) {
        case StratumError::JOB_NOT_FOUND:
            return "Job not found (stale)";
        case StratumError::DUPLICATE_SHARE:
            return "Duplicate share";
        case StratumError::LOW_DIFFICULTY:
            return "Low difficulty share";
        case StratumError::UNAUTHORIZED:
            return "Unauthorized worker";
        case StratumError::NOT_SUBSCRIBED:
            return "Not subscribed";
        default:
            return "Unknown error";
    }
}

} // namespace pool
} // namespace ohmy