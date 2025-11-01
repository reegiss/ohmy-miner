/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <string>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <asio.hpp>
#include <unordered_set>
#include <unordered_map>

namespace ohmy {
namespace pool {

// Forward declarations
class WorkPackage;
struct ShareResult;

/**
 * StratumClient implements the Stratum v1 mining protocol.
 * See: https://en.bitcoin.it/wiki/Stratum_mining_protocol
 */
class StratumClient {
public:
    using json = nlohmann::json;
    using WorkCallback = std::function<void(const WorkPackage&)>;
    using ShareCallback = std::function<void(const ShareResult&)>;
    using DifficultyCallback = std::function<void(double)>;

    /**
     * Constructs a new Stratum client
     * @param io_context ASIO io_context for async operations
     * @param url Pool URL in format hostname:port
     * @param worker_name Worker name/wallet for mining
     * @param password Pool password (usually 'x')
     * @param enable_extranonce_subscribe Enable mining.extranonce.subscribe for pools that support dynamic extranonce changes
     */
    StratumClient(
        asio::io_context& io_context,
        const std::string& url,
        const std::string& worker_name,
        const std::string& password = "x",
        bool enable_extranonce_subscribe = false
    );

    virtual ~StratumClient() = default;

    // Connection management
    void connect();
    void disconnect();
    bool is_connected() const;

    // Work submission
    void submit_share(const ShareResult& share);

    // Callbacks for mining integration
    void set_work_callback(WorkCallback callback);
    void set_share_callback(ShareCallback callback);
    void set_difficulty_callback(DifficultyCallback callback);

    // Advisory options to send after successful authorization
    void set_send_capabilities(bool enable, const std::string& suggested_target_hex = "");
    void set_suggest_target(std::string full_hex_target);
    void set_suggest_difficulty(double difficulty);

protected:
    // Core protocol methods
    virtual void subscribe();
    virtual void authorize();
    virtual void extranonce_subscribe();  // Request dynamic extranonce support
    virtual void suggest_difficulty(double difficulty);
    // Optional client->server methods
    virtual void capabilities(const std::string& suggested_target_hex = "");
    virtual void suggest_target(const std::string& full_hex_target);
    virtual void get_transactions(const std::string& job_id);
    virtual void handle_mining_notify(const json& params);
    virtual void handle_set_difficulty(const json& params);
    virtual void handle_set_extranonce(const json& params);  // Handle extranonce changes
    virtual void handle_submit_result(const json& result, bool accepted);

    // Message handling
    virtual void send_message(const json& message);
    virtual void handle_message(const std::string& message);
    virtual void send_response(uint64_t id, const json& result);

protected:
    // Network operations
    virtual void start_reading();
    
private:
    // Internal state
    bool subscribed_ = false;
    bool authorized_ = false;
    bool enable_extranonce_subscribe_ = false;  // Enable dynamic extranonce support
    double current_difficulty_ = 0.0;
    std::string extranonce1_;        // Extranonce1 from mining.subscribe
    int extranonce2_size_ = 8;       // Extranonce2 size (bytes, usually 8)

    // Network state 
    asio::io_context& io_context_;
    std::string url_;
    std::string worker_name_;
    std::string password_;
    uint64_t message_id_ = 0;

    // Socket and buffer
    std::unique_ptr<asio::ip::tcp::socket> socket_;
    asio::streambuf receive_buffer_;

    // Callbacks
    WorkCallback work_callback_;
    ShareCallback share_callback_;
    DifficultyCallback difficulty_callback_;

    // Track valid job IDs to avoid submitting stale shares
    std::unordered_set<std::string> valid_job_ids_;

    // Track pending share submissions by request id for better diagnostics
    struct PendingShare {
        std::string job_id;
        std::string extranonce2;
        std::string ntime;
        std::string nonce_hex;
    };
    std::unordered_map<uint64_t, PendingShare> pending_submits_;

    // Advisory flags (sent after authorize)
    bool send_capabilities_ = false;
    std::string cap_suggested_target_;
    bool have_suggest_diff_ = false;
    double suggested_diff_ = 0.0;
    std::string suggested_target_;
};

// Helper struct for mining work units
struct WorkPackage {
    std::string job_id;          // Pool's job identifier
    std::string previous_hash;   // Previous block hash
    std::string coinbase1;       // First part of coinbase
    std::string coinbase2;       // Second part of coinbase
    std::vector<std::string> merkle_branch;  // Merkle branch for block
    std::string version;         // Block version
    std::string bits;           // Target difficulty
    std::string time;           // Current time
    bool clean_jobs;            // If true, discard previous jobs
    std::string extranonce1;     // Extranonce1 from subscription
    std::string extranonce2;     // Generated extranonce2 for this work
    // Share target derived from mining.set_difficulty (64-hex, big-endian)
    std::string share_target_hex;
    // Share difficulty value provided by pool
    double share_difficulty = 0.0;
};

// Helper struct for share submission results
struct ShareResult {
    std::string job_id;     // Job this share is for
    uint32_t nonce;         // Nonce that was found
    std::string ntime;      // Timestamp when share was found
    std::string extranonce2;  // Extranonce2 for this share
    std::string hash;       // Share hash (if pool requires it)
    double difficulty;      // Share difficulty
    bool accepted;          // Whether pool accepted share
    std::string reason;     // If rejected, reason why
};

} // namespace pool
} // namespace ohmy