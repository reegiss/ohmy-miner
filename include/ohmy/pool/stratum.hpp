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
     */
    StratumClient(
        asio::io_context& io_context,
        const std::string& url,
        const std::string& worker_name,
        const std::string& password = "x"
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

protected:
    // Core protocol methods
    virtual void subscribe();
    virtual void authorize();
    virtual void suggest_difficulty(double difficulty);
    virtual void handle_mining_notify(const json& params);
    virtual void handle_set_difficulty(const json& params);
    virtual void handle_submit_result(const json& result, bool accepted);

    // Message handling
    virtual void send_message(const json& message);
    virtual void handle_message(const std::string& message);

protected:
    // Network operations
    virtual void start_reading();
    
private:
    // Internal state
    bool subscribed_ = false;
    bool authorized_ = false;
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