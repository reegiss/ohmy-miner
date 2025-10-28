/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#ifndef OHMY_MINER_POOL_CONNECTION_HPP
#define OHMY_MINER_POOL_CONNECTION_HPP

#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <nlohmann/json.hpp>
#include <asio.hpp>

namespace ohmy {

using json = nlohmann::json;

/**
 * @brief Mining job information received from pool
 */
struct MiningJob {
    std::string job_id;
    std::string prev_hash;
    std::string coinbase1;
    std::string coinbase2;
    std::vector<std::string> merkle_branch;
    std::string version;
    std::string nbits;
    std::string ntime;
    bool clean_jobs;
    
    bool is_valid() const {
        return !job_id.empty() && !prev_hash.empty();
    }
};

/**
 * @brief Stratum protocol connection to mining pool
 */
class PoolConnection {
public:
    /**
     * @brief Callback types for pool events
     */
    using JobCallback = std::function<void(const MiningJob&)>;
    using DifficultyCallback = std::function<void(double)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    /**
     * @brief Constructor
     * @param io_context ASIO io_context for async operations
     * @param pool_url Pool URL (host:port)
     * @param username Mining username (wallet.worker)
     * @param password Pool password
     */
    PoolConnection(asio::io_context& io_context,
                   const std::string& pool_url,
                   const std::string& username,
                   const std::string& password);

    ~PoolConnection();

    /**
     * @brief Connect to the mining pool
     * @return true if connection successful, false otherwise
     */
    bool connect();

    /**
     * @brief Disconnect from the pool
     */
    void disconnect();

    /**
     * @brief Check if connected to pool
     */
    bool is_connected() const { return connected_; }

    /**
     * @brief Subscribe to mining notifications
     * @return true if successful, false otherwise
     */
    bool subscribe();

    /**
     * @brief Authorize miner with pool
     * @return true if authorized, false otherwise
     */
    bool authorize();

    /**
     * @brief Submit a share to the pool
     * @param job_id Job ID
     * @param nonce Nonce value
     * @param ntime Block timestamp
     * @param extra_nonce2 Extra nonce 2
     * @return true if submission accepted, false otherwise
     */
    bool submit_share(const std::string& job_id,
                     const std::string& nonce,
                     const std::string& ntime,
                     const std::string& extra_nonce2);

    /**
     * @brief Set callback for new mining jobs
     */
    void set_job_callback(JobCallback callback) {
        job_callback_ = callback;
    }

    /**
     * @brief Set callback for difficulty changes
     */
    void set_difficulty_callback(DifficultyCallback callback) {
        difficulty_callback_ = callback;
    }

    /**
     * @brief Set callback for errors
     */
    void set_error_callback(ErrorCallback callback) {
        error_callback_ = callback;
    }

    /**
     * @brief Get current mining job
     */
    MiningJob get_current_job() const {
        std::lock_guard<std::mutex> lock(job_mutex_);
        return current_job_;
    }

    /**
     * @brief Get current difficulty
     */
    double get_difficulty() const { return difficulty_; }

    /**
     * @brief Get extra nonce 1
     */
    std::string get_extra_nonce1() const { return extra_nonce1_; }

    /**
     * @brief Get extra nonce 2 size
     */
    int get_extra_nonce2_size() const { return extra_nonce2_size_; }

    /**
     * @brief Start async receive loop
     */
    void start_receive_loop();

private:
    /**
     * @brief Parse pool URL into host and port
     */
    bool parse_url(const std::string& url, std::string& host, std::string& port);

    /**
     * @brief Send JSON-RPC request to pool
     * @param method Method name
     * @param params Parameters array
     * @param id Request ID
     */
    bool send_request(const std::string& method, const json& params, int id);

    /**
     * @brief Receive response from pool
     * @param timeout_seconds Timeout in seconds
     */
    json receive_response(int timeout_seconds = 10);

    /**
     * @brief Wait until a response with matching id is received; handle notifications in between
     * @param expected_id Request id to wait for
     * @param timeout_seconds Timeout in seconds (best-effort)
     * @return JSON response object for the matching id, or empty object on failure
     */
    json wait_for_response_for_id(int expected_id, int timeout_seconds = 10);

    /**
     * @brief Handle incoming message from pool
     */
    void handle_message(const std::string& message);

    /**
     * @brief Handle mining.notify method
     */
    void handle_mining_notify(const json& params);

    /**
     * @brief Handle mining.set_difficulty method
     */
    void handle_set_difficulty(const json& params);

    /**
     * @brief Async read handler
     */
    void async_read_handler(const asio::error_code& error, std::size_t bytes_transferred);

    /**
     * @brief Start async read operation
     */
    void do_async_read();

    // ASIO components
    asio::io_context& io_context_;
    std::unique_ptr<asio::ip::tcp::socket> socket_;
    asio::streambuf receive_buffer_;

    // Connection details
    std::string pool_url_;
    std::string host_;
    std::string port_;
    std::string username_;
    std::string password_;

    // State
    std::atomic<bool> connected_{false};
    std::atomic<bool> authorized_{false};
    std::atomic<bool> subscribed_{false};

    // Stratum session data
    std::string extra_nonce1_;
    int extra_nonce2_size_{0};
    std::string session_id_;
    std::atomic<double> difficulty_{1.0};
    
    // Current mining job
    mutable std::mutex job_mutex_;
    MiningJob current_job_;

    // Request counter
    std::atomic<int> request_id_{1};

    // Callbacks
    JobCallback job_callback_;
    DifficultyCallback difficulty_callback_;
    ErrorCallback error_callback_;

    // Thread safety
    std::mutex send_mutex_;
};

} // namespace ohmy

#endif // OHMY_MINER_POOL_CONNECTION_HPP
