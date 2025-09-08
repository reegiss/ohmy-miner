/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace miner {

/**
 * @brief Custom exception for network-related errors (connection, protocol, etc.).
 */
class NetException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace net {

/**
 * @brief Represents the state of the connection to the mining pool.
 */
enum class ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    AuthenticationFailed,
    Error
};

/**
 * @brief Represents a mining job received from the pool via 'mining.notify'.
 */
struct Job {
    std::string job_id;
    std::string prev_hash;
    std::string coinb1;
    std::string coinb2;
    std::vector<std::string> merkle_branch;
    std::string version;
    std::string nbits; // Difficulty target in hex format
    std::string ntime; // Block time in hex format
    bool clean_jobs;   // If true, discard all previous jobs
};

/**
 * @brief Represents a solution (share) to be submitted to the pool.
 */
struct Share {
    std::string job_id;
    std::string extranonce2;
    std::string ntime;
    std::string nonce;
};

/**
 * @class IStratumClient
 * @brief Abstract interface for a Stratum protocol client.
 *
 * Defines the contract for communicating with a mining pool.
 * This interface is designed to be asynchronous, using callbacks for events.
 */
class IStratumClient {
public:
    virtual ~IStratumClient() = default;

    /**
     * @brief Asynchronously connects to the pool and starts the session.
     * @param host The pool's hostname or IP address.
     * @param port The pool's port.
     * @param user The username or wallet address.
     * @param pass The password (often 'x' or worker name).
     */
    virtual void connect(const std::string& host, uint16_t port, const std::string& user, const std::string& pass) = 0;

    /**
     * @brief Disconnects from the pool.
     */
    virtual void disconnect() = 0;

    /**
     * @brief Asynchronously submits a share to the pool.
     * @param share The share to be submitted.
     */
    virtual void submit(const Share& share) = 0;

    /**
     * @brief Registers a callback to be invoked when a new job is received.
     * @param callback The function to call with the new Job data.
     */
    virtual void onNewJob(std::function<void(const Job&)> callback) = 0;

    /**
     * @brief Registers a callback to be invoked when the connection state changes.
     * @param callback The function to call with the new ConnectionState.
     */
    virtual void onConnectionStateChanged(std::function<void(ConnectionState)> callback) = 0;
};

} // namespace net
} // namespace miner