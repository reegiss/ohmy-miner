/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

namespace miner::net {

// Connection state machine states
enum class ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Subscribing,
    Authorizing,
    Mining,
    AuthenticationFailed,
    Error
};

// Authorization result
struct AuthResult {
    bool success;
    std::string error_message;
    std::string worker_name;
    std::string session_id;
};

// Authorization callback type
using AuthCallback = std::function<void(const AuthResult&)>;

} // namespace miner::net