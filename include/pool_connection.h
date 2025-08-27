#ifndef POOL_CONNECTION_H
#define POOL_CONNECTION_H

#include <string>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>

// Use these aliases for convenience
namespace asio = boost::asio;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

class PoolConnection {
public:
    // Constructor takes the host and port to connect to.
    PoolConnection(const std::string& host, uint16_t port);

    // Attempts to connect to the pool.
    // Returns true on success, false on failure.
    bool connect();

    // Performs the Stratum handshake (subscribe & authorize).
    bool handshake(const std::string& user, const std::string& pass);

private:
    std::string host_;
    uint16_t port_;

    // Writes a JSON-RPC message to the pool.
    bool write_json(const json& j);

    // Reads a line-delimited JSON-RPC message from the pool.
    json read_json();

    asio::io_context io_context_;
    tcp::socket socket_;
    asio::streambuf buffer_;

    // --- Stratum Session State ---
    std::string session_id_;
    std::string extranonce1_;
    int extranonce2_size_;
};

#endif // POOL_CONNECTION_H