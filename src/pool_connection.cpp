#include "pool_connection.h"
#include <iostream>
#include <istream>

PoolConnection::PoolConnection(const std::string& host, uint16_t port)
    : host_(host),
      port_(port),
      io_context_(),
      socket_(io_context_) {}

bool PoolConnection::connect() {
    try {
        tcp::resolver resolver(io_context_);
        boost::system::error_code ec;
        auto endpoints = resolver.resolve(host_, std::to_string(port_), ec);
        if (ec) {
            std::cerr << "Error: Could not resolve host '" << host_ << "': " << ec.message() << std::endl;
            return false;
        }
        std::cout << "Connecting to " << host_ << ":" << port_ << "..." << std::endl;
        asio::connect(socket_, endpoints, ec);
        if (ec) {
            std::cerr << "Error: Could not connect to pool: " << ec.message() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during connection: " << e.what() << std::endl;
        return false;
    }
    std::cout << "✅ Successfully connected to the pool!" << std::endl;
    return true;
}

bool PoolConnection::write_json(const json& j) {
    std::string message = j.dump() + "\n";
    std::cout << "[CLIENT] -> " << message;

    boost::system::error_code ec;
    asio::write(socket_, asio::buffer(message), ec);

    if (ec) {
        std::cerr << "Error writing to socket: " << ec.message() << std::endl;
        return false;
    }
    return true;
}

json PoolConnection::read_json() {
    boost::system::error_code ec;
    asio::read_until(socket_, buffer_, '\n', ec);

    if (ec) {
        std::cerr << "Error reading from socket: " << ec.message() << std::endl;
        return nullptr;
    }

    std::istream is(&buffer_);
    std::string line;
    std::getline(is, line);

    std::cout << "[POOL] <- " << line << std::endl;

    try {
        return json::parse(line);
    } catch (const json::parse_error& e) {
        std::cerr << "JSON Parse Error: " << e.what() << " on line: " << line << std::endl;
        return nullptr;
    }
}


bool PoolConnection::handshake(const std::string& user, const std::string& pass) {
    // --- Step 1: Subscribe ---
    json subscribe_req = {
        {"id", 1},
        {"method", "mining.subscribe"},
        {"params", {"qtcminer/0.1"}}
    };
    if (!write_json(subscribe_req)) return false;

    json subscribe_res = read_json();
    if (subscribe_res.is_null() || subscribe_res.value("error", json(nullptr)) != nullptr) {
        std::cerr << "Subscription failed. Pool response: " << subscribe_res.dump(2) << std::endl;
        return false;
    }

    auto result = subscribe_res["result"];
    extranonce1_ = result[1].get<std::string>();
    extranonce2_size_ = result[2].get<int>();
    std::cout << "Subscription successful. Extranonce1: " << extranonce1_ << std::endl;

    // --- Step 2: Authorize ---
    json authorize_req = {
        {"id", 2},
        {"method", "mining.authorize"},
        {"params", {user, pass}}
    };
    if (!write_json(authorize_req)) return false;

    json authorize_res = read_json();

    // ----- THE CORRECTED LOOP -----
    // Loop until we find the response with id == 2.
    while (true) {
        if (authorize_res.is_null()) return false; // Exit on read error

        // Robustly check if "id" is a number and equals 2
        if (authorize_res.contains("id") && authorize_res["id"].is_number() && authorize_res["id"].get<int>() == 2) {
            break; // Found our response, exit the loop
        }

        std::cout << "Received a notification, waiting for auth response..." << std::endl;
        // This was a notification (like set_difficulty), so we ignore it and read the next message.
        authorize_res = read_json();
    }
    // ----------------------------

    if (authorize_res.value("error", json(nullptr)) != nullptr || !authorize_res.value("result", false)) {
        std::cerr << "Authorization failed. Pool response: " << authorize_res.dump(2) << std::endl;
        return false;
    }
    
    std::cout << "✅ Authorization successful!" << std::endl;
    return true;
}