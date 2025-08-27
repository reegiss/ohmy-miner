#include "pool_connection.h"
#include <iostream>
#include <istream> // Required for std::istream

// The constructor now initializes the job queue reference.
PoolConnection::PoolConnection(const std::string& host, uint16_t port, ThreadSafeQueue<MiningJob>& job_queue)
    : host_(host),
      port_(port),
      job_queue_(job_queue), // Initialize the reference to the shared queue
      io_context_(),
      socket_(io_context_) {}

// Main run loop for the network thread.
void PoolConnection::run(const std::string& user, const std::string& pass) {
    if (!connect() || !handshake(user, pass)) {
        std::cerr << "Failed to initialize pool connection. Exiting network thread." << std::endl;
        return;
    }

    std::cout << "[NETWORK] Handshake complete. Listening for jobs..." << std::endl;

    // Main network thread loop
    while (socket_.is_open()) {
        json msg = read_json();
        if (msg.is_null()) {
            std::cerr << "Connection lost or invalid message. Exiting network thread." << std::endl;
            break;
        }

        if (msg.value("method", "") == "mining.notify") {
            std::cout << "[NETWORK] Received new mining job." << std::endl;
            try {
                auto params = msg["params"];
                MiningJob job;
                job.job_id = params[0].get<std::string>();
                job.prev_hash = params[1].get<std::string>();
                job.coinb1 = params[2].get<std::string>();
                job.coinb2 = params[3].get<std::string>();
                job.merkle_branches = params[4].get<std::vector<std::string>>();
                job.version = params[5].get<std::string>();
                job.nbits = params[6].get<std::string>();
                job.ntime = params[7].get<std::string>();
                job.clean_jobs = params[8].get<bool>();

                job_queue_.push(job); // Push the new job to the queue for the miner thread
            } catch (const json::exception& e) {
                std::cerr << "[NETWORK] Error parsing mining.notify: " << e.what() << std::endl;
            }
        } else if (msg.value("method", "") == "mining.set_difficulty") {
            try {
                double new_diff = msg["params"][0].get<double>();
                std::cout << "[NETWORK] Pool set new difficulty: " << new_diff << std::endl;
                // Logic to update the miner's difficulty will go here in the future
            } catch (const json::exception& e) {
                std::cerr << "[NETWORK] Error parsing set_difficulty: " << e.what() << std::endl;
            }
        }
    }
}

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
        if (ec == asio::error::eof) {
             std::cout << "[NETWORK] Connection closed by peer." << std::endl;
        } else {
             std::cerr << "Error reading from socket: " << ec.message() << std::endl;
        }
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

    try {
        auto result = subscribe_res["result"];
        extranonce1_ = result[1].get<std::string>();
        extranonce2_size_ = result[2].get<int>();
        std::cout << "Subscription successful. Extranonce1: " << extranonce1_ << std::endl;
    } catch (const json::exception& e) {
        std::cerr << "Error parsing subscription response: " << e.what() << std::endl;
        return false;
    }

    // --- Step 2: Authorize ---
    json authorize_req = {
        {"id", 2},
        {"method", "mining.authorize"},
        {"params", {user, pass}}
    };
    if (!write_json(authorize_req)) return false;

    json authorize_res = read_json();

    // Loop until we find the response with id == 2.
    while (true) {
        if (authorize_res.is_null()) return false; // Exit on read error

        // Robustly check if "id" is a number and equals 2
        if (authorize_res.contains("id") && authorize_res["id"].is_number() && authorize_res["id"].get<int>() == 2) {
            break; // Found our response, exit the loop
        }

        std::cout << "[NETWORK] Received a notification, waiting for auth response..." << std::endl;
        // This was a notification (like set_difficulty), so we ignore it and read the next message.
        authorize_res = read_json();
    }

    if (authorize_res.value("error", json(nullptr)) != nullptr || !authorize_res.value("result", false)) {
        std::cerr << "Authorization failed. Pool response: " << authorize_res.dump(2) << std::endl;
        return false;
    }
    
    std::cout << "✅ Authorization successful!" << std::endl;
    return true;
}

void PoolConnection::close() {
    if (socket_.is_open()) {
        boost::system::error_code ec;
        // Desliga o envio e recebimento para evitar condições de corrida
        socket_.shutdown(tcp::socket::shutdown_both, ec);
        socket_.close(ec);
    }
}