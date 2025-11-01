/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <asio.hpp>

#include "ohmy/pool/stratum.hpp"

using json = nlohmann::json;
using ohmy::pool::StratumClient;

// Test double: capture outgoing messages instead of using a socket
class TestClient : public StratumClient {
public:
    explicit TestClient(asio::io_context& io, const std::string& url)
        : StratumClient(io, url, "worker", "x", false) {}

    const std::vector<json>& outbox() const { return outbox_; }
    void clear_outbox() { outbox_.clear(); }

    // Public shims to access protected API for testing
    void pub_capabilities(const std::string& t) { capabilities(t); }
    void pub_suggest_target(const std::string& t) { suggest_target(t); }
    void pub_get_transactions(const std::string& job) { get_transactions(job); }
    void pub_handle_message(const std::string& s) { handle_message(s); }

protected:
    void send_message(const json& message) override {
        outbox_.push_back(message);
    }

private:
    std::vector<json> outbox_;
};

static void test_capabilities_and_suggest_target() {
    asio::io_context io;
    TestClient c(io, "pool.example.com:3333");

    // capabilities should be a notification with caps object
    c.pub_capabilities("00ff");
    assert(!c.outbox().empty());
    auto msg = c.outbox().back();
    assert(msg["method"] == "mining.capabilities");
    assert(msg["id"].is_null());
    assert(msg["params"].is_array());
    assert(msg["params"].size() == 1);
    assert(msg["params"][0].is_object());
    assert(msg["params"][0].contains("notify"));
    assert(msg["params"][0].contains("set_difficulty"));
    assert(msg["params"][0].contains("suggested_target"));

    // suggest_target should be a notification with full hex
    c.pub_suggest_target("ffffffff");
    auto msg2 = c.outbox().back();
    assert(msg2["method"] == "mining.suggest_target");
    assert(msg2["id"].is_null());
    assert(msg2["params"].is_array());
    assert(msg2["params"].size() == 1);
    assert(msg2["params"][0] == "ffffffff");
}

static void test_get_transactions_and_ids() {
    asio::io_context io;
    TestClient c(io, "pool.example.com:3333");

    c.pub_get_transactions("job123");
    auto msg = c.outbox().back();
    assert(msg["method"] == "mining.get_transactions");
    assert(msg["id"].is_number_integer());
    assert(msg["params"].is_array());
    assert(msg["params"].size() == 1);
    assert(msg["params"][0] == "job123");
}

static void test_server_requests_and_responses() {
    asio::io_context io;
    TestClient c(io, "pool.example.com:3333");

    // client.get_version
    c.clear_outbox();
    json req = {
        {"jsonrpc", "2.0"},
        {"id", 42},
        {"method", "client.get_version"},
        {"params", json::array()}
    };
    c.pub_handle_message(req.dump());
    assert(!c.outbox().empty());
    auto resp = c.outbox().back();
    assert(resp["id"] == 42);
    assert(resp["result"].is_string());

    // client.show_message
    c.clear_outbox();
    json msg = {
        {"jsonrpc", "2.0"},
        {"id", 43},
        {"method", "client.show_message"},
        {"params", json::array({"hello miner"})}
    };
    c.pub_handle_message(msg.dump());
    assert(!c.outbox().empty());
    auto resp2 = c.outbox().back();
    assert(resp2["id"] == 43);
    assert(resp2["result"] == true);

    // client.reconnect same host allowed
    c.clear_outbox();
    json rec1 = {
        {"jsonrpc", "2.0"},
        {"id", 44},
        {"method", "client.reconnect"},
        {"params", json::array({"pool.example.com", 4444, 1})}
    };
    c.pub_handle_message(rec1.dump());
    assert(!c.outbox().empty());
    auto resp3 = c.outbox().back();
    assert(resp3["id"] == 44);
    assert(resp3["result"] == true);

    // client.reconnect different host denied
    c.clear_outbox();
    json rec2 = {
        {"jsonrpc", "2.0"},
        {"id", 45},
        {"method", "client.reconnect"},
        {"params", json::array({"evil.example.com", 4444, 1})}
    };
    c.pub_handle_message(rec2.dump());
    assert(!c.outbox().empty());
    auto resp4 = c.outbox().back();
    assert(resp4["id"] == 45);
    assert(resp4["result"] == false);
}

int main() {
    test_capabilities_and_suggest_target();
    test_get_transactions_and_ids();
    test_server_requests_and_responses();
    std::cout << "test_stratum_optional: OK\n";
    return 0;
}
