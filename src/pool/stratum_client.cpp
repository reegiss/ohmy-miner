/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/pool/stratum.hpp"
#include "ohmy/pool/messages.hpp"
#include <fmt/format.h>
#include <iostream>
#include <optional>
#include <vector>
#include <cctype>
#include "ohmy/log.hpp"

namespace ohmy {
namespace pool {

// Parse BIP34 block height from the beginning of coinbase scriptSig contained in coinbase1.
// Returns empty if parsing fails.
static std::optional<uint32_t> parse_bip34_height_from_coinbase1_hex(const std::string& coinbase1_hex) {
    auto hex_to_bytes = [](const std::string& hex) -> std::vector<uint8_t> {
        std::vector<uint8_t> out;
        if (hex.size() % 2 != 0) return out;
        out.reserve(hex.size() / 2);
        for (size_t i = 0; i < hex.size(); i += 2) {
            auto hexval = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                return -1;
            };
            int hi = hexval(hex[i]);
            int lo = hexval(hex[i+1]);
            if (hi < 0 || lo < 0) return {};
            out.push_back(static_cast<uint8_t>((hi << 4) | lo));
        }
        return out;
    };

    const auto cb = hex_to_bytes(coinbase1_hex);
    if (cb.size() < 42) return std::nullopt; // must contain header + start of script

    // Walk: version(4) + vin_count(varint) + prevout(32+4) + script_len(varint)
    size_t i = 0;
    auto read_varint = [&](uint64_t& val) -> bool {
        if (i >= cb.size()) return false;
        uint8_t first = cb[i++];
        if (first < 0xFD) { val = first; return true; }
        if (first == 0xFD) { if (i + 2 > cb.size()) return false; val = cb[i] | (cb[i+1] << 8); i += 2; return true; }
        if (first == 0xFE) { if (i + 4 > cb.size()) return false; val = cb[i] | (cb[i+1] << 8) | (cb[i+2] << 16) | (static_cast<uint64_t>(cb[i+3]) << 24); i += 4; return true; }
        if (first == 0xFF) { if (i + 8 > cb.size()) return false; uint64_t v=0; for (int b=0;b<8;++b) v |= (static_cast<uint64_t>(cb[i+b]) << (8*b)); i+=8; val=v; return true; }
        return false;
    };

    if (i + 4 > cb.size()) return std::nullopt; // version
    i += 4;

    uint64_t vin_count = 0;
    if (!read_varint(vin_count) || vin_count == 0) return std::nullopt;

    // prevout (hash + index)
    if (i + 32 + 4 > cb.size()) return std::nullopt;
    i += 32 + 4;

    uint64_t script_len = 0;
    if (!read_varint(script_len)) return std::nullopt;

    // Now at start of scriptSig; first opcode should be PUSHDATA with height
    if (i >= cb.size()) return std::nullopt;
    uint8_t op = cb[i++];
    size_t push_len = 0;
    if (op >= 0x01 && op <= 0x4b) {
        push_len = op;
    } else if (op == 0x4c) { // OP_PUSHDATA1
        if (i >= cb.size()) return std::nullopt;
        push_len = cb[i++];
    } else {
        return std::nullopt;
    }
    if (i + push_len > cb.size()) return std::nullopt;

    // Height is little-endian in these bytes
    uint32_t height = 0;
    for (size_t b = 0; b < push_len && b < 4; ++b) {
        height |= static_cast<uint32_t>(cb[i + b]) << (8 * b);
    }
    return height;
}

StratumClient::StratumClient(
    asio::io_context& io_context,
    const std::string& url,
    const std::string& worker_name,
    const std::string& password
)   : io_context_(io_context)
    , url_(url)
    , worker_name_(worker_name)
    , password_(password)
{
    // Reserve message IDs 1 and 2 for subscribe/authorize used explicitly below
    // Ensures subsequent requests (e.g., share submits) start from ID >= 3
    message_id_ = 2;
}

void StratumClient::connect() {
    // Parse URL into host and port
    size_t colon_pos = url_.find(':');
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("Invalid pool URL format. Expected hostname:port");
    }

    std::string host = url_.substr(0, colon_pos);
    std::string port = url_.substr(colon_pos + 1);

    // Resolve endpoint
    asio::ip::tcp::resolver resolver(io_context_);
    auto endpoints = resolver.resolve(host, port);

    // Create and connect socket
    socket_ = std::make_unique<asio::ip::tcp::socket>(io_context_);
    
    asio::async_connect(*socket_, endpoints,
        [this, host](std::error_code ec, asio::ip::tcp::endpoint ep) {
            if (!ec) {
                // Connection successful, subscribe to mining
                ohmy::log::line("use pool {} {}", url_, ep.address().to_string());
                subscribe();
                start_reading();
            } else {
                ohmy::log::line("Connection failed: {}", ec.message());
                // Schedule reconnect after delay
                auto timer = std::make_shared<asio::steady_timer>(
                    io_context_, std::chrono::seconds(5));
                timer->async_wait([this, timer](std::error_code) {
                    connect();
                });
            }
        });
}

void StratumClient::disconnect() {
    if (socket_ && socket_->is_open()) {
        socket_->close();
    }
    subscribed_ = false;
    authorized_ = false;
}

bool StratumClient::is_connected() const {
    return socket_ && socket_->is_open() && subscribed_ && authorized_;
}

void StratumClient::start_reading() {
    // Read until newline delimiter
    asio::async_read_until(*socket_, receive_buffer_, '\n',
        [this](std::error_code ec, [[maybe_unused]] std::size_t length) {
            if (!ec) {
                // Extract message from buffer
                std::string message;
                std::istream is(&receive_buffer_);
                std::getline(is, message);

                // Handle the message
                handle_message(message);

                // Continue reading
                start_reading();
            } else {
                fmt::print("Read error: {}\n", ec.message());
                disconnect();
                
                // Schedule reconnect
                auto timer = std::make_shared<asio::steady_timer>(
                    io_context_, std::chrono::seconds(5));
                timer->async_wait([this, timer](std::error_code) {
                    connect();
                });
            }
        });
}

void StratumClient::send_message(const json& message) {
    if (!socket_ || !socket_->is_open()) {
        return;
    }

    std::string message_str = message.dump() + "\n";
    // Log outgoing request IDs and methods for troubleshooting
    try {
        if (message.contains("id") && message.contains("method")) {
            fmt::print("Sending request id={} method={}\n", (uint64_t)message["id"], std::string(message["method"]));
        }
    } catch (...) {
        // ignore logging errors
    }
    asio::async_write(*socket_, asio::buffer(message_str),
        [this](std::error_code ec, std::size_t) {
            if (ec) {
                fmt::print("Write error: {}\n", ec.message());
                disconnect();
            }
        });
}

void StratumClient::handle_message(const std::string& message) {
    try {
        json j = json::parse(message);

        if (StratumMessages::is_notification(j)) {
            // Handle notifications from server
            const std::string& method = j["method"];
            
            if (method == "mining.notify") {
                handle_mining_notify(j["params"]);
            } else if (method == "mining.set_difficulty") {
                handle_set_difficulty(j["params"]);
            } else {
                // ignore unknown notifications
            }
        }
        else if (StratumMessages::is_response(j)) {
            // Handle method responses
            uint64_t id = j["id"];
            
            if (id == 1) {  // Response to subscribe
                if (!StratumMessages::is_error(j)) {
                    subscribed_ = true;
                    
                    // Extract extranonce1 and extranonce2_size from subscription result
                    // Result format: [[["mining.set_difficulty", "subscription_id"], ...], "extranonce1", extranonce2_size]
                    if (j["result"].is_array() && j["result"].size() >= 3) {
                        extranonce1_ = j["result"][1].get<std::string>();
                        extranonce2_size_ = j["result"][2].get<int>();
                        // silent success
                    } 
                    
                    // Now authorize
                    authorize();
                } else {
                    ohmy::log::line("Subscription failed: {}", j["error"].dump());
                }
            }
            else if (id == 2) {  // Response to authorize
                if (!StratumMessages::is_error(j)) {
                    authorized_ = true;
                    ohmy::log::line("Start mining");
                } else {
                    ohmy::log::line("Authorization failed: {}", j["error"].dump());
                }
            }
            else {  // Response to share submission
                bool accepted = !StratumMessages::is_error(j);
                handle_submit_result(j, accepted);
            }
        } else {
            // ignore non-json or unknown
        }

    } catch (const std::exception& e) {
        // Ignore malformed lines quietly
    }
}

void StratumClient::subscribe() {
    send_message(StratumMessages::subscribe(1));
}

void StratumClient::authorize() {
    send_message(StratumMessages::authorize(2, worker_name_, password_));
}

void StratumClient::submit_share(const ShareResult& share) {
    // Guard: only submit if we're connected, subscribed and authorized
    if (!is_connected()) {
        fmt::print("Cannot submit share: not connected/authorized\n");
        return;
    }

    // Guard: avoid stale shares if job id isn't currently valid
    if (!share.job_id.empty() && !valid_job_ids_.empty() && !valid_job_ids_.count(share.job_id)) {
        fmt::print("Skipping stale share for job {} (not in valid set)\n", share.job_id);
        return;
    }

    // Ensure extranonce2 has correct hex length (2 chars per byte)
    std::string extranonce2 = share.extranonce2;
    const size_t expected_len = static_cast<size_t>(std::max(0, extranonce2_size_)) * 2;
    if (expected_len > 0) {
        if (extranonce2.size() < expected_len) {
            // Left-pad with zeros
            extranonce2 = std::string(expected_len - extranonce2.size(), '0') + extranonce2;
            fmt::print("Adjusted extranonce2 (padded) to {} hex chars\n", extranonce2.size());
        } else if (extranonce2.size() > expected_len) {
            // Trim left to match expected width
            extranonce2 = extranonce2.substr(extranonce2.size() - expected_len);
            fmt::print("Adjusted extranonce2 (trimmed) to {} hex chars\n", extranonce2.size());
        }
    }

    // Normalize to lowercase hex
    auto to_lower_hex = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        return s;
    };
    std::string ntime = to_lower_hex(share.ntime);
    extranonce2 = to_lower_hex(extranonce2);
    std::string nonce_hex = fmt::format("{:08x}", share.nonce);

    // Format share submission
    uint64_t req_id = ++message_id_;
    auto msg = StratumMessages::submit(
        req_id,
        worker_name_,
        share.job_id,
        extranonce2,
        ntime,
        nonce_hex
    );
    
    // Track pending submit for diagnostics
    pending_submits_[req_id] = PendingShare{share.job_id, extranonce2, ntime, nonce_hex};

    send_message(msg);
}

void StratumClient::handle_mining_notify(const json& params) {
    try {
        // Extract work package from notification
        WorkPackage work;
        
        // Array of parameters as per Stratum spec
        work.job_id = params[0];
        work.previous_hash = params[1];
        work.coinbase1 = params[2];
        work.coinbase2 = params[3];
        work.merkle_branch = params[4];
        work.version = params[5];
        work.bits = params[6];
        work.time = params[7];
        work.clean_jobs = params[8];
        
    // Add extranonce1 from subscription
        work.extranonce1 = extranonce1_;
        
        // Generate extranonce2 (for now, just zeros - workers will modify this)
        work.extranonce2 = std::string(extranonce2_size_ * 2, '0');

        // Derive share target from current difficulty (Stratum share target)
        // Compute a FULL 256-bit target (64-hex, big-endian) from difficulty via compact bits
        work.share_difficulty = current_difficulty_;
        if (current_difficulty_ > 0.0) {
            auto bits_from_difficulty = [](double diff) -> uint32_t {
                // Base: diff1 target uses bits 0x1d00ffff
                const int base_exp = 0x1d;           // 29
                const double base_coeff = 65535.0;   // 0x00ffff
                int exp = base_exp;
                double coeff = base_coeff / diff;     // scale by 1/diff
                // Normalize mantissa into [0x008000, 0x007fffff]
                while (coeff < 0x008000 && exp > 1) {
                    coeff *= 256.0;
                    --exp;
                }
                while (coeff > 0x007fffff) {
                    coeff /= 256.0;
                    ++exp;
                }
                uint32_t c = static_cast<uint32_t>(std::floor(coeff + 0.5)); // round to nearest
                if (c < 0x000800) c = 0x000800; // clamp lower bound to avoid subnormal
                if (c > 0x007fffff) c = 0x007fffff; // clamp upper bound
                return (static_cast<uint32_t>(exp) << 24) | (c & 0x007FFFFF);
            };

            auto expand_compact_to_full = [](uint32_t bits) -> std::array<uint8_t,32> {
                std::array<uint8_t,32> target{}; // zero-initialized (big-endian)
                uint32_t exponent = bits >> 24;
                uint32_t coefficient = bits & 0x00FFFFFF;
                if (exponent <= 3) {
                    // Right-align in last 3 bytes
                    target[29] = static_cast<uint8_t>((coefficient >> 16) & 0xFF);
                    target[30] = static_cast<uint8_t>((coefficient >> 8) & 0xFF);
                    target[31] = static_cast<uint8_t>(coefficient & 0xFF);
                } else if (exponent <= 32) {
                    int pos = 32 - static_cast<int>(exponent);
                    if (pos >= 0 && pos + 2 < 32) {
                        target[static_cast<size_t>(pos)]     = static_cast<uint8_t>((coefficient >> 16) & 0xFF);
                        target[static_cast<size_t>(pos + 1)] = static_cast<uint8_t>((coefficient >> 8) & 0xFF);
                        target[static_cast<size_t>(pos + 2)] = static_cast<uint8_t>(coefficient & 0xFF);
                    }
                }
                return target;
            };

            auto to_hex64 = [](const std::array<uint8_t,32>& bytes) {
                static const char* hex = "0123456789abcdef";
                std::string s;
                s.resize(64);
                for (size_t i = 0; i < 32; ++i) {
                    s[2*i]   = hex[(bytes[i] >> 4) & 0xF];
                    s[2*i+1] = hex[bytes[i] & 0xF];
                }
                return s;
            };

            uint32_t share_bits = bits_from_difficulty(current_difficulty_);
            auto full_target = expand_compact_to_full(share_bits);
            work.share_target_hex = to_hex64(full_target);
        }

        // Maintain valid job id set to prevent stale share submissions
        if (work.clean_jobs) {
            valid_job_ids_.clear();
        }
        valid_job_ids_.insert(work.job_id);

        // Compute network difficulty from compact bits
        auto difficulty_from_bits = [](const std::string& bits_hex) -> double {
            if (bits_hex.size() < 8) return 0.0;
            uint32_t bits_value = 0;
            try { bits_value = static_cast<uint32_t>(std::stoul(bits_hex, nullptr, 16)); }
            catch (...) { return 0.0; }
            const uint32_t exp = bits_value >> 24;
            const uint32_t coeff = bits_value & 0x007fffff;
            // diff = diff1_target / target; diff1_target = 0x1d00ffff
            const double diff1 = 0x00ffff * std::pow(256.0, 0x1d - 3);
            const double target = static_cast<double>(coeff) * std::pow(256.0, static_cast<int>(exp) - 3);
            if (target <= 0.0) return 0.0;
            return diff1 / target;
        };

        auto human_diff = [](double d) -> std::string {
            const char* units[] = {"", "K", "M", "G", "T", "P", "E"};
            int i = 0;
            while (d >= 1000.0 && i < 6) { d /= 1000.0; ++i; }
            return fmt::format("{:.2f}{}", d, units[i]);
        };

        double net_diff = difficulty_from_bits(work.bits);
        double share_diff = current_difficulty_;

        // Block height (best-effort from coinbase1/BIP34)
        auto height_opt = parse_bip34_height_from_coinbase1_hex(work.coinbase1);

        // Compact job line(s)
        if (share_diff > 0.0 && net_diff > 0.0) {
            ohmy::log::line("new job from {} diff {}/{}", url_, human_diff(share_diff), human_diff(net_diff));
        } else {
            ohmy::log::line("new job from {}", url_);
        }
        if (height_opt.has_value()) {
            ohmy::log::line("block: {}", *height_opt);
        }
        if (!work.share_target_hex.empty()) {
            ohmy::log::line("job target: 0x{}", work.share_target_hex.substr(0, 16));
        }

        // Notify miner of new work
        if (work_callback_) {
            work_callback_(work);
        }
    } catch (const std::exception& e) {
        // ignore
    }
}

void StratumClient::handle_set_difficulty(const json& params) {
    try {
        current_difficulty_ = params[0];
        ohmy::log::line("Stratum set raw difficulty to {:.4f}", current_difficulty_);
        
        // Notify the work manager about difficulty change
        if (difficulty_callback_) {
            difficulty_callback_(current_difficulty_);
        }
    } catch (const std::exception& e) {
        // ignore parse error
    }
}

void StratumClient::handle_submit_result(const json& result, bool accepted) {
    uint64_t id = 0;
    try {
        if (result.contains("id") && !result["id"].is_null()) {
            id = result["id"].get<uint64_t>();
        }
    } catch (...) {}

    std::string details;
    auto it = pending_submits_.find(id);
    if (it != pending_submits_.end()) {
        details = fmt::format("job={}, ntime={}, en2={}, nonce={}", it->second.job_id, it->second.ntime, it->second.extranonce2, it->second.nonce_hex);
    }

    if (accepted) {
        ohmy::log::line("Share accepted");
    } else {
        std::string reason_text;
        if (StratumMessages::is_error(result)) {
            try {
                auto error = result["error"];
                if (!error.is_null() && error.is_array() && error.size() >= 1) {
                    int code = error[0].get<int>();
                    reason_text = get_error_message(static_cast<StratumError>(code));
                }
            } catch (...) {}
        }
        ohmy::log::line("Share rejected{}", reason_text.empty() ? std::string("") : fmt::format(" (reason: {})", reason_text));
    }

    // Forward minimal result to external callback
    ShareResult share_cb;
    share_cb.accepted = accepted;
    if (!accepted && StratumMessages::is_error(result)) {
        auto error = result["error"];
        if (!error.is_null()) {
            int code = error[0];
            share_cb.reason = get_error_message(static_cast<StratumError>(code));
        }
    }
    if (share_callback_) {
        share_callback_(share_cb);
    }

    // Clean up pending map
    if (it != pending_submits_.end()) pending_submits_.erase(it);
}

void StratumClient::set_work_callback(WorkCallback callback) {
    work_callback_ = std::move(callback);
}

void StratumClient::set_share_callback(ShareCallback callback) {
    share_callback_ = std::move(callback);
}

void StratumClient::set_difficulty_callback(DifficultyCallback callback) {
    difficulty_callback_ = std::move(callback);
}

} // namespace pool
} // namespace ohmy