#include <ohmy/pool/stratum.hpp>
#include <ohmy/pool/stratum_messages.hpp>

#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/connect.hpp>
#include <asio/steady_timer.hpp>
#include <asio/error.hpp>
#include <asio/read_until.hpp>
#include <asio/write.hpp>
#include <asio/streambuf.hpp>

#include <nlohmann/json.hpp>

using namespace std::literals;
using json = nlohmann::json;

namespace ohmy::pool {

StratumClient::StratumClient(ohmy::logging::Logger& log, StratumOptions opts)
    : log_(log), opts_(std::move(opts)) {}

StratumClient::~StratumClient() { stop(); }

void StratumClient::start() {
    if (io_thread_) return;
    ioc_ = std::make_unique<asio::io_context>();
    io_thread_ = std::make_unique<std::thread>([this]{ run_io_(); });
}

void StratumClient::stop() {
    if (ioc_) ioc_->stop();
    if (io_thread_ && io_thread_->joinable()) io_thread_->join();
    io_thread_.reset();
    ioc_.reset();
}

void StratumClient::on_notify(std::function<void(std::string_view)> cb) {
    notify_cb_ = std::move(cb);
}

void StratumClient::run_io_() {
    try {
        connect_and_handshake_();
        if (ioc_) ioc_->run();
    } catch (const std::exception& e) {
        log_.error(std::string("Stratum IO error: ") + e.what());
    }
}

void StratumClient::connect_and_handshake_() {
    // Placeholder: networking to be implemented next iteration.
    log_.info("Stratum: connect_and_handshake() not yet implemented");
}

bool StratumClient::probe_connect() {
    // Implement a short timeout (3s) using async resolve/connect + steady_timer.
    try {
        asio::io_context ioc;
        asio::ip::tcp::resolver resolver{ioc};
        asio::ip::tcp::socket socket{ioc};
        asio::steady_timer timer{ioc};

        using namespace std::chrono_literals;
        const auto timeout = 3s;
        std::atomic<bool> timed_out{false};
        std::atomic<bool> success{false};
        std::string err_msg;

        log_.info(std::string("Stratum probe: resolving ") + opts_.host + ":" + opts_.port);

        // Timer handler: on timeout, cancel resolver and close socket to abort connect
        timer.expires_after(timeout);
        timer.async_wait([&](const std::error_code& ec){
            if (ec) return; // timer cancelled
            timed_out.store(true);
            std::error_code ignored;
            resolver.cancel();
            socket.close(ignored);
        });

        // Resolve asynchronously
        resolver.async_resolve(opts_.host, opts_.port,
            [&](const std::error_code& ec, asio::ip::tcp::resolver::results_type results){
                if (ec) {
                    if (!timed_out.load()) err_msg = std::string("resolve: ") + ec.message();
                    return; // let io_context finish
                }
                log_.info("Stratum probe: connecting...");
                asio::async_connect(socket, results,
                    [&](const std::error_code& ec2, const asio::ip::tcp::endpoint&){
                        if (!ec2) {
                            log_.info("Stratum probe: connected, sending subscribe...");
                            // Send mining.subscribe
                            auto sub_line = stratum_messages::build_subscribe(opts_.client, 1);
                            asio::async_write(socket, asio::buffer(sub_line),
                                [&](const std::error_code& ec_write, std::size_t){
                                    if (ec_write) {
                                        if (!timed_out.load()) err_msg = std::string("write: ") + ec_write.message();
                                        return;
                                    }
                                    log_.info("Stratum probe: subscribe sent, reading response...");
                                    // Read one line
                                    auto buf = std::make_shared<asio::streambuf>();
                                    asio::async_read_until(socket, *buf, '\n',
                                        [&, buf](const std::error_code& ec_read, std::size_t){
                                            if (!ec_read) {
                                                std::istream is(buf.get());
                                                std::string line;
                                                std::getline(is, line);
                                                log_.info(std::string("Stratum probe: received: ") + line);
                                                
                                                // Parse subscribe response
                                                try {
                                                    auto j = json::parse(line);
                                                    if (j.contains("error") && !j["error"].is_null()) {
                                                        log_.error(std::string("Stratum subscribe error: ") + j["error"].dump());
                                                        err_msg = "subscribe returned error";
                                                        return;
                                                    }
                                                    if (j.contains("result") && j["result"].is_array() && j["result"].size() >= 3) {
                                                        auto& result = j["result"];
                                                        std::string extranonce1 = result[1].get<std::string>();
                                                        int extranonce2_size = result[2].get<int>();
                                                        log_.info(std::string("Stratum subscribe OK: extranonce1=") + extranonce1 
                                                                  + ", extranonce2_size=" + std::to_string(extranonce2_size));
                                                        
                                                        // Now send mining.authorize
                                                        log_.info("Stratum probe: sending authorize...");
                                                        auto auth_line = stratum_messages::build_authorize(opts_.user, opts_.pass, 2);
                                                        asio::async_write(socket, asio::buffer(auth_line),
                                                            [&](const std::error_code& ec_write2, std::size_t){
                                                                if (ec_write2) {
                                                                    if (!timed_out.load()) err_msg = std::string("write authorize: ") + ec_write2.message();
                                                                    return;
                                                                }
                                                                log_.info("Stratum probe: authorize sent, reading response...");
                                                                // Pool may send unsolicited messages first; read lines until we get id=2
                                                                auto buf2 = std::make_shared<asio::streambuf>();
                                                                auto read_next = std::make_shared<std::function<void()>>();
                                                                *read_next = [&, buf2, read_next](){
                                                                    asio::async_read_until(socket, *buf2, '\n',
                                                                        [&, buf2, read_next](const std::error_code& ec_read2, std::size_t){
                                                                            if (!ec_read2) {
                                                                                std::istream is2(buf2.get());
                                                                                std::string line2;
                                                                                std::getline(is2, line2);
                                                                                log_.info(std::string("Stratum probe: <- ") + line2);
                                                                                
                                                                                try {
                                                                                    auto j2 = json::parse(line2);
                                                                                    // Check if this is the authorize response (id=2)
                                                                                    if (j2.contains("id") && j2["id"].is_number_integer() && j2["id"].get<int>() == 2) {
                                                                                        if (j2.contains("error") && !j2["error"].is_null()) {
                                                                                            log_.error(std::string("Stratum authorize error: ") + j2["error"].dump());
                                                                                            err_msg = "authorize returned error";
                                                                                            return;
                                                                                        }
                                                                                        if (j2.contains("result") && j2["result"].is_boolean() && j2["result"].get<bool>()) {
                                                                                            log_.info("Stratum authorize OK");
                                                                                            success.store(true);
                                                                                            std::error_code ignored;
                                                                                            timer.cancel(ignored);
                                                                                            socket.close(ignored);
                                                                                        } else {
                                                                                            log_.error("Stratum authorize: result is not true");
                                                                                            err_msg = "authorize result failed";
                                                                                        }
                                                                                    } else {
                                                                                        // Not the authorize response; read next line
                                                                                        log_.debug("Stratum probe: ignoring unsolicited message, reading next...");
                                                                                        (*read_next)();
                                                                                    }
                                                                                } catch (const json::exception& e) {
                                                                                    log_.error(std::string("Stratum parse error: ") + e.what());
                                                                                    err_msg = std::string("JSON parse: ") + e.what();
                                                                                }
                                                                            } else if (!timed_out.load()) {
                                                                                err_msg = std::string("read authorize: ") + ec_read2.message();
                                                                            }
                                                                        }
                                                                    );
                                                                };
                                                                (*read_next)();
                                                            }
                                                        );
                                                    } else {
                                                        log_.error("Stratum subscribe: unexpected result format");
                                                        err_msg = "subscribe result format invalid";
                                                    }
                                                } catch (const json::exception& e) {
                                                    log_.error(std::string("Stratum subscribe parse error: ") + e.what());
                                                    err_msg = std::string("JSON parse: ") + e.what();
                                                }
                                            } else if (!timed_out.load()) {
                                                err_msg = std::string("read: ") + ec_read.message();
                                            }
                                        }
                                    );
                                }
                            );
                        } else if (!timed_out.load()) {
                            err_msg = std::string("connect: ") + ec2.message();
                        }
                    }
                );
            }
        );

        ioc.run();

        if (success.load()) {
            log_.info("Stratum probe: handshake complete");
            return true;
        }
        if (timed_out.load()) {
            log_.error("Stratum probe failed: timeout (3s)");
        } else {
            log_.error(std::string("Stratum probe failed: ") + (err_msg.empty() ? "unknown" : err_msg));
        }
        return false;
    } catch (const std::exception& e) {
        log_.error(std::string("Stratum probe failed: ") + e.what());
        return false;
    }
}

bool StratumClient::listen_mode(int duration_sec) {
    // Similar to probe_connect but keeps connection open to read mining.notify continuously
    try {
        asio::io_context ioc;
        asio::ip::tcp::resolver resolver{ioc};
        asio::ip::tcp::socket socket{ioc};
        asio::steady_timer connect_timer{ioc};
        asio::steady_timer listen_timer{ioc};

        const auto connect_timeout = 3s;
        std::atomic<bool> timed_out{false};
        std::atomic<bool> handshake_ok{false};
        std::atomic<bool> listen_done{false};
        std::string err_msg;

        log_.info(std::string("Stratum listen: resolving ") + opts_.host + ":" + opts_.port);

        // Connect timeout (handshake phase)
        connect_timer.expires_after(connect_timeout);
        connect_timer.async_wait([&](const std::error_code& ec){
            if (ec) return;
            timed_out.store(true);
            std::error_code ignored;
            resolver.cancel();
            socket.close(ignored);
        });

        // Resolve and connect
        resolver.async_resolve(opts_.host, opts_.port,
            [&](const std::error_code& ec, asio::ip::tcp::resolver::results_type results){
                if (ec) {
                    if (!timed_out.load()) err_msg = std::string("resolve: ") + ec.message();
                    return;
                }
                log_.info("Stratum listen: connecting...");
                asio::async_connect(socket, results,
                    [&](const std::error_code& ec2, const asio::ip::tcp::endpoint&){
                        if (!ec2) {
                            log_.info("Stratum listen: connected, sending subscribe...");
                            auto sub_line = stratum_messages::build_subscribe(opts_.client, 1);
                            asio::async_write(socket, asio::buffer(sub_line),
                                [&](const std::error_code& ec_write, std::size_t){
                                    if (ec_write) {
                                        if (!timed_out.load()) err_msg = std::string("write subscribe: ") + ec_write.message();
                                        return;
                                    }
                                    auto buf = std::make_shared<asio::streambuf>();
                                    asio::async_read_until(socket, *buf, '\n',
                                        [&, buf](const std::error_code& ec_read, std::size_t){
                                            if (!ec_read) {
                                                std::istream is(buf.get());
                                                std::string line;
                                                std::getline(is, line);
                                                try {
                                                    auto j = json::parse(line);
                                                    if (j.contains("error") && !j["error"].is_null()) {
                                                        err_msg = "subscribe error";
                                                        return;
                                                    }
                                                    if (j.contains("result") && j["result"].is_array() && j["result"].size() >= 3) {
                                                        std::string extranonce1 = j["result"][1].get<std::string>();
                                                        int extranonce2_size = j["result"][2].get<int>();
                                                        log_.info(std::string("Stratum listen: subscribe OK, extranonce1=") + extranonce1 
                                                                  + ", extranonce2_size=" + std::to_string(extranonce2_size));
                                                        
                                                        // Send authorize
                                                        auto auth_line = stratum_messages::build_authorize(opts_.user, opts_.pass, 2);
                                                        asio::async_write(socket, asio::buffer(auth_line),
                                                            [&](const std::error_code& ec_auth, std::size_t){
                                                                if (ec_auth) {
                                                                    if (!timed_out.load()) err_msg = std::string("write authorize: ") + ec_auth.message();
                                                                    return;
                                                                }
                                                                
                                                                // Read continuously until authorize response
                                                                auto buf2 = std::make_shared<asio::streambuf>();
                                                                auto read_until_auth = std::make_shared<std::function<void()>>();
                                                                *read_until_auth = [&, buf2, read_until_auth](){
                                                                    asio::async_read_until(socket, *buf2, '\n',
                                                                        [&, buf2, read_until_auth](const std::error_code& ec2, std::size_t){
                                                                            if (!ec2) {
                                                                                std::istream is2(buf2.get());
                                                                                std::string line2;
                                                                                std::getline(is2, line2);
                                                                                log_.info(std::string("Stratum listen: <- ") + line2);
                                                                                
                                                                                try {
                                                                                    auto j2 = json::parse(line2);
                                                                                    if (j2.contains("id") && j2["id"].is_number_integer() && j2["id"].get<int>() == 2) {
                                                                                        if (j2.contains("result") && j2["result"].get<bool>()) {
                                                                                            log_.info("Stratum listen: authorize OK, entering listen mode...");
                                                                                            handshake_ok.store(true);
                                                                                            std::error_code ignored;
                                                                                            connect_timer.cancel(ignored);
                                                                                            
                                                                                            // Start listen duration timer
                                                                                            listen_timer.expires_after(std::chrono::seconds(duration_sec));
                                                                                            listen_timer.async_wait([&](const std::error_code& ec_timer){
                                                                                                if (!ec_timer) {
                                                                                                    log_.info(std::string("Stratum listen: duration expired (") + std::to_string(duration_sec) + "s)");
                                                                                                    listen_done.store(true);
                                                                                                    std::error_code ignored2;
                                                                                                    socket.close(ignored2);
                                                                                                }
                                                                                            });
                                                                                            
                                                                                            // Continue reading notify messages
                                                                                            auto read_notify = std::make_shared<std::function<void()>>();
                                                                                            *read_notify = [&, buf2, read_notify](){
                                                                                                asio::async_read_until(socket, *buf2, '\n',
                                                                                                    [&, buf2, read_notify](const std::error_code& ec3, std::size_t){
                                                                                                        if (!ec3) {
                                                                                                            std::istream is3(buf2.get());
                                                                                                            std::string line3;
                                                                                                            std::getline(is3, line3);
                                                                                                            try {
                                                                                                                auto j3 = json::parse(line3);
                                                                                                                if (j3.contains("method") && j3["method"] == "mining.notify") {
                                                                                                                    std::string job_id = j3["params"][0].get<std::string>();
                                                                                                                    log_.info(std::string("Stratum listen: NOTIFY job_id=") + job_id);
                                                                                                                } else {
                                                                                                                    log_.info(std::string("Stratum listen: <- ") + line3);
                                                                                                                }
                                                                                                            } catch (...) {
                                                                                                                log_.debug(std::string("Stratum listen: unparseable: ") + line3);
                                                                                                            }
                                                                                                            (*read_notify)();
                                                                                                        }
                                                                                                    }
                                                                                                );
                                                                                            };
                                                                                            (*read_notify)();
                                                                                        } else {
                                                                                            err_msg = "authorize failed";
                                                                                        }
                                                                                    } else {
                                                                                        (*read_until_auth)();
                                                                                    }
                                                                                } catch (...) {
                                                                                    (*read_until_auth)();
                                                                                }
                                                                            } else if (!timed_out.load()) {
                                                                                err_msg = std::string("read: ") + ec2.message();
                                                                            }
                                                                        }
                                                                    );
                                                                };
                                                                (*read_until_auth)();
                                                            }
                                                        );
                                                    } else {
                                                        err_msg = "subscribe format invalid";
                                                    }
                                                } catch (const json::exception& e) {
                                                    err_msg = std::string("JSON: ") + e.what();
                                                }
                                            } else if (!timed_out.load()) {
                                                err_msg = std::string("read subscribe: ") + ec_read.message();
                                            }
                                        }
                                    );
                                }
                            );
                        } else if (!timed_out.load()) {
                            err_msg = std::string("connect: ") + ec2.message();
                        }
                    }
                );
            }
        );

        ioc.run();

        if (listen_done.load()) {
            return true;
        }
        if (timed_out.load()) {
            log_.error("Stratum listen failed: timeout during handshake");
        } else if (!err_msg.empty()) {
            log_.error(std::string("Stratum listen failed: ") + err_msg);
        }
        return false;
    } catch (const std::exception& e) {
        log_.error(std::string("Stratum listen exception: ") + e.what());
        return false;
    }
}

} // namespace ohmy::pool
