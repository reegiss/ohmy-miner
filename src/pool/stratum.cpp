#include <ohmy/pool/stratum.hpp>
#include <ohmy/pool/stratum_messages.hpp>
#include <ohmy/pool/mining_job.hpp>

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

// Internal helper: perform Stratum handshake (subscribe + authorize)
// Calls on_complete with result when done (or on timeout/error)
static void do_handshake_async(ohmy::logging::Logger& log,
                               const StratumOptions& opts,
                               asio::ip::tcp::socket& socket,
                               std::atomic<bool>& timed_out,
                               std::function<void(HandshakeResult)> on_complete) {
    // Send mining.subscribe
    auto sub_line = stratum_messages::build_subscribe(opts.client, 1);
    log.info("Sending subscribe...");
    
    asio::async_write(socket, asio::buffer(sub_line),
        [&log, &opts, &socket, &timed_out, on_complete](const std::error_code& ec, std::size_t){
            if (ec) {
                if (!timed_out.load()) {
                    HandshakeResult result;
                    result.error_msg = "write subscribe: " + ec.message();
                    on_complete(result);
                }
                return;
            }
            
            // Read subscribe response
            auto buf = std::make_shared<asio::streambuf>();
            asio::async_read_until(socket, *buf, '\n',
                [&log, &opts, &socket, &timed_out, on_complete, buf](const std::error_code& ec, std::size_t){
                    if (ec) {
                        if (!timed_out.load()) {
                            HandshakeResult result;
                            result.error_msg = "read subscribe: " + ec.message();
                            on_complete(result);
                        }
                        return;
                    }
                    
                    std::istream is(buf.get());
                    std::string line;
                    std::getline(is, line);
                    log.info(std::string("Received: ") + line);
                    
                    try {
                        auto j = json::parse(line);
                        if (j.contains("error") && !j["error"].is_null()) {
                            log.error(std::string("Subscribe error: ") + j["error"].dump());
                            HandshakeResult result;
                            result.error_msg = "subscribe returned error";
                            on_complete(result);
                            return;
                        }
                        if (j.contains("result") && j["result"].is_array() && j["result"].size() >= 3) {
                            std::string extranonce1 = j["result"][1].get<std::string>();
                            int extranonce2_size = j["result"][2].get<int>();
                            log.info(std::string("Subscribe OK: extranonce1=") + extranonce1
                                      + ", extranonce2_size=" + std::to_string(extranonce2_size));
                            
                            // Now send authorize
                            auto auth_line = stratum_messages::build_authorize(opts.user, opts.pass, 2);
                            log.info("Sending authorize...");
                            
                            asio::async_write(socket, asio::buffer(auth_line),
                                [&log, &socket, &timed_out, on_complete, extranonce1, extranonce2_size]
                                (const std::error_code& ec, std::size_t){
                                    if (ec) {
                                        if (!timed_out.load()) {
                                            HandshakeResult result;
                                            result.error_msg = "write authorize: " + ec.message();
                                            on_complete(result);
                                        }
                                        return;
                                    }
                                    
                                    // Read authorize response (may need to skip unsolicited messages)
                                    auto buf2 = std::make_shared<asio::streambuf>();
                                    auto read_next = std::make_shared<std::function<void()>>();
                                    
                                    *read_next = [&log, &socket, &timed_out, on_complete, 
                                                  extranonce1, extranonce2_size, buf2, read_next](){
                                        asio::async_read_until(socket, *buf2, '\n',
                                            [&log, &timed_out, on_complete, extranonce1, extranonce2_size, 
                                             buf2, read_next](const std::error_code& ec, std::size_t){
                                                if (ec) {
                                                    if (!timed_out.load()) {
                                                        HandshakeResult result;
                                                        result.error_msg = "read authorize: " + ec.message();
                                                        on_complete(result);
                                                    }
                                                    return;
                                                }
                                                
                                                std::istream is2(buf2.get());
                                                std::string line2;
                                                std::getline(is2, line2);
                                                log.info(std::string("<- ") + line2);
                                                
                                                try {
                                                    auto j2 = json::parse(line2);
                                                    if (j2.contains("id") && j2["id"].is_number_integer() && 
                                                        j2["id"].get<int>() == 2) {
                                                        // This is the authorize response
                                                        if (j2.contains("error") && !j2["error"].is_null()) {
                                                            log.error(std::string("Authorize error: ") + j2["error"].dump());
                                                            HandshakeResult result;
                                                            result.error_msg = "authorize returned error";
                                                            on_complete(result);
                                                            return;
                                                        }
                                                        if (j2.contains("result") && j2["result"].is_boolean() && 
                                                            j2["result"].get<bool>()) {
                                                            log.info("Authorize OK");
                                                            HandshakeResult result;
                                                            result.success = true;
                                                            result.extranonce1 = extranonce1;
                                                            result.extranonce2_size = extranonce2_size;
                                                            on_complete(result);
                                                        } else {
                                                            log.error("Authorize: result is not true");
                                                            HandshakeResult result;
                                                            result.error_msg = "authorize result failed";
                                                            on_complete(result);
                                                        }
                                                    } else {
                                                        // Not the authorize response, read next
                                                        log.debug("Ignoring unsolicited message, reading next...");
                                                        (*read_next)();
                                                    }
                                                } catch (const json::exception& e) {
                                                    log.error(std::string("Parse error: ") + e.what());
                                                    HandshakeResult result;
                                                    result.error_msg = "JSON parse: " + std::string(e.what());
                                                    on_complete(result);
                                                }
                                            }
                                        );
                                    };
                                    (*read_next)();
                                }
                            );
                        } else {
                            log.error("Subscribe: unexpected result format");
                            HandshakeResult result;
                            result.error_msg = "subscribe result invalid";
                            on_complete(result);
                        }
                    } catch (const json::exception& e) {
                        log.error(std::string("Subscribe parse error: ") + e.what());
                        HandshakeResult result;
                        result.error_msg = "JSON parse: " + std::string(e.what());
                        on_complete(result);
                    }
                }
            );
        }
    );
}

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
    try {
        asio::io_context ioc;
        asio::ip::tcp::resolver resolver{ioc};
        asio::ip::tcp::socket socket{ioc};
        asio::steady_timer timer{ioc};

        const auto timeout = 3s;
        std::atomic<bool> timed_out{false};
        std::atomic<bool> success{false};
        std::string err_msg;

        log_.info(std::string("Stratum probe: resolving ") + opts_.host + ":" + opts_.port);

        // Setup timeout
        timer.expires_after(timeout);
        timer.async_wait([&](const std::error_code& ec){
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
                    if (!timed_out.load()) err_msg = "resolve: " + ec.message();
                    return;
                }
                log_.info("Stratum probe: connecting...");
                asio::async_connect(socket, results,
                    [&](const std::error_code& ec2, const asio::ip::tcp::endpoint&){
                        if (ec2) {
                            if (!timed_out.load()) err_msg = "connect: " + ec2.message();
                            return;
                        }
                        log_.info("Stratum probe: connected");
                        
                        // Perform handshake (subscribe + authorize)
                        do_handshake_async(log_, opts_, socket, timed_out, [&](HandshakeResult result){
                            if (result.success) {
                                success.store(true);
                                std::error_code ignored;
                                timer.cancel(ignored);
                                socket.close(ignored);
                            } else {
                                if (!timed_out.load()) err_msg = result.error_msg;
                            }
                        });
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
            log_.error(std::string("Stratum probe failed: ") + 
                      (err_msg.empty() ? "unknown" : err_msg));
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
                                                                                                                    if (j3.contains("params") && j3["params"].is_array()) {
                                                                                                                        auto job_opt = parse_mining_notify(j3["params"].get<std::vector<json>>());
                                                                                                                        if (job_opt) {
                                                                                                                            const auto& job = *job_opt;
                                                                                                                            log_.info(std::string("Stratum listen: NOTIFY job_id=") + job.job_id 
                                                                                                                                      + ", version=" + job.version
                                                                                                                                      + ", nbits=" + job.nbits
                                                                                                                                      + ", ntime=" + job.ntime
                                                                                                                                      + ", clean=" + (job.clean_jobs ? "true" : "false")
                                                                                                                                      + ", merkle_branches=" + std::to_string(job.merkle_branch.size()));
                                                                                                                            log_.debug(std::string("  prev_hash=") + job.prev_hash);
                                                                                                                            log_.debug(std::string("  coinbase1_len=") + std::to_string(job.coinbase1.size()/2) + " bytes");
                                                                                                                            log_.debug(std::string("  coinbase2_len=") + std::to_string(job.coinbase2.size()/2) + " bytes");
                                                                                                                        } else {
                                                                                                                            log_.error("Stratum listen: failed to parse mining.notify params");
                                                                                                                        }
                                                                                                                    } else {
                                                                                                                        log_.error("Stratum listen: mining.notify missing params array");
                                                                                                                    }
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
