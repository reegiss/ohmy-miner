#ifndef POOL_CONNECTION_H
#define POOL_CONNECTION_H

#include <string>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include "thread_safe_queue.h"
#include "mining_job.h"
#include "found_share.h"

namespace asio = boost::asio;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

class PoolConnection {
public:
    PoolConnection(const std::string& host, uint16_t port,
                   ThreadSafeQueue<MiningJob>& job_queue,
                   ThreadSafeQueue<FoundShare>& result_queue);

    void run(const std::string& user, const std::string& pass);
    void close();

private:
    // --- Métodos Síncronos (para inicialização) ---
    bool connect();
    bool handshake(const std::string& user, const std::string& pass);
    bool write_json(const json& j);
    json read_json();

    // --- MÉTODOS ASSÍNCRONOS (para o loop principal) ---
    void start_async_read();
    void handle_read(const boost::system::error_code& ec, std::size_t bytes_transferred);
    void check_submit_queue(const std::string& user);
    void process_pool_message(const json& msg);
    // ---------------------------------------------------------

    std::string host_;
    uint16_t port_;
    ThreadSafeQueue<MiningJob>& job_queue_;
    ThreadSafeQueue<FoundShare>& result_queue_;

    asio::io_context io_context_;
    tcp::socket socket_;
    asio::streambuf buffer_;

    std::string extranonce1_;
    int extranonce2_size_;
};

#endif // POOL_CONNECTION_H