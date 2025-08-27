#ifndef POOL_CONNECTION_H
#define POOL_CONNECTION_H

#include <string>
#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include "thread_safe_queue.h" // Inclui a fila
#include "mining_job.h"      // Inclui a struct do job

namespace asio = boost::asio;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

class PoolConnection {
public:
    // O construtor agora recebe uma referência para a fila de jobs
    PoolConnection(const std::string& host, uint16_t port, ThreadSafeQueue<MiningJob>& job_queue);

    // O método run() conterá o loop principal de rede
    void run(const std::string& user, const std::string& pass);
    void close();

private:
    bool connect();
    bool handshake(const std::string& user, const std::string& pass);
    bool write_json(const json& j);
    json read_json();

    std::string host_;
    uint16_t port_;
    ThreadSafeQueue<MiningJob>& job_queue_; // Referência para a fila

    asio::io_context io_context_;
    tcp::socket socket_;
    asio::streambuf buffer_;

    std::string session_id_;
    std::string extranonce1_;
    int extranonce2_size_;
};

#endif // POOL_CONNECTION_H