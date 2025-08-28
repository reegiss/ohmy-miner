#include "pool_connection.h"
#include "found_share.h"
#include <iostream>
#include <istream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>

// Flag global de desligamento, declarada em main.cpp
extern std::atomic<bool> g_shutdown;

PoolConnection::PoolConnection(const std::string& host, uint16_t port,
                               ThreadSafeQueue<MiningJob>& job_queue,
                               ThreadSafeQueue<FoundShare>& result_queue)
    : host_(host),
      port_(port),
      job_queue_(job_queue),
      result_queue_(result_queue),
      io_context_(),
      socket_(io_context_) {}

// --- O loop principal, agora totalmente assíncrono ---
void PoolConnection::run(const std::string& user, const std::string& pass) {
    if (!connect() || !handshake(user, pass)) {
        std::cerr << "Falha ao inicializar a conexão com a pool. Encerrando a thread de rede." << std::endl;
        g_shutdown = true;
        return;
    }

    std::cout << "[NETWORK] Handshake completo. Escutando por trabalhos..." << std::endl;

    start_async_read(); // Inicia o loop de leitura assíncrona
    check_submit_queue(user); // Inicia o loop de verificação de submissão

    io_context_.run(); // Bloqueia até que o io_context seja parado

    std::cout << "[NETWORK] Thread de rede encerrando." << std::endl;
}

// --- Novos métodos assíncronos ---

void PoolConnection::start_async_read() {
    asio::async_read_until(socket_, buffer_, '\n',
        [this](const boost::system::error_code& ec, std::size_t bytes_transferred) {
            handle_read(ec, bytes_transferred);
        });
}

void PoolConnection::handle_read(const boost::system::error_code& ec, std::size_t bytes_transferred) {
    if (!ec && bytes_transferred > 0) {
        std::istream is(&buffer_);
        std::string line;
        std::getline(is, line);

        std::cout << "[POOL] <- " << line << std::endl;

        try {
            json msg = json::parse(line);
            process_pool_message(msg);
        } catch (const json::parse_error& e) {
            std::cerr << "Erro de parsing JSON: " << e.what() << std::endl;
        }

        if (!g_shutdown) {
            start_async_read(); // Continua o loop de leitura
        }
    } else {
        if (ec != asio::error::eof && ec) {
            std::cerr << "Erro ao ler do socket: " << ec.message() << std::endl;
        } else {
            std::cout << "[NETWORK] Conexão fechada pela pool." << std::endl;
        }
        g_shutdown = true; // Sinaliza o desligamento em qualquer erro/fechamento
    }
}

void PoolConnection::check_submit_queue(const std::string& user) {
    if (g_shutdown) return;

    FoundShare share;
    if (result_queue_.try_pop(share)) {
        std::cout << "[NETWORK] Submetendo share encontrado para o trabalho " << share.job_id << "..." << std::endl;
        json submit_req = {
            {"method", "mining.submit"},
            {"params", {user, share.job_id, share.extranonce2, share.ntime, share.nonce_hex}},
            {"id", 4} // Em um minerador real, este ID deve ser um contador
        };
        write_json(submit_req);
    }

    // Agenda a próxima verificação para criar um loop não-bloqueante
    asio::post(io_context_, [this, user]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Verifica a cada 50ms
        check_submit_queue(user);
    });
}

void PoolConnection::process_pool_message(const json& msg) {
    static uint32_t extranonce2_counter = 0;

    if (msg.value("method", "") == "mining.notify") {
        std::cout << "[NETWORK] Recebido novo trabalho de mineração." << std::endl;
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
            job.extranonce1 = extranonce1_;

            std::stringstream ss;
            ss << std::hex << std::setw(extranonce2_size_ * 2) << std::setfill('0') << extranonce2_counter++;
            job.extranonce2 = ss.str();
            job_queue_.push(job);
        } catch (const json::exception& e) {
            std::cerr << "[NETWORK] Erro ao processar mining.notify: " << e.what() << std::endl;
        }
    } else if (msg.value("method", "") == "mining.set_difficulty") {
        try {
            double new_diff = msg["params"][0].get<double>();
            std::cout << "[NETWORK] Pool definiu nova dificuldade: " << new_diff << std::endl;
            // TODO: Lógica para atualizar a dificuldade do minerador
        } catch (const json::exception& e) {
            std::cerr << "[NETWORK] Erro ao processar set_difficulty: " << e.what() << std::endl;
        }
    } else if (msg.contains("id") && msg.contains("result")) {
        bool result_ok = msg.value("result", false);
        if (result_ok) {
            std::cout << "✅ [NETWORK] Share ACEITO pela pool." << std::endl;
        } else {
            auto error = msg.value("error", json::array());
            std::cerr << "❌ [NETWORK] Share REJEITADO pela pool. Motivo: " << error.dump() << std::endl;
        }
    }
}

// --- Métodos Síncronos (para inicialização) ---

bool PoolConnection::connect() {
    try {
        tcp::resolver resolver(io_context_);
        boost::system::error_code ec;
        auto endpoints = resolver.resolve(host_, std::to_string(port_), ec);
        if (ec) {
            std::cerr << "Erro: Não foi possível resolver o host '" << host_ << "': " << ec.message() << std::endl;
            return false;
        }
        std::cout << "Conectando a " << host_ << ":" << port_ << "..." << std::endl;
        asio::connect(socket_, endpoints, ec);
        if (ec) {
            std::cerr << "Erro: Não foi possível conectar à pool: " << ec.message() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Uma exceção ocorreu durante a conexão: " << e.what() << std::endl;
        return false;
    }
    std::cout << "✅ Conectado com sucesso à pool!" << std::endl;
    return true;
}

bool PoolConnection::write_json(const json& j) {
    std::string message = j.dump() + "\n";
    std::cout << "[CLIENT] -> " << message;
    boost::system::error_code ec;
    asio::write(socket_, asio::buffer(message), ec);
    if (ec) {
        std::cerr << "Erro ao escrever no socket: " << ec.message() << std::endl;
        return false;
    }
    return true;
}

json PoolConnection::read_json() {
    boost::system::error_code ec;
    asio::read_until(socket_, buffer_, '\n', ec);
    if (ec) {
        return nullptr;
    }
    std::istream is(&buffer_);
    std::string line;
    std::getline(is, line);
    std::cout << "[POOL] <- " << line << std::endl;
    try {
        return json::parse(line);
    } catch (const json::parse_error& e) {
        return nullptr;
    }
}

bool PoolConnection::handshake(const std::string& user, const std::string& pass) {
    try {
        json subscribe_req = {{"id", 1}, {"method", "mining.subscribe"}, {"params", {"qtcminer/0.1"}}};
        if (!write_json(subscribe_req)) return false;
        json subscribe_res = read_json();
        if (subscribe_res.is_null() || subscribe_res.value("error", json(nullptr)) != nullptr) return false;
        auto result = subscribe_res["result"];
        extranonce1_ = result[1].get<std::string>();
        extranonce2_size_ = result[2].get<int>();
        std::cout << "Inscrição bem-sucedida. Extranonce1: " << extranonce1_ << std::endl;

        json authorize_req = {{"id", 2}, {"method", "mining.authorize"}, {"params", {user, pass}}};
        if (!write_json(authorize_req)) return false;
        json authorize_res = read_json();
        while (true) {
            if (authorize_res.is_null()) return false;
            if (authorize_res.contains("id") && authorize_res["id"].is_number() && authorize_res["id"].get<int>() == 2) break;
            process_pool_message(authorize_res);
            authorize_res = read_json();
        }
        if (authorize_res.value("error", json(nullptr)) != nullptr || !authorize_res.value("result", false)) return false;
    } catch (const std::exception& e) {
        std::cerr << "Exceção durante o handshake: " << e.what() << std::endl;
        return false;
    }
    std::cout << "✅ Autorização bem-sucedida!" << std::endl;
    return true;
}

void PoolConnection::close() {
    asio::post(io_context_, [this]() {
        if (socket_.is_open()) {
            boost::system::error_code ec;
            socket_.shutdown(tcp::socket::shutdown_both, ec);
            socket_.close(ec);
        }
        io_context_.stop();
    });
}