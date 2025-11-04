#include <ohmy/config/validator.hpp>

#include <algorithm>
#include <cctype>

namespace ohmy::config {

bool is_valid_hostname(const std::string& host, std::string& err) {
    if (host.empty()) { err = "hostname vazio"; return false; }
    if (host.size() > 253) { err = "hostname muito longo (>253)"; return false; }
    std::size_t start = 0;
    while (start < host.size()) {
        auto dot = host.find('.', start);
        std::size_t end = (dot == std::string::npos) ? host.size() : dot;
        std::size_t len = end - start;
        if (len == 0) { err = "label de hostname vazia"; return false; }
        if (len > 63) { err = "label do hostname muito longa (>63)"; return false; }
        if (host[start] == '-' || host[end-1] == '-') { err = "label do hostname não pode começar/terminar com '-'"; return false; }
        for (std::size_t i = start; i < end; ++i) {
            char c = host[i];
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-')) {
                err = "hostname contém caracteres inválidos"; return false; }
        }
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return true;
}

bool validate_host_port(const std::string& url, std::string& err) {
    auto pos = url.rfind(':');
    if (pos == std::string::npos || pos == url.size() - 1) {
        err = "url deve ser no formato host:port"; return false; }
    std::string host = url.substr(0, pos);
    std::string port_s = url.substr(pos + 1);
    if (!std::all_of(port_s.begin(), port_s.end(), ::isdigit)) {
        err = "porta em url deve conter apenas dígitos"; return false; }
    unsigned long port = 0;
    try { port = std::stoul(port_s); } catch (...) { err = "porta inválida"; return false; }
    if (port == 0 || port > 65535UL) { err = "porta fora do intervalo (1-65535)"; return false; }
    if (!is_valid_hostname(host, err)) return false;
    return true;
}

} // namespace ohmy::config
