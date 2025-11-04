#include <ohmy/config/loader.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <ohmy/config/validator.hpp>

namespace ohmy::config {

static void load_key_value(MinerConfig& cfg, const std::string& text) {
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (key == "algo") cfg.algo = val;
        else if (key == "url") cfg.url = val;
        else if (key == "user") cfg.user = val;
        else if (key == "pass") cfg.pass = val;
    }
}

std::vector<std::string> load_from_file(MinerConfig& cfg, const std::string& path) {
    std::vector<std::string> errs;
    std::ifstream in(path);
    if (!in.good()) return errs; // optional

    std::stringstream buffer; buffer << in.rdbuf();
    std::string text = buffer.str();
    auto first_non_space = text.find_first_not_of(" \t\n\r");
    if (first_non_space == std::string::npos) return errs;

    if (text[first_non_space] == '{') {
        // JSON
        try {
            nlohmann::json j = nlohmann::json::parse(text);
            auto expect_string = [&](const char* key) {
                if (j.contains(key) && !j.at(key).is_string()) {
                    errs.push_back(fmt::format("'{}' deve ser string", key));
                }
            };
            expect_string("algo");
            expect_string("url");
            expect_string("user");
            expect_string("pass");
            if (j.contains("algo") && j.at("algo").is_string()) {
                std::string a = j.at("algo").get<std::string>();
                if (a != "qhash") errs.push_back("algo deve ser 'qhash'");
            }
            if (j.contains("url") && j.at("url").is_string()) {
                std::string u = j.at("url").get<std::string>();
                std::string e;
                if (!validate_host_port(u, e)) errs.push_back(e);
            }
            if (!errs.empty()) return errs;
            if (j.contains("algo")) cfg.algo = j.at("algo").get<std::string>();
            if (j.contains("url"))  cfg.url  = j.at("url").get<std::string>();
            if (j.contains("user")) cfg.user = j.at("user").get<std::string>();
            if (j.contains("pass")) cfg.pass = j.at("pass").get<std::string>();
        } catch (const std::exception& ex) {
            errs.push_back(fmt::format("Falha ao ler miner.conf: {}", ex.what()));
        }
    } else {
        load_key_value(cfg, text);
    }
    return errs;
}

void apply_env_overrides(MinerConfig& cfg) {
    if (const char* v = std::getenv("OMM_ALGO")) cfg.algo = v;
    if (const char* v = std::getenv("OMM_URL"))  cfg.url  = v;
    if (const char* v = std::getenv("OMM_USER")) cfg.user = v;
    if (const char* v = std::getenv("OMM_PASS")) cfg.pass = v;
}

std::vector<std::string> validate_final(const MinerConfig& cfg) {
    std::vector<std::string> errs;
    if (cfg.algo != "qhash") errs.push_back("algo deve ser 'qhash'");
    if (cfg.url.empty()) errs.push_back("url é obrigatório");
    if (cfg.user.empty()) errs.push_back("user é obrigatório");
    if (!cfg.url.empty()) {
        std::string e;
        if (!validate_host_port(cfg.url, e)) errs.push_back(e);
    }
    return errs;
}

} // namespace ohmy::config
