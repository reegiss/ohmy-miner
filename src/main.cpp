/*
 * ohmy-miner bootstrap
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#include <optional>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <cxxopts.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

struct CmdArgs {
    std::string algo;
    std::string url;
    std::string user;
    std::string pass{"x"};
};

struct ParseResult {
    std::optional<CmdArgs> args; // present when we should continue
    bool show_only{false};       // true if --help/--version printed
};

static void apply_env_overrides(CmdArgs& cfg) {
    if (const char* v = std::getenv("OMM_ALGO")) cfg.algo = v;
    if (const char* v = std::getenv("OMM_URL"))  cfg.url  = v;
    if (const char* v = std::getenv("OMM_USER")) cfg.user = v;
    if (const char* v = std::getenv("OMM_PASS")) cfg.pass = v;
}

static void apply_config_file(CmdArgs& cfg, const std::string& path = "miner.conf") {
    std::ifstream in(path);
    if (!in.good()) return; // optional
    // Read whole file
    std::stringstream buffer; buffer << in.rdbuf();
    std::string text = buffer.str();
    // Trim leading spaces
    auto first_non_space = text.find_first_not_of(" \t\n\r");
    if (first_non_space == std::string::npos) return;
    if (text[first_non_space] == '{') {
        // JSON format
        try {
            nlohmann::json j = nlohmann::json::parse(text);
            if (j.contains("algo")) cfg.algo = j.at("algo").get<std::string>();
            if (j.contains("url"))  cfg.url  = j.at("url").get<std::string>();
            if (j.contains("user")) cfg.user = j.at("user").get<std::string>();
            if (j.contains("pass")) cfg.pass = j.at("pass").get<std::string>();
        } catch (const std::exception&) {
            // ignore parse errors silently for now
        }
    } else {
        // key=value format
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
}

static ParseResult parse_args(int argc, char** argv) {
    cxxopts::Options options("ohmy-miner", "GPU miner for Qubitcoin (qhash)");
    // clang-format off
    options.add_options()
        ("algo", "Mining algorithm (only 'qhash' supported)", cxxopts::value<std::string>()->default_value("qhash"))
        ("url",  "Pool URL (host:port)", cxxopts::value<std::string>())
        ("user", "Wallet[.RIG]", cxxopts::value<std::string>())
        ("pass", "Pool password", cxxopts::value<std::string>()->default_value("x"))
        ("config", "Path to config file (miner.conf)", cxxopts::value<std::string>()->default_value("miner.conf"))
        ("v,version", "Show version and exit")
        ("h,help",    "Show help and exit");
    // clang-format on

    ParseResult pr;
    CmdArgs out;
    try {
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            pr.show_only = true;
            return pr;
        }
        if (result.count("version")) {
            fmt::print("ohmy-miner v{}\n", OHMY_MINER_VERSION);
            pr.show_only = true;
            return pr;
        }
        out.algo = result["algo"].as<std::string>();
        out.url  = result.count("url") ? result["url"].as<std::string>() : std::string{};
        out.user = result.count("user") ? result["user"].as<std::string>() : std::string{};
        out.pass = result["pass"].as<std::string>();

        // Apply config file then env (lowest to higher precedence before CLI overrides)
        const auto cfg_path = result["config"].as<std::string>();
        apply_config_file(out, cfg_path);
        apply_env_overrides(out);

        // Finally, CLI args (already applied) have highest precedence
    } catch (const std::exception& e) {
        fmt::print(stderr, "Argument error: {}\n\n{}\n", e.what(), options.help());
        return pr; // .args empty, show_only=false -> failure
    }

    if (out.algo != "qhash" || out.url.empty() || out.user.empty()) {
        fmt::print(stderr, "Missing or invalid required options.\n\n{}\n", options.help());
        return pr; // invalid
    }
    pr.args = out;
    return pr;
}

int main(int argc, char** argv) {
    auto parsed = parse_args(argc, argv);
    if (parsed.show_only) {
        return 0;
    }
    if (!parsed.args.has_value()) {
        return 1;
    }
    const auto& args = *parsed.args;

    fmt::print("ohmy-miner v{}\n", OHMY_MINER_VERSION);
    fmt::print("  algo    : {}\n", args.algo);
    fmt::print("  url     : {}\n", args.url);
    fmt::print("  user    : {}\n", args.user);
    fmt::print("  pass    : {}\n", args.pass);

    // CUDA device info (best-effort)
    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr == cudaSuccess && device_count > 0) {
        fmt::print("CUDA devices: {}\n", device_count);
        for (int d = 0; d < device_count; ++d) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
                fmt::print("  [{}] {} (SM {}.{}, {} MB)\n", d, prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));
            }
        }
    } else {
        fmt::print("CUDA not available or no devices found.\n");
    }
    fmt::print("(bootstrap mode: networking and CUDA kernels to be added)\n");

    // TODO: initialize CUDA, connect to pool (Stratum), start mining workers

    return 0;
}
