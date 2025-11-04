/*
 * ohmy-miner bootstrap
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#include <optional>
#include <string>

#include <cxxopts.hpp>
#include <fmt/core.h>

struct CmdArgs {
    std::string algo;
    std::string url;
    std::string user;
    std::string pass{"x"};
};

static std::optional<CmdArgs> parse_args(int argc, char** argv) {
    cxxopts::Options options("ohmy-miner", "GPU miner for Qubitcoin (qhash)");
    // clang-format off
    options.add_options()
        ("algo", "Mining algorithm (only 'qhash' supported)", cxxopts::value<std::string>()->default_value("qhash"))
        ("url",  "Pool URL (host:port)", cxxopts::value<std::string>())
        ("user", "Wallet[.RIG]", cxxopts::value<std::string>())
        ("pass", "Pool password", cxxopts::value<std::string>()->default_value("x"))
        ("v,version", "Show version and exit")
        ("h,help",    "Show help and exit");
    // clang-format on

    CmdArgs out;
    try {
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return std::nullopt;
        }
        if (result.count("version")) {
            fmt::print("ohmy-miner v{}\n", OHMY_MINER_VERSION);
            return std::nullopt;
        }
        out.algo = result["algo"].as<std::string>();
        out.url  = result.count("url") ? result["url"].as<std::string>() : std::string{};
        out.user = result.count("user") ? result["user"].as<std::string>() : std::string{};
        out.pass = result["pass"].as<std::string>();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Argument error: {}\n\n{}\n", e.what(), options.help());
        return std::nullopt;
    }

    if (out.algo != "qhash" || out.url.empty() || out.user.empty()) {
        fmt::print(stderr, "Missing or invalid required options.\n\n{}\n", options.help());
        return std::nullopt;
    }
    return out;
}

int main(int argc, char** argv) {
    auto parsed = parse_args(argc, argv);
    if (!parsed.has_value()) {
        return 1;
    }
    const auto& args = *parsed;

    fmt::print("ohmy-miner v{}\n", OHMY_MINER_VERSION);
    fmt::print("  algo    : {}\n", args.algo);
    fmt::print("  url     : {}\n", args.url);
    fmt::print("  user    : {}\n", args.user);
    fmt::print("  pass    : {}\n", args.pass);
    fmt::print("(bootstrap mode: networking and CUDA kernels to be added)\n");

    // TODO: initialize CUDA, connect to pool (Stratum), start mining workers

    return 0;
}
