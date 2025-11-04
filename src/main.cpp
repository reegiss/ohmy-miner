/*
 * ohmy-miner bootstrap
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */

#include <iostream>
#include <string>
#include <optional>

struct CmdArgs {
    std::string algo;
    std::string url;
    std::string user;
    std::string pass{"x"};
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --algo qhash --url host:port --user WALLET[.RIG] --pass x\n\n"
              << "Options:\n"
              << "  --algo qhash            Mining algorithm (only 'qhash' supported)\n"
              << "  --url HOST:PORT         Pool URL (e.g., qubitcoin.luckypool.io:8610)\n"
              << "  --user WALLET[.RIG]     Wallet address and optional rig name\n"
              << "  --pass PASSWORD         Pool password (default: x)\n"
              << std::endl;
}

static std::optional<CmdArgs> parse_args(int argc, char** argv) {
    CmdArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 < argc) return std::string(argv[++i]);
            return {};
        };
        if (a == "--algo") {
            args.algo = next();
        } else if (a == "--url") {
            args.url = next();
        } else if (a == "--user") {
            args.user = next();
        } else if (a == "--pass") {
            args.pass = next();
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            return std::nullopt;
        }
    }
    if (args.algo != "qhash" || args.url.empty() || args.user.empty()) {
        std::cerr << "Missing or invalid required options.\n";
        print_usage(argv[0]);
        return std::nullopt;
    }
    return args;
}

int main(int argc, char** argv) {
    auto parsed = parse_args(argc, argv);
    if (!parsed.has_value()) {
        return 1;
    }
    const auto& args = *parsed;

    std::cout << "ohmy-miner starting...\n"
              << "  algo    : " << args.algo << "\n"
              << "  url     : " << args.url << "\n"
              << "  user    : " << args.user << "\n"
              << "  pass    : " << args.pass << "\n"
              << "(bootstrap mode: networking and CUDA kernels to be added)\n";

    // TODO: initialize CUDA, connect to pool (Stratum), start mining workers

    return 0;
}
