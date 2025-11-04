#include <ohmy/cli/args.hpp>

#include <string>

#include <cxxopts.hpp>
#include <fmt/core.h>

#ifndef OHMY_MINER_VERSION
#define OHMY_MINER_VERSION "0.0.0"
#endif

namespace ohmy::cli {

ohmy::config::ParseResult parse(int argc, char** argv) {
    ohmy::config::ParseResult pr;
    cxxopts::Options options("ohmy-miner", "GPU miner for Qubitcoin (qhash)");
    options.add_options()
        ("algo",   "Mining algorithm (only 'qhash' supported)", cxxopts::value<std::string>()->default_value("qhash"))
        ("url",    "Pool URL (host:port)", cxxopts::value<std::string>())
        ("user",   "Wallet[.RIG]", cxxopts::value<std::string>())
        ("pass",   "Pool password", cxxopts::value<std::string>()->default_value("x"))
        ("config", "Path to config file (miner.conf)", cxxopts::value<std::string>()->default_value("miner.conf"))
        ("v,version", "Show version and exit")
        ("h,help",    "Show help and exit");

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
        ohmy::config::MinerConfig cfg;
        cfg.algo = result["algo"].as<std::string>();
        if (result.count("url"))  cfg.url  = result["url"].as<std::string>();
        if (result.count("user")) cfg.user = result["user"].as<std::string>();
        cfg.pass = result["pass"].as<std::string>();
        pr.config_path = result["config"].as<std::string>();
        pr.cfg = cfg;
    } catch (const std::exception& e) {
        fmt::print(stderr, "Argument error: {}\n\n{}\n", e.what(), options.help());
        return pr;
    }
    return pr;
}

} // namespace ohmy::cli
