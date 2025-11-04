/*
 * ohmy-miner bootstrap
 * Copyright (C) 2025 Regis Araujo Melo
 * GPL-3.0-only
 */
// Keep main minimal by delegating responsibilities to separate modules

#include <fmt/core.h>

#include <ohmy/cli/args.hpp>
#include <ohmy/config/loader.hpp>
#include <ohmy/system/cuda.hpp>
#include <ohmy/logging/fmt_logger.hpp>
#include <ohmy/pool/stratum.hpp>

int main(int argc, char** argv) {
    ohmy::logging::FmtLogger logger;
    // 1) Parse CLI
    auto pr = ohmy::cli::parse(argc, argv, logger);
    // Enable verbose logging as early as possible
    logger.set_debug(pr.debug);
    if (pr.debug) logger.debug("debug logging enabled");
    if (pr.show_only) return 0;
    if (!pr.cfg.has_value()) return 1;

    // 2) Merge config file then env, keeping CLI precedence
    auto cfg = *pr.cfg;
    auto file_errs = ohmy::config::load_from_file(cfg, pr.config_path);
    for (const auto& e : file_errs) logger.error(fmt::format("config: {}", e));
    ohmy::config::apply_env_overrides(cfg);

    // 3) Final validation
    auto verrs = ohmy::config::validate_final(cfg);
    if (!verrs.empty()) {
        for (const auto& e : verrs) logger.error(e);
        return 1;
    }

    // 4) Startup banner
    logger.info(fmt::format("ohmy-miner v{}", OHMY_MINER_VERSION));
    logger.info(fmt::format("  algo    : {}", cfg.algo));
    logger.info(fmt::format("  url     : {}", cfg.url));
    logger.info(fmt::format("  user    : {}", cfg.user));
    logger.info(fmt::format("  pass    : {}", cfg.pass));

    // 4.1) Optional one-shot Stratum connectivity probe
    if (pr.stratum_connect) {
        auto pos = cfg.url.rfind(':');
        if (pos == std::string::npos || pos == cfg.url.size()-1) {
            logger.error("--stratum-connect: url deve ser host:port");
            return 1;
        }
        ohmy::pool::StratumOptions sopts;
        sopts.host = cfg.url.substr(0, pos);
        sopts.port = cfg.url.substr(pos + 1);
        sopts.user = cfg.user;
        sopts.pass = cfg.pass;
        ohmy::pool::StratumClient client(logger, std::move(sopts));
        bool ok = client.probe_connect();
        return ok ? 0 : 2;
    }

    // 4.2) Optional Stratum listen mode (keeps connection open for mining.notify)
    if (pr.stratum_listen) {
        auto pos = cfg.url.rfind(':');
        if (pos == std::string::npos || pos == cfg.url.size()-1) {
            logger.error("--stratum-listen: url deve ser host:port");
            return 1;
        }
        ohmy::pool::StratumOptions sopts;
        sopts.host = cfg.url.substr(0, pos);
        sopts.port = cfg.url.substr(pos + 1);
        sopts.user = cfg.user;
        sopts.pass = cfg.pass;
        ohmy::pool::StratumClient client(logger, std::move(sopts));
        bool ok = client.listen_mode(10); // Listen for 10 seconds
        return ok ? 0 : 2;
    }

    // 5) CUDA info (best-effort)
    ohmy::system::print_cuda_info(logger);

    logger.info("(bootstrap mode: networking and CUDA kernels to be added)");
    return 0;
}

