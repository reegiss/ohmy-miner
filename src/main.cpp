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

int main(int argc, char** argv) {
    ohmy::logging::FmtLogger logger;
    // 1) Parse CLI
    auto pr = ohmy::cli::parse(argc, argv, logger);
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

    // 5) CUDA info (best-effort)
    ohmy::system::print_cuda_info(logger);

    logger.info("(bootstrap mode: networking and CUDA kernels to be added)");
    return 0;
}

