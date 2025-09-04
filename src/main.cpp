// src/main.cpp

/*
 * Copyright (C) 2025 Regis Araujo Melo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <string>
#include <optional>

#include <fmt/core.h>
#include <fmt/color.h>
#include <cuda_runtime.h>

#include "miner/Config.hpp"
#include "miner/Device.hpp"
#include "miner/IAlgorithm.hpp"

void print_welcome_message();

int main(int argc, char** argv) {
    print_welcome_message();

    std::optional<miner::Config> config_opt = miner::parse_arguments(argc, argv);
    if (!config_opt.has_value()) {
        // Error message was already printed by the parser, or user requested help.
        // A non-zero exit code indicates an issue, while 0 is for clean exits like --help.
        return 1;
    }

    const auto& config = config_opt.value();

    fmt::print(fg(fmt::color::green), "Configuration loaded:\n");
    fmt::print("  - Algorithm: {}\n", config.algo);
    fmt::print("  - Pool URL:  {}\n", config.url);
    fmt::print("  - User:      {}\n", config.user);
    fmt::print("  - Pass:      {}\n", std::string(config.pass.length(), '*')); // Obfuscate password in log

    try {
        miner::DeviceManager device_manager;
        device_manager.list_devices();

        // Future steps:
        // 1. Select device(s) based on config.
        // 2. Load algorithm plugin.
        // 3. Initialize Stratum client.
        // 4. Start mining loop.

    } catch (const std::runtime_error& e) {
        fmt::print(stderr, fg(fmt::color::red), "A critical CUDA error occurred: {}\n", e.what());
        return 1;
    }

    fmt::print("\nInitialization complete. Exiting for now.\n");

    return 0;
}

void print_welcome_message() {
    // Get compiler info
    std::string compiler_info;
#if defined(__clang__)
    compiler_info = fmt::format("Clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    compiler_info = fmt::format("GCC {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    compiler_info = "Unknown C++ Compiler";
#endif

    // Get CUDA version info from the runtime header
    std::string cuda_version_str = fmt::format("{}.{}", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

    // Print formatted welcome message
    fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold, " * Oh My Miner v0.1.0\n");
    fmt::print(" * ------------------\n");
    fmt::print(" * License:      GPL-3.0\n");
    fmt::print(" * Build:        {} {} ({})\n", __DATE__, __TIME__, compiler_info);
    fmt::print(" * CUDA Version: {}\n", cuda_version_str);
    fmt::print(" * Dev Fee:      This software has a 2% developer fee.\n");
    fmt::print(" *               (2 minute of mining every 100 minutes)\n");
    fmt::print(fg(fmt::color::yellow), " * INFO:         This is pre-alpha software. Use at your own risk.\n");
    fmt::print("\n");
}