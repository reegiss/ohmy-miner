// src/main.cpp

/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "miner/device.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <optional>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>

#include <cuda_runtime.h>

#include "miner/Config.hpp"

#include "miner/net.hpp"
#include <future>
#include <atomic>

#include "miner/IAlgorithm.hpp"

void print_welcome_message();

void run_telemetry_monitor() {
    // This function's only job is to run the monitor loop.
    // It should not handle program termination. If an error occurs here
    // that it cannot handle (like in getTelemetry), it should let the
    // exception propagate up to main.

    auto& manager = miner::device::DeviceManager::instance();

    fmt::print("--- Found {} CUDA Device(s) ---\n", manager.getDevices().size());

    for (const auto& device : manager.getDevices()) {
        const auto& info = device->getInfo();
        fmt::print("[GPU {}] Name: {}, Memory: {:.2f} GB\n",
                   info.id, info.name,
                   static_cast<double>(info.memory_total_bytes) / (1024 * 1024 * 1024));
    }
    fmt::print("----------------------------------\n\n");

    // Telemetry loop (runs until process exit)
    while (true) {
        fmt::print("--- Telemetry Update @ {} ---\n", std::chrono::system_clock::now());
        for (auto& device : manager.getDevices()) {
            try {
                const auto telemetry = device->getTelemetry();
                fmt::print(
                    "[GPU {}] Temp: {}C | Power: {}W | Fan: {}% | Core Clock: {} MHz | Mem Clock: {} MHz | Util: {}%\n",
                    device->getInfo().id, telemetry.temperature_c, telemetry.power_usage_watts,
                    telemetry.fan_speed, telemetry.sm_clock_mhz, telemetry.mem_clock_mhz,
                    telemetry.utilization_gpu);
            } catch (const miner::DeviceException& e) {
                // Handle non-critical errors locally if possible, e.g., log and continue.
                fmt::print(stderr, "[GPU {}] Error fetching telemetry: {}\n", device->getInfo().id, e.what());
            }
        }
        fmt::print("\n");
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

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
        auto& manager = miner::device::DeviceManager::instance();
        manager.initialize();
        // Create and connect Stratum client using CLI config
        auto client = miner::net::createStratumClient();

        // Promise/future to observe authentication result (set by connection callback)
        auto auth_promise = std::make_shared<std::promise<bool>>();
        auto auth_future = auth_promise->get_future();
        auto auth_done = std::make_shared<std::atomic<bool>>(false);

        client->onConnectionStateChanged([auth_promise, auth_done](miner::net::ConnectionState s) {
            const char* name = "Unknown";
            switch (s) {
                case miner::net::ConnectionState::Disconnected: name = "Disconnected"; break;
                case miner::net::ConnectionState::Connecting: name = "Connecting"; break;
                case miner::net::ConnectionState::Connected: name = "Connected"; break;
                case miner::net::ConnectionState::AuthenticationFailed: name = "AuthenticationFailed"; break;
                case miner::net::ConnectionState::Error: name = "Error"; break;
            }
            fmt::print("[Net] State: {}\n", name);

            // Only set promise once
            if (!auth_done->exchange(true)) {
                if (s == miner::net::ConnectionState::Connected) {
                    try { auth_promise->set_value(true); } catch(...) {}
                } else if (s == miner::net::ConnectionState::AuthenticationFailed || s == miner::net::ConnectionState::Error) {
                    try { auth_promise->set_value(false); } catch(...) {}
                }
            }
        });

        client->onNewJob([&manager](const miner::net::Job& job) {
            fmt::print("[Net] New job: id={} prev_hash={} ntime={} clean={}\n",
                       job.job_id, job.prev_hash, job.ntime, job.clean_jobs);
            // In a full implementation we'd dispatch the job to algorithm/device queues here.
        });

        // Parse host:port from config.url (expect format host:port)
        std::string host;
        uint16_t port = 3333; // default
        const auto pos = config.url.find(':');
        if (pos != std::string::npos) {
            host = config.url.substr(0, pos);
            try {
                port = static_cast<uint16_t>(std::stoi(config.url.substr(pos + 1)));
            } catch (...) {
                fmt::print(stderr, "Invalid port in URL: {}\n", config.url);
            }
        } else {
            host = config.url;
        }

        fmt::print("[Net] Connecting to {}:{}\n", host, port);
        client->connect(host, port, config.user, config.pass);

        // Wait for authentication/connect result (timeout configurable via MINER_CONN_TIMEOUT, default 30s)
        using namespace std::chrono_literals;
        int timeout_seconds = 30;
        const char* env_timeout = std::getenv("MINER_CONN_TIMEOUT");
        if (env_timeout) {
            try { timeout_seconds = std::stoi(env_timeout); } catch(...) { /* ignore */ }
        }
        auto status = auth_future.wait_for(std::chrono::seconds(timeout_seconds));
        if (status == std::future_status::ready) {
            bool ok = false;
            try { ok = auth_future.get(); } catch(...) { ok = false; }
            if (ok) {
                fmt::print(fg(fmt::color::green), "[Net] Connected and authenticated with pool.\n");
            } else {
                fmt::print(fg(fmt::color::red), "[Net] Authentication with pool failed.\n");
            }
        } else {
            fmt::print(fg(fmt::color::yellow), "[Net] Connection/auth timed out (no reply within 10s).\n");
        }

        run_telemetry_monitor();
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