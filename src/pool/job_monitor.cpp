/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/pool/monitor.hpp"
#include "ohmy/mining/fused_qhash_worker.hpp"
#include <fmt/format.h>
#include "ohmy/log.hpp"
#include <cuda_runtime.h>
#include <nvml.h>

namespace ohmy {
namespace pool {

JobMonitor::JobMonitor(std::shared_ptr<WorkManager> work_manager, 
                       std::shared_ptr<JobDispatcher> dispatcher)
    : work_manager_(work_manager)
    , dispatcher_(dispatcher) {
}

JobMonitor::~JobMonitor() {
    stop_monitoring();
}

void JobMonitor::start_monitoring() {
    if (monitoring_.load()) {
        return;
    }
    
    monitoring_.store(true);
    monitor_thread_ = std::thread(&JobMonitor::monitor_loop, this);
    ohmy::log::line("Job monitor started (stats interval: {}s)", stats_interval_.count());
}

void JobMonitor::stop_monitoring() {
    if (!monitoring_.load()) {
        return;
    }
    
    monitoring_.store(false);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    ohmy::log::line("Job monitor stopped");
}

bool JobMonitor::is_monitoring() const {
    return monitoring_.load();
}

void JobMonitor::print_stats() const {
    print_job_stats();
    print_performance_stats();
}

void JobMonitor::set_stats_interval(std::chrono::seconds interval) {
    stats_interval_ = interval;
}

void JobMonitor::monitor_loop() {
    while (monitoring_.load()) {
        print_stats();
        
        // Sleep for the stats interval
        for (int i = 0; i < stats_interval_.count() && monitoring_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void JobMonitor::print_job_stats() const {
    auto work_stats = work_manager_->get_stats();
    auto dispatcher_stats = dispatcher_->get_stats();
    
    // Compact, miner-style job line
    ohmy::log::line("Jobs rcvd: {}, processed: {}, pending: {}, workers: {}",
                    work_stats.jobs_received, work_stats.jobs_processed,
                    work_stats.pending_jobs, dispatcher_stats.active_workers);
}

void JobMonitor::print_performance_stats() const {
    auto work_stats = work_manager_->get_stats();
    
    // Aggregate hashrate from all workers
    double total_hashrate = 0.0;
    uint64_t total_hashes = 0;
    auto workers = dispatcher_->get_workers();
    
    for (const auto& worker : workers) {
        // Prefer fused kernel worker stats if available
        if (auto fused_worker = std::dynamic_pointer_cast<ohmy::mining::FusedQHashWorker>(worker)) {
            auto worker_stats = fused_worker->get_stats();
            total_hashrate += worker_stats.hashrate;
            total_hashes += worker_stats.hashes_computed;
            continue;
        }
        // (Legacy) Batched worker path retained for compatibility if present
        // Note: header not included; this clause intentionally omitted to avoid link-time deps
    }
    
    // GPU properties (best-effort)
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::string gpu_name = "GPU";
    if (device_count > 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            gpu_name = prop.name;
        }
    }

    // Format hashrate to string with unit
    auto hr_str = [&]() -> std::string {
        if (total_hashrate < 1000) return fmt::format("{:.2f} H/s", total_hashrate);
        if (total_hashrate < 1e6) return fmt::format("{:.2f} KH/s", total_hashrate/1e3);
        if (total_hashrate < 1e9) return fmt::format("{:.2f} MH/s", total_hashrate/1e6);
        return fmt::format("{:.2f} GH/s", total_hashrate/1e9);
    }();

    // Query NVML metrics (best-effort)
    int tempC = -1, fanPct = -1, pwrW = -1, gfxClk = -1, memClk = -1;
    static bool nvml_inited = false;
    static nvmlDevice_t nvml_dev;
    if (!nvml_inited) {
        if (nvmlInit_v2() == NVML_SUCCESS) {
            if (nvmlDeviceGetHandleByIndex(0, &nvml_dev) == NVML_SUCCESS) {
                nvml_inited = true;
            }
        }
    }
    if (nvml_inited) {
        unsigned int ui = 0;
        if (nvmlDeviceGetTemperature(nvml_dev, NVML_TEMPERATURE_GPU, &ui) == NVML_SUCCESS) tempC = static_cast<int>(ui);
        if (nvmlDeviceGetFanSpeed(nvml_dev, &ui) == NVML_SUCCESS) fanPct = static_cast<int>(ui);
        unsigned int pwrMw = 0;
        if (nvmlDeviceGetPowerUsage(nvml_dev, &pwrMw) == NVML_SUCCESS) pwrW = static_cast<int>(pwrMw / 1000);
        unsigned int clk = 0;
        if (nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_GRAPHICS, &clk) == NVML_SUCCESS) gfxClk = static_cast<int>(clk);
        if (nvmlDeviceGetClockInfo(nvml_dev, NVML_CLOCK_MEM, &clk) == NVML_SUCCESS) memClk = static_cast<int>(clk);
    }

    // Efficiency: KH/W for all hashrate levels (more intuitive for current perf)
    std::string eff = "-";
    if (pwrW > 0 && total_hashrate > 0.0) {
        double khs = total_hashrate / 1e3; // Convert to KH/s
        double effv = khs / pwrW;
        eff = fmt::format("{:.2f}", effv);
    }

    // Statistics table (wildrig-like)
    fmt::print("-------------------------------------[Statistics]-------------------------------------\n");
    fmt::print(" {:>2} {:<28} {:>12} {:>5} {:>5} {:>5} {:>5} {:>4} {:>5} {:>5} {:>4} {:>4}\n",
               "ID", "Name", "Hashrate", "Temp", "Fan", "Pwr", "Eff", "CClk", "MClk", "A", "R", "I");
    fmt::print("--------------------------------------------------------------------------------------\n");
    fmt::print(" #{:<1} {:<28} {:>12} {:>5} {:>5} {:>5} {:>5} {:>4} {:>5} {:>5} {:>4} {:>4}\n",
               0, gpu_name.substr(0, 28), hr_str,
               (tempC>=0?fmt::format("{}C", tempC):"-"),
               (fanPct>=0?fmt::format("{}%", fanPct):"-"),
               (pwrW>=0?fmt::format("{}W", pwrW):"-"),
               eff,
               (gfxClk>=0?fmt::format("{}", gfxClk):"-"),
               (memClk>=0?fmt::format("{}", memClk):"-"),
               work_stats.shares_accepted,
               work_stats.shares_rejected,
               0);
    fmt::print("--------------------------------------------------------------------------------------\n");
    fmt::print(" 10s: {:>28} {:>10} {:>14} {:>12}\n", hr_str, (pwrW>=0?fmt::format("Power: {}W", pwrW):"Power:"), "Accepted:", work_stats.shares_accepted);
    fmt::print(" 60s: {:>28} {:>10} {:>14} {:>12}\n", "n/a MH/s", "", "Rejected:", work_stats.shares_rejected);
    fmt::print(" 15m: {:>28} {:>10} {:>14} {:>12}\n", "n/a MH/s", "", "Ignored:", 0);
}

} // namespace pool
} // namespace ohmy