/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/pool/monitor.hpp"
#include <fmt/format.h>
#include <fmt/color.h>

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
    fmt::print("Job monitor started (stats interval: {}s)\n", stats_interval_.count());
}

void JobMonitor::stop_monitoring() {
    if (!monitoring_.load()) {
        return;
    }
    
    monitoring_.store(false);
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    fmt::print("Job monitor stopped\n");
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
    
    fmt::print(fg(fmt::color::cyan), "\n=== Job Statistics ===\n");
    fmt::print("Jobs Received: {}\n", work_stats.jobs_received);
    fmt::print("Jobs Processed: {}\n", work_stats.jobs_processed);
    fmt::print("Jobs Pending: {}\n", work_stats.pending_jobs);
    fmt::print("Jobs Dispatched: {}\n", dispatcher_stats.jobs_dispatched);
    fmt::print("Active Workers: {}\n", dispatcher_stats.active_workers);
    fmt::print("Dispatch Rate: {:.2f} jobs/sec\n", dispatcher_stats.jobs_per_second);
}

void JobMonitor::print_performance_stats() const {
    auto work_stats = work_manager_->get_stats();
    
    fmt::print(fg(fmt::color::yellow), "\n=== Mining Statistics ===\n");
    fmt::print("Current Difficulty: {}\n", work_stats.current_difficulty);
    fmt::print("Hashrate: {:.2f} H/s\n", work_stats.hashrate);
    fmt::print("Shares Accepted: {} ", work_stats.shares_accepted);
    fmt::print(fg(fmt::color::green), "✓\n");
    fmt::print("Shares Rejected: {} ", work_stats.shares_rejected);
    
    if (work_stats.shares_rejected > 0) {
        fmt::print(fg(fmt::color::red), "✗\n");
    } else {
        fmt::print("✗\n");
    }
    
    // Calculate acceptance rate
    uint64_t total_shares = work_stats.shares_accepted + work_stats.shares_rejected;
    if (total_shares > 0) {
        double acceptance_rate = (double)work_stats.shares_accepted / total_shares * 100.0;
        fmt::print("Acceptance Rate: {:.1f}%\n", acceptance_rate);
    }
    
    fmt::print(fg(fmt::color::cyan), "========================\n\n");
}

} // namespace pool
} // namespace ohmy