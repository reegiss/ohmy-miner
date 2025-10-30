/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include "ohmy/pool/work.hpp"

namespace ohmy {
namespace pool {

/**
 * Monitors job processing and provides statistics
 */
class JobMonitor {
public:
    JobMonitor(std::shared_ptr<WorkManager> work_manager, 
               std::shared_ptr<JobDispatcher> dispatcher);
    ~JobMonitor();

    // Monitoring control
    void start_monitoring();
    void stop_monitoring();
    bool is_monitoring() const;

    // Statistics reporting
    void print_stats() const;
    void set_stats_interval(std::chrono::seconds interval);

private:
    void monitor_loop();
    void print_job_stats() const;
    void print_performance_stats() const;

    std::shared_ptr<WorkManager> work_manager_;
    std::shared_ptr<JobDispatcher> dispatcher_;
    
    std::atomic<bool> monitoring_{false};
    std::thread monitor_thread_;
    std::chrono::seconds stats_interval_{10}; // Print stats every 10 seconds
};

} // namespace pool
} // namespace ohmy