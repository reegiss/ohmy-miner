/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include "ohmy/pool/stratum.hpp"

namespace ohmy {
namespace pool {

/**
 * Manages current mining work and share submissions
 */
class WorkManager {
public:
    WorkManager() = default;
    ~WorkManager() = default;

    // Job queue management
    void add_job(const WorkPackage& work);
    bool has_pending_jobs() const;
    WorkPackage get_next_job();
    void clear_jobs();  // Called when clean_jobs flag is set
    size_t pending_job_count() const;

    // Current work handling
    void set_current_work(const WorkPackage& work);
    bool has_current_work() const;
    WorkPackage current_work() const;

    // Share handling
    void submit_share(const ShareResult& share);
    std::vector<ShareResult> get_pending_shares();
    void track_share_result(const ShareResult& share);  // Track pool responses

    // Difficulty management
    void set_difficulty(double difficulty);
    double get_difficulty() const;

    // Statistics
    struct Stats {
        uint64_t jobs_received = 0;
        uint64_t jobs_processed = 0;
        uint64_t shares_accepted = 0;
        uint64_t shares_rejected = 0;
        double current_difficulty = 0.0;
        double hashrate = 0.0;  // Hashes per second
        size_t pending_jobs = 0;
    };

    Stats get_stats() const;
    void update_hashrate(double hashrate);

    // Job validation
    bool validate_job(const WorkPackage& work) const;

private:
    mutable std::mutex mutex_;
    
    // Job queue
    std::queue<WorkPackage> job_queue_;
    WorkPackage current_work_;
    bool has_current_work_ = false;
    
    // Share queue
    std::queue<ShareResult> pending_shares_;
    
    // Mining state
    double current_difficulty_ = 0.0;
    Stats stats_;
};

/**
 * Interface for work consumers (miners)
 */
class IWorker {
public:
    virtual ~IWorker() = default;

    // Work processing
    virtual void process_work(const WorkPackage& work) = 0;
    virtual void stop_work() = 0;

    // Share submission callback
    virtual void set_share_callback(std::function<void(const ShareResult&)> callback) = 0;
};

/**
 * Job dispatcher manages work distribution to mining workers
 */
class JobDispatcher {
public:
    JobDispatcher(std::shared_ptr<WorkManager> work_manager);
    ~JobDispatcher();

    // Worker management
    void add_worker(std::shared_ptr<IWorker> worker);
    void remove_worker(std::shared_ptr<IWorker> worker);
    void stop_all_workers();
    std::vector<std::shared_ptr<IWorker>> get_workers() const;

    // Job dispatching
    void start_dispatching();
    void stop_dispatching();
    bool is_running() const;

    // Statistics
    struct DispatcherStats {
        size_t active_workers = 0;
        size_t jobs_dispatched = 0;
        double jobs_per_second = 0.0;
    };

    DispatcherStats get_stats() const;

private:
    void dispatch_loop();
    void dispatch_job_to_worker(const WorkPackage& job, std::shared_ptr<IWorker> worker);

    std::shared_ptr<WorkManager> work_manager_;
    std::vector<std::shared_ptr<IWorker>> workers_;
    std::atomic<bool> running_{false};
    std::thread dispatch_thread_;
    mutable std::mutex workers_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    DispatcherStats stats_;
};

} // namespace pool
} // namespace ohmy