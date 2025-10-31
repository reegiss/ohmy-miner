/*
 * Copyright (C) 2025 Regis Araujo Melo
 * This program is free software under the GPL-3.0 license. See LICENSE file.
 */

#include "ohmy/pool/work.hpp"
#include <fmt/format.h>
#include <thread>
#include <chrono>
#include <algorithm>

namespace ohmy {
namespace pool {

void WorkManager::add_job(const WorkPackage& work) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If clean_jobs is set, clear existing jobs
    if (work.clean_jobs) {
        std::queue<WorkPackage> empty;
        job_queue_.swap(empty);
        fmt::print("Cleared job queue due to clean_jobs flag\n");
    }
    
    // Validate job before adding
    if (!validate_job(work)) {
        fmt::print("Invalid job received, discarding: {}\n", work.job_id);
        return;
    }
    
    // Add to queue
    job_queue_.push(work);
    stats_.jobs_received++;
    
    fmt::print("Added job to queue: {} (queue size: {})\n", 
               work.job_id, job_queue_.size());
}

bool WorkManager::has_pending_jobs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !job_queue_.empty();
}

WorkPackage WorkManager::get_next_job() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (job_queue_.empty()) {
        throw std::runtime_error("No jobs available");
    }
    
    WorkPackage job = job_queue_.front();
    job_queue_.pop();
    stats_.jobs_processed++;
    
    // Set as current work
    current_work_ = job;
    has_current_work_ = true;
    
    fmt::print("Processing job: {} (remaining in queue: {})\n", 
               job.job_id, job_queue_.size());
    
    return job;
}

void WorkManager::clear_jobs() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<WorkPackage> empty;
    job_queue_.swap(empty);
    fmt::print("Manually cleared job queue\n");
}

size_t WorkManager::pending_job_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return job_queue_.size();
}

void WorkManager::set_current_work(const WorkPackage& work) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_work_ = work;
    has_current_work_ = true;
}

bool WorkManager::has_current_work() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_current_work_;
}

WorkPackage WorkManager::current_work() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_work_;
}

void WorkManager::submit_share(const ShareResult& share) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update statistics
    if (share.accepted) {
        stats_.shares_accepted++;
    } else {
        stats_.shares_rejected++;
    }

    // Queue share for submission
    pending_shares_.push(share);
}

std::vector<ShareResult> WorkManager::get_pending_shares() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<ShareResult> shares;
    while (!pending_shares_.empty()) {
        shares.push_back(pending_shares_.front());
        pending_shares_.pop();
    }
    
    return shares;
}

void WorkManager::track_share_result(const ShareResult& share) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update final statistics based on pool response
    if (share.accepted) {
        fmt::print("Share {} accepted by pool\n", share.nonce);
        stats_.shares_accepted++;
    } else {
        fmt::print("Share {} rejected by pool: {}\n", share.nonce, share.reason);
        stats_.shares_rejected++;
    }
}

void WorkManager::set_difficulty(double difficulty) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_difficulty_ = difficulty;
    stats_.current_difficulty = difficulty;
    fmt::print("Difficulty updated to: {}\n", difficulty);
}

double WorkManager::get_difficulty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_difficulty_;
}

WorkManager::Stats WorkManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats current_stats = stats_;
    current_stats.pending_jobs = job_queue_.size();
    return current_stats;
}

void WorkManager::update_hashrate(double hashrate) {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.hashrate = hashrate;
}

bool WorkManager::validate_job(const WorkPackage& work) const {
    // Basic validation for qhash jobs
    if (work.job_id.empty()) {
        return false;
    }
    
    if (work.previous_hash.empty() || work.previous_hash.length() != 64) {
        return false;
    }
    
    if (work.version.empty() || work.bits.empty() || work.time.empty()) {
        return false;
    }
    
    // All basic checks passed
    return true;
}

// JobDispatcher implementation
JobDispatcher::JobDispatcher(std::shared_ptr<WorkManager> work_manager)
    : work_manager_(work_manager) {
}

JobDispatcher::~JobDispatcher() {
    stop_dispatching();
}

void JobDispatcher::add_worker(std::shared_ptr<IWorker> worker) {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    workers_.push_back(worker);
    
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.active_workers = workers_.size();
    
    fmt::print("Added worker, total workers: {}\n", workers_.size());
}

void JobDispatcher::remove_worker(std::shared_ptr<IWorker> worker) {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    workers_.erase(
        std::remove(workers_.begin(), workers_.end(), worker),
        workers_.end()
    );
    
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.active_workers = workers_.size();
    
    fmt::print("Removed worker, total workers: {}\n", workers_.size());
}

void JobDispatcher::stop_all_workers() {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    for (auto& worker : workers_) {
        worker->stop_work();
    }
    fmt::print("Stopped all {} workers\n", workers_.size());
}

std::vector<std::shared_ptr<IWorker>> JobDispatcher::get_workers() const {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    return workers_;
}

void JobDispatcher::start_dispatching() {
    if (running_.load()) {
        return;
    }
    
    running_.store(true);
    dispatch_thread_ = std::thread(&JobDispatcher::dispatch_loop, this);
    fmt::print("Job dispatcher started\n");
}

void JobDispatcher::stop_dispatching() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    if (dispatch_thread_.joinable()) {
        dispatch_thread_.join();
    }
    fmt::print("Job dispatcher stopped\n");
}

bool JobDispatcher::is_running() const {
    return running_.load();
}

JobDispatcher::DispatcherStats JobDispatcher::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void JobDispatcher::dispatch_loop() {
    auto last_dispatch_time = std::chrono::steady_clock::now();
    size_t jobs_dispatched_in_period = 0;
    
    while (running_.load()) {
        // Check for pending jobs
        if (!work_manager_->has_pending_jobs()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Get available workers
        std::vector<std::shared_ptr<IWorker>> available_workers;
        {
            std::lock_guard<std::mutex> lock(workers_mutex_);
            available_workers = workers_;
        }
        
        if (available_workers.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Dispatch job to ALL available workers in parallel
        try {
            WorkPackage job = work_manager_->get_next_job();
            
            // Send same job to all workers (they'll work on different nonce ranges)
            for (const auto& worker : available_workers) {
                dispatch_job_to_worker(job, worker);
            }
            
            fmt::print("Dispatched job {} to {} workers in parallel\n", 
                      job.job_id, available_workers.size());
            
            jobs_dispatched_in_period++;
            
            // Update statistics
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.jobs_dispatched++;
            }
            
        } catch (const std::exception& e) {
            fmt::print("Error dispatching job: {}\n", e.what());
        }
        
        // Update jobs per second statistics every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_dispatch_time);
        if (elapsed.count() >= 1) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.jobs_per_second = static_cast<double>(jobs_dispatched_in_period) / elapsed.count();
            
            last_dispatch_time = now;
            jobs_dispatched_in_period = 0;
        }
        
        // Small delay to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void JobDispatcher::dispatch_job_to_worker(const WorkPackage& job, std::shared_ptr<IWorker> worker) {
    // Note: Multiple workers will receive same job and work on different nonce ranges
    worker->process_work(job);
}

} // namespace pool
} // namespace ohmy