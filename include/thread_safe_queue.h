// Copyright (c) 2025 The GPU-Miner Authors. All rights reserved.
// Use of this source code is governed by a GPL-3.0-style license that can be
// found in the LICENSE file.

#ifndef THREAD_SAFE_QUEUE_H_
#define THREAD_SAFE_QUEUE_H_

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() : shutdown_(false) {}

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) return;
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty() || shutdown_) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // --- FIX: Add the missing clear() method ---
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        queue_.swap(empty);
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cond_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_;
    std::atomic<bool> shutdown_;
};

#endif // THREAD_SAFE_QUEUE_H_